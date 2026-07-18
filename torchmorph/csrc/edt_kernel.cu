#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <tuple>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <algorithm>

// ==============================================================================
// Configuration
// ==============================================================================
#define INF_VAL 1e20f
#define MAX_THREADS 256
#define SHARED_MEM_LIMIT 2048  // Max dimension size for shared memory path (48KB limit)

// ==============================================================================
// 2D Optimized: Initialization kernel
// ==============================================================================
__global__ void init_distance_2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ distance,
    int* __restrict__ indices_y,
    int* __restrict__ indices_x,
    int height,
    int width,
    int64_t batch_stride,
    bool compute_indices
) {
    int64_t batch_idx = blockIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= height || x >= width) return;

    int64_t idx = batch_idx * batch_stride + y * width + x;

    float val = input[idx];
    distance[idx] = (val != 0.0f) ? INF_VAL : 0.0f;

    if (compute_indices) {
        indices_y[idx] = y;
        indices_x[idx] = x;
    }
}

// ==============================================================================
// 2D Optimized: Row-wise EDT (X direction) - contiguous access
// Each block processes one row
// ==============================================================================
__global__ void edt_2d_rows_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ input_idx_y,
    const int* __restrict__ input_idx_x,
    int* __restrict__ output_idx_y,
    int* __restrict__ output_idx_x,
    int height,
    int width,
    int64_t batch_stride,
    float spacing,
    bool compute_indices
) {
    // blockIdx.x = batch_idx * height + row_idx
    int64_t linear_idx = blockIdx.x;
    int row_idx = linear_idx % height;
    int64_t batch_idx = linear_idx / height;

    int64_t row_base = batch_idx * batch_stride + row_idx * width;

    extern __shared__ char shared_mem[];
    float* v_val = (float*)shared_mem;
    int* v_idx = (int*)(v_val + width);
    float* z = (float*)(v_idx + width);

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Load row into shared memory (contiguous access - optimal)
    for (int i = tid; i < width; i += num_threads) {
        v_val[i] = input[row_base + i];
    }
    __syncthreads();

    // Build lower envelope (thread 0 only)
    __shared__ int k_shared;

    if (tid == 0) {
        int k = -1;

        for (int q = 0; q < width; q++) {
            float fq = v_val[q];
            if (fq >= INF_VAL * 0.5f) continue;

            float q_pos = (float)q * spacing;
            float q_pos_sq = q_pos * q_pos;

            while (k >= 0) {
                int vk = v_idx[k];
                float vk_pos = (float)vk * spacing;
                float fvk = v_val[vk];
                float s = ((fq + q_pos_sq) - (fvk + vk_pos * vk_pos)) / (2.0f * (q_pos - vk_pos));

                if (s > z[k]) break;
                k--;
            }

            k++;
            v_idx[k] = q;

            if (k == 0) {
                z[0] = -INF_VAL;
            } else {
                int vk_prev = v_idx[k - 1];
                float vk_prev_pos = (float)vk_prev * spacing;
                float fvk_prev = v_val[vk_prev];
                z[k] = ((fq + q_pos_sq) - (fvk_prev + vk_prev_pos * vk_prev_pos)) /
                       (2.0f * (q_pos - vk_prev_pos));
            }
            z[k + 1] = INF_VAL;
        }
        k_shared = k;
    }
    __syncthreads();

    int k = k_shared;

    // Parallel fill with binary search
    for (int q = tid; q < width; q += num_threads) {
        int64_t out_idx = row_base + q;

        if (k < 0) {
            output[out_idx] = INF_VAL;
            if (compute_indices) {
                output_idx_y[out_idx] = row_idx;
                output_idx_x[out_idx] = 0;
            }
        } else {
            float q_pos = (float)q * spacing;

            // Binary search
            int lo = 0, hi = k;
            while (lo < hi) {
                int mid = (lo + hi + 1) / 2;
                if (z[mid] <= q_pos) lo = mid;
                else hi = mid - 1;
            }

            int nearest = v_idx[lo];
            float nearest_pos = (float)nearest * spacing;
            float diff = q_pos - nearest_pos;
            float dist_sq = diff * diff + v_val[nearest];

            output[out_idx] = dist_sq;  // Keep squared for next pass

            if (compute_indices) {
                output_idx_y[out_idx] = row_idx;  // Y unchanged in X pass
                output_idx_x[out_idx] = nearest;
            }
        }
    }
}

// ==============================================================================
// 2D Optimized: Column-wise EDT (Y direction) - strided access with shared memory
// Each block processes one column
// ==============================================================================
__global__ void edt_2d_cols_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ input_idx_y,
    const int* __restrict__ input_idx_x,
    int* __restrict__ output_idx_y,
    int* __restrict__ output_idx_x,
    int height,
    int width,
    int64_t batch_stride,
    float spacing,
    float spacing_x,
    bool is_final,
    bool compute_indices
) {
    // blockIdx.x = batch_idx * width + col_idx
    int64_t linear_idx = blockIdx.x;
    int col_idx = linear_idx % width;
    int64_t batch_idx = linear_idx / width;

    int64_t col_base = batch_idx * batch_stride + col_idx;
    int stride = width;  // Stride to next row

    extern __shared__ char shared_mem[];
    float* v_val = (float*)shared_mem;
    int* v_idx = (int*)(v_val + height);
    float* z = (float*)(v_idx + height);
    int* src_x = (int*)(z + height + 1);  // Store source X indices for index propagation

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Load column into shared memory (strided access - but only once)
    for (int i = tid; i < height; i += num_threads) {
        v_val[i] = input[col_base + i * stride];
        if (compute_indices) {
            src_x[i] = input_idx_x[col_base + i * stride];
        }
    }
    __syncthreads();

    // Build lower envelope (thread 0 only)
    __shared__ int k_shared;

    if (tid == 0) {
        int k = -1;

        for (int q = 0; q < height; q++) {
            float fq = v_val[q];
            if (fq >= INF_VAL * 0.5f) continue;

            float q_pos = (float)q * spacing;
            float q_pos_sq = q_pos * q_pos;

            while (k >= 0) {
                int vk = v_idx[k];
                float vk_pos = (float)vk * spacing;
                float fvk = v_val[vk];
                float s = ((fq + q_pos_sq) - (fvk + vk_pos * vk_pos)) / (2.0f * (q_pos - vk_pos));

                if (s > z[k]) break;
                k--;
            }

            k++;
            v_idx[k] = q;

            if (k == 0) {
                z[0] = -INF_VAL;
            } else {
                int vk_prev = v_idx[k - 1];
                float vk_prev_pos = (float)vk_prev * spacing;
                float fvk_prev = v_val[vk_prev];
                z[k] = ((fq + q_pos_sq) - (fvk_prev + vk_prev_pos * vk_prev_pos)) /
                       (2.0f * (q_pos - vk_prev_pos));
            }
            z[k + 1] = INF_VAL;
        }
        k_shared = k;
    }
    __syncthreads();

    int k = k_shared;

    // Parallel fill with binary search
    for (int q = tid; q < height; q += num_threads) {
        int64_t out_idx = col_base + q * stride;

        if (k < 0) {
            float delta_y = (float)(q + 1) * spacing;
            float delta_x = (float)col_idx * spacing_x;
            output[out_idx] = sqrtf(delta_y * delta_y + delta_x * delta_x);
            if (compute_indices) {
                output_idx_y[out_idx] = -1;
                output_idx_x[out_idx] = 0;
            }
        } else {
            float q_pos = (float)q * spacing;

            // Binary search
            int lo = 0, hi = k;
            while (lo < hi) {
                int mid = (lo + hi + 1) / 2;
                if (z[mid] <= q_pos) lo = mid;
                else hi = mid - 1;
            }

            int nearest = v_idx[lo];
            float nearest_pos = (float)nearest * spacing;
            float diff = q_pos - nearest_pos;
            float dist_sq = diff * diff + v_val[nearest];

            output[out_idx] = is_final ? sqrtf(dist_sq) : dist_sq;

            if (compute_indices) {
                output_idx_y[out_idx] = nearest;
                output_idx_x[out_idx] = src_x[nearest];  // Propagate X from source
            }
        }
    }
}

// ==============================================================================
// 2D Optimized: Host function (shared memory only, for dimensions <= 2048)
// ==============================================================================
std::tuple<torch::Tensor, torch::Tensor> run_edt_2d_optimized(
    torch::Tensor input,
    float spacing_y,
    float spacing_x,
    bool return_indices
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    input = input.contiguous();
    const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());

    int total_ndim = input.dim();
    TORCH_CHECK(total_ndim >= 2, "Input must have at least 2 dimensions");

    auto shape = input.sizes().vec();
    int height = shape[total_ndim - 2];
    int width = shape[total_ndim - 1];
    int64_t batch_stride = (int64_t)height * width;
    int64_t batch_size = input.numel() / batch_stride;

    // This function should only be called when both dimensions fit in shared memory
    TORCH_CHECK(height <= SHARED_MEM_LIMIT && width <= SHARED_MEM_LIMIT,
                "Dimensions too large for 2D optimized path, use general N-D version");

    // Create output tensors
    auto distance = torch::empty_like(input);
    auto temp = torch::empty_like(input);

    torch::Tensor indices_y, indices_x, temp_idx_y, temp_idx_x;
    if (return_indices) {
        indices_y = torch::empty_like(input, input.options().dtype(torch::kInt32));
        indices_x = torch::empty_like(input, input.options().dtype(torch::kInt32));
        temp_idx_y = torch::empty_like(indices_y);
        temp_idx_x = torch::empty_like(indices_x);
    }

    // Step 1: Initialize
    {
        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16, batch_size);

        init_distance_2d_kernel<<<grid, block, 0, stream>>>(
            input.data_ptr<float>(),
            distance.data_ptr<float>(),
            return_indices ? indices_y.data_ptr<int>() : nullptr,
            return_indices ? indices_x.data_ptr<int>() : nullptr,
            height, width, batch_stride,
            return_indices
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Step 2: Row-wise EDT (X direction) - shared memory
    {
        int64_t num_rows = batch_size * height;
        int threads = min(width, MAX_THREADS);
        size_t shared_mem_size = width * sizeof(float) +      // v_val
                                  width * sizeof(int) +        // v_idx
                                  (width + 1) * sizeof(float); // z

        edt_2d_rows_kernel<<<num_rows, threads, shared_mem_size, stream>>>(
            distance.data_ptr<float>(),
            temp.data_ptr<float>(),
            return_indices ? indices_y.data_ptr<int>() : nullptr,
            return_indices ? indices_x.data_ptr<int>() : nullptr,
            return_indices ? temp_idx_y.data_ptr<int>() : nullptr,
            return_indices ? temp_idx_x.data_ptr<int>() : nullptr,
            height, width, batch_stride,
            spacing_x,
            return_indices
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Step 3: Column-wise EDT (Y direction) - shared memory
    {
        int64_t num_cols = batch_size * width;
        int threads = min(height, MAX_THREADS);
        size_t shared_mem_size = height * sizeof(float) +       // v_val
                                  height * sizeof(int) +         // v_idx
                                  (height + 1) * sizeof(float);  // z
        if (return_indices) {
            shared_mem_size += height * sizeof(int);  // src_x
        }

        edt_2d_cols_kernel<<<num_cols, threads, shared_mem_size, stream>>>(
            temp.data_ptr<float>(),
            distance.data_ptr<float>(),
            return_indices ? temp_idx_y.data_ptr<int>() : nullptr,
            return_indices ? temp_idx_x.data_ptr<int>() : nullptr,
            return_indices ? indices_y.data_ptr<int>() : nullptr,
            return_indices ? indices_x.data_ptr<int>() : nullptr,
            height, width, batch_stride,
            spacing_y,
            spacing_x,
            true,  // is_final
            return_indices
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Combine indices into single tensor with shape [2, ...]
    torch::Tensor indices;
    if (return_indices) {
        std::vector<int64_t> idx_shape = {2};
        for (auto s : shape) idx_shape.push_back(s);
        indices = torch::empty(idx_shape, input.options().dtype(torch::kInt32));

        // Copy Y and X indices
        indices.select(0, 0).copy_(indices_y);
        indices.select(0, 1).copy_(indices_x);
    }

    return std::make_tuple(distance, indices);
}

__device__ __forceinline__ float virtual_background_distance(
    int64_t slice_idx,
    int64_t slices_per_sample,
    int q,
    int spatial_ndim,
    const int64_t* __restrict__ spatial_shape,
    const float* __restrict__ sampling
) {
    int64_t rem = slice_idx % slices_per_sample;
    float squared_distance = 0.0f;

    for (int d = spatial_ndim - 2; d >= 0; d--) {
        int64_t coordinate = rem % spatial_shape[d];
        rem /= spatial_shape[d];
        float shifted_coordinate = (d == 0) ? (float)(coordinate + 1) : (float)coordinate;
        float delta = shifted_coordinate * sampling[d];
        squared_distance += delta * delta;
    }

    float last_coordinate = (spatial_ndim == 1) ? (float)(q + 1) : (float)q;
    float last_delta = last_coordinate * sampling[spatial_ndim - 1];
    return sqrtf(squared_distance + last_delta * last_delta);
}

// ==============================================================================
// 1D EDT kernel using GLOBAL memory (for large dimensions)
// ==============================================================================
__global__ void edt_1d_global_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ input_idx,
    int* __restrict__ output_idx,
    float* __restrict__ g_v_val,
    int* __restrict__ g_v_idx,
    float* __restrict__ g_z,
    int* __restrict__ g_k,
    int64_t num_slices,
    int64_t slice_len,
    int64_t num_pixels,
    int spatial_ndim,
    int current_dim,
    float spacing,
    const int64_t* __restrict__ spatial_shape,
    const float* __restrict__ sampling,
    int64_t slices_per_sample,
    bool is_final,
    bool compute_indices
) {
    int64_t slice_idx = blockIdx.x;
    if (slice_idx >= num_slices) return;

    int64_t base_offset = slice_idx * slice_len;

    float* v_val = g_v_val + base_offset;
    int* v_idx = g_v_idx + base_offset;
    float* z = g_z + slice_idx * (slice_len + 1);
    int* k_ptr = g_k + slice_idx;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Load input values
    for (int i = tid; i < slice_len; i += num_threads) {
        v_val[i] = input[base_offset + i];
    }
    __syncthreads();

    // Build lower envelope (thread 0 only)
    if (tid == 0) {
        int k = -1;

        for (int q = 0; q < slice_len; q++) {
            float fq = v_val[q];
            if (fq >= INF_VAL * 0.5f) continue;

            float q_pos = (float)q * spacing;
            float q_pos_sq = q_pos * q_pos;

            while (k >= 0) {
                int vk = v_idx[k];
                float vk_pos = (float)vk * spacing;
                float fvk = v_val[vk];
                float s = ((fq + q_pos_sq) - (fvk + vk_pos * vk_pos)) / (2.0f * (q_pos - vk_pos));

                if (s > z[k]) break;
                k--;
            }

            k++;
            v_idx[k] = q;

            if (k == 0) {
                z[0] = -INF_VAL;
            } else {
                int vk_prev = v_idx[k - 1];
                float vk_prev_pos = (float)vk_prev * spacing;
                float fvk_prev = v_val[vk_prev];
                z[k] = ((fq + q_pos_sq) - (fvk_prev + vk_prev_pos * vk_prev_pos)) /
                       (2.0f * (q_pos - vk_prev_pos));
            }
            z[k + 1] = INF_VAL;
        }
        *k_ptr = k;
    }
    __syncthreads();

    int k = *k_ptr;

    // Parallel fill with binary search
    for (int q = tid; q < slice_len; q += num_threads) {
        int64_t out_idx = base_offset + q;

        if (k < 0) {
            output[out_idx] = is_final
                ? virtual_background_distance(
                      slice_idx,
                      slices_per_sample,
                      q,
                      spatial_ndim,
                      spatial_shape,
                      sampling)
                : INF_VAL;
            if (compute_indices) {
                for (int d = 0; d < spatial_ndim; d++) {
                    output_idx[d * num_pixels + out_idx] = (is_final && d == 0) ? -1 : 0;
                }
            }
        } else {
            float q_pos = (float)q * spacing;

            // Binary search
            int lo = 0, hi = k;
            while (lo < hi) {
                int mid = (lo + hi + 1) / 2;
                if (z[mid] <= q_pos) lo = mid;
                else hi = mid - 1;
            }

            int nearest = v_idx[lo];
            float nearest_pos = (float)nearest * spacing;
            float diff = q_pos - nearest_pos;
            float dist_sq = diff * diff + v_val[nearest];

            output[out_idx] = is_final ? sqrtf(dist_sq) : dist_sq;

            if (compute_indices) {
                int64_t src_idx = base_offset + nearest;
                for (int d = 0; d < spatial_ndim; d++) {
                    if (d == current_dim) {
                        output_idx[d * num_pixels + out_idx] = nearest;
                    } else {
                        output_idx[d * num_pixels + out_idx] = input_idx[d * num_pixels + src_idx];
                    }
                }
            }
        }
    }
}

// ==============================================================================
// 1D Euclidean Distance Transform (Felzenszwalb & Huttenlocher)
// ==============================================================================
__global__ void edt_1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ input_idx,
    int* __restrict__ output_idx,
    int64_t num_slices,
    int64_t slice_len,
    int64_t num_pixels,
    int spatial_ndim,
    int current_dim,
    float spacing,
    const int64_t* __restrict__ spatial_shape,
    const float* __restrict__ sampling,
    int64_t slices_per_sample,
    bool is_final,
    bool compute_indices
) {
    int64_t slice_idx = blockIdx.x;
    if (slice_idx >= num_slices) return;

    int64_t base_offset = slice_idx * slice_len;

    extern __shared__ char shared_mem[];
    float* v_val = (float*)shared_mem;
    int* v_idx = (int*)(v_val + slice_len);
    float* z = (float*)(v_idx + slice_len);

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Load input values into shared memory
    for (int i = tid; i < slice_len; i += num_threads) {
        v_val[i] = input[base_offset + i];
    }
    __syncthreads();

    // Build lower envelope (thread 0 only)
    __shared__ int k_shared;

    if (tid == 0) {
        int k = -1;

        for (int q = 0; q < slice_len; q++) {
            float fq = v_val[q];
            if (fq >= INF_VAL * 0.5f) continue;

            float q_pos = (float)q * spacing;
            float q_pos_sq = q_pos * q_pos;

            while (k >= 0) {
                int vk = v_idx[k];
                float vk_pos = (float)vk * spacing;
                float fvk = v_val[vk];
                float s = ((fq + q_pos_sq) - (fvk + vk_pos * vk_pos)) / (2.0f * (q_pos - vk_pos));

                if (s > z[k]) break;
                k--;
            }

            k++;
            v_idx[k] = q;

            if (k == 0) {
                z[0] = -INF_VAL;
            } else {
                int vk_prev = v_idx[k - 1];
                float vk_prev_pos = (float)vk_prev * spacing;
                float fvk_prev = v_val[vk_prev];
                z[k] = ((fq + q_pos_sq) - (fvk_prev + vk_prev_pos * vk_prev_pos)) /
                       (2.0f * (q_pos - vk_prev_pos));
            }
            z[k + 1] = INF_VAL;
        }
        k_shared = k;
    }
    __syncthreads();

    int k = k_shared;

    // Parallel fill
    for (int q = tid; q < slice_len; q += num_threads) {
        int64_t out_idx = base_offset + q;

        if (k < 0) {
            output[out_idx] = is_final
                ? virtual_background_distance(
                      slice_idx,
                      slices_per_sample,
                      q,
                      spatial_ndim,
                      spatial_shape,
                      sampling)
                : INF_VAL;
            if (compute_indices) {
                for (int d = 0; d < spatial_ndim; d++) {
                    output_idx[d * num_pixels + out_idx] = (is_final && d == 0) ? -1 : 0;
                }
            }
        } else {
            float q_pos = (float)q * spacing;

            // Binary search
            int lo = 0, hi = k;
            while (lo < hi) {
                int mid = (lo + hi + 1) / 2;
                if (z[mid] <= q_pos) lo = mid;
                else hi = mid - 1;
            }

            int nearest = v_idx[lo];
            float nearest_pos = (float)nearest * spacing;
            float diff = q_pos - nearest_pos;
            float dist_sq = diff * diff + v_val[nearest];

            output[out_idx] = is_final ? sqrtf(dist_sq) : dist_sq;

            if (compute_indices) {
                int64_t src_idx = base_offset + nearest;
                for (int d = 0; d < spatial_ndim; d++) {
                    if (d == current_dim) {
                        output_idx[d * num_pixels + out_idx] = nearest;
                    } else {
                        output_idx[d * num_pixels + out_idx] = input_idx[d * num_pixels + src_idx];
                    }
                }
            }
        }
    }
}

// ==============================================================================
// Initialization kernel: set up initial distances and indices
// ==============================================================================
__global__ void init_distance_kernel(
    const float* __restrict__ input,
    float* __restrict__ distance,
    int* __restrict__ indices,
    int64_t total_pixels,
    int total_ndim,
    int spatial_ndim,
    const int64_t* __restrict__ shape,
    bool compute_indices
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    // Set distance: 0 for background (input == 0), INF for foreground (input != 0)
    float val = input[idx];
    distance[idx] = (val != 0.0f) ? INF_VAL : 0.0f;

    // Initialize indices to current coordinates
    if (compute_indices) {
        int64_t temp = idx;
        int coords[16];  // Support up to 16D

        // Compute coordinates from linear index
        for (int d = total_ndim - 1; d >= 0; d--) {
            int64_t dim_size = shape[d];
            coords[d] = temp % dim_size;
            temp /= dim_size;
        }

        // Store spatial coordinates
        int start_dim = total_ndim - spatial_ndim;
        for (int s = 0; s < spatial_ndim; s++) {
            indices[s * total_pixels + idx] = coords[start_dim + s];
        }
    }
}

// ==============================================================================
// Host function to run separable EDT
// ==============================================================================
std::tuple<torch::Tensor, torch::Tensor> run_edt_separable(
    torch::Tensor input,
    const std::vector<float>& sampling,
    bool return_indices
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    input = input.contiguous();
    const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());

    int total_ndim = input.dim();
    int spatial_ndim = sampling.size();
    int start_dim = total_ndim - spatial_ndim;

    auto shape = input.sizes().vec();
    int64_t total_pixels = input.numel();

    // Create output tensors
    auto distance = torch::empty_like(input);
    torch::Tensor indices;
    if (return_indices) {
        std::vector<int64_t> idx_shape = {spatial_ndim};
        for (auto s : shape) idx_shape.push_back(s);
        indices = torch::empty(idx_shape, input.options().dtype(torch::kInt32));
    }

    // Copy shape to device
    auto shape_tensor = torch::tensor(std::vector<int64_t>(shape.begin(), shape.end()),
                                       torch::TensorOptions().dtype(torch::kInt64).device(input.device()));
    auto sampling_tensor = torch::tensor(sampling, input.options());
    int64_t batch_size = shape[0] * shape[1];

    // Initialize distances and indices
    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;

    init_distance_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(),
        distance.data_ptr<float>(),
        return_indices ? indices.data_ptr<int>() : nullptr,
        total_pixels,
        total_ndim,
        spatial_ndim,
        shape_tensor.data_ptr<int64_t>(),
        return_indices
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Global memory buffers (allocated lazily for large dimensions)
    torch::Tensor g_v_val, g_v_idx, g_z, g_k;

    // Process each spatial dimension
    for (int dim_idx = 0; dim_idx < spatial_ndim; dim_idx++) {
        int actual_dim = start_dim + dim_idx;
        bool is_final = (dim_idx == spatial_ndim - 1);
        float spacing = sampling[dim_idx];

        // Transpose to make current dimension last
        auto dist_transposed = distance.transpose(actual_dim, total_ndim - 1).contiguous();
        auto dist_out = torch::empty_like(dist_transposed);

        torch::Tensor idx_transposed, idx_out;
        if (return_indices) {
            // Indices have an extra leading dimension
            idx_transposed = indices.transpose(actual_dim + 1, total_ndim).contiguous();
            idx_out = torch::empty_like(idx_transposed);
        }

        // Get dimensions after transpose
        int64_t slice_len = dist_transposed.size(-1);
        int64_t num_slices = dist_transposed.numel() / slice_len;

        int kernel_threads = min((int)slice_len, MAX_THREADS);

        // Choose between shared memory and global memory kernel
        bool use_shared = (slice_len <= SHARED_MEM_LIMIT);

        if (use_shared) {
            // Calculate shared memory size
            size_t shared_mem_size = slice_len * sizeof(float) +  // v_val
                                      slice_len * sizeof(int) +    // v_idx
                                      (slice_len + 1) * sizeof(float);  // z

            edt_1d_kernel<<<num_slices, kernel_threads, shared_mem_size, stream>>>(
                dist_transposed.data_ptr<float>(),
                dist_out.data_ptr<float>(),
                return_indices ? idx_transposed.data_ptr<int>() : nullptr,
                return_indices ? idx_out.data_ptr<int>() : nullptr,
                num_slices,
                slice_len,
                dist_transposed.numel(),
                spatial_ndim,
                dim_idx,
                spacing,
                shape_tensor.data_ptr<int64_t>() + start_dim,
                sampling_tensor.data_ptr<float>(),
                num_slices / batch_size,
                is_final,
                return_indices
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            // Allocate global memory buffers if needed
            int64_t total_elements = dist_transposed.numel();
            if (!g_v_val.defined() || g_v_val.numel() < total_elements) {
                g_v_val = torch::empty({total_elements}, dist_transposed.options());
                g_v_idx = torch::empty({total_elements}, dist_transposed.options().dtype(torch::kInt32));
            }
            if (!g_z.defined() || g_z.numel() < num_slices * (slice_len + 1)) {
                g_z = torch::empty({num_slices * (slice_len + 1)}, dist_transposed.options());
            }
            if (!g_k.defined() || g_k.numel() < num_slices) {
                g_k = torch::empty({num_slices}, dist_transposed.options().dtype(torch::kInt32));
            }

            edt_1d_global_kernel<<<num_slices, kernel_threads, 0, stream>>>(
                dist_transposed.data_ptr<float>(),
                dist_out.data_ptr<float>(),
                return_indices ? idx_transposed.data_ptr<int>() : nullptr,
                return_indices ? idx_out.data_ptr<int>() : nullptr,
                g_v_val.data_ptr<float>(),
                g_v_idx.data_ptr<int>(),
                g_z.data_ptr<float>(),
                g_k.data_ptr<int>(),
                num_slices,
                slice_len,
                dist_transposed.numel(),
                spatial_ndim,
                dim_idx,
                spacing,
                shape_tensor.data_ptr<int64_t>() + start_dim,
                sampling_tensor.data_ptr<float>(),
                num_slices / batch_size,
                is_final,
                return_indices
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        // Transpose back
        distance = dist_out.transpose(actual_dim, total_ndim - 1);
        if (return_indices) {
            indices = idx_out.transpose(actual_dim + 1, total_ndim);
        }
    }

    return std::make_tuple(distance.contiguous(), return_indices ? indices.contiguous() : torch::Tensor());
}

// ==============================================================================
// Python binding entry point
// ==============================================================================

std::tuple<torch::Tensor, torch::Tensor> edt_cuda(
    torch::Tensor input,
    std::vector<float> sampling,
    bool return_distances,
    bool return_indices
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(input.device());

    int total_ndim = input.dim();

    // Handle empty sampling (default to unit spacing for all spatial dimensions)
    if (sampling.empty()) {
        // Assume all dimensions are spatial if no sampling provided
        // But typically input is (B, C, spatial...) so use total_ndim - 2
        int spatial_ndim = total_ndim >= 3 ? total_ndim - 2 : total_ndim;
        sampling.resize(spatial_ndim, 1.0f);
    }

    int spatial_ndim = sampling.size();

    // Use exact (Felzenszwalb) algorithm
    // Use 2D optimized path only when both dimensions fit in shared memory
    // For larger dimensions, the N-D general version with transpose is faster
    if (spatial_ndim == 2) {
        auto shape = input.sizes().vec();
        int height = shape[total_ndim - 2];
        int width = shape[total_ndim - 1];

        // Only use 2D optimized path when shared memory can be used for both directions
        if (height <= SHARED_MEM_LIMIT && width <= SHARED_MEM_LIMIT) {
            float spacing_y = sampling[0];
            float spacing_x = sampling[1];

            auto [distances, indices_result] = run_edt_2d_optimized(
                input, spacing_y, spacing_x, return_indices
            );

            if (!return_indices) {
                indices_result = torch::Tensor();
            }

            return std::make_tuple(distances, indices_result);
        }
    }

    // Fall back to general N-D implementation
    auto [distances, indices_result] = run_edt_separable(input, sampling, return_indices);

    if (!return_indices) {
        indices_result = torch::Tensor();
    }

    return std::make_tuple(distances, indices_result);
}
