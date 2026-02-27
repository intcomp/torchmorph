#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <vector>

#define CDT_BLOCK_SIZE 256
#define CDT_INF_VAL 1000000000
#define MAX_NDIM 16

// ============================================================================
// High-performance N-dimensional CDT using dimension-separable parallel scans
// For each dimension, we do forward and backward sweeps that can be parallelized
// across all other dimensions and batch elements.
// ============================================================================

// Initialize distance and indices
__global__ void cdt_init_kernel(
    const float* __restrict__ input,
    int32_t* __restrict__ dist,
    int32_t* __restrict__ indices,  // [spatial_ndim, total_elements]
    int64_t total_elements,
    int spatial_ndim,
    int64_t spatial_elements,
    const int64_t* __restrict__ spatial_strides,
    bool compute_indices
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    if (input[tid] == 0.0f) {
        dist[tid] = 0;
        if (compute_indices) {
            int64_t spatial_idx = tid % spatial_elements;
            int64_t rem = spatial_idx;
            for (int d = 0; d < spatial_ndim; d++) {
                int64_t coord = rem / spatial_strides[d];
                rem = rem % spatial_strides[d];
                indices[d * total_elements + tid] = (int32_t)coord;
            }
        }
    } else {
        dist[tid] = CDT_INF_VAL;
        if (compute_indices) {
            for (int d = 0; d < spatial_ndim; d++) {
                indices[d * total_elements + tid] = -1;
            }
        }
    }
}

// ============================================================================
// Dimension-wise sweep kernels for chessboard metric
// Each thread handles one "line" along the scan dimension
// ============================================================================

// Forward sweep along dimension d (from 0 to size-1)
// For chessboard: check neighbor at offset -1 in dimension d, and diagonal neighbors
__global__ void cdt_sweep_forward_chessboard_kernel(
    int32_t* __restrict__ dist,
    int32_t* __restrict__ indices,
    int64_t total_elements,
    int64_t num_lines,           // number of parallel lines
    int64_t line_stride,         // stride between elements in the same line
    int64_t line_length,         // number of elements in one line
    int64_t batch_stride,        // stride between batches
    int64_t spatial_elements,
    int spatial_ndim,
    int scan_dim,
    const int64_t* __restrict__ spatial_strides,
    const int64_t* __restrict__ spatial_shape,
    bool compute_indices
) {
    int64_t line_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (line_idx >= num_lines) return;

    // Compute starting position for this line
    int64_t batch_idx = line_idx / (spatial_elements / line_length);
    int64_t within_batch = line_idx % (spatial_elements / line_length);

    // Convert within_batch to actual spatial offset (excluding scan dimension)
    int64_t spatial_offset = 0;
    int64_t rem = within_batch;
    for (int d = 0; d < spatial_ndim; d++) {
        if (d == scan_dim) continue;
        int64_t dim_size = spatial_shape[d];
        int64_t coord = rem % dim_size;
        rem /= dim_size;
        spatial_offset += coord * spatial_strides[d];
    }

    int64_t base = batch_idx * spatial_elements + spatial_offset;

    // Forward sweep: i = 0 to line_length-1
    for (int64_t i = 1; i < line_length; i++) {
        int64_t curr_idx = base + i * line_stride;
        int32_t curr_dist = dist[curr_idx];

        if (curr_dist == 0) continue;

        // Check previous element in this dimension
        int64_t prev_idx = base + (i - 1) * line_stride;
        int32_t prev_dist = dist[prev_idx];

        if (prev_dist < CDT_INF_VAL) {
            int32_t new_dist = prev_dist + 1;
            if (new_dist < curr_dist) {
                dist[curr_idx] = new_dist;
                if (compute_indices) {
                    for (int d = 0; d < spatial_ndim; d++) {
                        indices[d * total_elements + curr_idx] = indices[d * total_elements + prev_idx];
                    }
                }
            }
        }
    }
}

// Backward sweep along dimension d (from size-1 to 0)
__global__ void cdt_sweep_backward_chessboard_kernel(
    int32_t* __restrict__ dist,
    int32_t* __restrict__ indices,
    int64_t total_elements,
    int64_t num_lines,
    int64_t line_stride,
    int64_t line_length,
    int64_t batch_stride,
    int64_t spatial_elements,
    int spatial_ndim,
    int scan_dim,
    const int64_t* __restrict__ spatial_strides,
    const int64_t* __restrict__ spatial_shape,
    bool compute_indices
) {
    int64_t line_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (line_idx >= num_lines) return;

    int64_t batch_idx = line_idx / (spatial_elements / line_length);
    int64_t within_batch = line_idx % (spatial_elements / line_length);

    int64_t spatial_offset = 0;
    int64_t rem = within_batch;
    for (int d = 0; d < spatial_ndim; d++) {
        if (d == scan_dim) continue;
        int64_t dim_size = spatial_shape[d];
        int64_t coord = rem % dim_size;
        rem /= dim_size;
        spatial_offset += coord * spatial_strides[d];
    }

    int64_t base = batch_idx * spatial_elements + spatial_offset;

    // Backward sweep: i = line_length-2 down to 0
    for (int64_t i = line_length - 2; i >= 0; i--) {
        int64_t curr_idx = base + i * line_stride;
        int32_t curr_dist = dist[curr_idx];

        if (curr_dist == 0) continue;

        int64_t next_idx = base + (i + 1) * line_stride;
        int32_t next_dist = dist[next_idx];

        if (next_dist < CDT_INF_VAL) {
            int32_t new_dist = next_dist + 1;
            if (new_dist < curr_dist) {
                dist[curr_idx] = new_dist;
                if (compute_indices) {
                    for (int d = 0; d < spatial_ndim; d++) {
                        indices[d * total_elements + curr_idx] = indices[d * total_elements + next_idx];
                    }
                }
            }
        }
    }
}

// ============================================================================
// Diagonal sweep kernels for chessboard metric (handles corner neighbors)
// ============================================================================

// Check all neighbors at distance 1 in chessboard metric
__global__ void cdt_diagonal_pass_kernel(
    int32_t* __restrict__ dist,
    int32_t* __restrict__ indices,
    int64_t total_elements,
    int64_t batch_size,
    int64_t spatial_elements,
    int spatial_ndim,
    const int64_t* __restrict__ spatial_strides,
    const int64_t* __restrict__ spatial_shape,
    const int32_t* __restrict__ offsets,
    int num_offsets,
    bool compute_indices,
    bool forward  // true for forward pass, false for backward
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    int32_t curr_dist = dist[tid];
    if (curr_dist == 0) return;

    int64_t batch_idx = tid / spatial_elements;
    int64_t spatial_idx = tid % spatial_elements;
    int64_t base = batch_idx * spatial_elements;

    // Compute current coordinates
    int32_t coords[MAX_NDIM];
    int64_t rem = spatial_idx;
    for (int d = 0; d < spatial_ndim; d++) {
        coords[d] = (int32_t)(rem / spatial_strides[d]);
        rem = rem % spatial_strides[d];
    }

    int32_t min_dist = curr_dist;
    int best_neighbor = -1;

    for (int n = 0; n < num_offsets; n++) {
        int64_t neighbor_spatial = spatial_idx + offsets[n];

        // Check bounds and no wrap-around
        if (neighbor_spatial < 0 || neighbor_spatial >= spatial_elements) continue;

        // Verify no wrap-around by checking coordinate differences
        int64_t n_rem = neighbor_spatial;
        bool valid = true;
        for (int d = 0; d < spatial_ndim; d++) {
            int32_t n_coord = (int32_t)(n_rem / spatial_strides[d]);
            n_rem = n_rem % spatial_strides[d];
            int32_t diff = coords[d] - n_coord;
            if (diff < -1 || diff > 1) {
                valid = false;
                break;
            }
        }
        if (!valid) continue;

        int64_t neighbor_idx = base + neighbor_spatial;
        int32_t neighbor_dist = dist[neighbor_idx];

        if (neighbor_dist < CDT_INF_VAL) {
            int32_t new_dist = neighbor_dist + 1;
            if (new_dist < min_dist) {
                min_dist = new_dist;
                best_neighbor = n;
            }
        }
    }

    if (min_dist < curr_dist) {
        dist[tid] = min_dist;
        if (compute_indices && best_neighbor >= 0) {
            int64_t src_idx = base + spatial_idx + offsets[best_neighbor];
            for (int d = 0; d < spatial_ndim; d++) {
                indices[d * total_elements + tid] = indices[d * total_elements + src_idx];
            }
        }
    }
}

// ============================================================================
// Taxicab metric uses simpler dimension-separable sweeps (no diagonals)
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor> cdt_cuda(
    torch::Tensor input,
    const std::string& metric,
    bool return_distances,
    bool return_indices
) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(metric == "chessboard" || metric == "taxicab",
                "metric must be 'chessboard' or 'taxicab'");
    TORCH_CHECK(return_distances || return_indices,
                "At least one of return_distances or return_indices must be True");

    input = input.contiguous();

    bool is_taxicab = (metric == "taxicab");
    int total_ndim = input.dim();

    TORCH_CHECK(total_ndim >= 3, "Input must be (B, C, Spatial...) format with at least 3 dimensions");

    auto shape_vec = input.sizes().vec();
    int64_t batch_size = shape_vec[0] * shape_vec[1];
    int spatial_ndim = total_ndim - 2;

    TORCH_CHECK(spatial_ndim >= 1 && spatial_ndim <= MAX_NDIM,
                "CDT supports 1D-" + std::to_string(MAX_NDIM) + "D spatial dimensions");

    std::vector<int64_t> spatial_shape(spatial_ndim);
    std::vector<int64_t> spatial_strides(spatial_ndim);

    int64_t spatial_elements = 1;
    for (int d = 0; d < spatial_ndim; d++) {
        spatial_shape[d] = shape_vec[d + 2];
        spatial_elements *= spatial_shape[d];
    }

    spatial_strides[spatial_ndim - 1] = 1;
    for (int d = spatial_ndim - 2; d >= 0; d--) {
        spatial_strides[d] = spatial_strides[d + 1] * spatial_shape[d + 1];
    }

    int64_t total_elements = input.numel();

    // Allocate output tensors
    auto dist = torch::empty({total_elements}, input.options().dtype(torch::kInt32));
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({spatial_ndim, total_elements}, input.options().dtype(torch::kInt32));
    }

    // Copy shape/strides to device
    auto spatial_shape_tensor = torch::tensor(spatial_shape, torch::TensorOptions().dtype(torch::kInt64).device(input.device()));
    auto spatial_strides_tensor = torch::tensor(spatial_strides, torch::TensorOptions().dtype(torch::kInt64).device(input.device()));

    // Initialize
    int block = CDT_BLOCK_SIZE;
    int grid = (total_elements + block - 1) / block;

    cdt_init_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        dist.data_ptr<int32_t>(),
        return_indices ? indices.data_ptr<int32_t>() : nullptr,
        total_elements,
        spatial_ndim,
        spatial_elements,
        spatial_strides_tensor.data_ptr<int64_t>(),
        return_indices
    );

    // For each dimension, do forward and backward sweeps
    for (int d = 0; d < spatial_ndim; d++) {
        int64_t line_length = spatial_shape[d];
        int64_t line_stride = spatial_strides[d];
        int64_t num_lines = batch_size * (spatial_elements / line_length);

        int sweep_block = CDT_BLOCK_SIZE;
        int sweep_grid = (num_lines + sweep_block - 1) / sweep_block;

        // Forward sweep
        cdt_sweep_forward_chessboard_kernel<<<sweep_grid, sweep_block>>>(
            dist.data_ptr<int32_t>(),
            return_indices ? indices.data_ptr<int32_t>() : nullptr,
            total_elements,
            num_lines,
            line_stride,
            line_length,
            spatial_elements,
            spatial_elements,
            spatial_ndim,
            d,
            spatial_strides_tensor.data_ptr<int64_t>(),
            spatial_shape_tensor.data_ptr<int64_t>(),
            return_indices
        );

        // Backward sweep
        cdt_sweep_backward_chessboard_kernel<<<sweep_grid, sweep_block>>>(
            dist.data_ptr<int32_t>(),
            return_indices ? indices.data_ptr<int32_t>() : nullptr,
            total_elements,
            num_lines,
            line_stride,
            line_length,
            spatial_elements,
            spatial_elements,
            spatial_ndim,
            d,
            spatial_strides_tensor.data_ptr<int64_t>(),
            spatial_shape_tensor.data_ptr<int64_t>(),
            return_indices
        );
    }

    // For chessboard metric, we need additional diagonal passes
    if (!is_taxicab && spatial_ndim >= 2) {
        // Generate diagonal offsets
        std::vector<int32_t> diagonal_offsets;

        // All neighbors in 3^ndim hypercube except axis-aligned ones
        int total_combos = 1;
        for (int d = 0; d < spatial_ndim; d++) total_combos *= 3;

        for (int i = 0; i < total_combos; i++) {
            int temp = i;
            int64_t offset = 0;
            int non_zero_count = 0;

            for (int d = 0; d < spatial_ndim; d++) {
                int dir = (temp % 3) - 1;
                temp /= 3;
                if (dir != 0) non_zero_count++;
                offset += dir * spatial_strides[d];
            }

            // Only include diagonal neighbors (more than one non-zero direction)
            if (non_zero_count >= 2) {
                diagonal_offsets.push_back((int32_t)offset);
            }
        }

        if (!diagonal_offsets.empty()) {
            auto offsets_tensor = torch::tensor(diagonal_offsets, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));

            // Multiple passes to propagate diagonal distances
            // Need more passes for higher dimensions to ensure full propagation
            int num_passes = spatial_ndim * 2;  // Scale with dimensions
            for (int pass = 0; pass < num_passes; pass++) {
                cdt_diagonal_pass_kernel<<<grid, block>>>(
                    dist.data_ptr<int32_t>(),
                    return_indices ? indices.data_ptr<int32_t>() : nullptr,
                    total_elements,
                    batch_size,
                    spatial_elements,
                    spatial_ndim,
                    spatial_strides_tensor.data_ptr<int64_t>(),
                    spatial_shape_tensor.data_ptr<int64_t>(),
                    offsets_tensor.data_ptr<int32_t>(),
                    diagonal_offsets.size(),
                    return_indices,
                    pass % 2 == 0
                );
            }
        }
    }

    // Prepare output
    torch::Tensor result_dist;
    torch::Tensor result_indices;

    if (return_distances) {
        result_dist = dist.to(torch::kFloat32).view(input.sizes());
    }

    if (return_indices) {
        std::vector<int64_t> idx_shape = {spatial_ndim};
        for (int d = 0; d < total_ndim; d++) {
            idx_shape.push_back(shape_vec[d]);
        }
        result_indices = indices.view(idx_shape);
    }

    return std::make_tuple(result_dist, result_indices);
}
