#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

// ------------------------------------------------------------------
// Configuration Constants
// ------------------------------------------------------------------
#define INF_VAL 1e8f
#define MAX_THREADS 1024
// Shared memory limit: typically 48 KB.
// Each pixel requires: float(value) + int(idx1) + int(idx2) = 12 bytes.
// 4096 * 12 = 48 KB.
#define SMEM_LIMIT_ELEMENTS 4096 

// ------------------------------------------------------------------
// Device Helper Functions
// ------------------------------------------------------------------

__device__ __forceinline__ float sqr(float x) { return x * x; }

// Compute the JFA cost: (q - p)^2 + weight[p]
__device__ __forceinline__ float compute_cost(int q, int p, float val_p) {
    if (p < 0) return INF_VAL; 
    return sqr((float)q - (float)p) + val_p;
}

// ------------------------------------------------------------------
// JFA Core Logic (Device Only)
// ------------------------------------------------------------------
// Core JFA logic, independent of data location (works with both Shared and Global memory).
__device__ void run_jfa_core(
    int N,
    int tid,
    const float* __restrict__ vals,  // input weight (read-only)
    int* __restrict__ idx_curr,      // Ping-Pong Buffer A
    int* __restrict__ idx_next       // Ping-Pong Buffer B
) {
    // 1. Initialization: determine whether each pixel is a valid source based on vals.
    for (int i = tid; i < N; i += blockDim.x) {
        if (vals[i] >= INF_VAL * 0.9f) {
            idx_curr[i] = -1; // background
        } else {
            idx_curr[i] = i;  // For each object/source point, the initial index points to itself.
        }
    }
    __syncthreads();

    // 2. Iterative Propagation (Step = 1, 2, 4, ... < N)
    int* idx_in = idx_curr;
    int* idx_out = idx_next;

    for (int step = 1; step < N; step *= 2) {
        for (int i = tid; i < N; i += blockDim.x) {
            int my_best_p = idx_in[i];
            float min_cost = INF_VAL;

            // Check its current best solution
            if (my_best_p != -1) {
                min_cost = compute_cost(i, my_best_p, vals[my_best_p]);
            }

            // Check Left Neighbor (-step)
            int left = i - step;
            if (left >= 0) {
                int left_p = idx_in[left];
                if (left_p != -1) {
                    float c = compute_cost(i, left_p, vals[left_p]);
                    if (c < min_cost) {
                        min_cost = c;
                        my_best_p = left_p;
                    }
                }
            }

            // Check Right Neighbor (+step)
            int right = i + step;
            if (right < N) {
                int right_p = idx_in[right];
                if (right_p != -1) {
                    float c = compute_cost(i, right_p, vals[right_p]);
                    if (c < min_cost) {
                        min_cost = c;
                        my_best_p = right_p;
                    }
                }
            }
            idx_out[i] = my_best_p;
        }
        
        // Swap Pointers
        int* temp = idx_in;
        idx_in = idx_out;
        idx_out = temp;
        __syncthreads();
    }

    // 3. Ensure the final result is stored in idx_curr (if the loop ends with idx_next, copy it back).
    if (idx_in != idx_curr) {
        for (int i = tid; i < N; i += blockDim.x) {
            idx_curr[i] = idx_next[i];
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------------
// Kernel 1: Shared Memory JFA (Fast Path)
// ------------------------------------------------------------------
// Template parameter NDim: when NDim > 0, the compiler performs loop unrolling optimizations.
// Runtime parameter runtime_ndim: when NDim == 0 (default behavior), this parameter specifies the dimension.
template <bool IsFinal, int NDim>
__global__ void edt_kernel_shared(
    const float* __restrict__ in_data,       // input Dist^2
    const int32_t* __restrict__ in_indices,  // output Indices
    float* __restrict__ out_dist,            // output Dist (IsFinal ? sqrt : sqr)
    int32_t* __restrict__ out_indices,       // output Indices
    int64_t L,               // Size of the current dimension
    int64_t total_elements,  // Total number of elements
    int runtime_ndim         // Runtime dimension (used as fallback)
) {
    // Determine the effective dimension
    const int D = (NDim > 0) ? NDim : runtime_ndim;

    // Compute row offset
    int64_t row_idx = blockIdx.x;
    int64_t offset = row_idx * L;

    if (offset >= total_elements) return;

    // Shared memory layout
    extern __shared__ char s_buffer[];
    float* s_vals = (float*)s_buffer;
    int*   s_idx1 = (int*)(s_vals + L);
    int*   s_idx2 = (int*)(s_idx1 + L);

    // 1. Load distances into Shared Memory
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        s_vals[i] = __ldg(&in_data[offset + i]);
    }
    __syncthreads();

    // 2. Run the core JFA logic
    run_jfa_core(L, threadIdx.x, s_vals, s_idx1, s_idx2);

    // 3. Write back the results
    for (int q = threadIdx.x; q < L; q += blockDim.x) {
        int p = s_idx1[q];  // Nearest point (local index within 0..L-1)
        float dist_val;

        // Compute updated distance
        if (p != -1) {
            float dist_sq = sqr((float)q - (float)p) + s_vals[p];
            dist_val = IsFinal ? sqrtf(dist_sq) : dist_sq;
        } else {
            // No source point found (e.g., entire row is background)
            dist_val = IsFinal ? INF_VAL : sqr(INF_VAL);
            p = 0;  // Prevent out-of-bounds access
        }
        out_dist[offset + q] = dist_val;

        // Propagate indices: copy a vector of size [D]
        if (p != -1) {
            int64_t src_offset = (offset + p) * D;
            int64_t dst_offset = (offset + q) * D;

            // When NDim > 0, this loop is fully unrolled by the compiler
            for (int d = 0; d < D; ++d) {
                out_indices[dst_offset + d] = in_indices[src_offset + d];
            }
        } else {
            // Fallback: no source available
            int64_t dst_offset = (offset + q) * D;
            for (int d = 0; d < D; ++d) out_indices[dst_offset + d] = 0;
        }
    }
}

// ------------------------------------------------------------------
// Kernel 2: Global Memory JFA (Fallback Path)
// ------------------------------------------------------------------
// Same logic as above, but uses Global Memory as the ping-pong buffer
template <bool IsFinal, int NDim>
__global__ void edt_kernel_global(
    const float* __restrict__ in_data,
    const int32_t* __restrict__ in_indices,
    float* __restrict__ out_dist,
    int32_t* __restrict__ out_indices,
    int* __restrict__ global_buffer_1,
    int* __restrict__ global_buffer_2,
    int64_t L,
    int64_t total_elements,
    int runtime_ndim
) {
    const int D = (NDim > 0) ? NDim : runtime_ndim;

    int64_t row_idx = blockIdx.x;
    int64_t offset = row_idx * L;
    
    if (offset >= total_elements) return;

    // Pointers to Global Memory
    int* g_idx1 = global_buffer_1 + offset;
    int* g_idx2 = global_buffer_2 + offset;
    
    // 1. & 2. Run the JFA core (operating directly on Global Memory)
    run_jfa_core(L, threadIdx.x, in_data + offset, g_idx1, g_idx2);

    // 3. Write back results
    for (int q = threadIdx.x; q < L; q += blockDim.x) {
        int p = g_idx1[q];
        float dist_val;

        if (p != -1) {
            float val_p = in_data[offset + p]; 
            float dist_sq = sqr((float)q - (float)p) + val_p;
            dist_val = IsFinal ? sqrtf(dist_sq) : dist_sq;
        } else {
            dist_val = IsFinal ? INF_VAL : sqr(INF_VAL);
            p = 0; 
        }

        out_dist[offset + q] = dist_val;

        if (p != -1) {
            int64_t src_offset = (offset + p) * D;
            int64_t dst_offset = (offset + q) * D;
            for (int d = 0; d < D; ++d) {
                out_indices[dst_offset + d] = in_indices[src_offset + d];
            }
        } else {
             int64_t dst_offset = (offset + q) * D;
             for (int d = 0; d < D; ++d) out_indices[dst_offset + d] = 0;
        }
    }
}


// ------------------------------------------------------------------
// Kernel 3: Initialize Indices
// ------------------------------------------------------------------
// Initialize index tensor as grid coordinates
// indices shape: (..., D)
__global__ void init_indices_kernel(
    int32_t* indices, 
    int64_t total_pixels, 
    int NDim, 
    const int64_t* __restrict__ shape_ptr // shape of spatial dimensions
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    // Unravel Index
    // idx is the flat index of each pixel
    // We need to compute its coordinate in spatial_shape
    
    int64_t temp = idx;
    // Use local register array to avoid repeated global memory reads (assume max 10 dims)
    int32_t coords[10]; 

    // Example: spatial_shape = [D0, D1, D2]
    // compute by modulo from last dimension
    for (int d = NDim - 1; d >= 0; --d) {
        int64_t dim_size = shape_ptr[d];
        coords[d] = temp % dim_size;
        temp /= dim_size;
    }

    // Write to Global Memory
    // Indices tensor is flattened as (TotalPixels, NDim)
    int64_t out_ptr = idx * NDim;
    for (int d = 0; d < NDim; ++d) {
        indices[out_ptr + d] = coords[d];
    }
}

// ------------------------------------------------------------------
// Host Function: C++ Entry Point
// ------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> distance_transform_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA device.");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32.");
    
    input = input.contiguous();
    
    // Handle batch dimension: if input is 1D (L), treat as no batch but internally add a batch dimension.
    // Convention: input shape is (Batch, D1, D2, ..., Dn)
    // Algorithm treats batch and other dims identically (batch is just another leading dimension)
    // But index initialization needs to know which are "spatial dimensions".
    // Assumption: all dims except dim 0 (Batch) are spatial.
    
    const int ndim = input.dim(); 
    // If ndim=1, assume (L) → sample_ndim=1
    // If ndim=4 (B, C, H, W), sample_ndim=3 (C,H,W treated as spatial? Channels often processed independently)
    // Correction: classical EDT usually runs on (H,W) or (D,H,W).
    // If channels exist, typically each channel is processed independently.
    // For maximum generality, we treat **all dims except dim 0** as spatial dims.
    // If input has no batch dim, user should use unsqueeze(0) in Python.
    
    const int sample_ndim = ndim - 1; 
    TORCH_CHECK(sample_ndim > 0, "Input tensor must have at least 2 dimensions (Batch, ...)");
    
    auto shape = input.sizes().vec();
    int64_t num_pixels = input.numel();

    if (num_pixels == 0) {
        auto index_shape = shape;
        index_shape.push_back(sample_ndim);
        return std::make_tuple(torch::empty_like(input), 
                               torch::empty(index_shape, input.options().dtype(torch::kInt32)));
    }

    // 1. Initialize Distance Tensor
    // 0 -> 0, 1 -> INF
    auto current_dist = torch::where(input == 0, 
                                     torch::tensor(0.0f, input.options()), 
                                     torch::tensor(INF_VAL, input.options()));
    
    // 2. Initialize Index Tensor
    // Shape: (Batch, D1, ..., Dn, sample_ndim)
    auto index_shape = shape;
    index_shape.push_back(sample_ndim);
    auto current_idx = torch::empty(index_shape, input.options().dtype(torch::kInt32));
    
    // 2.1 Prepare shape tensor for kernel
    std::vector<int64_t> spatial_shape(shape.begin() + 1, shape.end());
    auto shape_tensor = torch::tensor(spatial_shape, torch::kInt64).to(input.device());

    // 2.2 Launch initialization kernel
    {
        int threads = 256;
        int blocks = (num_pixels + threads - 1) / threads;
        init_indices_kernel<<<blocks, threads>>>(
            current_idx.data_ptr<int32_t>(),
            num_pixels,
            sample_ndim,
            shape_tensor.data_ptr<int64_t>()
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Init Kernel Failed: %s\n", cudaGetErrorString(err));
        }
    }

    // Pre-allocate Global Memory Buffers (lazy)
    torch::Tensor global_buf1, global_buf2;

    // 3. Process each spatial dimension (Separable JFA)
    // Iterate through each spatial dimension (1 to ndim-1)
    for (int d = 1; d < ndim; ++d) {
        bool is_final_pass = (d == ndim - 1);
        
        // --- Step A: Transpose current dim to last ---
        // Resulting shape: (..., L)
        auto dist_in = current_dist.transpose(d, ndim - 1).contiguous();
        auto idx_in  = current_idx.transpose(d, ndim - 1).contiguous(); 
        
        int64_t L = dist_in.size(-1); 
        int64_t total_slices = dist_in.numel() / L; 
        
        auto dist_out = torch::empty_like(dist_in);
        auto idx_out  = torch::empty_like(idx_in);

        // --- Step B: Kernel Dispatch ---
        int threads = std::min((int64_t)MAX_THREADS, L);
        
        // Check whether Shared Memory can be used
        if (L <= SMEM_LIMIT_ELEMENTS) {
            size_t smem_size = L * (sizeof(float) + 2 * sizeof(int));
            
            // Switch macro to handle template dimension specialization
            #define DISPATCH_SHARED(IS_FINAL) \
                switch(sample_ndim) { \
                    case 1: edt_kernel_shared<IS_FINAL, 1><<<total_slices, threads, smem_size>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        L, dist_in.numel(), sample_ndim); break; \
                    case 2: edt_kernel_shared<IS_FINAL, 2><<<total_slices, threads, smem_size>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        L, dist_in.numel(), sample_ndim); break; \
                    case 3: edt_kernel_shared<IS_FINAL, 3><<<total_slices, threads, smem_size>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        L, dist_in.numel(), sample_ndim); break; \
                    case 4: edt_kernel_shared<IS_FINAL, 4><<<total_slices, threads, smem_size>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        L, dist_in.numel(), sample_ndim); break; \
                    case 5: edt_kernel_shared<IS_FINAL, 5><<<total_slices, threads, smem_size>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        L, dist_in.numel(), sample_ndim); break; \
                    case 6: edt_kernel_shared<IS_FINAL, 6><<<total_slices, threads, smem_size>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        L, dist_in.numel(), sample_ndim); break; \
                    default: /* Fallback for > 6D */ \
                        edt_kernel_shared<IS_FINAL, 0><<<total_slices, threads, smem_size>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        L, dist_in.numel(), sample_ndim); break; \
                }

            if (is_final_pass) { DISPATCH_SHARED(true); } 
            else { DISPATCH_SHARED(false); }

        } else {
            // Global Memory fallback (L > 4096)
            if (global_buf1.numel() < dist_in.numel()) {
                global_buf1 = torch::empty({dist_in.numel()}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
                global_buf2 = torch::empty({dist_in.numel()}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
            }

            #define DISPATCH_GLOBAL(IS_FINAL) \
                switch(sample_ndim) { \
                    case 1: edt_kernel_global<IS_FINAL, 1><<<total_slices, threads>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        global_buf1.data_ptr<int>(), global_buf2.data_ptr<int>(), \
                        L, dist_in.numel(), sample_ndim); break; \
                    case 2: edt_kernel_global<IS_FINAL, 2><<<total_slices, threads>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        global_buf1.data_ptr<int>(), global_buf2.data_ptr<int>(), \
                        L, dist_in.numel(), sample_ndim); break; \
                    case 3: edt_kernel_global<IS_FINAL, 3><<<total_slices, threads>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        global_buf1.data_ptr<int>(), global_buf2.data_ptr<int>(), \
                        L, dist_in.numel(), sample_ndim); break; \
                    case 4: edt_kernel_global<IS_FINAL, 4><<<total_slices, threads>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        global_buf1.data_ptr<int>(), global_buf2.data_ptr<int>(), \
                        L, dist_in.numel(), sample_ndim); break; \
                    case 5: edt_kernel_global<IS_FINAL, 5><<<total_slices, threads>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        global_buf1.data_ptr<int>(), global_buf2.data_ptr<int>(), \
                        L, dist_in.numel(), sample_ndim); break; \
                    case 6: edt_kernel_global<IS_FINAL, 6><<<total_slices, threads>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        global_buf1.data_ptr<int>(), global_buf2.data_ptr<int>(), \
                        L, dist_in.numel(), sample_ndim); break; \
                    default: /* Fallback */ \
                        edt_kernel_global<IS_FINAL, 0><<<total_slices, threads>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        global_buf1.data_ptr<int>(), global_buf2.data_ptr<int>(), \
                        L, dist_in.numel(), sample_ndim); break; \
                }

            if (is_final_pass) { DISPATCH_GLOBAL(true); } 
            else { DISPATCH_GLOBAL(false); }
        }

        // --- Step C: Transpose Back ---
        current_dist = dist_out.transpose(d, ndim - 1);
        current_idx  = idx_out.transpose(d, ndim - 1);
    }

    return std::make_tuple(current_dist, current_idx);
}

