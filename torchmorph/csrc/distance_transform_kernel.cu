#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <limits>
#include <cuda.h>
#include <cuda_runtime.h>

// ------------------------------------------------------------------
// Configuration
// ------------------------------------------------------------------
// Use a large enough value to avoid overflow, but safe for float addition
#define INF_VAL 1e20f 
#define MAX_THREADS 1024
#define SMEM_LIMIT_ELEMENTS 4096 

// ------------------------------------------------------------------
// Device Helper
// ------------------------------------------------------------------

__device__ __forceinline__ float sqr(float x) { return x * x; }

__device__ __forceinline__ float compute_cost(int q, int p, float val_p) {
    // Safety check for boundaries and INF propagation
    if (p < 0 || val_p >= INF_VAL) return INF_VAL; 
    return sqr((float)q - (float)p) + val_p;
}

// ------------------------------------------------------------------
// JFA Core Logic (Device Only)
// ------------------------------------------------------------------
__device__ void run_jfa_core(
    int N,
    int tid,
    const float* __restrict__ vals,  
    int* __restrict__ idx_curr,      
    int* __restrict__ idx_next       
) {
    // 1. Initialization
    for (int i = tid; i < N; i += blockDim.x) {
        // Use a relative threshold to safely detect background
        if (vals[i] >= INF_VAL * 0.9f) {
            idx_curr[i] = -1; 
        } else {
            idx_curr[i] = i;  
        }
    }
    __syncthreads();

    // 2. Iterative Propagation (Pointer Jumping: 1 -> 2 -> 4...)
    int* idx_in = idx_curr;
    int* idx_out = idx_next;

    for (int step = 1; step < N; step *= 2) {
        for (int i = tid; i < N; i += blockDim.x) {
            int my_best_p = idx_in[i];
            float min_cost = INF_VAL;

            // Check self (current best)
            if (my_best_p != -1) {
                min_cost = compute_cost(i, my_best_p, vals[my_best_p]);
            }

            // Check Left Neighbor
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

            // Check Right Neighbor
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
        
        // Swap Ping-Pong buffers
        int* temp = idx_in;
        idx_in = idx_out;
        idx_out = temp;
        __syncthreads();
    }

    // 3. Final Copy Back (if needed)
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
// Note: We removed the template switch for NDim to reduce compile time.
// The performance impact is negligible for the copy loop.
template <bool IsFinal>
__global__ void edt_kernel_shared(
    const float* __restrict__ in_data,       // Contiguous Input
    const int32_t* __restrict__ in_indices,  // Contiguous Input
    float* __restrict__ out_dist,            
    int32_t* __restrict__ out_indices,       
    int64_t L,               
    int64_t total_elements,  
    int coord_ndim           
) {
    // 1 Block processes 1 Row (L elements)
    int64_t row_idx = blockIdx.x;
    int64_t offset = row_idx * L;

    if (offset >= total_elements) return;

    // Shared memory layout
    extern __shared__ char s_buffer[];
    float* s_vals = (float*)s_buffer;
    int*   s_idx1 = (int*)(s_vals + L);
    int*   s_idx2 = (int*)(s_idx1 + L);

    // 1. Load Data (Coalesced Read due to .contiguous() input)
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        s_vals[i] = __ldg(&in_data[offset + i]);
    }
    __syncthreads();

    // 2. Run JFA Core
    run_jfa_core(L, threadIdx.x, s_vals, s_idx1, s_idx2);

    // 3. Write Back Results
    for (int q = threadIdx.x; q < L; q += blockDim.x) {
        int p = s_idx1[q];
        float dist_val;

        // Calculate final distance
        if (p != -1) {
            float dist_sq = sqr((float)q - (float)p) + s_vals[p];
            dist_val = IsFinal ? sqrtf(dist_sq) : dist_sq;
        } else {
            dist_val = IsFinal ? INF_VAL : INF_VAL; // Use large val instead of sqr(INF)
            p = 0;
        }
        out_dist[offset + q] = dist_val;

        // Propagate Indices
        // out_indices shape is (TotalElements, coord_ndim) flattened
        // Using runtime loop instead of template unrolling
        int64_t dst_base = (offset + q) * coord_ndim;
        
        if (p != -1 && s_vals[p] < INF_VAL) {
            int64_t src_base = (offset + p) * coord_ndim;
            for (int d = 0; d < coord_ndim; ++d) {
                out_indices[dst_base + d] = in_indices[src_base + d];
            }
        } else {
            for (int d = 0; d < coord_ndim; ++d) {
                out_indices[dst_base + d] = 0;
            }
        }
    }
}

// ------------------------------------------------------------------
// Kernel 2: Global Memory JFA (Fallback Path)
// ------------------------------------------------------------------
template <bool IsFinal>
__global__ void edt_kernel_global(
    const float* __restrict__ in_data,
    const int32_t* __restrict__ in_indices,
    float* __restrict__ out_dist,
    int32_t* __restrict__ out_indices,
    int* __restrict__ global_buffer_1,
    int* __restrict__ global_buffer_2,
    int64_t L,
    int64_t total_elements,
    int coord_ndim
) {
    int64_t row_idx = blockIdx.x;
    int64_t offset = row_idx * L;
    
    if (offset >= total_elements) return;

    int* g_idx1 = global_buffer_1 + offset;
    int* g_idx2 = global_buffer_2 + offset;
    
    // Core Logic operates on Global Memory pointers
    run_jfa_core(L, threadIdx.x, in_data + offset, g_idx1, g_idx2);

    for (int q = threadIdx.x; q < L; q += blockDim.x) {
        int p = g_idx1[q];
        float dist_val;

        if (p != -1) {
            float val_p = in_data[offset + p]; 
            float dist_sq = sqr((float)q - (float)p) + val_p;
            dist_val = IsFinal ? sqrtf(dist_sq) : dist_sq;
        } else {
            dist_val = IsFinal ? INF_VAL : INF_VAL;
            p = 0; 
        }

        out_dist[offset + q] = dist_val;

        int64_t dst_base = (offset + q) * coord_ndim;
        if (p != -1 && in_data[offset + p] < INF_VAL) {
            int64_t src_base = (offset + p) * coord_ndim;
            for (int d = 0; d < coord_ndim; ++d) {
                out_indices[dst_base + d] = in_indices[src_base + d];
            }
        } else {
             for (int d = 0; d < coord_ndim; ++d) out_indices[dst_base + d] = 0;
        }
    }
}

// ------------------------------------------------------------------
// Initialization Kernel
// ------------------------------------------------------------------
__global__ void init_indices_kernel(
    int32_t* indices, 
    int64_t total_pixels, 
    int NDim, 
    const int64_t* __restrict__ shape_ptr
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    int64_t temp = idx;
    int32_t coords[8]; // Max 8 dims supported locally

    // Unravel index
    for (int d = NDim - 1; d >= 0; --d) {
        int64_t dim_size = shape_ptr[d];
        coords[d] = temp % dim_size;
        temp /= dim_size;
    }

    int64_t out_ptr = idx * NDim;
    for (int d = 0; d < NDim; ++d) {
        indices[out_ptr + d] = coords[d];
    }
}

// ------------------------------------------------------------------
// Host Function
// ------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> distance_transform_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA device.");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32.");
    
    // 1. Force Contiguous Input (Optimized Copy)
    // This is crucial for coalesced memory access in init kernel.
    input = input.contiguous();
    
    const int ndim = input.dim(); 
    const int sample_ndim = ndim - 1; 
    TORCH_CHECK(sample_ndim > 0 && sample_ndim <= 8, "Dims must be between 2 and 9 (Batch + 8 Spatial)");
    
    auto shape = input.sizes().vec();
    int64_t num_pixels = input.numel();

    // Handle empty input
    if (num_pixels == 0) {
        auto index_shape = shape;
        index_shape.push_back(sample_ndim);
        return std::make_tuple(torch::empty_like(input), 
                               torch::empty(index_shape, input.options().dtype(torch::kInt32)));
    }

    // 2. Init Distances
    auto current_dist = torch::where(input == 0, 
                                     torch::tensor(0.0f, input.options()), 
                                     torch::tensor(INF_VAL, input.options()));
    
    // 3. Init Indices
    auto index_shape = shape;
    index_shape.push_back(sample_ndim);
    auto current_idx = torch::empty(index_shape, input.options().dtype(torch::kInt32));
    
    {
        std::vector<int64_t> spatial_shape(shape.begin() + 1, shape.end());
        auto shape_tensor = torch::tensor(spatial_shape, torch::kInt64).to(input.device());

        int threads = 256;
        int blocks = (num_pixels + threads - 1) / threads;
        init_indices_kernel<<<blocks, threads>>>(
            current_idx.data_ptr<int32_t>(),
            num_pixels,
            sample_ndim,
            shape_tensor.data_ptr<int64_t>()
        );
    }

    // Lazy buffers
    torch::Tensor global_buf1, global_buf2;

    // 4. Dimensional Iteration
    for (int d = 1; d < ndim; ++d) {
        bool is_final_pass = (d == ndim - 1);
        
        // --- Step A: Transpose + Contiguous (The "Expensive" Copy) ---
        // We accept this copy because it enables fully coalesced memory access in the kernel.
        // Without this, the kernel bandwidth drops to <5%, which is much slower than the copy.
        auto dist_in = current_dist.transpose(d, ndim - 1).contiguous();
        auto idx_in  = current_idx.transpose(d, ndim - 1).contiguous(); 
        
        // Prepare Output (Contiguous)
        // Using empty() instead of empty_like() to ensure standard stride layout
        auto dist_out = torch::empty(dist_in.sizes(), dist_in.options());
        auto idx_out  = torch::empty(idx_in.sizes(), idx_in.options());

        int64_t L = dist_in.size(-1); 
        int64_t total_slices = dist_in.numel() / L; 
        int threads = std::min((int64_t)MAX_THREADS, L);
        
        // --- Step B: Kernel Dispatch ---
        if (L <= SMEM_LIMIT_ELEMENTS) {
            size_t smem_size = L * (sizeof(float) + 2 * sizeof(int));
            if (is_final_pass) {
                edt_kernel_shared<true><<<total_slices, threads, smem_size>>>(
                    dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(),
                    dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(),
                    L, dist_in.numel(), sample_ndim
                );
            } else {
                edt_kernel_shared<false><<<total_slices, threads, smem_size>>>(
                    dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(),
                    dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(),
                    L, dist_in.numel(), sample_ndim
                );
            }
        } else {
            // Global Memory Fallback
            if (global_buf1.numel() < dist_in.numel()) {
                global_buf1 = torch::empty({dist_in.numel()}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
                global_buf2 = torch::empty({dist_in.numel()}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
            }
            if (is_final_pass) {
                edt_kernel_global<true><<<total_slices, threads>>>(
                    dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(),
                    dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(),
                    global_buf1.data_ptr<int>(), global_buf2.data_ptr<int>(),
                    L, dist_in.numel(), sample_ndim
                );
            } else {
                edt_kernel_global<false><<<total_slices, threads>>>(
                    dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(),
                    dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(),
                    global_buf1.data_ptr<int>(), global_buf2.data_ptr<int>(),
                    L, dist_in.numel(), sample_ndim
                );
            }
        }

        // --- Step C: Logical Transpose ---
        // This is just a metadata swap, no copy. The next loop's .contiguous() will handle the copy.
        current_dist = dist_out.transpose(d, ndim - 1);
        current_idx  = idx_out.transpose(d, ndim - 1);
    }

    return std::make_tuple(current_dist, current_idx);
}