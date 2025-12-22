#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <tuple>

// ------------------------------------------------------------------
// Global Configuration
// ------------------------------------------------------------------
#define BLOCK_SIZE 256
#define INF_VAL 1e20f
#define MAX_THREADS 1024
#define SMEM_LIMIT_ELEMENTS 4096 

// Configuration for 2D Optimized Block-JFA
#define JFA_BLOCK_DIM 32       //  Tile size: 32x32
#define JFA_FUSED_STEPS 4      //  Fused steps: 1, 2, 4, 8
#define JFA_MAX_OFFSET 8       //  Max offset processed in shared memory (Step 8)
#define JFA_SMEM_DIM (JFA_BLOCK_DIM + 2 * JFA_MAX_OFFSET) // Shared mem size: 48x48 (includes halo)

// ------------------------------------------------------------------
// Device Helpers
// ------------------------------------------------------------------
__device__ __forceinline__ float sqr(float x) { return x * x; }

// Helper for JFA 2D calculation
__device__ __forceinline__ float dist_sq_2d(int y1, int x1, int y2, int x2) {
    return sqr((float)(y1 - y2)) + sqr((float)(x1 - x2));
}

// Helper for JFA 3D calculation
__device__ __forceinline__ float dist_sq_3d(int z1, int y1, int x1, int z2, int y2, int x2) {
    return sqr((float)(z1 - z2)) + sqr((float)(y1 - y2)) + sqr((float)(x1 - x2));
}

// Helper for Separable 1D cost calculation
__device__ __forceinline__ float compute_cost(int q, int p, float val_p) {
    if (p < 0 || val_p >= INF_VAL) return INF_VAL; 
    return sqr((float)q - (float)p) + val_p;
}

// Device Helpers for int2 (Vectorized Coordinate)
// seed.x represents Y, seed.y represents X
__device__ __forceinline__ float dist_sq_int2(int y, int x, int2 seed) {
    if (seed.x == -1) return INF_VAL; 
    float dy = (float)(y - seed.x);
    float dx = (float)(x - seed.y);
    return dy*dy + dx*dx;
}

// ==================================================================
// PART 1: JFA KERNELS (Optimized for 2D with Block-Shared Memory)
// ==================================================================

// --- 2D Initialization (Vectorized int2) ---
// Initializes the coordinate map. Pixels with value 0 become seeds.
__global__ void init_jfa_kernel_2d_opt(
    const float* __restrict__ input,
    int2* __restrict__ output, // Output treats (y,x) pair as a single int2
    int64_t total_elements,    // Total pixels (B*H*W)
    int H, int W
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    if (input[tid] == 0.0f) {
        int64_t spatial_size = (int64_t)H * W;
        int64_t rem = tid % spatial_size;
        int w = (int)(rem % W);
        int h = (int)(rem / W);
        // Store coordinates: .x=y (height), .y=x (width)
        output[tid] = make_int2(h, w); 
    } else {
        output[tid] = make_int2(-1, -1);
    }
}

// --- 2D Block-JFA Fused Step ---
// Innovation: Performs Steps 1, 2, 4, and 8 entirely within Shared Memory.
// Uses a "Halo" (Apron) region to avoid boundary checks during iteration.
__global__ void jfa_block_fused_kernel_2d(
    const int2* __restrict__ in_idx,
    int2* __restrict__ out_idx,
    int H, int W,
    int64_t num_images // Batch Size
) {
    // Shared Memory: 48x48 int2 array (~18KB)
    // Covers the 32x32 block plus an 8-pixel halo on all sides.
    __shared__ int2 smem[JFA_SMEM_DIM][JFA_SMEM_DIM];

    int tx = threadIdx.x; // 0..31
    int ty = threadIdx.y; // 0..31
    
    // Global Block Indices
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    int img_idx = blockIdx.z;
    int64_t batch_offset = (int64_t)img_idx * (H * W);

    int gx = bx + tx;
    int gy = by + ty;

    // --- Phase 1: Cooperative Load to Shared Memory (Tile + Halo) ---
    
    int smem_linear_size = JFA_SMEM_DIM * JFA_SMEM_DIM;
    int total_threads = blockDim.x * blockDim.y;
    int thread_linear_idx = ty * blockDim.x + tx;

    // Base coordinates for the top-left corner of the Halo region
    int base_x = bx - JFA_MAX_OFFSET;
    int base_y = by - JFA_MAX_OFFSET;

    // Loop to fill the entire shared memory buffer (larger than block size)
    for (int i = thread_linear_idx; i < smem_linear_size; i += total_threads) {
        int s_y = i / JFA_SMEM_DIM;
        int s_x = i % JFA_SMEM_DIM;

        int global_y = base_y + s_y;
        int global_x = base_x + s_x;

        int2 val = make_int2(-1, -1);
        if (global_y >= 0 && global_y < H && global_x >= 0 && global_x < W) {
            val = in_idx[batch_offset + global_y * W + global_x];
        }
        smem[s_y][s_x] = val;
    }
    __syncthreads();

    // --- Phase 2: Iterative JFA in Shared Memory ---
    // Only process valid pixels within the image bounds
    if (gx < W && gy < H) {
        // Map thread to the center region of Shared Memory
        int center_sy = ty + JFA_MAX_OFFSET;
        int center_sx = tx + JFA_MAX_OFFSET;

        int2 best_seed = smem[center_sy][center_sx];
        float best_dist = dist_sq_int2(gy, gx, best_seed);

        int step = 1;
        
        // Unroll steps 1, 2, 4, 8
        #pragma unroll
        for (int k = 0; k < JFA_FUSED_STEPS; ++k) { 
            
            #pragma unroll
            for (int dy = -1; dy <= 1; ++dy) {
                #pragma unroll
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dy == 0 && dx == 0) continue;
                    
                    // Access neighbor in SMEM directly. 
                    // No boundary check needed because the Halo covers the max offset (8).
                    int2 neighbor_seed = smem[center_sy + dy * step][center_sx + dx * step];
                    
                    if (neighbor_seed.x != -1) {
                        float d = dist_sq_int2(gy, gx, neighbor_seed);
                        if (d < best_dist) {
                            best_dist = d;
                            best_seed = neighbor_seed;
                        }
                    }
                }
            }
            __syncthreads();
            smem[center_sy][center_sx] = best_seed;
            __syncthreads();
            step *= 2;
        }

        // --- Phase 3: Write results back to Global Memory ---
        out_idx[batch_offset + gy * W + gx] = best_seed;
    }
}

// --- 2D Global Step (Vectorized int2) ---
// Handles larger steps (16, 32, ...) that exceed Shared Memory capacity.
__global__ void jfa_step_global_2d_opt(
    const int2* __restrict__ in_idx,
    int2* __restrict__ out_idx,
    int step,
    int H, int W, 
    int64_t total_pixels
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_pixels) return;

    int64_t spatial_size = (int64_t)H * W;
    int64_t rem = tid % spatial_size;
    int64_t batch_offset = tid - rem; 
    int w = (int)(rem % W);
    int h = (int)(rem / W);

    int2 best_seed = in_idx[tid];
    float best_dist = dist_sq_int2(h, w, best_seed);

    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue; 
            
            int ny = h + dy * step;
            int nx = w + dx * step;

            if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
                int2 neighbor_seed = in_idx[batch_offset + ny * W + nx];
                if (neighbor_seed.x != -1) {
                    float d = dist_sq_int2(h, w, neighbor_seed);
                    if (d < best_dist) {
                        best_dist = d;
                        best_seed = neighbor_seed;
                    }
                }
            }
        }
    }
    out_idx[tid] = best_seed;
}

// --- 2D Final Distance Calculation ---
__global__ void calc_dist_kernel_2d_opt(
    const int2* __restrict__ indices,
    float* __restrict__ dist_out,
    int64_t total_elements,
    int H, int W
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    int2 s = indices[tid];
    if (s.x == -1) { 
        dist_out[tid] = INF_VAL; 
    } else {
        int64_t spatial_size = (int64_t)H * W;
        int64_t rem = tid % spatial_size;
        int cur_w = (int)(rem % W);
        int cur_h = (int)(rem / W);
        dist_out[tid] = sqrtf(dist_sq_int2(cur_h, cur_w, s));
    }
}

// --- 3D JFA Initialization ---
template <typename IndexType>
__global__ void init_jfa_kernel_3d(
    const float* __restrict__ input,
    IndexType* __restrict__ indices, 
    int64_t total_elements,
    int D, int H, int W
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    if (input[tid] == 0.0f) {
        int64_t spatial_size = (int64_t)D * H * W;
        int64_t rem = tid % spatial_size;
        int w = (int)(rem % W);
        int h = (int)((rem / W) % H);
        int d = (int)(rem / (W * H));
        int64_t idx_ptr = tid * 3;
        indices[idx_ptr + 0] = (IndexType)d;
        indices[idx_ptr + 1] = (IndexType)h;
        indices[idx_ptr + 2] = (IndexType)w;
    } else {
        int64_t idx_ptr = tid * 3;
        indices[idx_ptr + 0] = (IndexType)-1;
        indices[idx_ptr + 1] = (IndexType)-1;
        indices[idx_ptr + 2] = (IndexType)-1;
    }
}

// --- 3D JFA Step ---
template <typename IndexType>
__global__ void jfa_step_3d(
    const IndexType* __restrict__ in_idx,
    IndexType* __restrict__ out_idx,
    int step,
    int D, int H, int W, 
    int64_t total_pixels
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_pixels) return;

    int64_t spatial_size = (int64_t)D * H * W;
    int64_t rem = tid % spatial_size;
    int64_t batch_offset = tid - rem; 
    int w = (int)(rem % W);
    int h = (int)((rem / W) % H);
    int d = (int)(rem / (W * H));

    int best_z = -1, best_y = -1, best_x = -1;
    float best_dist = INF_VAL;

    #pragma unroll
    for (int dz = -1; dz <= 1; ++dz) {
        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            #pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                int nz = d + dz * step;
                int ny = h + dy * step;
                int nx = w + dx * step;
                if (nz >= 0 && nz < D && ny >= 0 && ny < H && nx >= 0 && nx < W) {
                    int64_t n_ptr = (batch_offset + (int64_t)nz * (H * W) + ny * W + nx) * 3;
                    int seed_z = (int)in_idx[n_ptr + 0];
                    if (seed_z != -1) {
                        int seed_y = (int)in_idx[n_ptr + 1];
                        int seed_x = (int)in_idx[n_ptr + 2];
                        float dist = dist_sq_3d(d, h, w, seed_z, seed_y, seed_x);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_z = seed_z;
                            best_y = seed_y;
                            best_x = seed_x;
                        }
                    }
                }
            }
        }
    }
    int64_t out_ptr = tid * 3;
    out_idx[out_ptr + 0] = (IndexType)best_z;
    out_idx[out_ptr + 1] = (IndexType)best_y;
    out_idx[out_ptr + 2] = (IndexType)best_x;
}

// --- 3D JFA Distance Calculation ---
template <typename IndexType>
__global__ void calc_dist_kernel_3d(
    const IndexType* __restrict__ indices,
    float* __restrict__ dist_out,
    int64_t total_elements,
    int D, int H, int W
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    int64_t idx_ptr = tid * 3;
    int seed_d = (int)indices[idx_ptr + 0];
    if (seed_d == -1) { dist_out[tid] = INF_VAL; return; }
    int seed_h = (int)indices[idx_ptr + 1];
    int seed_w = (int)indices[idx_ptr + 2];

    int64_t spatial_size = (int64_t)D * H * W;
    int64_t rem = tid % spatial_size;
    int cur_w = (int)(rem % W);
    int cur_h = (int)((rem / W) % H);
    int cur_d = (int)(rem / (W * H));

    dist_out[tid] = sqrtf(dist_sq_3d(cur_d, cur_h, cur_w, seed_d, seed_h, seed_w));
}

// ==================================================================
// PART 2: SEPARABLE N-DIM KERNELS (For 4D+ Spatial)
// ==================================================================

// Core logic for 1D Voronoi scan
__device__ void run_separable_scan_core(
    int N,
    int tid,
    const float* __restrict__ vals,  
    int* __restrict__ idx_curr,      
    int* __restrict__ idx_next       
) {
    // 1. Initialization
    for (int i = tid; i < N; i += blockDim.x) {
        if (vals[i] >= INF_VAL * 0.9f) idx_curr[i] = -1; 
        else idx_curr[i] = i;  
    }
    __syncthreads();

    // 2. Iterative Propagation (Logarithmic steps)
    int* idx_in = idx_curr;
    int* idx_out = idx_next;

    for (int step = 1; step < N; step *= 2) {
        for (int i = tid; i < N; i += blockDim.x) {
            int my_best_p = idx_in[i];
            float min_cost = INF_VAL;

            if (my_best_p != -1) min_cost = compute_cost(i, my_best_p, vals[my_best_p]);

            // Check Left Neighbor
            int left = i - step;
            if (left >= 0) {
                int left_p = idx_in[left];
                if (left_p != -1) {
                    float c = compute_cost(i, left_p, vals[left_p]);
                    if (c < min_cost) { min_cost = c; my_best_p = left_p; }
                }
            }

            // Check Right Neighbor
            int right = i + step;
            if (right < N) {
                int right_p = idx_in[right];
                if (right_p != -1) {
                    float c = compute_cost(i, right_p, vals[right_p]);
                    if (c < min_cost) { min_cost = c; my_best_p = right_p; }
                }
            }
            idx_out[i] = my_best_p;
        }
        // Swap buffers
        int* temp = idx_in; idx_in = idx_out; idx_out = temp;
        __syncthreads();
    }

    // 3. Final Copy Back (ensure result is in idx_curr)
    if (idx_in != idx_curr) {
        for (int i = tid; i < N; i += blockDim.x) idx_curr[i] = idx_next[i];
        __syncthreads();
    }
}

// Separable Kernel: Optimized using Shared Memory
template <bool IsFinal>
__global__ void separable_kernel_shared(
    const float* __restrict__ in_data,
    const int32_t* __restrict__ in_indices,
    float* __restrict__ out_dist,            
    int32_t* __restrict__ out_indices,       
    int64_t L,               
    int64_t total_elements,  
    int coord_ndim           
) {
    int64_t row_idx = blockIdx.x;
    int64_t offset = row_idx * L;
    if (offset >= total_elements) return;

    // Dynamic Shared Memory allocation
    extern __shared__ char s_buffer[];
    float* s_vals = (float*)s_buffer;
    int*   s_idx1 = (int*)(s_vals + L);
    int*   s_idx2 = (int*)(s_idx1 + L);

    // Load data
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        s_vals[i] = __ldg(&in_data[offset + i]);
    }
    __syncthreads();

    run_separable_scan_core(L, threadIdx.x, s_vals, s_idx1, s_idx2);

    // Write back
    for (int q = threadIdx.x; q < L; q += blockDim.x) {
        int p = s_idx1[q];
        float dist_val;

        if (p != -1) {
            float dist_sq = sqr((float)q - (float)p) + s_vals[p];
            dist_val = IsFinal ? sqrtf(dist_sq) : dist_sq;
        } else {
            dist_val = IsFinal ? INF_VAL : INF_VAL;
            p = 0;
        }
        out_dist[offset + q] = dist_val;

        int64_t dst_base = (offset + q) * coord_ndim;
        if (p != -1 && s_vals[p] < INF_VAL) {
            int64_t src_base = (offset + p) * coord_ndim;
            for (int d = 0; d < coord_ndim; ++d) {
                out_indices[dst_base + d] = in_indices[src_base + d];
            }
        } else {
            for (int d = 0; d < coord_ndim; ++d) out_indices[dst_base + d] = 0;
        }
    }
}

// Separable Kernel: Global Memory Fallback (when dim size > Shared Mem)
template <bool IsFinal>
__global__ void separable_kernel_global(
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
    
    run_separable_scan_core(L, threadIdx.x, in_data + offset, g_idx1, g_idx2);

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

// Separable Initialization
__global__ void init_indices_separable_kernel(
    int32_t* indices, 
    int64_t total_pixels, 
    int NDim, 
    const int64_t* __restrict__ shape_ptr
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    int64_t temp = idx;
    int32_t coords[8]; 
    for (int d = NDim - 1; d >= 0; --d) {
        int64_t dim_size = shape_ptr[d];
        coords[d] = temp % dim_size;
        temp /= dim_size;
    }
    int64_t out_ptr = idx * NDim;
    for (int d = 0; d < NDim; ++d) indices[out_ptr + d] = coords[d];
}

// ==================================================================
// PART 3: DISPATCH HELPERS
// ==================================================================

// --- JFA 2D Dispatch ---
std::tuple<torch::Tensor, torch::Tensor> run_jfa_2d(
    torch::Tensor input, int64_t H, int64_t W, int grid, int block, int64_t numel
) {
    // Force Int32 to enable int2 vectorized loads/stores
    auto index_opts = input.options().dtype(torch::kInt32);
    
    // Create Double Buffer indices: (Batch, H, W, 2)
    auto idx_shape = input.sizes().vec();
    idx_shape.push_back(2); 
    auto curr_idx = torch::empty(idx_shape, index_opts);
    auto next_idx = torch::empty(idx_shape, index_opts);
    
    // Cast int32 pointer to int2 pointer.
    // Memory layout matches: consecutive pairs of (y, x) form int2.
    int2* d_curr = (int2*)curr_idx.data_ptr<int32_t>();
    int2* d_next = (int2*)next_idx.data_ptr<int32_t>();

    // 1. Initialization Kernel
    // numel equals the number of int2 elements (pixels)
    init_jfa_kernel_2d_opt<<<grid, block>>>(
        input.data_ptr<float>(), 
        d_curr, 
        numel, H, W
    );

    // 2. Block-JFA Fused Kernel (Optimized)
    // Runs Steps 1, 2, 4, 8 inside Shared Memory
    {
        dim3 dimBlock(JFA_BLOCK_DIM, JFA_BLOCK_DIM); 
        int64_t batch_size = numel / (H * W);
        dim3 dimGrid((W + JFA_BLOCK_DIM - 1) / JFA_BLOCK_DIM, 
                     (H + JFA_BLOCK_DIM - 1) / JFA_BLOCK_DIM, 
                     batch_size);
        
        jfa_block_fused_kernel_2d<<<dimGrid, dimBlock>>>(
            d_curr, 
            d_next, 
            H, W, batch_size
        );
        std::swap(d_curr, d_next);     // d_curr now holds result of Step 8
        std::swap(curr_idx, next_idx); // Keep Tensor pointers in sync
    }

    // 3. Global Loop (Steps 16, 32...)
    int max_dim = std::max((int)H, (int)W);
    int step = 16; 

    while (step < max_dim) {
        jfa_step_global_2d_opt<<<grid, block>>>(
            d_curr, 
            d_next, 
            step, 
            H, W, numel
        );
        std::swap(d_curr, d_next);
        std::swap(curr_idx, next_idx);
        step *= 2;
    }
    
    // 4. Final Distance Calculation
    auto final_dist = torch::empty_like(input);
    calc_dist_kernel_2d_opt<<<grid, block>>>(
        d_curr, 
        final_dist.data_ptr<float>(), 
        numel, H, W
    );

    return std::make_tuple(final_dist, curr_idx);
}

std::tuple<torch::Tensor, torch::Tensor> run_jfa_3d(
    torch::Tensor input, int64_t D, int64_t H, int64_t W, int grid, int block, int64_t numel
) {
    bool use_int16 = (D < 32767 && H < 32767 && W < 32767);
    auto index_opts = input.options().dtype(use_int16 ? torch::kInt16 : torch::kInt32);
    auto idx_shape = input.sizes().vec();
    idx_shape.push_back(3);
    auto curr_idx = torch::empty(idx_shape, index_opts);
    auto next_idx = torch::empty(idx_shape, index_opts);

    if (use_int16) init_jfa_kernel_3d<int16_t><<<grid, block>>>(input.data_ptr<float>(), (int16_t*)curr_idx.data_ptr(), numel, D, H, W);
    else init_jfa_kernel_3d<int32_t><<<grid, block>>>(input.data_ptr<float>(), (int32_t*)curr_idx.data_ptr(), numel, D, H, W);

    int max_dim = std::max({(int)D, (int)H, (int)W});
    int step = 1;
    while (step < max_dim) step *= 2;
    step /= 2;

    while (step >= 1) {
        if (use_int16) jfa_step_3d<int16_t><<<grid, block>>>((int16_t*)curr_idx.data_ptr(), (int16_t*)next_idx.data_ptr(), step, D, H, W, numel);
        else jfa_step_3d<int32_t><<<grid, block>>>((int32_t*)curr_idx.data_ptr(), (int32_t*)next_idx.data_ptr(), step, D, H, W, numel);
        std::swap(curr_idx, next_idx);
        step /= 2;
    }
    auto final_dist = torch::empty_like(input);
    if (use_int16) calc_dist_kernel_3d<int16_t><<<grid, block>>>((int16_t*)curr_idx.data_ptr(), final_dist.data_ptr<float>(), numel, D, H, W);
    else calc_dist_kernel_3d<int32_t><<<grid, block>>>((int32_t*)curr_idx.data_ptr(), final_dist.data_ptr<float>(), numel, D, H, W);

    return std::make_tuple(final_dist, curr_idx);
}

// --- Separable N-Dim Dispatch ---
std::tuple<torch::Tensor, torch::Tensor> run_separable_ndim(torch::Tensor input) {
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Separable N-Dim input must be float32.");
    input = input.contiguous();
    
    const int ndim = input.dim(); 
    const int sample_ndim = ndim - 1; // Assuming Dim 0 is Batch
    TORCH_CHECK(sample_ndim > 0 && sample_ndim <= 8, "Unsupported dims for Separable EDT");
    
    auto shape = input.sizes().vec();
    int64_t num_pixels = input.numel();

    // 1. Init Distances
    auto current_dist = torch::where(input == 0, 
                                    torch::tensor(0.0f, input.options()), 
                                    torch::tensor(INF_VAL, input.options()));
    
    // 2. Init Indices
    auto index_shape = shape;
    index_shape.push_back(sample_ndim);
    auto current_idx = torch::empty(index_shape, input.options().dtype(torch::kInt32));
    
    {
        std::vector<int64_t> spatial_shape(shape.begin() + 1, shape.end());
        auto shape_tensor = torch::tensor(spatial_shape, torch::kInt64).to(input.device());
        int threads = 256;
        int blocks = (num_pixels + threads - 1) / threads;
        init_indices_separable_kernel<<<blocks, threads>>>(
            current_idx.data_ptr<int32_t>(), num_pixels, sample_ndim, shape_tensor.data_ptr<int64_t>()
        );
    }

    torch::Tensor global_buf1, global_buf2;

    // 3. Dimensional Iteration (Apply 1D transform along each spatial axis)
    for (int d = 1; d < ndim; ++d) {
        bool is_final_pass = (d == ndim - 1);
        
        auto dist_in = current_dist.transpose(d, ndim - 1).contiguous();
        auto idx_in  = current_idx.transpose(d, ndim - 1).contiguous(); 
        
        auto dist_out = torch::empty(dist_in.sizes(), dist_in.options());
        auto idx_out  = torch::empty(idx_in.sizes(), idx_in.options());

        int64_t L = dist_in.size(-1); 
        int64_t total_slices = dist_in.numel() / L; 
        int threads = std::min((int64_t)MAX_THREADS, L);
        
        // Choose between Shared Memory or Global Memory kernel based on dimension size
        if (L <= SMEM_LIMIT_ELEMENTS) {
            size_t smem_size = L * (sizeof(float) + 2 * sizeof(int));
            if (is_final_pass) {
                separable_kernel_shared<true><<<total_slices, threads, smem_size>>>(
                    dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(),
                    dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(),
                    L, dist_in.numel(), sample_ndim
                );
            } else {
                separable_kernel_shared<false><<<total_slices, threads, smem_size>>>(
                    dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(),
                    dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(),
                    L, dist_in.numel(), sample_ndim
                );
            }
        } else {
            if (global_buf1.numel() < dist_in.numel()) {
                global_buf1 = torch::empty({dist_in.numel()}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
                global_buf2 = torch::empty({dist_in.numel()}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
            }
            if (is_final_pass) {
                separable_kernel_global<true><<<total_slices, threads>>>(
                    dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(),
                    dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(),
                    global_buf1.data_ptr<int>(), global_buf2.data_ptr<int>(),
                    L, dist_in.numel(), sample_ndim
                );
            } else {
                separable_kernel_global<false><<<total_slices, threads>>>(
                    dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(),
                    dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(),
                    global_buf1.data_ptr<int>(), global_buf2.data_ptr<int>(),
                    L, dist_in.numel(), sample_ndim
                );
            }
        }
        current_dist = dist_out.transpose(d, ndim - 1);
        current_idx  = idx_out.transpose(d, ndim - 1);
    }

    return std::make_tuple(current_dist, current_idx);
}

// ==================================================================
// PART 4: MAIN ENTRY POINT (INTEGRATED)
// ==================================================================

std::tuple<torch::Tensor, torch::Tensor> distance_transform_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    input = input.contiguous();

    int64_t dims = input.dim();
    int64_t numel = input.numel();
    int block = BLOCK_SIZE;
    int grid = (numel + block - 1) / block;

    // ------------------------------------------------------------------
    // CASE 1: High-Dimension (5D+) -> Use Separable N-Dim Algorithm
    // Input: (Batch, D1, D2, D3...) -> Treated as N-Dim spatial
    // ------------------------------------------------------------------
    if (dims >= 5) {
        return run_separable_ndim(input);
    }
    
    // ------------------------------------------------------------------
    // CASE 2: 4D Tensor -> (Batch, Dim1, H, W)
    // ------------------------------------------------------------------
    else if (dims == 4) {
        int64_t dim1 = input.size(1); 
        
        // [Fast Path]: (Batch, 1, H, W) -> Treat as 2D JFA
        if (dim1 == 1) {
            int64_t H = input.size(-2);
            int64_t W = input.size(-1);
            return run_jfa_2d(input, H, W, grid, block, numel);
        } 
        // [Standard Path]: (Batch, Depth, H, W) -> Use 3D JFA
        else {
            int64_t D = dim1;
            int64_t H = input.size(-2);
            int64_t W = input.size(-1);
            return run_jfa_3d(input, D, H, W, grid, block, numel);
        }
    }

    // ------------------------------------------------------------------
    // CASE 3: 3D Tensor -> (Batch, H, W) -> Use 2D JFA
    // ------------------------------------------------------------------
    else if (dims == 3) {
        int64_t H = input.size(-2);
        int64_t W = input.size(-1);
        return run_jfa_2d(input, H, W, grid, block, numel);
    }

    // ------------------------------------------------------------------
    // CASE 4: 2D Tensor -> (Batch, Length) -> 1D JFA (via 2D wrapper)
    // ------------------------------------------------------------------
    else if (dims == 2) {
        int64_t H = 1;
        int64_t W = input.size(-1);
        auto result = run_jfa_2d(input, H, W, grid, block, numel);
        
        // Post-process for 1D: slice out the dummy Y coordinate
        torch::Tensor dist = std::get<0>(result);
        torch::Tensor idx_2d = std::get<1>(result); 
        auto idx_1d = idx_2d.slice(/*dim=*/-1, /*start=*/1, /*end=*/2).contiguous();
        return std::make_tuple(dist, idx_1d);
    }

    else {
        TORCH_CHECK(false, "Unsupported dimensions.");
        return std::make_tuple(torch::Tensor(), torch::Tensor());
    }
}