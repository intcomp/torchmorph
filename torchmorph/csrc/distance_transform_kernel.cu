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

#define JFA_BLOCK_DIM 32       
#define JFA_FUSED_STEPS 4      
#define JFA_MAX_OFFSET 8       
#define JFA_SMEM_DIM (JFA_BLOCK_DIM + 2 * JFA_MAX_OFFSET) 

// 3D Config
#define JFA_3D_BLOCK 8
#define JFA_3D_HALO 1 

// ------------------------------------------------------------------
// Device Helpers
// ------------------------------------------------------------------
__device__ __forceinline__ float sqr(float x) { return x * x; }

// Helper for JFA 2D/3D (Standard)
__device__ __forceinline__ float dist_sq_2d(int y1, int x1, int y2, int x2) {
    return sqr((float)(y1 - y2)) + sqr((float)(x1 - x2));
}

// Helper for SoA 3D (Z, Y, X separate)
__device__ __forceinline__ float dist_sq_3d_soa(int z1, int y1, int x1, int z2, int y2, int x2) {
    if (z2 == -1) return INF_VAL;
    float dz = (float)(z1 - z2);
    float dy = (float)(y1 - y2);
    float dx = (float)(x1 - x2);
    return dz*dz + dy*dy + dx*dx;
}

// Helper for Separable 1D
__device__ __forceinline__ float compute_cost(int q, int p, float val_p) {
    if (p < 0 || val_p >= INF_VAL) return INF_VAL; 
    return sqr((float)q - (float)p) + val_p;
}

// Device Helpers for int2 (2D Vectorized)
__device__ __forceinline__ float dist_sq_int2(int y, int x, int2 seed) {
    if (seed.x == -1) return INF_VAL; 
    float dy = (float)(y - seed.x);
    float dx = (float)(x - seed.y);
    return dy*dy + dx*dx;
}

// ==================================================================
// PART 1: JFA KERNELS 2D (Vectorized int2 + Block Shared)
// ==================================================================

__global__ void init_jfa_kernel_2d_opt(
    const float* __restrict__ input,
    int2* __restrict__ output, 
    int64_t total_elements,    
    int H, int W
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    if (input[tid] == 0.0f) {
        int64_t spatial_size = (int64_t)H * W;
        int64_t rem = tid % spatial_size;
        int w = (int)(rem % W);
        int h = (int)(rem / W);
        output[tid] = make_int2(h, w); 
    } else {
        output[tid] = make_int2(-1, -1);
    }
}

__global__ void jfa_block_fused_kernel_2d(
    const int2* __restrict__ in_idx,
    int2* __restrict__ out_idx,
    int H, int W,
    int64_t num_images 
) {
    __shared__ int2 smem[JFA_SMEM_DIM][JFA_SMEM_DIM];

    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int img_idx = blockIdx.z;
    int64_t batch_offset = (int64_t)img_idx * (H * W);

    int gx = bx + tx;
    int gy = by + ty;

    // Phase 1: load data to Shared Memory
    int smem_linear_size = JFA_SMEM_DIM * JFA_SMEM_DIM;
    int total_threads = blockDim.x * blockDim.y;
    int thread_linear_idx = ty * blockDim.x + tx;

    int base_x = bx - JFA_MAX_OFFSET;
    int base_y = by - JFA_MAX_OFFSET;

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

    // Phase 2: Iterate in Shared Memory
    if (gx < W && gy < H) {
        int center_sy = ty + JFA_MAX_OFFSET;
        int center_sx = tx + JFA_MAX_OFFSET;

        int2 best_seed = smem[center_sy][center_sx];
        float best_dist = dist_sq_int2(gy, gx, best_seed);

        int step = 1;
        #pragma unroll
        for (int k = 0; k < JFA_FUSED_STEPS; ++k) { 
            #pragma unroll
            for (int dy = -1; dy <= 1; ++dy) {
                #pragma unroll
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dy == 0 && dx == 0) continue;
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
        out_idx[batch_offset + gy * W + gx] = best_seed;
    }
}

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

// ==================================================================
// PART 2: JFA KERNELS 3D (Optimized SoA Layout)
// ==================================================================

template <typename IndexType>
__global__ void init_jfa_kernel_3d_soa(
    const float* __restrict__ input,
    IndexType* __restrict__ indices_z,
    IndexType* __restrict__ indices_y,
    IndexType* __restrict__ indices_x,
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

        indices_z[tid] = (IndexType)d;
        indices_y[tid] = (IndexType)h;
        indices_x[tid] = (IndexType)w;
    } else {
        indices_z[tid] = (IndexType)-1;
        indices_y[tid] = (IndexType)-1;
        indices_x[tid] = (IndexType)-1;
    }
}

template <typename IndexType>
__global__ void jfa_block_fused_kernel_3d_soa(
    const IndexType* __restrict__ in_z,
    const IndexType* __restrict__ in_y,
    const IndexType* __restrict__ in_x,
    IndexType* __restrict__ out_z,
    IndexType* __restrict__ out_y,
    IndexType* __restrict__ out_x,
    int D, int H, int W,
    int blocks_per_d
) {
    const int BLOCK_DIM = 8;
    const int HALO = 3; 
    const int SMEM_DIM = BLOCK_DIM + 2 * HALO; // 14
    const int SMEM_SIZE = SMEM_DIM * SMEM_DIM * SMEM_DIM;

    extern __shared__ char smem_raw[];
    IndexType* smem_z = (IndexType*)smem_raw;
    IndexType* smem_y = smem_z + SMEM_SIZE;
    IndexType* smem_x = smem_y + SMEM_SIZE;

    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;

    int b_z_total = blockIdx.z;
    int batch_id = b_z_total / blocks_per_d;
    int b_z_local = b_z_total % blocks_per_d;
    
    int bx = blockIdx.x * BLOCK_DIM;
    int by = blockIdx.y * BLOCK_DIM;
    int bz = b_z_local * BLOCK_DIM;

    int64_t spatial_offset = (int64_t)batch_id * (D * H * W);

    // Phase 1: Load to SoA Shared Memory
    int tid = tz * 64 + ty * 8 + tx; 
    int base_x = bx - HALO;
    int base_y = by - HALO;
    int base_z = bz - HALO;

    for (int i = tid; i < SMEM_SIZE; i += 512) {
        int temp = i;
        int sx = temp % SMEM_DIM; temp /= SMEM_DIM;
        int sy = temp % SMEM_DIM;
        int sz = temp / SMEM_DIM;

        int gx = base_x + sx;
        int gy = base_y + sy;
        int gz = base_z + sz;

        IndexType val_z = -1, val_y = -1, val_x = -1;
        if (gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W) {
            int64_t idx = spatial_offset + (int64_t)gz * (H * W) + gy * W + gx;
            val_z = in_z[idx];
            val_y = in_y[idx];
            val_x = in_x[idx];
        }
        smem_z[i] = val_z;
        smem_y[i] = val_y;
        smem_x[i] = val_x;
    }
    __syncthreads();

    // Phase 2: Compute
    int center_sz = tz + HALO;
    int center_sy = ty + HALO;
    int center_sx = tx + HALO;
    int my_s_idx = (center_sz * SMEM_DIM + center_sy) * SMEM_DIM + center_sx;

    int best_z = (int)smem_z[my_s_idx];
    int best_y = (int)smem_y[my_s_idx];
    int best_x = (int)smem_x[my_s_idx];

    int g_cz = bz + tz;
    int g_cy = by + ty;
    int g_cx = bx + tx;

    float best_dist = dist_sq_3d_soa(g_cz, g_cy, g_cx, best_z, best_y, best_x);

    int step = 1;
    #pragma unroll
    for (int k = 0; k < 2; ++k) { 
        #pragma unroll
        for (int dz = -1; dz <= 1; ++dz) {
            #pragma unroll
            for (int dy = -1; dy <= 1; ++dy) {
                #pragma unroll
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dz == 0 && dy == 0 && dx == 0) continue;

                    int nz = center_sz + dz * step;
                    int ny = center_sy + dy * step;
                    int nx = center_sx + dx * step;
                    int n_idx = (nz * SMEM_DIM + ny) * SMEM_DIM + nx;

                    int sz_in = (int)smem_z[n_idx]; 
                    if (sz_in != -1) {
                        int sy_in = (int)smem_y[n_idx];
                        int sx_in = (int)smem_x[n_idx];
                        float d = dist_sq_3d_soa(g_cz, g_cy, g_cx, sz_in, sy_in, sx_in);
                        if (d < best_dist) {
                            best_dist = d;
                            best_z = sz_in;
                            best_y = sy_in;
                            best_x = sx_in;
                        }
                    }
                }
            }
        }
        __syncthreads();
        smem_z[my_s_idx] = (IndexType)best_z;
        smem_y[my_s_idx] = (IndexType)best_y;
        smem_x[my_s_idx] = (IndexType)best_x;
        __syncthreads();
        step *= 2;
    }

    if (g_cz < D && g_cy < H && g_cx < W) {
        int64_t out_idx_g = spatial_offset + (int64_t)g_cz * (H * W) + g_cy * W + g_cx;
        out_z[out_idx_g] = (IndexType)best_z;
        out_y[out_idx_g] = (IndexType)best_y;
        out_x[out_idx_g] = (IndexType)best_x;
    }
}

template <typename IndexType>
__global__ void jfa_step_3d_soa(
    const IndexType* __restrict__ in_z,
    const IndexType* __restrict__ in_y,
    const IndexType* __restrict__ in_x,
    IndexType* __restrict__ out_z,
    IndexType* __restrict__ out_y,
    IndexType* __restrict__ out_x,
    int step,
    int D, int H, int W, 
    int64_t total_pixels
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_pixels) return;

    int64_t spatial_size = (int64_t)D * H * W;
    int64_t rem = tid % spatial_size;
    int64_t batch_offset = tid - rem; 
    int cur_w = (int)(rem % W);
    int cur_h = (int)((rem / W) % H);
    int cur_d = (int)(rem / (W * H));

    int best_z = (int)in_z[tid];
    int best_y = (int)in_y[tid];
    int best_x = (int)in_x[tid];
    
    float best_dist = dist_sq_3d_soa(cur_d, cur_h, cur_w, best_z, best_y, best_x);

    #pragma unroll
    for (int dz = -1; dz <= 1; ++dz) {
        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            #pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                if (dz == 0 && dy == 0 && dx == 0) continue;

                int nz = cur_d + dz * step;
                int ny = cur_h + dy * step;
                int nx = cur_w + dx * step;

                if (nz >= 0 && nz < D && ny >= 0 && ny < H && nx >= 0 && nx < W) {
                    int64_t n_idx = batch_offset + (int64_t)nz * (H * W) + ny * W + nx;
                    
                    int seed_z = (int)in_z[n_idx];
                    if (seed_z != -1) {
                        float dz_val = (float)(cur_d - seed_z);
                        float dz_sq = dz_val * dz_val;
                        
                        if (dz_sq < best_dist) {
                            int seed_y = (int)in_y[n_idx];
                            int seed_x = (int)in_x[n_idx];
                            float dist = dz_sq + sqr((float)(cur_h - seed_y)) + sqr((float)(cur_w - seed_x));
                            
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
    }
    out_z[tid] = (IndexType)best_z;
    out_y[tid] = (IndexType)best_y;
    out_x[tid] = (IndexType)best_x;
}

template <typename IndexType>
__global__ void calc_dist_kernel_3d_soa(
    const IndexType* __restrict__ in_z,
    const IndexType* __restrict__ in_y,
    const IndexType* __restrict__ in_x,
    float* __restrict__ dist_out,
    int64_t total_elements,
    int D, int H, int W
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    int seed_d = (int)in_z[tid];
    if (seed_d == -1) { 
        dist_out[tid] = INF_VAL; 
    } else {
        int seed_h = (int)in_y[tid];
        int seed_w = (int)in_x[tid];

        int64_t spatial_size = (int64_t)D * H * W;
        int64_t rem = tid % spatial_size;
        int cur_w = (int)(rem % W);
        int cur_h = (int)((rem / W) % H);
        int cur_d = (int)(rem / (W * H));
        
        dist_out[tid] = sqrtf(dist_sq_3d_soa(cur_d, cur_h, cur_w, seed_d, seed_h, seed_w));
    }
}

// ==================================================================
// PART 3: SEPARABLE N-DIM KERNELS
// ==================================================================

__device__ void run_separable_scan_core(
    int N,
    int tid,
    const float* __restrict__ vals,  
    int* __restrict__ idx_curr,      
    int* __restrict__ idx_next       
) {
    for (int i = tid; i < N; i += blockDim.x) {
        if (vals[i] >= INF_VAL * 0.9f) idx_curr[i] = -1; 
        else idx_curr[i] = i;  
    }
    __syncthreads();

    int* idx_in = idx_curr;
    int* idx_out = idx_next;

    for (int step = 1; step < N; step *= 2) {
        for (int i = tid; i < N; i += blockDim.x) {
            int my_best_p = idx_in[i];
            float min_cost = INF_VAL;

            if (my_best_p != -1) min_cost = compute_cost(i, my_best_p, vals[my_best_p]);

            int left = i - step;
            if (left >= 0) {
                int left_p = idx_in[left];
                if (left_p != -1) {
                    float c = compute_cost(i, left_p, vals[left_p]);
                    if (c < min_cost) { min_cost = c; my_best_p = left_p; }
                }
            }

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
        int* temp = idx_in; idx_in = idx_out; idx_out = temp;
        __syncthreads();
    }

    if (idx_in != idx_curr) {
        for (int i = tid; i < N; i += blockDim.x) idx_curr[i] = idx_next[i];
        __syncthreads();
    }
}

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

    extern __shared__ char s_buffer[];
    float* s_vals = (float*)s_buffer;
    int*   s_idx1 = (int*)(s_vals + L);
    int*   s_idx2 = (int*)(s_idx1 + L);

    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        s_vals[i] = __ldg(&in_data[offset + i]);
    }
    __syncthreads();

    run_separable_scan_core(L, threadIdx.x, s_vals, s_idx1, s_idx2);

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
// PART 4: DISPATCH HELPERS
// ==================================================================

std::tuple<torch::Tensor, torch::Tensor> run_jfa_2d(
    torch::Tensor input, int64_t H, int64_t W, int grid, int block, int64_t numel
) {
    auto index_opts = input.options().dtype(torch::kInt32);
    auto idx_shape = input.sizes().vec();
    idx_shape.push_back(2); 
    auto curr_idx = torch::empty(idx_shape, index_opts);
    auto next_idx = torch::empty(idx_shape, index_opts);
    
    int2* d_curr = (int2*)curr_idx.data_ptr<int32_t>();
    int2* d_next = (int2*)next_idx.data_ptr<int32_t>();

    init_jfa_kernel_2d_opt<<<grid, block>>>(
        input.data_ptr<float>(), d_curr, numel, H, W
    );

    {
        dim3 dimBlock(JFA_BLOCK_DIM, JFA_BLOCK_DIM); 
        int64_t batch_size = numel / (H * W);
        dim3 dimGrid((W + JFA_BLOCK_DIM - 1) / JFA_BLOCK_DIM, 
                     (H + JFA_BLOCK_DIM - 1) / JFA_BLOCK_DIM, 
                     batch_size);
        
        jfa_block_fused_kernel_2d<<<dimGrid, dimBlock>>>(d_curr, d_next, H, W, batch_size);
        std::swap(d_curr, d_next); 
        std::swap(curr_idx, next_idx); 
    }

    int max_dim = std::max((int)H, (int)W);
    int step = 16; 

    while (step < max_dim) {
        jfa_step_global_2d_opt<<<grid, block>>>(d_curr, d_next, step, H, W, numel);
        std::swap(d_curr, d_next);
        std::swap(curr_idx, next_idx);
        step *= 2;
    }
    
    auto final_dist = torch::empty_like(input);
    calc_dist_kernel_2d_opt<<<grid, block>>>(d_curr, final_dist.data_ptr<float>(), numel, H, W);

    return std::make_tuple(final_dist, curr_idx);
}


std::tuple<torch::Tensor, torch::Tensor> run_jfa_3d(
    torch::Tensor input, int64_t D, int64_t H, int64_t W, int grid, int block, int64_t numel
) {
    bool use_int16 = (D < 32767 && H < 32767 && W < 32767);
    auto index_opts = input.options().dtype(use_int16 ? torch::kInt16 : torch::kInt32);
    
    int64_t batch = numel / (D * H * W);
    
    // (3, Batch, D, H, W)
    auto curr_idx_soa = torch::empty({3, batch, D, H, W}, index_opts);
    auto next_idx_soa = torch::empty({3, batch, D, H, W}, index_opts);
    
    void* d_curr = curr_idx_soa.data_ptr();
    void* d_next = next_idx_soa.data_ptr();
    int64_t plane_stride = numel; // B*D*H*W

    // 1. Init
    if (use_int16) {
        int16_t* ptr = (int16_t*)d_curr;
        init_jfa_kernel_3d_soa<int16_t><<<grid, block>>>(
            input.data_ptr<float>(), ptr, ptr + plane_stride, ptr + 2 * plane_stride, numel, D, H, W
        );
    } else {
        int32_t* ptr = (int32_t*)d_curr;
        init_jfa_kernel_3d_soa<int32_t><<<grid, block>>>(
            input.data_ptr<float>(), ptr, ptr + plane_stride, ptr + 2 * plane_stride, numel, D, H, W
        );
    }

    // 2. Fused Steps
    int block_dim = 8;
    int blocks_per_d = (D + block_dim - 1) / block_dim;
    dim3 fused_block(block_dim, block_dim, block_dim);
    dim3 fused_grid((W + block_dim - 1) / block_dim, (H + block_dim - 1) / block_dim, blocks_per_d * batch);
    size_t smem_bytes = (14*14*14) * 3 * (use_int16 ? 2 : 4);

    if (use_int16) {
        int16_t* c = (int16_t*)d_curr;
        int16_t* n = (int16_t*)d_next;
        jfa_block_fused_kernel_3d_soa<int16_t><<<fused_grid, fused_block, smem_bytes>>>(
            c, c + plane_stride, c + 2 * plane_stride, 
            n, n + plane_stride, n + 2 * plane_stride, 
            D, H, W, blocks_per_d
        );
    } else {
        int32_t* c = (int32_t*)d_curr;
        int32_t* n = (int32_t*)d_next;
        jfa_block_fused_kernel_3d_soa<int32_t><<<fused_grid, fused_block, smem_bytes>>>(
            c, c + plane_stride, c + 2 * plane_stride, 
            n, n + plane_stride, n + 2 * plane_stride, 
            D, H, W, blocks_per_d
        );
    }
    std::swap(d_curr, d_next); 

    // 3. Global Steps
    int max_dim = std::max({(int)D, (int)H, (int)W});
    int step = 4;
    while (step < max_dim) {
        if (use_int16) {
            int16_t* c = (int16_t*)d_curr;
            int16_t* n = (int16_t*)d_next;
            jfa_step_3d_soa<int16_t><<<grid, block>>>(
                c, c + plane_stride, c + 2 * plane_stride, 
                n, n + plane_stride, n + 2 * plane_stride, 
                step, D, H, W, numel
            );
        } else {
            int32_t* c = (int32_t*)d_curr;
            int32_t* n = (int32_t*)d_next;
            jfa_step_3d_soa<int32_t><<<grid, block>>>(
                c, c + plane_stride, c + 2 * plane_stride, 
                n, n + plane_stride, n + 2 * plane_stride, 
                step, D, H, W, numel
            );
        }
        std::swap(d_curr, d_next);
        step *= 2;
    }

    // 4. Final Dist
    auto final_dist = torch::empty_like(input);
    if (use_int16) {
        int16_t* c = (int16_t*)d_curr;
        calc_dist_kernel_3d_soa<int16_t><<<grid, block>>>(
            c, c + plane_stride, c + 2 * plane_stride, 
            final_dist.data_ptr<float>(), numel, D, H, W
        );
    } else {
        int32_t* c = (int32_t*)d_curr;
        calc_dist_kernel_3d_soa<int32_t><<<grid, block>>>(
            c, c + plane_stride, c + 2 * plane_stride, 
            final_dist.data_ptr<float>(), numel, D, H, W
        );
    }

    // Permute result indices back to (Batch, D, H, W, 3)
    torch::Tensor result_indices;
    if (d_curr == curr_idx_soa.data_ptr()) result_indices = curr_idx_soa;
    else result_indices = next_idx_soa;
    
    result_indices = result_indices.permute({1, 2, 3, 4, 0}).contiguous();

    return std::make_tuple(final_dist, result_indices);
}

std::tuple<torch::Tensor, torch::Tensor> run_separable_ndim(torch::Tensor input) {
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Separable N-Dim input must be float32.");
    input = input.contiguous();
    
    const int ndim = input.dim(); 
    const int sample_ndim = ndim - 1; 
    TORCH_CHECK(sample_ndim > 0 && sample_ndim <= 8, "Unsupported dims for Separable EDT");
    
    auto shape = input.sizes().vec();
    int64_t num_pixels = input.numel();

    auto current_dist = torch::where(input == 0, 
                                    torch::tensor(0.0f, input.options()), 
                                    torch::tensor(INF_VAL, input.options()));
    
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

    for (int d = 1; d < ndim; ++d) {
        bool is_final_pass = (d == ndim - 1);
        
        auto dist_in = current_dist.transpose(d, ndim - 1).contiguous();
        auto idx_in  = current_idx.transpose(d, ndim - 1).contiguous(); 
        
        auto dist_out = torch::empty(dist_in.sizes(), dist_in.options());
        auto idx_out  = torch::empty(idx_in.sizes(), idx_in.options());

        int64_t L = dist_in.size(-1); 
        int64_t total_slices = dist_in.numel() / L; 
        int threads = std::min((int64_t)MAX_THREADS, L);
        
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
// PART 5: MAIN ENTRY POINT
// ==================================================================

std::tuple<torch::Tensor, torch::Tensor> distance_transform_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    input = input.contiguous();

    int64_t dims = input.dim();
    int64_t numel = input.numel();
    int block = BLOCK_SIZE;
    int grid = (numel + block - 1) / block;

    if (dims >= 5) {
        return run_separable_ndim(input);
    }
    else if (dims == 4) {
        int64_t dim1 = input.size(1); 
        if (dim1 == 1) {
            int64_t H = input.size(-2);
            int64_t W = input.size(-1);
            return run_jfa_2d(input, H, W, grid, block, numel);
        } 
        else {
            int64_t D = dim1;
            int64_t H = input.size(-2);
            int64_t W = input.size(-1);
            return run_jfa_3d(input, D, H, W, grid, block, numel);
        }
    }
    else if (dims == 3) {
        int64_t H = input.size(-2);
        int64_t W = input.size(-1);
        return run_jfa_2d(input, H, W, grid, block, numel);
    }
    else if (dims == 2) {
        int64_t H = 1;
        int64_t W = input.size(-1);
        auto result = run_jfa_2d(input, H, W, grid, block, numel);
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