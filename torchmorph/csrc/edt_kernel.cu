#include <torch/extension.h>
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

// JFA Configuration
#define BLOCK_SIZE 256
#define SMEM_LIMIT_ELEMENTS 4096
#define JFA_BLOCK_DIM 32
#define JFA_FUSED_STEPS 4
#define JFA_MAX_OFFSET 8
#define JFA_SMEM_DIM (JFA_BLOCK_DIM + 2 * JFA_MAX_OFFSET)
#define JFA_3D_BLOCK 8
#define JFA_3D_HALO 1

// ==============================================================================
// JFA Device Helpers
// ==============================================================================
__device__ __forceinline__ float sqr(float x) { return x * x; }

__device__ __forceinline__ float dist_sq_2d(int y1, int x1, int y2, int x2) {
    return sqr((float)(y1 - y2)) + sqr((float)(x1 - x2));
}

__device__ __forceinline__ float dist_sq_3d_soa(int z1, int y1, int x1, int z2, int y2, int x2) {
    if (z2 == -1) return INF_VAL;
    float dz = (float)(z1 - z2);
    float dy = (float)(y1 - y2);
    float dx = (float)(x1 - x2);
    return dz*dz + dy*dy + dx*dx;
}

__device__ __forceinline__ float compute_cost(int q, int p, float val_p) {
    if (p < 0 || val_p >= INF_VAL) return INF_VAL;
    return sqr((float)q - (float)p) + val_p;
}

__device__ __forceinline__ float dist_sq_int2(int y, int x, int2 seed) {
    if (seed.x == -1) return INF_VAL;
    float dy = (float)(y - seed.x);
    float dx = (float)(x - seed.y);
    return dy*dy + dx*dx;
}

// ==============================================================================
// JFA 2D Kernels (Vectorized int2 + Block Shared)
// ==============================================================================
__global__ void init_jfa_2d_opt_kernel(
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

__global__ void jfa_block_fused_2d_kernel(
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

__global__ void jfa_step_global_2d_opt_kernel(
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

__global__ void calc_dist_2d_opt_kernel(
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

// ==============================================================================
// JFA 3D Kernels (Optimized SoA Layout)
// ==============================================================================
template <typename IndexType>
__global__ void init_jfa_3d_soa_kernel(
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
__global__ void jfa_block_fused_3d_soa_kernel(
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
__global__ void jfa_step_3d_soa_kernel(
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
__global__ void calc_dist_3d_soa_kernel(
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
            output[out_idx] = INF_VAL;
            if (compute_indices) {
                output_idx_y[out_idx] = 0;
                output_idx_x[out_idx] = col_idx;
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

        init_distance_2d_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            distance.data_ptr<float>(),
            return_indices ? indices_y.data_ptr<int>() : nullptr,
            return_indices ? indices_x.data_ptr<int>() : nullptr,
            height, width, batch_stride,
            return_indices
        );
    }

    // Step 2: Row-wise EDT (X direction) - shared memory
    {
        int64_t num_rows = batch_size * height;
        int threads = min(width, MAX_THREADS);
        size_t shared_mem_size = width * sizeof(float) +      // v_val
                                  width * sizeof(int) +        // v_idx
                                  (width + 1) * sizeof(float); // z

        edt_2d_rows_kernel<<<num_rows, threads, shared_mem_size>>>(
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

        edt_2d_cols_kernel<<<num_cols, threads, shared_mem_size>>>(
            temp.data_ptr<float>(),
            distance.data_ptr<float>(),
            return_indices ? temp_idx_y.data_ptr<int>() : nullptr,
            return_indices ? temp_idx_x.data_ptr<int>() : nullptr,
            return_indices ? indices_y.data_ptr<int>() : nullptr,
            return_indices ? indices_x.data_ptr<int>() : nullptr,
            height, width, batch_stride,
            spacing_y,
            true,  // is_final
            return_indices
        );
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
            output[out_idx] = INF_VAL;
            if (compute_indices) {
                for (int d = 0; d < spatial_ndim; d++) {
                    output_idx[d * num_pixels + out_idx] = 0;
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
            output[out_idx] = INF_VAL;
            if (compute_indices) {
                for (int d = 0; d < spatial_ndim; d++) {
                    output_idx[d * num_pixels + out_idx] = 0;
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

    // Initialize distances and indices
    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;

    init_distance_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        distance.data_ptr<float>(),
        return_indices ? indices.data_ptr<int>() : nullptr,
        total_pixels,
        total_ndim,
        spatial_ndim,
        shape_tensor.data_ptr<int64_t>(),
        return_indices
    );

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

            edt_1d_kernel<<<num_slices, kernel_threads, shared_mem_size>>>(
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
                is_final,
                return_indices
            );
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

            edt_1d_global_kernel<<<num_slices, kernel_threads>>>(
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
                is_final,
                return_indices
            );
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
// JFA Dispatch Helpers
// ==============================================================================
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

    init_jfa_2d_opt_kernel<<<grid, block>>>(
        input.data_ptr<float>(), d_curr, numel, H, W
    );

    {
        dim3 dimBlock(JFA_BLOCK_DIM, JFA_BLOCK_DIM);
        int64_t batch_size = numel / (H * W);
        dim3 dimGrid((W + JFA_BLOCK_DIM - 1) / JFA_BLOCK_DIM,
                     (H + JFA_BLOCK_DIM - 1) / JFA_BLOCK_DIM,
                     batch_size);

        jfa_block_fused_2d_kernel<<<dimGrid, dimBlock>>>(d_curr, d_next, H, W, batch_size);
        std::swap(d_curr, d_next);
        std::swap(curr_idx, next_idx);
    }

    int max_dim = std::max((int)H, (int)W);
    int step = 16;

    while (step < max_dim) {
        jfa_step_global_2d_opt_kernel<<<grid, block>>>(d_curr, d_next, step, H, W, numel);
        std::swap(d_curr, d_next);
        std::swap(curr_idx, next_idx);
        step *= 2;
    }

    auto final_dist = torch::empty_like(input);
    calc_dist_2d_opt_kernel<<<grid, block>>>(d_curr, final_dist.data_ptr<float>(), numel, H, W);

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
        init_jfa_3d_soa_kernel<int16_t><<<grid, block>>>(
            input.data_ptr<float>(), ptr, ptr + plane_stride, ptr + 2 * plane_stride, numel, D, H, W
        );
    } else {
        int32_t* ptr = (int32_t*)d_curr;
        init_jfa_3d_soa_kernel<int32_t><<<grid, block>>>(
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
        jfa_block_fused_3d_soa_kernel<int16_t><<<fused_grid, fused_block, smem_bytes>>>(
            c, c + plane_stride, c + 2 * plane_stride,
            n, n + plane_stride, n + 2 * plane_stride,
            D, H, W, blocks_per_d
        );
    } else {
        int32_t* c = (int32_t*)d_curr;
        int32_t* n = (int32_t*)d_next;
        jfa_block_fused_3d_soa_kernel<int32_t><<<fused_grid, fused_block, smem_bytes>>>(
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
            jfa_step_3d_soa_kernel<int16_t><<<grid, block>>>(
                c, c + plane_stride, c + 2 * plane_stride,
                n, n + plane_stride, n + 2 * plane_stride,
                step, D, H, W, numel
            );
        } else {
            int32_t* c = (int32_t*)d_curr;
            int32_t* n = (int32_t*)d_next;
            jfa_step_3d_soa_kernel<int32_t><<<grid, block>>>(
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
        calc_dist_3d_soa_kernel<int16_t><<<grid, block>>>(
            c, c + plane_stride, c + 2 * plane_stride,
            final_dist.data_ptr<float>(), numel, D, H, W
        );
    } else {
        int32_t* c = (int32_t*)d_curr;
        calc_dist_3d_soa_kernel<int32_t><<<grid, block>>>(
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

// ==============================================================================
// JFA Main Entry Point
// ==============================================================================
std::tuple<torch::Tensor, torch::Tensor> distance_transform_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    input = input.contiguous();

    int64_t dims = input.dim();
    int64_t numel = input.numel();
    int block = BLOCK_SIZE;
    int grid = (numel + block - 1) / block;

    if (dims >= 5) {
        // For 4D+ spatial, fall back to separable algorithm
        int spatial_ndim = dims - 1;
        std::vector<float> sampling(spatial_ndim, 1.0f);
        return run_edt_separable(input, sampling, true);
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

// ==============================================================================
// Python binding entry point
// ==============================================================================

std::tuple<torch::Tensor, torch::Tensor> edt_cuda(
    torch::Tensor input,
    std::vector<float> sampling,
    bool return_distances,
    bool return_indices,
    const std::string& algorithm
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");

    int total_ndim = input.dim();

    // Handle empty sampling (default to unit spacing for all spatial dimensions)
    if (sampling.empty()) {
        // Assume all dimensions are spatial if no sampling provided
        // But typically input is (B, C, spatial...) so use total_ndim - 2
        int spatial_ndim = total_ndim >= 3 ? total_ndim - 2 : total_ndim;
        sampling.resize(spatial_ndim, 1.0f);
    }

    int spatial_ndim = sampling.size();

    // Check if we can use JFA algorithm
    bool can_use_jfa = true;

    // JFA doesn't support non-unit sampling
    for (float s : sampling) {
        if (std::abs(s - 1.0f) > 1e-6f) {
            can_use_jfa = false;
            break;
        }
    }

    // JFA only supports 2D and 3D (spatial dimensions)
    if (spatial_ndim > 3) {
        can_use_jfa = false;
    }

    // Determine which algorithm to use
    bool use_jfa = false;
    if (algorithm == "jfa") {
        if (can_use_jfa) {
            use_jfa = true;
        } else {
            // Fall back to exact with warning (or we can throw)
            // For now, silently fall back to exact
            use_jfa = false;
        }
    } else if (algorithm == "exact") {
        use_jfa = false;
    } else if (algorithm == "auto") {
        // Auto mode: use JFA only for 2D with unit sampling
        // For 3D, exact algorithm performs better in practice
        use_jfa = can_use_jfa && (spatial_ndim == 2);
    } else {
        TORCH_CHECK(false, "algorithm must be 'exact', 'jfa', or 'auto', got: ", algorithm);
    }

    if (use_jfa) {
        // Use JFA algorithm
        auto [distances, indices_result] = distance_transform_cuda(input);

        if (!return_indices) {
            indices_result = torch::Tensor();
        }

        return std::make_tuple(distances, indices_result);
    }

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
