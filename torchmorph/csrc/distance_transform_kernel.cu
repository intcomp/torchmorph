#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

// ------------------------------------------------------------------
// 配置常量
// ------------------------------------------------------------------
#define INF_VAL 1e8f
#define MAX_THREADS 1024
// Shared Memory 限制: 48KB 一般安全。
// 每个像素需要: float(val) + int(idx1) + int(idx2) = 12 bytes
// 4096 * 12 = 48KB.
#define SMEM_LIMIT_ELEMENTS 4096 

// ------------------------------------------------------------------
// Device Helper Functions
// ------------------------------------------------------------------

__device__ __forceinline__ float sqr(float x) { return x * x; }

// 计算 JFA 代价: (q - p)^2 + weight[p]
__device__ __forceinline__ float compute_cost(int q, int p, float val_p) {
    if (p < 0) return INF_VAL; 
    return sqr((float)q - (float)p) + val_p;
}

// ------------------------------------------------------------------
// JFA Core Logic (Device Only)
// ------------------------------------------------------------------
// 核心 JFA 逻辑，与数据位置无关 (Shared 或 Global 均通用)
__device__ void run_jfa_core(
    int N,
    int tid,
    const float* __restrict__ vals,  // 输入权重 (只读)
    int* __restrict__ idx_curr,      // Ping-Pong Buffer A
    int* __restrict__ idx_next       // Ping-Pong Buffer B
) {
    // 1. 初始化: 根据 vals 决定是否是有效源点
    for (int i = tid; i < N; i += blockDim.x) {
        if (vals[i] >= INF_VAL * 0.9f) {
            idx_curr[i] = -1; // 背景
        } else {
            idx_curr[i] = i;  // 物体/源点，初始索引指向自己
        }
    }
    __syncthreads();

    // 2. 迭代传播 (Step = 1, 2, 4, ... < N)
    int* idx_in = idx_curr;
    int* idx_out = idx_next;

    for (int step = 1; step < N; step *= 2) {
        for (int i = tid; i < N; i += blockDim.x) {
            int my_best_p = idx_in[i];
            float min_cost = INF_VAL;

            // 检查自己当前的最优解
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

    // 3. 确保最终结果在 idx_curr (如果循环结束时在 next，则拷回)
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
// 模板参数 NDim: 如果 > 0，编译器会展开循环优化。
// 参数 runtime_ndim: 如果 NDim == 0 (Default case)，使用该参数作为维度。
template <bool IsFinal, int NDim>
__global__ void edt_kernel_shared(
    const float* __restrict__ in_data,       // 输入 Dist^2
    const int32_t* __restrict__ in_indices,  // 输入 Indices
    float* __restrict__ out_dist,            // 输出 Dist (IsFinal ? sqrt : sqr)
    int32_t* __restrict__ out_indices,       // 输出 Indices
    int64_t L,                               // 当前维度的长度
    int64_t total_elements,                  // 总像素数
    int runtime_ndim                         // 运行时维度 (fallback)
) {
    // 确定实际维度
    const int D = (NDim > 0) ? NDim : runtime_ndim;

    // 计算行偏移
    int64_t row_idx = blockIdx.x;
    int64_t offset = row_idx * L;
    
    if (offset >= total_elements) return;

    // Shared Memory 布局
    extern __shared__ char s_buffer[];
    float* s_vals = (float*)s_buffer;
    int*   s_idx1 = (int*)(s_vals + L);
    int*   s_idx2 = (int*)(s_idx1 + L);

    // 1. 加载 Dist 到 Shared Memory
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        s_vals[i] = __ldg(&in_data[offset + i]);
    }
    __syncthreads();

    // 2. 运行 JFA 核心
    run_jfa_core(L, threadIdx.x, s_vals, s_idx1, s_idx2);
    
    // 3. 写回结果
    for (int q = threadIdx.x; q < L; q += blockDim.x) {
        int p = s_idx1[q]; // 最近点在当前行内的局部索引 (0..L-1)
        float dist_val;

        // 计算新距离
        if (p != -1) {
            float dist_sq = sqr((float)q - (float)p) + s_vals[p];
            dist_val = IsFinal ? sqrtf(dist_sq) : dist_sq;
        } else {
            dist_val = IsFinal ? INF_VAL : sqr(INF_VAL);
            p = 0; // 防止越界，随便指一个
        }
        out_dist[offset + q] = dist_val;

        // 索引传播: Copy Vector [D]
        if (p != -1) {
            int64_t src_offset = (offset + p) * D;
            int64_t dst_offset = (offset + q) * D;
            
            // 如果 NDim > 0，这里会完全展开，非常快
            for (int d = 0; d < D; ++d) {
                out_indices[dst_offset + d] = in_indices[src_offset + d];
            }
        } else {
             // 找不到源点（全图都是背景的情况）
             int64_t dst_offset = (offset + q) * D;
             for (int d = 0; d < D; ++d) out_indices[dst_offset + d] = 0;
        }
    }
}

// ------------------------------------------------------------------
// Kernel 2: Global Memory JFA (Fallback Path)
// ------------------------------------------------------------------
// 逻辑同上，只是用 Global Memory 做 Ping-Pong Buffer
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

    // 指向 Global Memory 的指针
    int* g_idx1 = global_buffer_1 + offset;
    int* g_idx2 = global_buffer_2 + offset;
    
    // 1. & 2. 运行 JFA (直接在 Global Mem 上读写)
    run_jfa_core(L, threadIdx.x, in_data + offset, g_idx1, g_idx2);

    // 3. 写回结果
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
// 初始化索引张量为网格坐标
// indices shape: (..., D)
__global__ void init_indices_kernel(
    int32_t* indices, 
    int64_t total_pixels, 
    int NDim, 
    const int64_t* __restrict__ shape_ptr // shape of spatial dimensions
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    // 反解坐标 (Unravel Index)
    // idx 是每个像素的 flat index
    // 我们需要计算它在 spatial_shape 中的坐标
    
    int64_t temp = idx;
    // 使用本地寄存器数组避免多次全局内存读取 (假设最大 10 维)
    int32_t coords[10]; 

    // 假设 spatial_shape 是 [D0, D1, D2]
    // 倒序计算除余
    for (int d = NDim - 1; d >= 0; --d) {
        int64_t dim_size = shape_ptr[d];
        coords[d] = temp % dim_size;
        temp /= dim_size;
    }

    // 写入 Global Memory
    // Indices tensor 是 (TotalPixels, NDim) 扁平化的
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
    
    // 处理 Batch 维度：如果输入是 1D (L)，视为无 Batch，但在处理中统一加一个 Batch 维方便
    // 标准约定：Input shape (Batch, D1, D2, ..., Dn)
    // 算法对 Batch 维度和其他维度处理其实是一样的（视为无关维度）
    // 但索引初始化需要知道哪些是 "Spatial Dimensions"。
    // 这里假设：输入的所有维度除了 Batch (Dim 0) 外都是空间维度。
    
    const int ndim = input.dim(); 
    // 如果 ndim=1, 假设是 (L)，sample_ndim=1
    // 如果 ndim=4 (B, C, H, W)，sample_ndim=3 (C,H,W 都算空间? 通常 C 也是独立处理的)
    // **修正**: 标准 EDT 通常是在 (H, W) 或 (D, H, W) 上进行的。
    // 如果有 Channel，通常 Channel 也是独立的。
    // 为了最通用，我们将 **除了第0维(Batch)** 以外的所有维度都视为空间维度进行索引记录。
    // 如果用户输入没有 Batch 维，请在 Python 端 unsqueeze(0)。
    
    // 假设输入已经是 (Batch, ...Spatial...)
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

    // 1. 初始化 Distance Tensor
    // 0 -> 0, 1 -> INF
    auto current_dist = torch::where(input == 0, 
                                     torch::tensor(0.0f, input.options()), 
                                     torch::tensor(INF_VAL, input.options()));
    
    // 2. 初始化 Index Tensor
    // Shape: (Batch, D1, ..., Dn, sample_ndim)
    auto index_shape = shape;
    index_shape.push_back(sample_ndim);
    auto current_idx = torch::empty(index_shape, input.options().dtype(torch::kInt32));
    
    // 2.1 准备 Shape 数据传给 Kernel
    std::vector<int64_t> spatial_shape(shape.begin() + 1, shape.end());
    auto shape_tensor = torch::tensor(spatial_shape, torch::kInt64).to(input.device());

    // 2.2 运行初始化 Kernel
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

    // 预分配 Global Memory Buffer (懒加载)
    torch::Tensor global_buf1, global_buf2;

    // 3. 逐维处理 (Separable JFA)
    // 遍历每一个空间维度 (从 1 到 ndim-1)
    for (int d = 1; d < ndim; ++d) {
        bool is_final_pass = (d == ndim - 1);
        
        // --- Step A: Transpose current dim to last ---
        // 变换后 Shape: (..., L)
        auto dist_in = current_dist.transpose(d, ndim - 1).contiguous();
        auto idx_in  = current_idx.transpose(d, ndim - 1).contiguous(); 
        
        int64_t L = dist_in.size(-1); // 当前维度的长度
        int64_t total_slices = dist_in.numel() / L; 
        
        auto dist_out = torch::empty_like(dist_in);
        auto idx_out  = torch::empty_like(idx_in);

        // --- Step B: Kernel Dispatch ---
        int threads = std::min((int64_t)MAX_THREADS, L);
        
        // 检查是否可以使用 Shared Memory
        if (L <= SMEM_LIMIT_ELEMENTS) {
            size_t smem_size = L * (sizeof(float) + 2 * sizeof(int));
            
            // 使用 Switch 宏来处理常用的维度模板特化
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
            // Global Memory Fallback (L > 4096)
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

