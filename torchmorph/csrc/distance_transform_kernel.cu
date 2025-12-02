#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

#define INF_VAL 1e8f
#define MAX_THREADS 1024
#define SMEM_LIMIT_ELEMENTS 4096 // 48KB / 12 bytes (float+int+int) ~= 4096

__device__ __forceinline__ float sqr(float x) { return x * x; }

// 计算从像素 q 到源点 p 的距离代价
// val_p 是源点 p 在上一轮计算后的距离平方值 (weight)
__device__ __forceinline__ float compute_cost(int q, int p, float val_p) {
    if (p < 0) return INF_VAL; 
    return sqr((float)q - (float)p) + val_p;
}

// ------------------------------------------------------------------
// JFA 核心逻辑 (Device Function)
// ------------------------------------------------------------------
// 无论数据是在 Shared Memory 还是 Global Memory，逻辑是一样的
__device__ void run_jfa_core(
    int N,
    int tid,
    const float* __restrict__ vals,  // 输入权重 (只读)
    int* __restrict__ idx_curr,      // Ping-Pong Buffer A
    int* __restrict__ idx_next       // Ping-Pong Buffer B
) {
    // 1. 初始化
    for (int i = tid; i < N; i += blockDim.x) {
        // 如果输入值很大，说明是背景，没有初始源点
        if (vals[i] >= INF_VAL * 0.9f) {
            idx_curr[i] = -1;
        } else {
            idx_curr[i] = i;
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

            if (my_best_p != -1) {
                min_cost = compute_cost(i, my_best_p, vals[my_best_p]);
            }

            // Check Left
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

            // Check Right
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
// 适用于 N <= 4096
// ------------------------------------------------------------------
template <bool IsFinal, int NDim>
__global__ void edt_kernel_shared(
    const float* __restrict__ in_data,       // 当前维度的输入 (dist^2)
    const int32_t* __restrict__ in_indices,  // 上一轮的索引图 (N_slices, L, NDim)
    float* __restrict__ out_dist,            // 输出距离
    int32_t* __restrict__ out_indices,       // 输出索引图
    int64_t L,                               // 当前维度的长度 (Length)
    int64_t total_elements                   // Batch * ... * L
) {
    // 这里的 total_elements 是展平后的总像素数
    // 由于我们做了 transpose，数据布局是 [Batch_and_other_dims, L]
    // 每个 Block 处理一行 (长度 L)
    
    int64_t row_idx = blockIdx.x;
    int64_t offset = row_idx * L;
    
    if (offset >= total_elements) return;

    // Shared Memory 布局: float vals[L], int idx1[L], int idx2[L]
    extern __shared__ char s_buffer[];
    float* s_vals = (float*)s_buffer;
    int*   s_idx1 = (int*)(s_vals + L);
    int*   s_idx2 = (int*)(s_idx1 + L);

    // 1. 加载数据到 Shared Memory
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        float val = __ldg(&in_data[offset + i]);
        // 如果是初始 Pass (无输入索引)，val 为 0 或 INF
        // 如果是后续 Pass，val 为上一步的 dist^2
        s_vals[i] = val;
    }
    __syncthreads();

    // 2. 运行 JFA
    run_jfa_core(L, threadIdx.x, s_vals, s_idx1, s_idx2);
    
    // 3. 写回结果
    for (int q = threadIdx.x; q < L; q += blockDim.x) {
        int p = s_idx1[q]; // 最近点在当前行内的局部索引 (0..L-1)
        float dist_val;

        if (p != -1) {
            // 计算新距离: (q-p)^2 + val[p]
            float dist_sq = sqr((float)q - (float)p) + s_vals[p];
            dist_val = IsFinal ? sqrtf(dist_sq) : dist_sq;
        } else {
            dist_val = IsFinal ? INF_VAL : sqr(INF_VAL);
            p = 0; // fallback
        }

        out_dist[offset + q] = dist_val;

        // 4. 索引传播
        // 我们需要从 in_indices 查找完整的高维索引
        // in_indices 形状: [Batch..., L, NDim]
        // 这里的 offset 对应 [Batch..., 0]
        // p 是当前维度的偏移
        if (p != -1) {
            int64_t src_offset = (offset + p) * NDim;
            int64_t dst_offset = (offset + q) * NDim;
            
            // 手动展开拷贝，或者循环
            for (int d = 0; d < NDim; ++d) {
                out_indices[dst_offset + d] = in_indices[src_offset + d];
            }
        } else {
            // 保持原样或填0 (通常保持原样即可，或者为了安全填0)
             int64_t dst_offset = (offset + q) * NDim;
             for (int d = 0; d < NDim; ++d) out_indices[dst_offset + d] = 0;
        }
    }
}

// ------------------------------------------------------------------
// Kernel 2: Global Memory JFA (Fallback Path)
// 适用于 N > 4096，使用 Global Memory 作为 Ping-Pong Buffer
// ------------------------------------------------------------------
template <bool IsFinal, int NDim>
__global__ void edt_kernel_global(
    const float* __restrict__ in_data,
    const int32_t* __restrict__ in_indices,
    float* __restrict__ out_dist,
    int32_t* __restrict__ out_indices,
    int* __restrict__ global_buffer_1, // 临时 buffer A [TotalElements]
    int* __restrict__ global_buffer_2, // 临时 buffer B [TotalElements]
    int64_t L,
    int64_t total_elements
) {
    int64_t row_idx = blockIdx.x;
    int64_t offset = row_idx * L;
    
    if (offset >= total_elements) return;

    // 指向当前行在 Global Memory 中的位置
    // 注意：in_data 是只读的，我们需要把它当做 weight
    // JFA 需要两个 int buffer 来存 index
    int* g_idx1 = global_buffer_1 + offset;
    int* g_idx2 = global_buffer_2 + offset;
    
    // 直接在 Global Memory 上运行 JFA
    // 注意：这里 vals 指针直接指向 in_data (Global)，读取稍慢但无需拷贝
    run_jfa_core(L, threadIdx.x, in_data + offset, g_idx1, g_idx2);

    // 写回逻辑同上
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
            int64_t src_offset = (offset + p) * NDim;
            int64_t dst_offset = (offset + q) * NDim;
            for (int d = 0; d < NDim; ++d) {
                out_indices[dst_offset + d] = in_indices[src_offset + d];
            }
        } else {
             int64_t dst_offset = (offset + q) * NDim;
             for (int d = 0; d < NDim; ++d) out_indices[dst_offset + d] = 0;
        }
    }
}


// ------------------------------------------------------------------
// 辅助：初始化索引张量
// ------------------------------------------------------------------
// 将 index tensor 初始化为 grid grid coordinates
// shape: (..., D), 最后一个维度存坐标
__global__ void init_indices_kernel(int32_t* indices, int64_t total_elements, int NDim, 
                                    const int64_t* shape, const int64_t* strides) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // 反解坐标
    int64_t temp = idx;
    int32_t coords[10]; // max dims

    // strides 是针对 elements 展开的，但这里 indices 是 (Total, NDim)
    // 我们可以简单地根据 shape 反解
    // 注意：这里的 total_elements 是像素数，不是 indices 数组的大小
    
    // 假设 shape 是 [D0, D1, D2]
    // idx 对应 flat index
    
    for (int d = NDim - 1; d >= 0; --d) {
        coords[d] = temp % shape[d];
        temp /= shape[d];
    }

    // 写入
    for (int d = 0; d < NDim; ++d) {
        indices[idx * NDim + d] = coords[d];
    }
}

// ------------------------------------------------------------------
// Host 函数
// ------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> distance_transform_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA device.");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32.");
    
    input = input.contiguous();
    bool had_no_batch_dim = (input.dim() == 1);
    if (had_no_batch_dim) input = input.unsqueeze(0);

    const int ndim = input.dim(); // Include batch
    const int sample_ndim = ndim - 1; 
    auto shape = input.sizes().vec();
    int64_t num_pixels = input.numel();

    if (num_pixels == 0) {
        auto index_shape = shape;
        index_shape.push_back(sample_ndim > 0 ? sample_ndim : 1);
        return std::make_tuple(torch::empty_like(input), 
                               torch::empty(index_shape, input.options().dtype(torch::kInt32)));
    }

    // 1. 初始化输出 Tensor
    // current_dist 在迭代过程中存储 dist^2，最后开方
    // 初始状态：Input 里的 0 还是 0，其他非 0 (背景) 设为 INF
    auto current_dist = torch::where(input == 0, 
                                     torch::tensor(0.0f, input.options()), 
                                     torch::tensor(INF_VAL, input.options()));
    
    // 初始化索引 Map (Batch, ..., NDim)
    auto index_shape = shape;
    index_shape.push_back(sample_ndim);
    auto current_idx = torch::empty(index_shape, input.options().dtype(torch::kInt32));
    
    // 启动 Kernel 初始化索引
    // 为了反解坐标，我们需要把 shape 传进去
    {
        // 排除 batch 维度的 shape 用于坐标计算? 
        // 需求是：返回的索引是 (batch_idx, z, y, x) 还是只是 (z, y, x)?
        // 通常 EDT 返回的是 sample 内的坐标。所以我们忽略 batch 维度。
        std::vector<int64_t> sample_shape_vec(shape.begin() + 1, shape.end());
        auto sample_shape_tensor = torch::tensor(sample_shape_vec, torch::kInt64).to(input.device());
        // 这里的 strides 不需要，直接由 shape 反解
        
        int threads = 256;
        int blocks = (num_pixels + threads - 1) / threads;
        
        // 我们需要传递 sample_ndim
        init_indices_kernel<<<blocks, threads>>>(
            current_idx.data_ptr<int32_t>(), 
            num_pixels, 
            sample_ndim,
            sample_shape_tensor.data_ptr<int64_t>(),
            nullptr // strides not needed for simple unravel
        );
    }
    
    // 用于 Global Memory Fallback 的临时 buffer
    torch::Tensor global_buf1, global_buf2;

    // 2. 逐维处理 (Separable Phases)
    // 从最后一个维度倒着处理，或者顺序处理都可以。
    // 为了 Host Transpose 方便，我们遍历 sample 的每一个维度 (1 到 ndim-1)
    for (int d = 1; d < ndim; ++d) {
        bool is_final_pass = (d == ndim - 1);
        
        // -----------------------------------------------------------
        // Step A: Permute & Contiguous
        // 将当前处理维度 d 移到最后: (0, 1, ..., d, ..., N-1) -> (0, 1, ..., N-1, d)
        // 这样最后内存布局就是 [..., L]，stride=1
        // -----------------------------------------------------------
        
        // 这种 swap 策略比较简单: transpose(d, -1)
        // 注意：index tensor 也要变换，但 index tensor 最后一维是 coord_dim，不能乱动。
        // Index tensor 形状是 [..., sample_ndim]。
        // 我们需要变换的是前面的空间维度 [...]。
        
        auto dist_in = current_dist.transpose(d, ndim - 1).contiguous();
        auto idx_in  = current_idx.transpose(d, ndim - 1).contiguous(); 
        
        // 此时 dist_in shape: [..., L]
        // idx_in shape: [..., L, sample_ndim]
        
        int64_t L = dist_in.size(-1); // 当前维度的长度
        int64_t total_slices = dist_in.numel() / L; // 有多少行
        
        auto dist_out = torch::empty_like(dist_in);
        auto idx_out  = torch::empty_like(idx_in);

        // -----------------------------------------------------------
        // Step B: Kernel Dispatch
        // -----------------------------------------------------------
        int threads = std::min((int64_t)MAX_THREADS, L);
        
        // 检查 Shared Memory 需求
        // Need: float(4) + int(4) + int(4) = 12 bytes per pixel
        if (L <= SMEM_LIMIT_ELEMENTS) {
            size_t smem_size = L * (sizeof(float) + 2 * sizeof(int));
            
            // 模板参数 NDim 需要是编译期常量。
            // 动态分发 sample_ndim (1D, 2D, 3D usually)
            // 使用 switch case 覆盖常见维度 (1, 2, 3)
            #define DISPATCH_SHARED(IS_FINAL) \
                switch(sample_ndim) { \
                    case 1: edt_kernel_shared<IS_FINAL, 1><<<total_slices, threads, smem_size>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        L, dist_in.numel()); break; \
                    case 2: edt_kernel_shared<IS_FINAL, 2><<<total_slices, threads, smem_size>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        L, dist_in.numel()); break; \
                    case 3: edt_kernel_shared<IS_FINAL, 3><<<total_slices, threads, smem_size>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        L, dist_in.numel()); break; \
                    default: /* fallback for >3D */ break; \
                }

            if (is_final_pass) { DISPATCH_SHARED(true); } 
            else { DISPATCH_SHARED(false); }

        } else {
            // Fallback: Global Memory
            // 需要分配 buffer: [total_slices * L] = [numel]
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
                        L, dist_in.numel()); break; \
                    case 2: edt_kernel_global<IS_FINAL, 2><<<total_slices, threads>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        global_buf1.data_ptr<int>(), global_buf2.data_ptr<int>(), \
                        L, dist_in.numel()); break; \
                    case 3: edt_kernel_global<IS_FINAL, 3><<<total_slices, threads>>>( \
                        dist_in.data_ptr<float>(), idx_in.data_ptr<int32_t>(), \
                        dist_out.data_ptr<float>(), idx_out.data_ptr<int32_t>(), \
                        global_buf1.data_ptr<int>(), global_buf2.data_ptr<int>(), \
                        L, dist_in.numel()); break; \
                    default: break; \
                }

            if (is_final_pass) { DISPATCH_GLOBAL(true); }
            else { DISPATCH_GLOBAL(false); }
        }

        // -----------------------------------------------------------
        // Step C: Transpose Back
        // -----------------------------------------------------------
        current_dist = dist_out.transpose(d, ndim - 1); // View, non-contiguous is fine here as next step makes it contiguous
        current_idx  = idx_out.transpose(d, ndim - 1);
    }

    if (had_no_batch_dim) {
        return std::make_tuple(current_dist.squeeze(0), current_idx.squeeze(0));
    }
    return std::make_tuple(current_dist, current_idx);
}