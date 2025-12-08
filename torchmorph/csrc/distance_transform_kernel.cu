#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <cmath>
#include <limits>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_DIMS 10
#define INF_VAL 1e8f  
// 保证 blockDim 足以覆盖大多数常见维度大小，或者配合 Loop 处理
#define MAX_THREADS 1024 

__device__ __forceinline__ float sqr(float x) { return x * x; }

// 计算从像素 q 到源点 p 的距离代价 (考虑了 p 点本身的数值权重 val_p)
__device__ __forceinline__ float compute_cost(int q, int p, float val_p) {
    if (p < 0) return INF_VAL; // 无效点
    // dist = (q - p)^2 + f[p]
    return sqr((float)q - (float)p) + val_p;
}

// ------------------------------------------------------------------
// 核心逻辑: 1D Jump Flooding (JFA) / Doubling Algorithm
// 全并行求解最近点索引，替代串行的抛物线构建
// ------------------------------------------------------------------
__device__ void compute_1d_jfa(
    int N,
    float* __restrict__ s_vals,      // 输入数值 (dist^2)
    int*   __restrict__ s_idx_curr,  // ping-pong buffer 1
    int*   __restrict__ s_idx_next   // ping-pong buffer 2
) {
    int tid = threadIdx.x;

    // --- 1. 初始化 ---
    // 每个线程负责一个或多个像素的初始化
    for (int i = tid; i < N; i += blockDim.x) {
        // 如果当前位置的值很大，说明它是背景，没有初始源点 (-1)
        // 否则源点就是它自己 (i)
        if (s_vals[i] >= INF_VAL * 0.9f) {
            s_idx_curr[i] = -1;
        } else {
            s_idx_curr[i] = i;
        }
    }
    __syncthreads();

    // --- 2. 迭代传播 (Step = 1, 2, 4, 8...) ---
    // 类似于双调排序或倍增法
    int* idx_in = s_idx_curr;
    int* idx_out = s_idx_next;

    // 只要步长小于 N，就需要传播
    // 对于 N=1024, 只需要 10 次迭代，每次所有线程全并行
    for (int step = 1; step < N; step *= 2) {
        
        for (int i = tid; i < N; i += blockDim.x) {
            int my_best_p = idx_in[i];
            float min_cost = INF_VAL;

            // 获取当前最优点的代价
            if (my_best_p != -1) {
                min_cost = compute_cost(i, my_best_p, s_vals[my_best_p]);
            }

            // --- 检查左边邻居 (i - step) ---
            int left = i - step;
            if (left >= 0) {
                int left_p = idx_in[left]; // 邻居推荐的源点
                if (left_p != -1) {
                    float c = compute_cost(i, left_p, s_vals[left_p]);
                    if (c < min_cost) {
                        min_cost = c;
                        my_best_p = left_p;
                    }
                }
            }

            // --- 检查右边邻居 (i + step) ---
            int right = i + step;
            if (right < N) {
                int right_p = idx_in[right]; // 邻居推荐的源点
                if (right_p != -1) {
                    float c = compute_cost(i, right_p, s_vals[right_p]);
                    if (c < min_cost) {
                        min_cost = c;
                        my_best_p = right_p;
                    }
                }
            }

            // 写入下一轮 Buffer
            idx_out[i] = my_best_p;
        }
        
        // 交换 Buffer 指针
        int* temp = idx_in;
        idx_in = idx_out;
        idx_out = temp;
        
        __syncthreads();
    }

    // --- 3. 结果写回 ---
    // 如果最后结果在 s_idx_next 里 (循环次数是奇数)，需要拷回 s_idx_curr
    // 或者直接让调用者知道结果在哪。
    // 为了简单，我们统一把结果放在 s_idx_curr 指向的内存里。
    // 注意：idx_in 现在指向的是包含最新结果的 buffer。
    
    // 如果 idx_in 已经指向 s_idx_curr，那不用动。
    // 如果 idx_in 指向 s_idx_next，说明最新结果在 s_idx_next，我们需要把它拷贝回 s_idx_curr
    // 或者是调整后续代码读取的指针。
    
    // 这里采用简单拷贝回 s_idx_curr 的方式，确保后续逻辑一致
    if (idx_in != s_idx_curr) {
        for (int i = tid; i < N; i += blockDim.x) {
            s_idx_curr[i] = s_idx_next[i];
        }
        __syncthreads();
    }
}


// ------------------------------------------------------------------
// 内核 1: 初始 Pass (JFA Version)
// ------------------------------------------------------------------
template <bool IsFinal>
__global__ void edt_kernel_first_pass(
    const float* __restrict__ in_data,
    float* __restrict__ out_dist,
    int32_t* __restrict__ out_idx,
    const int64_t* __restrict__ shape,
    const int64_t* __restrict__ strides,
    int32_t ndim,
    int32_t process_dim_sample,
    int64_t total_slices,
    int64_t num_slices_per_sample
) {
    int64_t slice_idx = blockIdx.x;
    if (slice_idx >= total_slices) return;

    int64_t batch_idx = slice_idx / num_slices_per_sample;
    int64_t slice_idx_in_sample = slice_idx % num_slices_per_sample;
    int64_t current_offset = batch_idx * strides[0];

    int32_t base_coords[MAX_DIMS]; 
    int64_t temp_idx = slice_idx_in_sample;
    const int sample_ndim = ndim - 1;

    for (int32_t d = sample_ndim - 1; d >= 0; --d) {
        if (d == process_dim_sample) {
            base_coords[d] = 0; 
            continue; 
        }
        int64_t size_of_dim = shape[d + 1];
        int32_t coord = (int32_t)(temp_idx % size_of_dim);
        base_coords[d] = coord;
        current_offset += coord * strides[d + 1];
        temp_idx /= size_of_dim;
    }

    const int64_t process_dim_actual = process_dim_sample + 1;
    const int64_t N = shape[process_dim_actual];
    const int64_t stride = strides[process_dim_actual];
    
    if (N == 0) return;

    // Shared Memory Layout:
    // f: float[N] (Values)
    // idx1: int[N] (Buffer 1)
    // idx2: int[N] (Buffer 2)
    extern __shared__ char s_buffer[];
    float* f = (float*)s_buffer;
    int*   idx1 = (int*)(f + N);
    int*   idx2 = (int*)(idx1 + N);

    // Phase 1: 并行加载数据
    for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
        float val = __ldg(&in_data[current_offset + i * stride]);
        f[i] = (val == 0.0f) ? 0.0f : INF_VAL;
    }
    __syncthreads();

    // Phase 2: 并行 JFA 计算
    compute_1d_jfa(N, f, idx1, idx2);
    // 结果现在存储在 idx1 中

    // Phase 3: 并行写回
    for (int q = threadIdx.x; q < N; q += blockDim.x) {
        int p = idx1[q];
        float dist_val;
        int p_idx;

        if (p != -1) {
            // JFA 得到的是最近源点的索引 p
            // 距离 = (q-p)^2 + f[p]
            // 注意：在 First Pass 中，f[p] 要么是 0 要么是 INF。如果 p != -1，f[p] 必为 0。
            // 但为了通用性，还是加上 f[p]
            float dist_sq = sqr((float)q - (float)p) + f[p];
            dist_val = IsFinal ? sqrtf(dist_sq) : dist_sq;
            p_idx = p;
        } else {
            dist_val = IsFinal ? INF_VAL : sqr(INF_VAL);
            p_idx = 0; 
        }

        int64_t global_idx = current_offset + q * stride;
        out_dist[global_idx] = dist_val;

        int32_t* out_idx_ptr = out_idx + global_idx * sample_ndim;
        #pragma unroll
        for (int d = 0; d < sample_ndim; ++d) {
            out_idx_ptr[d] = (d == process_dim_sample) ? p_idx : base_coords[d];
        }
    }
}

// ------------------------------------------------------------------
// 内核 2: 后续 Pass (JFA Version)
// ------------------------------------------------------------------
template <bool IsFinal>
__global__ void edt_kernel_subsequent_pass(
    const float* __restrict__ in_dist,
    const int32_t* __restrict__ in_idx,
    float* __restrict__ out_dist,
    int32_t* __restrict__ out_idx,
    const int64_t* __restrict__ shape,
    const int64_t* __restrict__ strides,
    int32_t ndim,
    int32_t process_dim_sample,
    int64_t total_slices,
    int64_t num_slices_per_sample
) {
    int64_t slice_idx = blockIdx.x;
    if (slice_idx >= total_slices) return;

    int64_t batch_idx = slice_idx / num_slices_per_sample;
    int64_t slice_idx_in_sample = slice_idx % num_slices_per_sample;
    int64_t current_offset = batch_idx * strides[0];
    
    int64_t temp_idx = slice_idx_in_sample;
    const int sample_ndim = ndim - 1;

    for (int32_t d = sample_ndim - 1; d >= 0; --d) {
        if (d == process_dim_sample) continue;
        int64_t size_of_dim = shape[d + 1];
        current_offset += (temp_idx % size_of_dim) * strides[d + 1];
        temp_idx /= size_of_dim;
    }
    
    const int64_t process_dim_actual = process_dim_sample + 1;
    const int64_t N = shape[process_dim_actual];
    const int64_t stride = strides[process_dim_actual];

    if (N == 0) return;

    // Shared Memory Layout 同上
    extern __shared__ char s_buffer[];
    float* f = (float*)s_buffer;
    int*   idx1 = (int*)(f + N);
    int*   idx2 = (int*)(idx1 + N);

    // Phase 1: 加载
    for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
        f[i] = __ldg(&in_dist[current_offset + i * stride]);
    }
    __syncthreads();

    // Phase 2: 并行 JFA 计算
    // 这里的 f[i] 是上一轮计算出的距离平方，作为权重
    compute_1d_jfa(N, f, idx1, idx2);

    // Phase 3: 写回
    for (int q = threadIdx.x; q < N; q += blockDim.x) {
        int p = idx1[q]; // 最近源点在当前行的索引
        float dist_val;

        if (p != -1) {
            float dist_sq = sqr((float)q - (float)p) + f[p];
            dist_val = IsFinal ? sqrtf(dist_sq) : dist_sq;
        } else {
            dist_val = IsFinal ? INF_VAL : sqr(INF_VAL);
            p = 0; // fallback
        }

        int64_t q_global_offset = current_offset + q * stride;
        out_dist[q_global_offset] = dist_val;

        // 索引处理
        if (p != -1) {
            int64_t p_global_offset = current_offset + p * stride;
            const int32_t* src_idx_ptr = in_idx + p_global_offset * sample_ndim;
            int32_t* out_idx_ptr = out_idx + q_global_offset * sample_ndim;

            #pragma unroll
            for (int d = 0; d < sample_ndim; ++d) {
                out_idx_ptr[d] = src_idx_ptr[d]; 
            }
        }
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
    if (had_no_batch_dim) { 
        input = input.unsqueeze(0); 
    }

    const auto ndim = input.dim();
    const auto sample_ndim = ndim - 1;
    const auto batch_size = input.size(0);
    auto shape = input.sizes().vec();
    auto strides_vec = input.strides().vec();
    
    if (input.numel() == 0) { 
        auto index_shape = shape;
        index_shape.push_back(sample_ndim > 0 ? sample_ndim : 1);
        return std::make_tuple(torch::empty_like(input), 
                               torch::empty(index_shape, input.options().dtype(torch::kInt32))); 
    }

    auto distance = torch::empty_like(input);
    auto index_shape = shape;
    index_shape.push_back(sample_ndim); 
    auto index = torch::empty(index_shape, input.options().dtype(torch::kInt32));

    auto buffer_dist = torch::empty_like(distance);
    auto buffer_idx = torch::empty_like(index);

    auto shape_tensor = torch::tensor(shape, torch::TensorOptions().dtype(torch::kInt64).device(input.device()));
    auto strides_tensor = torch::tensor(strides_vec, torch::TensorOptions().dtype(torch::kInt64).device(input.device()));

    std::vector<std::pair<int64_t, int32_t>> dim_order_pairs;
    for (int32_t d = 0; d < sample_ndim; ++d) {
        dim_order_pairs.push_back({strides_vec[d + 1], d});
    }
    std::sort(dim_order_pairs.rbegin(), dim_order_pairs.rend());

    for (int pass = 0; pass < sample_ndim; ++pass) {
        int32_t d_sample = dim_order_pairs[pass].second;
        bool is_first_pass = (pass == 0);
        bool is_final_pass = (pass == sample_ndim - 1);

        torch::Tensor *in_d, *in_i, *out_d, *out_i;

        if (is_first_pass) {
            in_d = nullptr; in_i = nullptr; 
            out_d = is_final_pass ? &distance : &buffer_dist;
            out_i = is_final_pass ? &index : &buffer_idx;
        } else {
            if (pass % 2 != 0) {
                in_d = &buffer_dist; in_i = &buffer_idx;
                out_d = &distance;   out_i = &index;
            } else {
                in_d = &distance;    in_i = &index;
                out_d = &buffer_dist; out_i = &buffer_idx;
            }
            if (is_final_pass) {
                out_d = &distance; out_i = &index;
            }
        }

        int64_t num_slices_per_sample = 1;
        for(int i = 0; i < sample_ndim; ++i) { 
            if (i != d_sample) num_slices_per_sample *= shape[i + 1]; 
        }
        int64_t total_slices = batch_size * num_slices_per_sample;
        int64_t slice_len = shape[d_sample + 1];
        
        int threads = std::min((int64_t)MAX_THREADS, slice_len);
        
        // JFA 需要的 Shared Memory:
        // float f[N]
        // int idx1[N]
        // int idx2[N]
        // 总共 slice_len * (4 + 4 + 4) = 12 * slice_len bytes
        size_t smem = slice_len * sizeof(float) + 
                      slice_len * sizeof(int) * 2; 

        if (is_first_pass) {
            const float* in_ptr = input.data_ptr<float>();
            if (is_final_pass) {
                edt_kernel_first_pass<true><<<total_slices, threads, smem>>>(
                    in_ptr, out_d->data_ptr<float>(), out_i->data_ptr<int32_t>(),
                    shape_tensor.data_ptr<int64_t>(), strides_tensor.data_ptr<int64_t>(),
                    ndim, d_sample, total_slices, num_slices_per_sample
                );
            } else {
                edt_kernel_first_pass<false><<<total_slices, threads, smem>>>(
                    in_ptr, out_d->data_ptr<float>(), out_i->data_ptr<int32_t>(),
                    shape_tensor.data_ptr<int64_t>(), strides_tensor.data_ptr<int64_t>(),
                    ndim, d_sample, total_slices, num_slices_per_sample
                );
            }
        } else {
            if (is_final_pass) {
                edt_kernel_subsequent_pass<true><<<total_slices, threads, smem>>>(
                    in_d->data_ptr<float>(), in_i->data_ptr<int32_t>(),
                    out_d->data_ptr<float>(), out_i->data_ptr<int32_t>(),
                    shape_tensor.data_ptr<int64_t>(), strides_tensor.data_ptr<int64_t>(),
                    ndim, d_sample, total_slices, num_slices_per_sample
                );
            } else {
                edt_kernel_subsequent_pass<false><<<total_slices, threads, smem>>>(
                    in_d->data_ptr<float>(), in_i->data_ptr<int32_t>(),
                    out_d->data_ptr<float>(), out_i->data_ptr<int32_t>(),
                    shape_tensor.data_ptr<int64_t>(), strides_tensor.data_ptr<int64_t>(),
                    ndim, d_sample, total_slices, num_slices_per_sample
                );
            }
        }
    }
    
    if (had_no_batch_dim) { 
        return std::make_tuple(distance.squeeze(0), index.squeeze(0));
    }
    return std::make_tuple(distance, index);
}