#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <cmath>
#include <limits>

#define MAX_DIMS 10
#define INF_VAL 1e8f  // 使用 1e8 保证 float32 精度下的数值稳定性

__device__ __forceinline__ float sqr(float x) { return x * x; }

// ------------------------------------------------------------------
// 内核 1: 初始 Pass (First Pass)
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

    // 预计算基准坐标 (除了 process_dim 以外的维度坐标)
    int32_t base_coords[MAX_DIMS]; 
    int64_t temp_idx = slice_idx_in_sample;
    const int sample_ndim = ndim - 1;

    // 根据 slice_idx 反解坐标
    for (int32_t d = sample_ndim - 1; d >= 0; --d) {
        if (d == process_dim_sample) {
            base_coords[d] = 0; // 占位
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

    extern __shared__ char s_buffer[];
    float* f = (float*)s_buffer;
    int*   v = (int*)(f + N);
    float* z = (float*)((char*)v + (N + 1) * sizeof(int));

    // Phase 1: 加载数据
    for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
        float val = __ldg(&in_data[current_offset + i * stride]);
        f[i] = (val == 0.0f) ? 0.0f : INF_VAL;
    }
    __syncthreads();

    // Phase 2: 构建包络
    if (threadIdx.x == 0) {
        int k = 0;
        v[0] = 0;
        z[0] = -INF_VAL;
        z[1] = INF_VAL;
        
        for (int q = 1; q < N; q++) {
            // 显式跳过背景点，避免 INF 污染计算
            if (f[q] >= (INF_VAL * 0.9f)) continue;

            float fq = f[q];
            int k_curr = k;
            while (k_curr >= 0) {
                int p = v[k_curr];
                
                // --- 核心修复：数值稳定的交点公式 ---
                // 先计算差值再相加，防止大数吞小数
                float diff_f = fq - f[p];
                float diff_sq = (float)q*(float)q - (float)p*(float)p;
                float s = (diff_f + diff_sq) / (2.0f * (float)(q - p));
                
                if (s > z[k_curr]) {
                    k_curr++;
                    v[k_curr] = q;
                    z[k_curr] = s;
                    z[k_curr + 1] = INF_VAL;
                    k = k_curr;
                    break;
                }
                k_curr--;
            }
            if (k_curr < 0) {
                k = 0; v[0] = q; z[0] = -INF_VAL; z[1] = INF_VAL;
            }
        }
    }
    __syncthreads();

    // Phase 3: 计算距离
    for (int q = threadIdx.x; q < N; q += blockDim.x) {
        int k = 0;
        float q_float = (float)q;
        while (z[k + 1] < q_float) k++;
        
        int p = v[k]; 
        
        int64_t global_idx = current_offset + q * stride;
        float dist_sq = sqr(q_float - (float)p) + f[p];
        
        out_dist[global_idx] = IsFinal ? sqrtf(dist_sq) : dist_sq;

        // 写入索引
        int32_t* out_idx_ptr = out_idx + global_idx * sample_ndim;
        #pragma unroll
        for (int d = 0; d < sample_ndim; ++d) {
            // 只有当前处理的维度写入 p，其他维度写入基准坐标
            out_idx_ptr[d] = (d == process_dim_sample) ? p : base_coords[d];
        }
    }
}

// ------------------------------------------------------------------
// 内核 2: 后续 Pass (Subsequent Pass)
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

    extern __shared__ char s_buffer[];
    float* f = (float*)s_buffer;
    int*   v = (int*)(f + N);
    float* z = (float*)((char*)v + (N + 1) * sizeof(int));

    for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
        f[i] = __ldg(&in_dist[current_offset + i * stride]);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int k = 0;
        v[0] = 0; z[0] = -INF_VAL; z[1] = INF_VAL;
        
        for (int q = 1; q < N; q++) {
            if (f[q] >= (INF_VAL * 0.9f)) continue;

            float fq = f[q];
            int k_curr = k;
            while (k_curr >= 0) {
                int p = v[k_curr];
                float diff_f = fq - f[p];
                float diff_sq = (float)q*(float)q - (float)p*(float)p;
                float s = (diff_f + diff_sq) / (2.0f * (float)(q - p));
                if (s > z[k_curr]) {
                    k_curr++; v[k_curr] = q; z[k_curr] = s; z[k_curr + 1] = INF_VAL;
                    k = k_curr; break;
                }
                k_curr--;
            }
            if (k_curr < 0) { k = 0; v[0] = q; z[0] = -INF_VAL; z[1] = INF_VAL; }
        }
    }
    __syncthreads();

    for (int q = threadIdx.x; q < N; q += blockDim.x) {
        int k = 0;
        float q_float = (float)q;
        while (z[k + 1] < q_float) k++;
        
        int p = v[k];
        
        int64_t q_global_offset = current_offset + q * stride;
        int64_t p_global_offset = current_offset + p * stride;
        
        float dist_sq = sqr(q_float - (float)p) + f[p];
        out_dist[q_global_offset] = IsFinal ? sqrtf(dist_sq) : dist_sq;

        // 索引直接从 Global Memory 拷贝，无需 Shared Memory
        const int32_t* src_idx_ptr = in_idx + p_global_offset * sample_ndim;
        int32_t* out_idx_ptr = out_idx + q_global_offset * sample_ndim;

        #pragma unroll
        for (int d = 0; d < sample_ndim; ++d) {
            out_idx_ptr[d] = src_idx_ptr[d]; 
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

    // 自动处理 1D 输入：(L) -> (1, L)
    // 自动处理 1D 批处理：(N, L) 保持不变 (视为 N 个 1D 样本)
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
        
        int threads = std::min((int64_t)256, slice_len);
        size_t smem = slice_len * (sizeof(float) + sizeof(int)) + (slice_len + 1) * sizeof(float);

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