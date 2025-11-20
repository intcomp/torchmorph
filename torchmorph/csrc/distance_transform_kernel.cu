#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <cmath>
#include <limits>

// 优化策略：用4个独立的内核函数替代模板，完全消除分支

// 内核1: 第一个pass且是唯一pass (1D情况)
__global__ void edt_kernel_first_final(
    const float* in_data,
    float* out_dist,
    int32_t* out_idx,
    const int64_t* shape,
    const int64_t* strides,
    int32_t ndim,
    int32_t process_dim_sample,
    int64_t total_slices,
    int64_t num_slices_per_sample
) {
    int64_t slice_idx = blockIdx.x;
    if (slice_idx >= total_slices) return;

    int64_t batch_idx = slice_idx / num_slices_per_sample;
    int64_t slice_idx_in_sample = slice_idx % num_slices_per_sample;
    int64_t batch_offset = batch_idx * strides[0];
    int64_t sample_base_offset = 0;
    int64_t temp_idx = slice_idx_in_sample;
    const int sample_ndim = ndim - 1;

    for (int32_t d = sample_ndim - 1; d >= 0; --d) {
        if (d == process_dim_sample) continue;
        int64_t size_of_dim = shape[d + 1];
        if (size_of_dim > 0) {
            sample_base_offset += (temp_idx % size_of_dim) * strides[d + 1];
            temp_idx /= size_of_dim;
        }
    }
    
    const int64_t process_dim_actual = process_dim_sample + 1;
    const int64_t N = shape[process_dim_actual];
    const int64_t stride = strides[process_dim_actual];
    const int64_t base_offset = batch_offset + sample_base_offset;

    if (N == 0) return;

    extern __shared__ char s_buffer[];
    float* f = (float*)s_buffer;
    int*   v = (int*)(f + N);
    float* z = (float*)((char*)v + (N + 1) * sizeof(int));
    int32_t* s_idx = (int32_t*)((char*)z + (N + 2) * sizeof(float));

    // 加载数据 - 第一个pass
    for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
        int64_t global_offset = base_offset + i * stride;
        float val = __ldg(&in_data[global_offset]);
        int32_t* shared_idx_ptr = s_idx + i * sample_ndim;
        
        if (val == 0.0f) {
            f[i] = 0.0f;
            int64_t temp_coord = slice_idx_in_sample;
            for (int32_t d = sample_ndim - 1; d >= 0; --d) {
                if (d == process_dim_sample) continue;
                int64_t size_of_dim = shape[d + 1];
                if (size_of_dim > 0) {
                    shared_idx_ptr[d] = temp_coord % size_of_dim;
                    temp_coord /= size_of_dim;
                } else {
                    shared_idx_ptr[d] = 0;
                }
            }
            shared_idx_ptr[process_dim_sample] = i;
        } else {
            f[i] = 1e20f;
            for (int d = 0; d < sample_ndim; ++d) shared_idx_ptr[d] = -1;
        }
    }
    __syncthreads();

    // 构建包络
    if (threadIdx.x == 0) {
        int k = 0;
        v[0] = 0;
        z[0] = -1e20f;
        z[1] = 1e20f;
        
        for (int q = 1; q < N; q++) {
            float fq = f[q];
            int q_sq = q * q;
            
            while (k >= 0) {
                int p = v[k];
                float s = ((fq + q_sq) - (f[p] + p * p)) / (2.0f * (q - p));
                if (s > z[k]) {
                    k++;
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = 1e20f;
                    break;
                }
                k--;
                if (k < 0) {
                    k = 0;
                    v[0] = q;
                    z[0] = -1e20f;
                    z[1] = 1e20f;
                    break;
                }
            }
        }
    }
    __syncthreads();

    // 计算距离 - 最后一个pass，直接开方
    for (int q = threadIdx.x; q < N; q += blockDim.x) {
        int k = 0;
        float q_float = (float)q;
        while (z[k + 1] < q_float) k++;
        
        int p = v[k];
        int64_t global_offset = base_offset + q * stride;
        float dist_sq = (float)(q - p) * (q - p) + f[p];
        
        out_dist[global_offset] = sqrtf(dist_sq);  // 直接开方

        int32_t* out_idx_ptr = out_idx + global_offset * sample_ndim;
        const int32_t* src_idx_ptr = s_idx + p * sample_ndim;
        for (int d = 0; d < sample_ndim; ++d) {
            out_idx_ptr[d] = src_idx_ptr[d];
        }
    }
}

// 内核2: 第一个pass但不是最后
__global__ void edt_kernel_first_only(
    const float* in_data,
    float* out_dist,
    int32_t* out_idx,
    const int64_t* shape,
    const int64_t* strides,
    int32_t ndim,
    int32_t process_dim_sample,
    int64_t total_slices,
    int64_t num_slices_per_sample
) {
    int64_t slice_idx = blockIdx.x;
    if (slice_idx >= total_slices) return;

    int64_t batch_idx = slice_idx / num_slices_per_sample;
    int64_t slice_idx_in_sample = slice_idx % num_slices_per_sample;
    int64_t batch_offset = batch_idx * strides[0];
    int64_t sample_base_offset = 0;
    int64_t temp_idx = slice_idx_in_sample;
    const int sample_ndim = ndim - 1;

    for (int32_t d = sample_ndim - 1; d >= 0; --d) {
        if (d == process_dim_sample) continue;
        int64_t size_of_dim = shape[d + 1];
        if (size_of_dim > 0) {
            sample_base_offset += (temp_idx % size_of_dim) * strides[d + 1];
            temp_idx /= size_of_dim;
        }
    }
    
    const int64_t process_dim_actual = process_dim_sample + 1;
    const int64_t N = shape[process_dim_actual];
    const int64_t stride = strides[process_dim_actual];
    const int64_t base_offset = batch_offset + sample_base_offset;

    if (N == 0) return;

    extern __shared__ char s_buffer[];
    float* f = (float*)s_buffer;
    int*   v = (int*)(f + N);
    float* z = (float*)((char*)v + (N + 1) * sizeof(int));
    int32_t* s_idx = (int32_t*)((char*)z + (N + 2) * sizeof(float));

    for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
        int64_t global_offset = base_offset + i * stride;
        float val = __ldg(&in_data[global_offset]);
        int32_t* shared_idx_ptr = s_idx + i * sample_ndim;
        
        if (val == 0.0f) {
            f[i] = 0.0f;
            int64_t temp_coord = slice_idx_in_sample;
            for (int32_t d = sample_ndim - 1; d >= 0; --d) {
                if (d == process_dim_sample) continue;
                int64_t size_of_dim = shape[d + 1];
                if (size_of_dim > 0) {
                    shared_idx_ptr[d] = temp_coord % size_of_dim;
                    temp_coord /= size_of_dim;
                } else {
                    shared_idx_ptr[d] = 0;
                }
            }
            shared_idx_ptr[process_dim_sample] = i;
        } else {
            f[i] = 1e20f;
            for (int d = 0; d < sample_ndim; ++d) shared_idx_ptr[d] = -1;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int k = 0;
        v[0] = 0;
        z[0] = -1e20f;
        z[1] = 1e20f;
        
        for (int q = 1; q < N; q++) {
            float fq = f[q];
            int q_sq = q * q;
            
            while (k >= 0) {
                int p = v[k];
                float s = ((fq + q_sq) - (f[p] + p * p)) / (2.0f * (q - p));
                if (s > z[k]) {
                    k++;
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = 1e20f;
                    break;
                }
                k--;
                if (k < 0) {
                    k = 0;
                    v[0] = q;
                    z[0] = -1e20f;
                    z[1] = 1e20f;
                    break;
                }
            }
        }
    }
    __syncthreads();

    for (int q = threadIdx.x; q < N; q += blockDim.x) {
        int k = 0;
        float q_float = (float)q;
        while (z[k + 1] < q_float) k++;
        
        int p = v[k];
        int64_t global_offset = base_offset + q * stride;
        float dist_sq = (float)(q - p) * (q - p) + f[p];
        
        out_dist[global_offset] = dist_sq;  // 不开方

        int32_t* out_idx_ptr = out_idx + global_offset * sample_ndim;
        const int32_t* src_idx_ptr = s_idx + p * sample_ndim;
        for (int d = 0; d < sample_ndim; ++d) {
            out_idx_ptr[d] = src_idx_ptr[d];
        }
    }
}

// 内核3: 中间pass
__global__ void edt_kernel_middle(
    const float* in_dist,
    const int32_t* in_idx,
    float* out_dist,
    int32_t* out_idx,
    const int64_t* shape,
    const int64_t* strides,
    int32_t ndim,
    int32_t process_dim_sample,
    int64_t total_slices,
    int64_t num_slices_per_sample
) {
    int64_t slice_idx = blockIdx.x;
    if (slice_idx >= total_slices) return;

    int64_t batch_idx = slice_idx / num_slices_per_sample;
    int64_t slice_idx_in_sample = slice_idx % num_slices_per_sample;
    int64_t batch_offset = batch_idx * strides[0];
    int64_t sample_base_offset = 0;
    int64_t temp_idx = slice_idx_in_sample;
    const int sample_ndim = ndim - 1;

    for (int32_t d = sample_ndim - 1; d >= 0; --d) {
        if (d == process_dim_sample) continue;
        int64_t size_of_dim = shape[d + 1];
        if (size_of_dim > 0) {
            sample_base_offset += (temp_idx % size_of_dim) * strides[d + 1];
            temp_idx /= size_of_dim;
        }
    }
    
    const int64_t process_dim_actual = process_dim_sample + 1;
    const int64_t N = shape[process_dim_actual];
    const int64_t stride = strides[process_dim_actual];
    const int64_t base_offset = batch_offset + sample_base_offset;

    if (N == 0) return;

    extern __shared__ char s_buffer[];
    float* f = (float*)s_buffer;
    int*   v = (int*)(f + N);
    float* z = (float*)((char*)v + (N + 1) * sizeof(int));
    int32_t* s_idx = (int32_t*)((char*)z + (N + 2) * sizeof(float));

    for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
        int64_t global_offset = base_offset + i * stride;
        f[i] = __ldg(&in_dist[global_offset]);
        
        const int32_t* global_idx_ptr = in_idx + global_offset * sample_ndim;
        int32_t* shared_idx_ptr = s_idx + i * sample_ndim;
        for (int d = 0; d < sample_ndim; ++d) {
            shared_idx_ptr[d] = __ldg(&global_idx_ptr[d]);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int k = 0;
        v[0] = 0;
        z[0] = -1e20f;
        z[1] = 1e20f;
        
        for (int q = 1; q < N; q++) {
            float fq = f[q];
            int q_sq = q * q;
            
            while (k >= 0) {
                int p = v[k];
                float s = ((fq + q_sq) - (f[p] + p * p)) / (2.0f * (q - p));
                if (s > z[k]) {
                    k++;
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = 1e20f;
                    break;
                }
                k--;
                if (k < 0) {
                    k = 0;
                    v[0] = q;
                    z[0] = -1e20f;
                    z[1] = 1e20f;
                    break;
                }
            }
        }
    }
    __syncthreads();

    for (int q = threadIdx.x; q < N; q += blockDim.x) {
        int k = 0;
        float q_float = (float)q;
        while (z[k + 1] < q_float) k++;
        
        int p = v[k];
        int64_t global_offset = base_offset + q * stride;
        float dist_sq = (float)(q - p) * (q - p) + f[p];
        
        out_dist[global_offset] = dist_sq;  // 不开方

        int32_t* out_idx_ptr = out_idx + global_offset * sample_ndim;
        const int32_t* src_idx_ptr = s_idx + p * sample_ndim;
        for (int d = 0; d < sample_ndim; ++d) {
            out_idx_ptr[d] = src_idx_ptr[d];
        }
    }
}

// 内核4: 最后一个pass
__global__ void edt_kernel_final(
    const float* in_dist,
    const int32_t* in_idx,
    float* out_dist,
    int32_t* out_idx,
    const int64_t* shape,
    const int64_t* strides,
    int32_t ndim,
    int32_t process_dim_sample,
    int64_t total_slices,
    int64_t num_slices_per_sample
) {
    int64_t slice_idx = blockIdx.x;
    if (slice_idx >= total_slices) return;

    int64_t batch_idx = slice_idx / num_slices_per_sample;
    int64_t slice_idx_in_sample = slice_idx % num_slices_per_sample;
    int64_t batch_offset = batch_idx * strides[0];
    int64_t sample_base_offset = 0;
    int64_t temp_idx = slice_idx_in_sample;
    const int sample_ndim = ndim - 1;

    for (int32_t d = sample_ndim - 1; d >= 0; --d) {
        if (d == process_dim_sample) continue;
        int64_t size_of_dim = shape[d + 1];
        if (size_of_dim > 0) {
            sample_base_offset += (temp_idx % size_of_dim) * strides[d + 1];
            temp_idx /= size_of_dim;
        }
    }
    
    const int64_t process_dim_actual = process_dim_sample + 1;
    const int64_t N = shape[process_dim_actual];
    const int64_t stride = strides[process_dim_actual];
    const int64_t base_offset = batch_offset + sample_base_offset;

    if (N == 0) return;

    extern __shared__ char s_buffer[];
    float* f = (float*)s_buffer;
    int*   v = (int*)(f + N);
    float* z = (float*)((char*)v + (N + 1) * sizeof(int));
    int32_t* s_idx = (int32_t*)((char*)z + (N + 2) * sizeof(float));

    for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
        int64_t global_offset = base_offset + i * stride;
        f[i] = __ldg(&in_dist[global_offset]);
        
        const int32_t* global_idx_ptr = in_idx + global_offset * sample_ndim;
        int32_t* shared_idx_ptr = s_idx + i * sample_ndim;
        for (int d = 0; d < sample_ndim; ++d) {
            shared_idx_ptr[d] = __ldg(&global_idx_ptr[d]);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int k = 0;
        v[0] = 0;
        z[0] = -1e20f;
        z[1] = 1e20f;
        
        for (int q = 1; q < N; q++) {
            float fq = f[q];
            int q_sq = q * q;
            
            while (k >= 0) {
                int p = v[k];
                float s = ((fq + q_sq) - (f[p] + p * p)) / (2.0f * (q - p));
                if (s > z[k]) {
                    k++;
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = 1e20f;
                    break;
                }
                k--;
                if (k < 0) {
                    k = 0;
                    v[0] = q;
                    z[0] = -1e20f;
                    z[1] = 1e20f;
                    break;
                }
            }
        }
    }
    __syncthreads();

    for (int q = threadIdx.x; q < N; q += blockDim.x) {
        int k = 0;
        float q_float = (float)q;
        while (z[k + 1] < q_float) k++;
        
        int p = v[k];
        int64_t global_offset = base_offset + q * stride;
        float dist_sq = (float)(q - p) * (q - p) + f[p];
        
        out_dist[global_offset] = sqrtf(dist_sq);  // 最后开方

        int32_t* out_idx_ptr = out_idx + global_offset * sample_ndim;
        const int32_t* src_idx_ptr = s_idx + p * sample_ndim;
        for (int d = 0; d < sample_ndim; ++d) {
            out_idx_ptr[d] = src_idx_ptr[d];
        }
    }
}

// Host函数
std::tuple<torch::Tensor, torch::Tensor> distance_transform_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be on a CUDA device.");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float tensor.");
    input = input.contiguous();

    bool had_no_batch_dim = (input.dim() <= 2); 
    if (had_no_batch_dim) { input = input.unsqueeze(0); }

    const auto ndim = input.dim();
    const auto sample_ndim = ndim - 1;
    const auto batch_size = input.size(0);
    
    auto shape = input.sizes().vec();
    auto strides_vec = input.strides().vec();
    
    if (input.numel() == 0) { 
        auto distance = torch::empty_like(input);
        auto index_shape = shape;
        index_shape.push_back(sample_ndim > 0 ? sample_ndim : 1);
        auto index = torch::empty(index_shape, input.options().dtype(torch::kInt32));
        if (had_no_batch_dim) return std::make_tuple(distance.squeeze(0), index.squeeze(0));
        return std::make_tuple(distance, index); 
    }
    
    auto distance = torch::empty_like(input);
    auto index_options = input.options().dtype(torch::kInt32);
    auto index_shape = shape;
    index_shape.push_back(sample_ndim > 0 ? sample_ndim : 1);
    auto index = torch::empty(index_shape, index_options);

    if (torch::all(input != 0).item<bool>()) {
        distance.fill_(std::numeric_limits<float>::infinity());
        index.fill_(-1);
        if (had_no_batch_dim) {
            return std::make_tuple(distance.squeeze(0), index.squeeze(0));
        }
        return std::make_tuple(distance, index);
    }

    auto shape_tensor = torch::tensor(shape, 
        torch::TensorOptions().dtype(torch::kInt64).device(input.device()));
    auto strides_tensor = torch::tensor(strides_vec, 
        torch::TensorOptions().dtype(torch::kInt64).device(input.device()));
    
    const int64_t* shape_gpu = shape_tensor.data_ptr<int64_t>();
    const int64_t* strides_gpu = strides_tensor.data_ptr<int64_t>();

    std::vector<std::pair<int64_t, int32_t>> dim_order_pairs;
    for (int32_t d_sample = 0; d_sample < sample_ndim; ++d_sample) {
        dim_order_pairs.push_back({strides_vec[d_sample + 1], d_sample});
    }
    std::sort(dim_order_pairs.rbegin(), dim_order_pairs.rend());

    if (sample_ndim == 0) {
        int64_t total_slices = batch_size;
        int64_t slice_len = (shape.size() > 1) ? shape[1] : 0;
        int threads = std::min((int64_t)256, slice_len);
        size_t smem = slice_len * sizeof(float) + (slice_len + 1) * sizeof(int) + 
                      (slice_len + 2) * sizeof(float) + slice_len * 1 * sizeof(int32_t);
        
        edt_kernel_first_final<<<total_slices, threads, smem>>>(
           input.data_ptr<float>(),
           distance.data_ptr<float>(), index.data_ptr<int32_t>(),
           shape_gpu, strides_gpu, ndim, 0, total_slices, 1
       );
    } else {
        auto buffer_dist = torch::empty_like(distance);
        auto buffer_idx = torch::empty_like(index);
        
        for (int pass = 0; pass < sample_ndim; ++pass) {
            int32_t d_sample = dim_order_pairs[pass].second;
            bool is_first = (pass == 0);
            bool is_final = (pass == sample_ndim - 1);

            torch::Tensor *in_dist, *in_idx, *out_dist, *out_idx;
            
            if (pass % 2 == 0) {
                in_dist = &distance; in_idx = &index;
                out_dist = &buffer_dist; out_idx = &buffer_idx;
            } else {
                in_dist = &buffer_dist; in_idx = &buffer_idx;
                out_dist = &distance; out_idx = &index;
            }
            
            int64_t num_slices_per_sample = 1;
            for(int i = 0; i < sample_ndim; ++i) { 
                if (i != d_sample) num_slices_per_sample *= shape[i + 1]; 
            }
            int64_t total_slices = batch_size * num_slices_per_sample;
            int64_t slice_len = shape[d_sample + 1];
            
            int threads = std::min((int64_t)256, slice_len);
            size_t smem = slice_len * sizeof(float) + (slice_len + 1) * sizeof(int) + 
                          (slice_len + 2) * sizeof(float) + slice_len * sample_ndim * sizeof(int32_t);

            if (is_first && is_final) {
                edt_kernel_first_final<<<total_slices, threads, smem>>>(
                    input.data_ptr<float>(),
                    out_dist->data_ptr<float>(), out_idx->data_ptr<int32_t>(),
                    shape_gpu, strides_gpu, ndim, d_sample, total_slices, num_slices_per_sample
                );
            } else if (is_first) {
                edt_kernel_first_only<<<total_slices, threads, smem>>>(
                    input.data_ptr<float>(),
                    out_dist->data_ptr<float>(), out_idx->data_ptr<int32_t>(),
                    shape_gpu, strides_gpu, ndim, d_sample, total_slices, num_slices_per_sample
                );
            } else if (is_final) {
                edt_kernel_final<<<total_slices, threads, smem>>>(
                    in_dist->data_ptr<float>(), in_idx->data_ptr<int32_t>(),
                    out_dist->data_ptr<float>(), out_idx->data_ptr<int32_t>(),
                    shape_gpu, strides_gpu, ndim, d_sample, total_slices, num_slices_per_sample
                );
            } else {
                edt_kernel_middle<<<total_slices, threads, smem>>>(
                    in_dist->data_ptr<float>(), in_idx->data_ptr<int32_t>(),
                    out_dist->data_ptr<float>(), out_idx->data_ptr<int32_t>(),
                    shape_gpu, strides_gpu, ndim, d_sample, total_slices, num_slices_per_sample
                );
            }
        }
        
        if (sample_ndim % 2 != 0) {
            distance.copy_(buffer_dist);
            index.copy_(buffer_idx);
        }
    }
    
    if (had_no_batch_dim) { 
        return std::make_tuple(distance.squeeze(0), index.squeeze(0));
    }
    
    return std::make_tuple(distance, index);
}
