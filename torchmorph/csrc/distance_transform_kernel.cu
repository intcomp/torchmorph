#include <torch/extension.h>
#include <vector>

// --- Kernel 1: 二值化内核  ---
/*
 * @brief 对输入张量进行逐元素二值化，为距离变换做准备。
 * @details 将前景点(in[idx] == 0)的初始距离设为0，
 *          背景点的初始距离设为一个极大值(1e20f)，这在距离变换的上下文中
 *          可以被认为是无穷大。
 * @param in 输入张量的数据指针。
 * @param out 输出张量的数据指针。
 * @param N 张量中的元素总数。
 */
__global__ void initialize_distance_kernel(const float* in, float* out, int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 如果输入像素为0，则为前景点，其距离为0。
        // 如果输入像素非0，则为背景点，其初始距离为无穷大。
        out[idx] = (in[idx] == 0.0f) ? 0.0f : 1e20f;
    }
}

// --- Kernel 2: 1D Pass 距离平方计算内核 ---
/**
 * @brief 沿着一个指定的空间维度，对N维张量执行一维抛物线下包络算法。
 * @details 这是Felzenszwalb和Huttenlocher EDT算法的核心。它通过将N维问题分解为N个
 *          一维问题来解决。此内核负责处理其中一个维度。它只计算距离的平方，以避免
 *          昂贵的开方运算并保持数值精度。
 *          每个CUDA线程块（block）负责处理一条完整的一维扫描线（slice）。
 * @param in_data 输入张量数据指针。
 * @param out_data 输出张量数据指针。
 * @param shape 描述输入张量形状的数组指针 (在GPU上)。
 * @param strides 描述输入张量步幅的数组指针 (在GPU上)。
 * @param ndim 张量的总维度数 (包括批处理维度)。
 * @param process_dim_sample 当前正在处理的空间维度索引 (0代表第一个空间维度，依此类推)。
 * @param total_slices 需要处理的一维扫描线总数 (batch_size * num_slices_per_sample)。
 * @param num_slices_per_sample 每个样本中，垂直于当前处理维度的扫描线数量。
 */
__global__ void edt_1d_pass_sq_kernel(
    const float* in_data, float* out_data, 
    const int64_t* shape, const int64_t* strides, 
    int32_t ndim, int32_t process_dim_sample, 
    int64_t total_slices, int64_t num_slices_per_sample
) {
    // 每个线程块处理一条一维扫描线
    int64_t slice_idx = blockIdx.x;
    if (slice_idx >= total_slices) return;


    int64_t batch_idx = slice_idx / num_slices_per_sample;
    int64_t slice_idx_in_sample = slice_idx % num_slices_per_sample;
    int64_t batch_offset = batch_idx * strides[0]; // 获取批处理的基地址
    int64_t sample_base_offset = 0;
    int64_t temp_idx = slice_idx_in_sample;
    int sample_ndim = ndim - 1;

    // 从非处理维度中计算出样本内的基地址偏移
    for (int32_t d = sample_ndim - 1; d >= 0; --d) {
        if (d == process_dim_sample) continue; // 跳过当前正在处理的维度
        int64_t size_of_dim = shape[d + 1];
        if (size_of_dim == 0) continue;
        int64_t coord_in_dim = temp_idx % size_of_dim;
        temp_idx /= size_of_dim;
        sample_base_offset += coord_in_dim * strides[d + 1];
    }
    
    const int64_t process_dim_actual = process_dim_sample + 1; // 加上批处理维度的实际索引
    const int64_t N = shape[process_dim_actual]; // 当前处理维度的长度
    const int64_t stride = strides[process_dim_actual]; // 沿当前维度移动一个元素所需的步幅
    const int64_t base_offset = batch_offset + sample_base_offset; // 最终的起始地址


    extern __shared__ float sdata[];
    float* f = sdata;                       // 存储函数值 g(p) = f(p) + p^2
    int*   v = (int*)(sdata + N);           // 存储抛物线顶点的索引
    float* z = (float*)(v + N + 1);         // 存储相邻抛物线的交点

    // 块内的所有线程协同将数据从全局内存加载到共享内存
    for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
        f[i] = in_data[base_offset + i * stride];
    }
    __syncthreads(); // 等待所有线程完成加载

    //计算抛物线的下包络 
    if (threadIdx.x == 0 && N > 0) {
        int k = 0; // 下包络中的抛物线数量
        v[0] = 0;  // 第一个抛物线的顶点索引为0
        z[0] = -1e20f; z[1] = 1e20f; // 初始化交点为负无穷和正无穷

        // 遍历所有点，构建下包络
        for (int q = 1; q < N; q++) {
            float s;
            // 寻找新的抛物线q应该插入的位置
            while (true) {
                int p = v[k]; if (q == p) break;
                // s 是抛物线 p 和 q 的交点的横坐标
                s = ((f[q] + q * q) - (f[p] + p * p)) / (2.0f * (q - p));
                // 如果交点在当前区间的右侧，则找到了插入点
                if (s > z[k]) { break; }
                // 否则，抛物线p被q完全覆盖，需要移除p
                if (k == 0) { break; } 
                k--;
            }
            // 插入新的抛物线q
            k++; 
            v[k] = q; 
            z[k] = s; 
            z[k + 1] = 1e20f;
        }
        // 计算距离平方
        k = 0;
        // 遍历所有点，找到其头顶上方的下包络线段，并计算距离
        for (int q = 0; q < N; q++) {
            while (z[k + 1] < q) k++; // 找到点q所属的区间
            int p = v[k]; // 获取该区间的抛物线顶点索引
            // 计算距离平方: D(q)^2 = (q - p)^2 + g(p)
            out_data[base_offset + q * stride] = (q - p) * (q - p) + f[p];
        }
    }
}

// --- Kernel 3: 开平方根内核 ---
/**
 * @brief 对张量中的每个元素计算平方根。
 * @details 这是一个简单的逐元素操作。由于之前的1D pass计算的是距离的平方，
 *          此内核在所有维度处理完毕后被调用，以得到最终的欧氏距离。
 * @param data 需要进行开方操作的张量数据指针。
 * @param N 张量中的元素总数。
 */
__global__ void sqrt_kernel(float* data, int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = sqrtf(data[idx]);
    }
}

// --- 主调函数 (Host) ---
/**
 * @brief 执行N维欧氏距离变换。
 * @param input 一个N维的PyTorch张量，第一个维度被视为批处理（batch）维度。
 * @return 一个与输入形状相同的张量，包含每个点到最近前景点（值>=0.5）的欧氏距离。
 */
std::tuple<torch::Tensor, torch::Tensor> distance_transform_cuda(torch::Tensor input) {
    auto original_input = input;
    
    // --- 预处理: 统一输入格式 ---
    // 确保所有输入都至少是3D的 (B, ...)，方便后续统一处理。
    // 如果输入是 (H, W) 或 (L)，则变为 (1, H, W) 或 (1, L)。
    bool had_no_batch_dim = (input.dim() <= 2); 
    if (had_no_batch_dim) { input = input.unsqueeze(0); }

    // 检查输入张量是否在CUDA上并且是内存连续的
    TORCH_CHECK(input.is_cuda(), "Input must be on a CUDA device.");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous.");

    
    // --- 获取张量元数据 ---
    const auto ndim = input.dim();
    const auto sample_ndim = ndim - 1; // 空间维度 = 总维度 - 1 (batch)
    const auto batch_size = input.size(0);
    const int64_t N_total = input.numel();
    
    auto shape = input.sizes().vec();
    auto index_shape = shape;
    index_shape.push_back(sample_ndim);

    auto strides_vec = input.strides().vec();
    
    // --- 内存分配: 使用Ping-Pong缓冲策略 ---
    // 分配两个缓冲区，在处理每个维度时交替作为输入和输出，避免原地读写冲突。
    auto distance = torch::zeros_like(input);
    auto index = torch::zeros(index_shape);
    auto buffer = (sample_ndim > 0) ? torch::empty_like(input) : distance;

    if (input.numel() == 0) { return std::make_tuple(distance, index); }

    //二值化
    int threads = 256; // 定义每个线程块的线程数
    int blocks = (N_total + threads - 1) / threads; // 计算启动的线程块数
    initialize_distance_kernel<<<blocks, threads>>>(input.data_ptr<float>(), buffer.data_ptr<float>(), N_total);

    //循环调用 edt_1d_pass_sq_kernel
    // 将shape和strides信息从CPU内存拷贝到GPU内存，以便内核可以访问
    int64_t *shape_gpu, *strides_gpu;
    cudaMalloc(&shape_gpu, ndim * sizeof(int64_t));
    cudaMalloc(&strides_gpu, ndim * sizeof(int64_t));
    cudaMemcpy(shape_gpu, shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(strides_gpu, strides_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    torch::Tensor current_input = buffer;
    torch::Tensor current_output = distance;

    // 遍历所有空间维度
    for (int32_t d_sample = 0; d_sample < sample_ndim; ++d_sample) {
        // 为当前处理的维度计算启动内核所需的参数
        int64_t num_slices_per_sample = 1;
        for(int i = 0; i < sample_ndim; ++i) { 
            if (i != d_sample) num_slices_per_sample *= shape[i + 1]; 
        }
        int64_t total_slices = batch_size * num_slices_per_sample;
        int64_t slice_len = shape[d_sample + 1];
        
        // 动态设置线程数和共享内存大小
        int threads_pass = (slice_len > 0 && slice_len < 256) ? slice_len : 256;
        if (threads_pass == 0) threads_pass = 1;
        size_t shared_mem_size = slice_len * sizeof(float) + (slice_len + 1) * sizeof(int) + (slice_len + 2) * sizeof(float);
        
        edt_1d_pass_sq_kernel<<<total_slices, threads_pass, shared_mem_size>>>(
            current_input.data_ptr<float>(), current_output.data_ptr<float>(),
            shape_gpu, strides_gpu, ndim, d_sample, total_slices, num_slices_per_sample
        );
        // 交换输入和输出缓冲区，为下一个维度做准备
        std::swap(current_input, current_output);
    }
    
    cudaFree(shape_gpu);
    cudaFree(strides_gpu);

    //计算最终距离
    // 经过循环后，current_input 指向的是包含最终距离平方结果的张量
    sqrt_kernel<<<blocks, threads>>>(current_input.data_ptr<float>(), N_total);
    
    // 如果最后一轮的输出不在我们期望的 output 张量里，就做一次拷贝
    if (current_input.data_ptr() != distance.data_ptr()){
        distance.copy_(current_input);
    }
    
    // 如果最初没有批处理维度，则移除我们添加的维度
    if (had_no_batch_dim) { distance = distance.squeeze(0); }
    
    return std::make_tuple(distance, index);
}
