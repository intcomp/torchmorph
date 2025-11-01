// =========================================================================
//  内容保存到: torchmorph/csrc/torchmorph.cpp
// =========================================================================

#include <torch/extension.h>

// 函数声明：告诉 C++ 编译器，这两个 CUDA 内核函数是在别的文件里定义的
// 这样 C++ 代码才能成功调用 .cu 文件里的内核
__global__ void edt_pass1_rows(const float* input, float* temp, int H, int W);
__global__ void edt_pass2_cols(const float* temp, float* output, int H, int W);



// 主调函数 (运行在 CPU 上)
torch::Tensor distance_transform_cuda(torch::Tensor input) {
    // 检查输入张量是否在 CUDA 上，以及是否为二维
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "Only 2D tensors supported");

    int H = input.size(0);
    int W = input.size(1);

    // 创建临时的和最终的输出张量
    auto temp = torch::empty_like(input);
    auto output = torch::empty_like(input);

    // 计算动态共享内存的大小
    size_t shared_mem_pass1 = W * sizeof(float) + (W + 1) * sizeof(int) + (W + 2) * sizeof(float);
    size_t shared_mem_pass2 = H * sizeof(float) + (H + 1) * sizeof(int) + (H + 2) * sizeof(float);

    // 设置每个块的线程数
    int threads_per_block = 32;

    // <<<...>>> 语法：启动 CUDA 内核
    // 参数：Grid大小, Block大小, 共享内存大小, (可选的流)
    
    // Pass 1: 每行启动一个 block
    edt_pass1_rows<<<H, threads_per_block, shared_mem_pass1>>>(
        input.data_ptr<float>(), temp.data_ptr<float>(), H, W);

    // Pass 2: 每列启动一个 block
    edt_pass2_cols<<<W, threads_per_block, shared_mem_pass2>>>(
        temp.data_ptr<float>(), output.data_ptr<float>(), H, W);

    return output;
}

// 使用 PYBIND11 将 C++ 函数绑定到 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("distance_transform", &distance_transform_cuda, "CUDA-accelerated Exact Euclidean Distance Transform");
}