#include <torch/extension.h>

__global__ void add_kernel(const float* in, float* out, float scalar, int64_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx] + scalar;
    }
}

torch::Tensor add_cuda(torch::Tensor input, float scalar) {
    auto output = torch::empty_like(input);
    int64_t N = input.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scalar,
        N
    );

    return output;
}

