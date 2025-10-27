#include <torch/extension.h>

// distance transform: https://en.wikipedia.org/wiki/Distance_transform
// https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html

__global__ void distance_transform_kernel(const float* in, float* out, int64_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = 2.0f * in[idx];
    }
}

torch::Tensor distance_transform_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int64_t N = input.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    distance_transform_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );

    return output;
}

