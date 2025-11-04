#include <torch/extension.h>

// Declare CUDA implementations
torch::Tensor add_cuda(torch::Tensor input, float scalar);
std::tuple<torch::Tensor, torch::Tensor> distance_transform_cuda(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_cuda", &add_cuda, "Add tensor with scalar");
    m.def("distance_transform_cuda", &distance_transform_cuda, "Distance transform");
}
