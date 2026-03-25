#include <torch/extension.h>

// Declare CUDA implementations
torch::Tensor add_cuda(torch::Tensor input, float scalar);

// Distance Transform functions
std::tuple<torch::Tensor, torch::Tensor> edt_cuda(
    torch::Tensor input,
    std::vector<float> sampling,
    bool return_distances,
    bool return_indices,
    const std::string& algorithm
);

std::tuple<torch::Tensor, torch::Tensor> cdt_cuda(
    torch::Tensor input,
    const std::string& metric,
    bool return_distances,
    bool return_indices
);

std::tuple<torch::Tensor, torch::Tensor> bfdt_cuda(
    torch::Tensor input,
    const std::string& metric,
    std::vector<float> sampling,
    bool return_distances,
    bool return_indices
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_cuda", &add_cuda, "Add tensor with scalar");

    // Distance Transform
    m.def("edt_cuda", &edt_cuda,
          "Exact Euclidean Distance Transform (Felzenszwalb algorithm)",
          py::arg("input"),
          py::arg("sampling"),
          py::arg("return_distances") = true,
          py::arg("return_indices") = false,
          py::arg("algorithm") = "exact");
    m.def("cdt_cuda", &cdt_cuda,
          "Chamfer Distance Transform",
          py::arg("input"),
          py::arg("metric") = "chessboard",
          py::arg("return_distances") = true,
          py::arg("return_indices") = false);

    m.def("bfdt_cuda", &bfdt_cuda,
          "Brute-Force Distance Transform",
          py::arg("input"),
          py::arg("metric") = "euclidean",
          py::arg("sampling") = std::vector<float>(),
          py::arg("return_distances") = true,
          py::arg("return_indices") = false);
}
