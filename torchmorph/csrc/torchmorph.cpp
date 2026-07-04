#include <torch/extension.h>

torch::Tensor add_cuda(torch::Tensor input, float scalar);

torch::Tensor grey_erosion_cuda(
    torch::Tensor input,
    torch::Tensor structure,
    torch::Tensor footprint,
    std::vector<int64_t> origin,
    int mode,
    float cval
);

torch::Tensor grey_dilation_cuda(
    torch::Tensor input,
    torch::Tensor structure,
    torch::Tensor footprint,
    std::vector<int64_t> origin,
    int mode,
    float cval
);

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

    m.def("grey_erosion_cuda", &grey_erosion_cuda,
          "N-dimensional fused grey erosion",
          py::arg("input"),
          py::arg("structure"),
          py::arg("footprint"),
          py::arg("origin"),
          py::arg("mode"),
          py::arg("cval"));

    m.def("grey_dilation_cuda", &grey_dilation_cuda,
          "N-dimensional fused grey dilation",
          py::arg("input"),
          py::arg("structure"),
          py::arg("footprint"),
          py::arg("origin"),
          py::arg("mode"),
          py::arg("cval"));

    m.def("edt_cuda", &edt_cuda,
          "Exact Euclidean Distance Transform (Felzenszwalb algorithm)",
          py::arg("input"),
          py::arg("sampling"),
          py::arg("return_distances") = true,
          py::arg("return_indices") = false,
          py::arg("algorithm") = "exact");
    m.def("cdt_cuda", &cdt_cuda,
          "Chessboard/Manhattan distance transform",
          py::arg("input"),
          py::arg("metric") = "chessboard",
          py::arg("return_distances") = true,
          py::arg("return_indices") = false);

    m.def("bfdt_cuda", &bfdt_cuda,
          "Brute-force distance transform",
          py::arg("input"),
          py::arg("metric") = "euclidean",
          py::arg("sampling") = std::vector<float>(),
          py::arg("return_distances") = true,
          py::arg("return_indices") = false);
}
