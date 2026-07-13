#include <torch/extension.h>

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

// Optimal Transport functions
std::tuple<torch::Tensor, torch::Tensor> sinkhorn_fastiter(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& K,
    const torch::Tensor& K_T,
    torch::Tensor u,
    torch::Tensor v,
    int64_t n_iter
);

std::tuple<torch::Tensor, torch::Tensor> sinkhorn_logiter(
    const torch::Tensor& log_a,
    const torch::Tensor& log_b,
    const torch::Tensor& M,
    const torch::Tensor& M_T,
    torch::Tensor log_u,
    torch::Tensor log_v,
    int64_t n_iter,
    double epsilon
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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

    // Optimal Transport
    m.def("sinkhorn_fastiter", &sinkhorn_fastiter,
          "Sinkhorn scaling-form iterations (CUDA)",
          py::arg("a"),
          py::arg("b"),
          py::arg("K"),
          py::arg("K_T"),
          py::arg("u"),
          py::arg("v"),
          py::arg("n_iter"));

    m.def("sinkhorn_logiter", &sinkhorn_logiter,
          "Sinkhorn log-domain iterations (CUDA)",
          py::arg("log_a"),
          py::arg("log_b"),
          py::arg("M"),
          py::arg("M_T"),
          py::arg("log_u"),
          py::arg("log_v"),
          py::arg("n_iter"),
          py::arg("epsilon"));
}

