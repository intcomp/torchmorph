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

// Optimal Transport functions
std::tuple<torch::Tensor, torch::Tensor> sinkhorn_fastiter(
    const torch::Tensor source,
    const torch::Tensor target,
    const torch::Tensor k,
    torch::Tensor u,
    torch::Tensor v,
    int itrstep,
    int N
);

std::tuple<torch::Tensor, torch::Tensor> sinkhorn_logiter(
    const torch::Tensor log_a,
    const torch::Tensor log_b,
    const torch::Tensor M,
    torch::Tensor log_u,
    torch::Tensor log_v,
    int itrstep,
    int N,
    float reg
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

    // Optimal Transport
    m.def("sinkhorn_fastiter", &sinkhorn_fastiter,
          "Sinkhorn Fast Iteration CUDA",
          py::arg("source"),
          py::arg("target"),
          py::arg("k"),
          py::arg("u"),
          py::arg("v"),
          py::arg("itrstep"),
          py::arg("N"));

    m.def("sinkhorn_logiter", &sinkhorn_logiter,
          "Sinkhorn Log-Domain Iteration CUDA",
          py::arg("log_a"),
          py::arg("log_b"),
          py::arg("M"),
          py::arg("log_u"),
          py::arg("log_v"),
          py::arg("itrstep"),
          py::arg("N"),
          py::arg("reg")
    );
}

