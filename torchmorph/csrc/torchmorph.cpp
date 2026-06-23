#include <torch/extension.h>

// CUDA 实现声明
torch::Tensor add_cuda(torch::Tensor input, float scalar);

// 灰度腐蚀（融合 kernel，边界处理在 CUDA 内部完成）
torch::Tensor grey_erosion_cuda(
    torch::Tensor input,
    torch::Tensor structure,
    torch::Tensor footprint,
    std::vector<int64_t> origin,
    int mode,
    float cval
);

// 距离变换函数
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
    m.def("add_cuda", &add_cuda, "标量加法");

    // 灰度形态学
    m.def("grey_erosion_cuda", &grey_erosion_cuda,
          "N维融合灰度腐蚀（CUDA，边界处理在内部完成）",
          py::arg("input"),
          py::arg("structure"),
          py::arg("footprint"),
          py::arg("origin"),
          py::arg("mode"),
          py::arg("cval"));

    // 距离变换
    m.def("edt_cuda", &edt_cuda,
          "精确欧氏距离变换（Felzenszwalb 算法）",
          py::arg("input"),
          py::arg("sampling"),
          py::arg("return_distances") = true,
          py::arg("return_indices") = false,
          py::arg("algorithm") = "exact");
    m.def("cdt_cuda", &cdt_cuda,
          "棋盘/曼哈顿距离变换",
          py::arg("input"),
          py::arg("metric") = "chessboard",
          py::arg("return_distances") = true,
          py::arg("return_indices") = false);

    m.def("bfdt_cuda", &bfdt_cuda,
          "暴力距离变换",
          py::arg("input"),
          py::arg("metric") = "euclidean",
          py::arg("sampling") = std::vector<float>(),
          py::arg("return_distances") = true,
          py::arg("return_indices") = false);
}
