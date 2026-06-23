#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cfloat>

// 支持的最大空间维度数
#define GREYERO_MAX_NDIM 8

struct GreyErosionGeometry {
    int64_t spatial_size[GREYERO_MAX_NDIM];
    int64_t spatial_stride[GREYERO_MAX_NDIM];
    int64_t min_deltas[GREYERO_MAX_NDIM];
    int64_t max_deltas[GREYERO_MAX_NDIM];
};

// 边界填充模式枚举，与 scipy.ndimage 的 mode 参数对应
enum BorderMode {
    MODE_CONSTANT = 0,  // 常数填充
    MODE_REFLECT  = 1,  // 镜像反射（不含边界像素本身）
    MODE_NEAREST  = 2,  // 最近邻重复
    MODE_MIRROR   = 3,  // 镜像翻转（含边界像素本身）
    MODE_WRAP     = 4,  // 环形/周期填充
};

// 将越界坐标映射回有效范围
__device__ __forceinline__ int64_t map_coord(
    int64_t coord, int64_t size, BorderMode mode, bool* use_cval
) {
    *use_cval = false;
    if (coord >= 0 && coord < size) return coord;

    switch (mode) {
        case MODE_CONSTANT:
            // 常数模式：标记使用 cval，返回值无效
            *use_cval = true;
            return 0;
        case MODE_NEAREST:
            // 最近邻：截断到边界
            return coord < 0 ? 0 : size - 1;
        case MODE_REFLECT: {
            // 反射模式：关于边界反射，不含边界像素
            // 例：[a b c d] -> [b a | a b c d | c d]
            int64_t period = 2 * size;
            int64_t c = ((coord % period) + period) % period;
            if (c >= size) c = period - 1 - c;
            return c;
        }
        case MODE_MIRROR: {
            // 镜像模式：关于边界反射，含边界像素
            // 例：[a b c d] -> [c b | a b c d | b c]
            int64_t period = 2 * (size - 1);
            if (period <= 0) return 0;
            int64_t c = ((coord % period) + period) % period;
            if (c >= size) c = period - c;
            return c;
        }
        case MODE_WRAP: {
            // 环形模式：取模运算实现周期填充
            return ((coord % size) + size) % size;
        }
        default:
            *use_cval = true;
            return 0;
    }
}

// 融合 kernel：边界处理 + 最小值计算在一次遍历中完成
// 每个线程处理一个输出像素
// struct_meta: 每个结构元素位置的各维度偏移量，末尾附加 flat 偏移量
__global__ void grey_erosion_fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ struct_vals,       // 结构元素的值（非平坦时使用）
    const int64_t* __restrict__ struct_meta,     // 每个结构元素的维度偏移量 + flat 偏移量
    const int num_struct,                        // 结构元素的元素总数
    const int ndim_spatial,                      // 空间维度数
    const GreyErosionGeometry geom,              // 空间尺寸、步长、内部区域边界
    const int64_t total_spatial,                 // 空间元素总数
    const int64_t batch_channel,                 // batch * channel 总数
    const BorderMode mode,                       // 边界模式
    const float cval                             // 常数模式的填充值
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_channel * total_spatial) return;

    // 计算当前线程属于哪个 batch*channel 和空间位置
    int64_t bc = idx / total_spatial;
    int64_t sp = idx % total_spatial;

    // 将展平的空间索引分解为 N 维坐标
    int64_t coords[GREYERO_MAX_NDIM];
    int64_t rem = sp;
    for (int d = 0; d < ndim_spatial; d++) {
        coords[d] = rem / geom.spatial_stride[d];
        rem %= geom.spatial_stride[d];
    }

    float min_val = FLT_MAX;
    const float* in_ptr = input + bc * total_spatial;
    const int meta_stride = ndim_spatial + 1;

    bool is_interior = true;
    for (int d = 0; d < ndim_spatial; d++) {
        if (coords[d] + geom.min_deltas[d] < 0 ||
            coords[d] + geom.max_deltas[d] >= geom.spatial_size[d]) {
            is_interior = false;
            break;
        }
    }

    if (is_interior) {
        for (int i = 0; i < num_struct; i++) {
            float val = in_ptr[sp + struct_meta[i * meta_stride + ndim_spatial]];
            min_val = fminf(min_val, val - struct_vals[i]);
        }
        output[idx] = min_val;
        return;
    }

    // 遍历结构元素的所有位置，取最小值
    for (int i = 0; i < num_struct; i++) {
        const int64_t* meta = struct_meta + i * meta_stride;

        // 对每个维度，将坐标加上偏移后通过边界模式映射回有效范围
        bool use_cval = false;
        int64_t flat_idx = 0;
        for (int d = 0; d < ndim_spatial; d++) {
            bool uc = false;
            int64_t mapped = map_coord(
                coords[d] + meta[d], geom.spatial_size[d], mode, &uc
            );
            if (uc) { use_cval = true; break; }
            flat_idx += mapped * geom.spatial_stride[d];
        }

        // 取输入值（或 cval）减去结构元素值，更新最小值
        float val = use_cval ? cval : in_ptr[flat_idx];
        min_val = fminf(min_val, val - struct_vals[i]);
    }

    output[idx] = min_val;
}

// ==============================================================================
// 主机端入口函数
// ==============================================================================
torch::Tensor grey_erosion_cuda(
    torch::Tensor input,       // (B, C, *spatial) float32, 连续, CUDA
    torch::Tensor structure,   // (*spatial) float32, CPU
    torch::Tensor footprint,   // (*spatial) bool CPU；空 tensor 表示全有效
    std::vector<int64_t> origin_vec,
    int mode_int,
    float cval
) {
    TORCH_CHECK(input.is_cuda(), "input 必须是 CUDA 张量");
    TORCH_CHECK(!structure.is_cuda(), "structure 必须是 CPU 张量");
    TORCH_CHECK(!footprint.is_cuda(), "footprint 必须是 CPU 张量");
    TORCH_CHECK(input.is_contiguous(), "input 必须是连续的");
    TORCH_CHECK(structure.is_contiguous(), "structure 必须是连续的");
    TORCH_CHECK(footprint.is_contiguous(), "footprint 必须是连续的");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input 必须是 float32");
    TORCH_CHECK(structure.dtype() == torch::kFloat32, "structure 必须是 float32");
    TORCH_CHECK(footprint.dtype() == torch::kBool, "footprint 必须是 bool");

    int ndim_spatial = structure.dim();
    TORCH_CHECK(ndim_spatial > 0 && ndim_spatial <= GREYERO_MAX_NDIM,
                "structure 维度必须在 1-", GREYERO_MAX_NDIM,
                " 之间，实际为 ", ndim_spatial);
    TORCH_CHECK((int)origin_vec.size() == ndim_spatial,
                "origin 长度必须与 structure 维度一致");
    TORCH_CHECK(input.dim() == ndim_spatial + 2,
                "input 空间维度必须与 structure 维度一致");

    const bool use_footprint = footprint.numel() != 0;
    if (use_footprint) {
        TORCH_CHECK(footprint.dim() == ndim_spatial,
                    "footprint 维度必须与 structure 维度一致");
        for (int d = 0; d < ndim_spatial; d++) {
            TORCH_CHECK(footprint.size(d) == structure.size(d),
                        "footprint 尺寸必须与 structure 尺寸一致");
        }
    }

    auto output = torch::empty_like(input);

    int B = input.size(0);
    int C = input.size(1);
    int64_t batch_channel = (int64_t)B * C;

    // 计算逻辑空间尺寸和步长
    std::vector<int64_t> h_spatial_size(ndim_spatial);
    std::vector<int64_t> h_spatial_stride(ndim_spatial);
    for (int d = 0; d < ndim_spatial; d++) {
        h_spatial_size[d] = input.size(d + 2);
    }
    h_spatial_stride[ndim_spatial - 1] = 1;
    for (int d = ndim_spatial - 2; d >= 0; d--) {
        h_spatial_stride[d] = h_spatial_stride[d + 1] * h_spatial_size[d + 1];
    }

    int64_t total_spatial = 1;
    for (int d = 0; d < ndim_spatial; d++) {
        total_spatial *= h_spatial_size[d];
    }

    GreyErosionGeometry h_geom;
    for (int d = 0; d < GREYERO_MAX_NDIM; d++) {
        h_geom.spatial_size[d] = 0;
        h_geom.spatial_stride[d] = 0;
        h_geom.min_deltas[d] = 0;
        h_geom.max_deltas[d] = 0;
    }
    for (int d = 0; d < ndim_spatial; d++) {
        h_geom.spatial_size[d] = h_spatial_size[d];
        h_geom.spatial_stride[d] = h_spatial_stride[d];
        h_geom.min_deltas[d] = INT64_MAX;
        h_geom.max_deltas[d] = INT64_MIN;
    }

    // 展平结构元素，仅保留 footprint 中有效的位置。
    auto struct_flat = structure.flatten().contiguous();
    const float* struct_ptr = struct_flat.data_ptr<float>();
    const int64_t total_struct = struct_flat.numel();

    torch::Tensor footprint_flat;
    const bool* footprint_ptr = nullptr;
    if (use_footprint) {
        footprint_flat = footprint.flatten().contiguous();
        footprint_ptr = footprint_flat.data_ptr<bool>();
    }

    std::vector<int64_t> h_structure_stride(ndim_spatial);
    h_structure_stride[ndim_spatial - 1] = 1;
    for (int d = ndim_spatial - 2; d >= 0; d--) {
        h_structure_stride[d] =
            h_structure_stride[d + 1] * structure.size(d + 1);
    }

    std::vector<float> h_struct_vals;
    std::vector<int64_t> h_struct_meta;
    h_struct_vals.reserve(total_struct);
    h_struct_meta.reserve(total_struct * (ndim_spatial + 1));

    for (int64_t i = 0; i < total_struct; i++) {
        if (use_footprint && !footprint_ptr[i]) {
            continue;
        }

        h_struct_vals.push_back(struct_ptr[i]);

        int64_t tmp = i;
        int64_t flat_offset = 0;
        for (int d = 0; d < ndim_spatial; d++) {
            int64_t coord_d = tmp / h_structure_stride[d];
            tmp %= h_structure_stride[d];

            // 偏移量 = 坐标 - 中心（中心考虑了 origin 偏移）
            int64_t center_d = structure.size(d) / 2 + origin_vec[d];
            int64_t delta = coord_d - center_d;
            h_struct_meta.push_back(delta);
            flat_offset += delta * h_spatial_stride[d];
            h_geom.min_deltas[d] = std::min(h_geom.min_deltas[d], delta);
            h_geom.max_deltas[d] = std::max(h_geom.max_deltas[d], delta);
        }
        h_struct_meta.push_back(flat_offset);
    }

    int num_struct = static_cast<int>(h_struct_vals.size());
    TORCH_CHECK(num_struct > 0, "footprint 至少需要包含一个有效位置");

    // 将数据传到 GPU：先创建 CPU tensor，再 .to(cuda) 做真正的拷贝
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32);
    auto opts_i = torch::TensorOptions().dtype(torch::kInt64);

    auto d_struct_vals = torch::from_blob(
        h_struct_vals.data(), {num_struct}, opts_f).to(input.device());
    auto d_struct_meta = torch::from_blob(
        h_struct_meta.data(), {num_struct * (ndim_spatial + 1)}, opts_i
    ).to(input.device());

    BorderMode bmode = static_cast<BorderMode>(mode_int);

    // 启动 kernel
    int64_t total_threads = batch_channel * total_spatial;
    int threads = 256;
    int blocks = (int)((total_threads + threads - 1) / threads);

    grey_erosion_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        d_struct_vals.data_ptr<float>(),
        d_struct_meta.data_ptr<int64_t>(),
        num_struct,
        ndim_spatial,
        h_geom,
        total_spatial,
        batch_channel,
        bmode,
        cval
    );

    return output;
}
