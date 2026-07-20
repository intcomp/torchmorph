#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdint>

#define BINARY_MORPH_MAX_NDIM 8

struct BinaryMorphologyGeometry {
    int64_t spatial_size[BINARY_MORPH_MAX_NDIM];
    int64_t spatial_stride[BINARY_MORPH_MAX_NDIM];
    int64_t min_deltas[BINARY_MORPH_MAX_NDIM];
    int64_t max_deltas[BINARY_MORPH_MAX_NDIM];
};

struct BinaryErosionOp {
    __device__ __forceinline__ static bool identity() {
        return true;
    }

    __device__ __forceinline__ static bool done(bool result) {
        return !result;
    }

    __device__ __forceinline__ static bool combine(bool result, bool value) {
        return result && value;
    }

    __host__ __forceinline__ static int64_t offset(int64_t delta) {
        return delta;
    }
};

struct BinaryDilationOp {
    __device__ __forceinline__ static bool identity() {
        return false;
    }

    __device__ __forceinline__ static bool done(bool result) {
        return result;
    }

    __device__ __forceinline__ static bool combine(bool result, bool value) {
        return result || value;
    }

    __host__ __forceinline__ static int64_t offset(int64_t delta) {
        return -delta;
    }
};

template <typename MorphologyOp>
__global__ void binary_morphology_fused_kernel(
    const bool* __restrict__ input,
    bool* __restrict__ output,
    const int64_t* __restrict__ struct_meta,
    const int num_struct,
    const int ndim_spatial,
    const BinaryMorphologyGeometry geom,
    const int64_t total_spatial,
    const int64_t batch_channel,
    const bool border_value
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_channel * total_spatial) return;

    int64_t bc = idx / total_spatial;
    int64_t sp = idx % total_spatial;

    int64_t coords[BINARY_MORPH_MAX_NDIM];
    int64_t rem = sp;
    for (int d = 0; d < ndim_spatial; d++) {
        coords[d] = rem / geom.spatial_stride[d];
        rem %= geom.spatial_stride[d];
    }

    bool result = MorphologyOp::identity();
    const bool* in_ptr = input + bc * total_spatial;
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
            int64_t offset = struct_meta[i * meta_stride + ndim_spatial];
            result = MorphologyOp::combine(result, in_ptr[sp + offset]);
            if (MorphologyOp::done(result)) break;
        }
        output[idx] = result;
        return;
    }

    for (int i = 0; i < num_struct; i++) {
        const int64_t* meta = struct_meta + i * meta_stride;

        bool value = border_value;
        bool in_bounds = true;
        int64_t flat_idx = 0;
        for (int d = 0; d < ndim_spatial; d++) {
            int64_t coord = coords[d] + meta[d];
            if (coord < 0 || coord >= geom.spatial_size[d]) {
                in_bounds = false;
                break;
            }
            flat_idx += coord * geom.spatial_stride[d];
        }

        if (in_bounds) {
            value = in_ptr[flat_idx];
        }
        result = MorphologyOp::combine(result, value);
        if (MorphologyOp::done(result)) break;
    }

    output[idx] = result;
}

template <typename MorphologyOp>
static torch::Tensor binary_morphology_cuda(
    torch::Tensor input,
    torch::Tensor structure,
    std::vector<int64_t> origin_vec,
    bool border_value
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(input.device());

    TORCH_CHECK(!structure.is_cuda(), "structure must be a CPU tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(structure.is_contiguous(), "structure must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kBool, "input must be bool");
    TORCH_CHECK(structure.dtype() == torch::kBool, "structure must be bool");

    int ndim_spatial = structure.dim();
    TORCH_CHECK(ndim_spatial > 0 && ndim_spatial <= BINARY_MORPH_MAX_NDIM,
                "structure dimension must be in 1-", BINARY_MORPH_MAX_NDIM,
                ", got ", ndim_spatial);
    TORCH_CHECK((int)origin_vec.size() == ndim_spatial,
                "origin length must match structure dimensions");
    TORCH_CHECK(input.dim() == ndim_spatial + 2,
                "input spatial dimensions must match structure dimensions");

    auto output = torch::empty_like(input);

    int B = input.size(0);
    int C = input.size(1);
    int64_t batch_channel = (int64_t)B * C;

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

    BinaryMorphologyGeometry h_geom;
    for (int d = 0; d < BINARY_MORPH_MAX_NDIM; d++) {
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

    auto struct_flat = structure.flatten().contiguous();
    const bool* struct_ptr = struct_flat.data_ptr<bool>();
    const int64_t total_struct = struct_flat.numel();

    std::vector<int64_t> h_structure_stride(ndim_spatial);
    h_structure_stride[ndim_spatial - 1] = 1;
    for (int d = ndim_spatial - 2; d >= 0; d--) {
        h_structure_stride[d] = h_structure_stride[d + 1] * structure.size(d + 1);
    }

    std::vector<int64_t> h_struct_meta;
    h_struct_meta.reserve(total_struct * (ndim_spatial + 1));

    for (int64_t i = 0; i < total_struct; i++) {
        if (!struct_ptr[i]) {
            continue;
        }

        int64_t tmp = i;
        int64_t flat_offset = 0;
        for (int d = 0; d < ndim_spatial; d++) {
            int64_t coord_d = tmp / h_structure_stride[d];
            tmp %= h_structure_stride[d];

            int64_t center_d = structure.size(d) / 2 + origin_vec[d];
            int64_t delta = MorphologyOp::offset(coord_d - center_d);
            h_struct_meta.push_back(delta);
            flat_offset += delta * h_spatial_stride[d];
            h_geom.min_deltas[d] = std::min(h_geom.min_deltas[d], delta);
            h_geom.max_deltas[d] = std::max(h_geom.max_deltas[d], delta);
        }
        h_struct_meta.push_back(flat_offset);
    }

    int num_struct = static_cast<int>(h_struct_meta.size() / (ndim_spatial + 1));
    TORCH_CHECK(num_struct > 0, "structure must contain at least one active position");

    auto opts_i = torch::TensorOptions().dtype(torch::kInt64);
    auto d_struct_meta = torch::from_blob(
        h_struct_meta.data(), {num_struct * (ndim_spatial + 1)}, opts_i
    ).to(input.device());

    int64_t total_threads = batch_channel * total_spatial;
    int threads = 256;
    int blocks = (int)((total_threads + threads - 1) / threads);
    const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());

    binary_morphology_fused_kernel<MorphologyOp><<<blocks, threads, 0, stream>>>(
        input.data_ptr<bool>(),
        output.data_ptr<bool>(),
        d_struct_meta.data_ptr<int64_t>(),
        num_struct,
        ndim_spatial,
        h_geom,
        total_spatial,
        batch_channel,
        border_value
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

torch::Tensor binary_erosion_cuda(
    torch::Tensor input,
    torch::Tensor structure,
    std::vector<int64_t> origin_vec,
    bool border_value
) {
    return binary_morphology_cuda<BinaryErosionOp>(
        input, structure, origin_vec, border_value
    );
}

torch::Tensor binary_dilation_cuda(
    torch::Tensor input,
    torch::Tensor structure,
    std::vector<int64_t> origin_vec,
    bool border_value
) {
    return binary_morphology_cuda<BinaryDilationOp>(
        input, structure, origin_vec, border_value
    );
}
