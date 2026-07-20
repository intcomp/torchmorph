#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cfloat>

#define GREY_MORPH_MAX_NDIM 8

struct GreyMorphologyGeometry {
    int64_t spatial_size[GREY_MORPH_MAX_NDIM];
    int64_t spatial_stride[GREY_MORPH_MAX_NDIM];
    int64_t min_deltas[GREY_MORPH_MAX_NDIM];
    int64_t max_deltas[GREY_MORPH_MAX_NDIM];
};

// Matches scipy.ndimage border modes.
enum BorderMode {
    MODE_CONSTANT = 0,  // Constant padding
    MODE_REFLECT  = 1,  // Half-sample symmetric
    MODE_NEAREST  = 2,  // Repeat nearest edge value
    MODE_MIRROR   = 3,  // Whole-sample symmetric
    MODE_WRAP     = 4,  // Periodic wrap padding
};

struct GreyErosionOp {
    __device__ __forceinline__ static float identity() {
        return CUDART_INF_F;
    }

    __device__ __forceinline__ static float combine(
        float result, float value, float structure
    ) {
        return fminf(result, value - structure);
    }

    __host__ __forceinline__ static int64_t offset(int64_t delta) {
        return delta;
    }
};

struct GreyDilationOp {
    __device__ __forceinline__ static float identity() {
        return -CUDART_INF_F;
    }

    __device__ __forceinline__ static float combine(
        float result, float value, float structure
    ) {
        return fmaxf(result, value + structure);
    }

    __host__ __forceinline__ static int64_t offset(int64_t delta) {
        return -delta;
    }
};

__device__ __forceinline__ int64_t map_coord(
    int64_t coord, int64_t size, BorderMode mode, bool* use_cval
) {
    *use_cval = false;
    if (coord >= 0 && coord < size) return coord;

    switch (mode) {
        case MODE_CONSTANT:
            *use_cval = true;
            return 0;
        case MODE_NEAREST:
            return coord < 0 ? 0 : size - 1;
        case MODE_REFLECT: {
            // Half-sample symmetric: [a b c d] -> [b a | a b c d | c d]
            int64_t period = 2 * size;
            int64_t c = ((coord % period) + period) % period;
            if (c >= size) c = period - 1 - c;
            return c;
        }
        case MODE_MIRROR: {
            // Whole-sample symmetric: [a b c d] -> [c b | a b c d | b c]
            int64_t period = 2 * (size - 1);
            if (period <= 0) return 0;
            int64_t c = ((coord % period) + period) % period;
            if (c >= size) c = period - c;
            return c;
        }
        case MODE_WRAP: {
            return ((coord % size) + size) % size;
        }
        default:
            *use_cval = true;
            return 0;
    }
}

// Fused grey morphology kernel. struct_meta stores per-element dimension offsets
// followed by the precomputed flat offset used by the interior fast path.
template <typename MorphologyOp>
__global__ void grey_morphology_fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ struct_vals,
    const int64_t* __restrict__ struct_meta,
    const int num_struct,
    const int ndim_spatial,
    const GreyMorphologyGeometry geom,
    const int64_t total_spatial,
    const int64_t batch_channel,
    const BorderMode mode,
    const float cval
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_channel * total_spatial) return;

    int64_t bc = idx / total_spatial;
    int64_t sp = idx % total_spatial;

    int64_t coords[GREY_MORPH_MAX_NDIM];
    int64_t rem = sp;
    for (int d = 0; d < ndim_spatial; d++) {
        coords[d] = rem / geom.spatial_stride[d];
        rem %= geom.spatial_stride[d];
    }

    float result = MorphologyOp::identity();
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
            int64_t offset = struct_meta[i * meta_stride + ndim_spatial];
            float val = in_ptr[sp + offset];
            result = MorphologyOp::combine(result, val, struct_vals[i]);
        }
        output[idx] = result;
        return;
    }

    for (int i = 0; i < num_struct; i++) {
        const int64_t* meta = struct_meta + i * meta_stride;

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

        float val = use_cval ? cval : in_ptr[flat_idx];
        result = MorphologyOp::combine(result, val, struct_vals[i]);
    }

    output[idx] = result;
}

template <typename MorphologyOp>
static torch::Tensor grey_morphology_cuda(
    torch::Tensor input,
    torch::Tensor structure,
    torch::Tensor footprint,
    std::vector<int64_t> origin_vec,
    int mode_int,
    float cval
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(input.device());

    TORCH_CHECK(!structure.is_cuda(), "structure must be a CPU tensor");
    TORCH_CHECK(!footprint.is_cuda(), "footprint must be a CPU tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(structure.is_contiguous(), "structure must be contiguous");
    TORCH_CHECK(footprint.is_contiguous(), "footprint must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(structure.dtype() == torch::kFloat32, "structure must be float32");
    TORCH_CHECK(footprint.dtype() == torch::kBool, "footprint must be bool");

    int ndim_spatial = structure.dim();
    TORCH_CHECK(ndim_spatial > 0 && ndim_spatial <= GREY_MORPH_MAX_NDIM,
                "structure dimension must be in 1-", GREY_MORPH_MAX_NDIM,
                ", got ", ndim_spatial);
    TORCH_CHECK((int)origin_vec.size() == ndim_spatial,
                "origin length must match structure dimensions");
    TORCH_CHECK(input.dim() == ndim_spatial + 2,
                "input spatial dimensions must match structure dimensions");

    const bool use_footprint = footprint.numel() != 0;
    if (use_footprint) {
        TORCH_CHECK(footprint.dim() == ndim_spatial,
                    "footprint dimensions must match structure dimensions");
        for (int d = 0; d < ndim_spatial; d++) {
            TORCH_CHECK(footprint.size(d) == structure.size(d),
                        "footprint shape must match structure shape");
        }
    }

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

    GreyMorphologyGeometry h_geom;
    for (int d = 0; d < GREY_MORPH_MAX_NDIM; d++) {
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

    // Keep only active footprint positions and precompute offsets.
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

            int64_t center_d = structure.size(d) / 2 + origin_vec[d];
            int64_t delta = MorphologyOp::offset(coord_d - center_d);
            h_struct_meta.push_back(delta);
            flat_offset += delta * h_spatial_stride[d];
            h_geom.min_deltas[d] = std::min(h_geom.min_deltas[d], delta);
            h_geom.max_deltas[d] = std::max(h_geom.max_deltas[d], delta);
        }
        h_struct_meta.push_back(flat_offset);
    }

    int num_struct = static_cast<int>(h_struct_vals.size());
    TORCH_CHECK(num_struct > 0,
                "footprint must contain at least one active position");

    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32);
    auto opts_i = torch::TensorOptions().dtype(torch::kInt64);

    auto d_struct_vals = torch::from_blob(
        h_struct_vals.data(), {num_struct}, opts_f).to(input.device());
    auto d_struct_meta = torch::from_blob(
        h_struct_meta.data(), {num_struct * (ndim_spatial + 1)}, opts_i
    ).to(input.device());

    BorderMode bmode = static_cast<BorderMode>(mode_int);

    int64_t total_threads = batch_channel * total_spatial;
    int threads = 256;
    int blocks = (int)((total_threads + threads - 1) / threads);
    const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());

    grey_morphology_fused_kernel<MorphologyOp><<<blocks, threads, 0, stream>>>(
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
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

torch::Tensor grey_erosion_cuda(
    torch::Tensor input,
    torch::Tensor structure,
    torch::Tensor footprint,
    std::vector<int64_t> origin_vec,
    int mode_int,
    float cval
) {
    return grey_morphology_cuda<GreyErosionOp>(
        input, structure, footprint, origin_vec, mode_int, cval
    );
}

torch::Tensor grey_dilation_cuda(
    torch::Tensor input,
    torch::Tensor structure,
    torch::Tensor footprint,
    std::vector<int64_t> origin_vec,
    int mode_int,
    float cval
) {
    return grey_morphology_cuda<GreyDilationOp>(
        input, structure, footprint, origin_vec, mode_int, cval
    );
}
