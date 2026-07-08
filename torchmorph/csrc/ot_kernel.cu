#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>

namespace {

constexpr int kThreads = 256;  // power of two, assumed by the reductions below

// One Sinkhorn scaling update. Vectors are batched (n, d); the (d, d) matrix
// is shared across the batch. One block per (row, batch) pair computes:
//   out[b, i] = num[b, i] / (sum_j mat[i, j] * scale[b, j] + 1e-12)
// The u-update passes (a, K, v); the v-update passes (b, K^T, u).
__global__ void scaling_update(
    const float* __restrict__ num,
    const float* __restrict__ mat,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int d
){
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t offset = (size_t)blockIdx.y * d;

    float local_sum = 0.0f;
    for (int j = tid; j < d; j += blockDim.x){
        local_sum += mat[(size_t)row * d + j] * scale[offset + j];
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) out[offset + row] = num[offset + row] / (sdata[0] + 1e-12f);
}

// One log-domain Sinkhorn update, one block per (row, batch) pair:
//   log_out[b, i] = log_num[b, i] - logsumexp_j(-cost[i, j] / eps + log_scale[b, j])
// evaluated with a two-pass block reduction (max, then sum of shifted exps).
// The u-update passes (log_a, M, log_v); the v-update passes (log_b, M^T, log_u).
__global__ void log_scaling_update(
    const float* __restrict__ log_num,
    const float* __restrict__ cost,
    const float* __restrict__ log_scale,
    float* __restrict__ log_out,
    int d,
    float inv_eps
){
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t offset = (size_t)blockIdx.y * d;

    float local_max = -INFINITY;
    for (int j = tid; j < d; j += blockDim.x){
        local_max = fmaxf(local_max, log_scale[offset + j] - cost[(size_t)row * d + j] * inv_eps);
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    const float row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < d; j += blockDim.x){
        local_sum += expf(log_scale[offset + j] - cost[(size_t)row * d + j] * inv_eps - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // row_max == -inf means the opposite marginal is all zero; pin the
    // potential instead of propagating -inf - (-inf) = NaN.
    if (tid == 0){
        log_out[offset + row] = row_max == -INFINITY
            ? -INFINITY
            : log_num[offset + row] - row_max - logf(sdata[0]);
    }
}

void check_input(const torch::Tensor& t, int64_t numel, const char* name){
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.numel() == numel, name, " must have ", numel, " elements, got ", t.numel());
}

// Validate batched (n, d) vectors against a shared (d, d) matrix and return
// the launch grid: one block per (row, batch) pair.
dim3 make_grid(const torch::Tensor& u, const torch::Tensor& mat){
    TORCH_CHECK(mat.dim() == 2 && mat.size(0) == mat.size(1), "cost/kernel matrix must be square");
    TORCH_CHECK(u.dim() == 2 && u.size(1) == mat.size(0),
                "scaling vectors must be (n, d) with d matching the matrix");
    return dim3(static_cast<unsigned>(u.size(1)), static_cast<unsigned>(u.size(0)));
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> sinkhorn_fastiter(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& K,
    const torch::Tensor& K_T,
    torch::Tensor u,
    torch::Tensor v,
    int64_t n_iter
){
    const dim3 grid = make_grid(u, K);
    const int64_t d = K.size(0);
    check_input(a, u.numel(), "a");
    check_input(b, u.numel(), "b");
    check_input(K, d * d, "K");
    check_input(K_T, d * d, "K_T");
    check_input(u, u.numel(), "u");
    check_input(v, u.numel(), "v");

    const at::cuda::CUDAGuard guard(u.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int shmem = kThreads * sizeof(float);

    for (int64_t i = 0; i < n_iter; i++){
        scaling_update<<<grid, kThreads, shmem, stream>>>(
            a.data_ptr<float>(), K.data_ptr<float>(), v.data_ptr<float>(),
            u.data_ptr<float>(), static_cast<int>(d));
        scaling_update<<<grid, kThreads, shmem, stream>>>(
            b.data_ptr<float>(), K_T.data_ptr<float>(), u.data_ptr<float>(),
            v.data_ptr<float>(), static_cast<int>(d));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(u, v);
}

std::tuple<torch::Tensor, torch::Tensor> sinkhorn_logiter(
    const torch::Tensor& log_a,
    const torch::Tensor& log_b,
    const torch::Tensor& M,
    const torch::Tensor& M_T,
    torch::Tensor log_u,
    torch::Tensor log_v,
    int64_t n_iter,
    double epsilon
){
    TORCH_CHECK(epsilon > 0, "epsilon must be positive, got ", epsilon);
    const dim3 grid = make_grid(log_u, M);
    const int64_t d = M.size(0);
    check_input(log_a, log_u.numel(), "log_a");
    check_input(log_b, log_u.numel(), "log_b");
    check_input(M, d * d, "M");
    check_input(M_T, d * d, "M_T");
    check_input(log_u, log_u.numel(), "log_u");
    check_input(log_v, log_u.numel(), "log_v");

    const at::cuda::CUDAGuard guard(log_u.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int shmem = kThreads * sizeof(float);
    const float inv_eps = static_cast<float>(1.0 / epsilon);

    for (int64_t i = 0; i < n_iter; i++){
        log_scaling_update<<<grid, kThreads, shmem, stream>>>(
            log_a.data_ptr<float>(), M.data_ptr<float>(), log_v.data_ptr<float>(),
            log_u.data_ptr<float>(), static_cast<int>(d), inv_eps);
        log_scaling_update<<<grid, kThreads, shmem, stream>>>(
            log_b.data_ptr<float>(), M_T.data_ptr<float>(), log_u.data_ptr<float>(),
            log_v.data_ptr<float>(), static_cast<int>(d), inv_eps);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(log_u, log_v);
}
