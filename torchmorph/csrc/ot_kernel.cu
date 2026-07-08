#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>

namespace {

constexpr int kThreads = 256;  // multiple of the warp size, assumed below
constexpr int kBatchTile = 8;  // batch items processed per block
constexpr int kWarps = kThreads / 32;

__device__ inline float warp_reduce_sum(float v){
    for (int offset = 16; offset > 0; offset >>= 1){
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// Block-level sum reduction; the result is valid in thread 0 only.
__device__ inline float block_reduce_sum(float v){
    __shared__ float partial[kWarps];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) partial[warp] = v;
    __syncthreads();
    if (warp == 0){
        v = lane < kWarps ? partial[lane] : 0.0f;
        v = warp_reduce_sum(v);
    }
    __syncthreads();  // let the next call reuse `partial`
    return v;
}

// Merge two online-logsumexp states; (m, s) represents m + log(s).
// s is 0 exactly when m is -inf (the empty state), which the guards below
// keep NaN-free.
__device__ inline void lse_merge(float& m, float& s, float m_other, float s_other){
    const float hi = fmaxf(m, m_other);
    if (hi == -INFINITY){
        m = -INFINITY;
        s = 0.0f;
        return;
    }
    s = (m == -INFINITY ? 0.0f : s * expf(m - hi)) +
        (m_other == -INFINITY ? 0.0f : s_other * expf(m_other - hi));
    m = hi;
}

__device__ inline void warp_reduce_lse(float& m, float& s){
    for (int offset = 16; offset > 0; offset >>= 1){
        const float m_other = __shfl_down_sync(0xffffffff, m, offset);
        const float s_other = __shfl_down_sync(0xffffffff, s, offset);
        lse_merge(m, s, m_other, s_other);
    }
}

// Block-level logsumexp reduction; the result is valid in thread 0 only.
__device__ inline void block_reduce_lse(float& m, float& s){
    __shared__ float partial_m[kWarps];
    __shared__ float partial_s[kWarps];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    warp_reduce_lse(m, s);
    if (lane == 0){
        partial_m[warp] = m;
        partial_s[warp] = s;
    }
    __syncthreads();
    if (warp == 0){
        m = lane < kWarps ? partial_m[lane] : -INFINITY;
        s = lane < kWarps ? partial_s[lane] : 0.0f;
        warp_reduce_lse(m, s);
    }
    __syncthreads();  // let the next call reuse the partial arrays
}

// One Sinkhorn scaling update. One block per (row, batch tile): the matrix
// row is streamed once and applied to TILE batch items, so the d^2 matrix is
// read ceil(n / TILE) times per update instead of n times.
//   out[b, i] = num[b, i] / (sum_j mat[i, j] * scale[b, j] + 1e-12)
// The u-update passes (a, K, v); the v-update passes (b, K^T, u).
template <int TILE>
__global__ void scaling_update(
    const float* __restrict__ num,
    const float* __restrict__ mat,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int d,
    int n_batch
){
    const int row = blockIdx.x;
    const int batch0 = blockIdx.y * TILE;
    const int tile = min(TILE, n_batch - batch0);
    const float* mat_row = mat + (size_t)row * d;

    float acc[TILE];
#pragma unroll
    for (int t = 0; t < TILE; t++) acc[t] = 0.0f;

    for (int j = threadIdx.x; j < d; j += blockDim.x){
        const float k = mat_row[j];
#pragma unroll
        for (int t = 0; t < TILE; t++){
            if (t < tile) acc[t] += k * scale[(size_t)(batch0 + t) * d + j];
        }
    }

    for (int t = 0; t < tile; t++){
        const float total = block_reduce_sum(acc[t]);
        if (threadIdx.x == 0){
            const size_t idx = (size_t)(batch0 + t) * d + row;
            out[idx] = num[idx] / (total + 1e-12f);
        }
    }
}

// One log-domain Sinkhorn update with the same batch tiling:
//   log_out[b, i] = log_num[b, i] - logsumexp_j(-cost[i, j] / eps + log_scale[b, j])
// The logsumexp is evaluated online in a single pass over the row (running
// max with a rescaled running sum), halving the matrix traffic of the
// classic two-pass max-then-sum evaluation.
// The u-update passes (log_a, M, log_v); the v-update passes (log_b, M^T, log_u).
template <int TILE>
__global__ void log_scaling_update(
    const float* __restrict__ log_num,
    const float* __restrict__ cost,
    const float* __restrict__ log_scale,
    float* __restrict__ log_out,
    int d,
    int n_batch,
    float inv_eps
){
    const int row = blockIdx.x;
    const int batch0 = blockIdx.y * TILE;
    const int tile = min(TILE, n_batch - batch0);
    const float* cost_row = cost + (size_t)row * d;

    float m[TILE], s[TILE];
#pragma unroll
    for (int t = 0; t < TILE; t++){
        m[t] = -INFINITY;
        s[t] = 0.0f;
    }

    for (int j = threadIdx.x; j < d; j += blockDim.x){
        const float c = -cost_row[j] * inv_eps;
#pragma unroll
        for (int t = 0; t < TILE; t++){
            if (t >= tile) continue;
            const float z = c + log_scale[(size_t)(batch0 + t) * d + j];
            if (z > m[t]){
                s[t] = s[t] * expf(m[t] - z) + 1.0f;  // expf(-inf) == 0 covers the first element
                m[t] = z;
            } else if (m[t] != -INFINITY){  // skip z == m == -inf, whose contribution is 0
                s[t] += expf(z - m[t]);
            }
        }
    }

    for (int t = 0; t < tile; t++){
        block_reduce_lse(m[t], s[t]);
        if (threadIdx.x == 0){
            const size_t idx = (size_t)(batch0 + t) * d + row;
            // m == -inf means the opposite marginal is all zero; pin the
            // potential instead of producing NaN.
            log_out[idx] = m[t] == -INFINITY ? -INFINITY : log_num[idx] - m[t] - logf(s[t]);
        }
    }
}

void check_input(const torch::Tensor& t, int64_t numel, const char* name){
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.numel() == numel, name, " must have ", numel, " elements, got ", t.numel());
}

// Validate batched (n, d) vectors against a shared (d, d) matrix.
void check_shapes(const torch::Tensor& u, const torch::Tensor& mat){
    TORCH_CHECK(mat.dim() == 2 && mat.size(0) == mat.size(1), "cost/kernel matrix must be square");
    TORCH_CHECK(u.dim() == 2 && u.size(1) == mat.size(0),
                "scaling vectors must be (n, d) with d matching the matrix");
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
    check_shapes(u, K);
    const int64_t n = u.size(0);
    const int64_t d = u.size(1);
    check_input(a, n * d, "a");
    check_input(b, n * d, "b");
    check_input(K, d * d, "K");
    check_input(K_T, d * d, "K_T");
    check_input(u, n * d, "u");
    check_input(v, n * d, "v");

    const at::cuda::CUDAGuard guard(u.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const bool single = n == 1;
    const dim3 grid(static_cast<unsigned>(d),
                    single ? 1u : static_cast<unsigned>((n + kBatchTile - 1) / kBatchTile));

    for (int64_t i = 0; i < n_iter; i++){
        if (single){
            scaling_update<1><<<grid, kThreads, 0, stream>>>(
                a.data_ptr<float>(), K.data_ptr<float>(), v.data_ptr<float>(),
                u.data_ptr<float>(), (int)d, (int)n);
            scaling_update<1><<<grid, kThreads, 0, stream>>>(
                b.data_ptr<float>(), K_T.data_ptr<float>(), u.data_ptr<float>(),
                v.data_ptr<float>(), (int)d, (int)n);
        } else {
            scaling_update<kBatchTile><<<grid, kThreads, 0, stream>>>(
                a.data_ptr<float>(), K.data_ptr<float>(), v.data_ptr<float>(),
                u.data_ptr<float>(), (int)d, (int)n);
            scaling_update<kBatchTile><<<grid, kThreads, 0, stream>>>(
                b.data_ptr<float>(), K_T.data_ptr<float>(), u.data_ptr<float>(),
                v.data_ptr<float>(), (int)d, (int)n);
        }
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
    check_shapes(log_u, M);
    const int64_t n = log_u.size(0);
    const int64_t d = log_u.size(1);
    check_input(log_a, n * d, "log_a");
    check_input(log_b, n * d, "log_b");
    check_input(M, d * d, "M");
    check_input(M_T, d * d, "M_T");
    check_input(log_u, n * d, "log_u");
    check_input(log_v, n * d, "log_v");

    const at::cuda::CUDAGuard guard(log_u.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const float inv_eps = static_cast<float>(1.0 / epsilon);
    const bool single = n == 1;
    const dim3 grid(static_cast<unsigned>(d),
                    single ? 1u : static_cast<unsigned>((n + kBatchTile - 1) / kBatchTile));

    for (int64_t i = 0; i < n_iter; i++){
        if (single){
            log_scaling_update<1><<<grid, kThreads, 0, stream>>>(
                log_a.data_ptr<float>(), M.data_ptr<float>(), log_v.data_ptr<float>(),
                log_u.data_ptr<float>(), (int)d, (int)n, inv_eps);
            log_scaling_update<1><<<grid, kThreads, 0, stream>>>(
                log_b.data_ptr<float>(), M_T.data_ptr<float>(), log_u.data_ptr<float>(),
                log_v.data_ptr<float>(), (int)d, (int)n, inv_eps);
        } else {
            log_scaling_update<kBatchTile><<<grid, kThreads, 0, stream>>>(
                log_a.data_ptr<float>(), M.data_ptr<float>(), log_v.data_ptr<float>(),
                log_u.data_ptr<float>(), (int)d, (int)n, inv_eps);
            log_scaling_update<kBatchTile><<<grid, kThreads, 0, stream>>>(
                log_b.data_ptr<float>(), M_T.data_ptr<float>(), log_u.data_ptr<float>(),
                log_v.data_ptr<float>(), (int)d, (int)n, inv_eps);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(log_u, log_v);
}
