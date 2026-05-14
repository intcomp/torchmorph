#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <cmath>
#include <cfloat>


__global__ void u_update(
    const float* a,
    const float* b,
    const float* K,
    float* u,
    float* v,
    int N
){
    int bidx = blockIdx.x;
    int blk = blockDim.x;
    int tid = threadIdx.x;
    int total = N;
    float local_sum = 0.0f;

    if(bidx >= total){
        return;
    }

    extern __shared__ float sdata[];

    //get ready for sinkhorn:u= a / (k v)
    for (int j = tid; j < N; j += blk){
        local_sum += K[bidx*N+j] * v[j];
    }//calculate one column

    sdata[tid]=local_sum;//get one element of one column
    __syncthreads();

    
    for (int stride=blk/2; stride > 0; stride /= 2){
        if (tid < stride){
        sdata[tid]+=sdata[tid+stride];
        }
        __syncthreads();
    }

    if (tid == 0){
        u[bidx] = a[bidx] / (sdata[0]+1e-12f);
    }
}


__global__ void v_update(
    const float* a,
    const float* b,
    const float* K,
    float* u,
    float* v,
    int N
){
    int bidx = blockIdx.x;
    int blk = blockDim.x;
    int tid = threadIdx.x;
    int total = N;
    float local_sum = 0.0f;

    if(bidx >= total){
        return;
    }

    extern __shared__ float sdata[];

    //get ready for sinkhorn:u= b / (k^t u)
    for (int j = tid; j < N; j += blk){
        local_sum += K[bidx+j*N] * u[j];
    }//calculate one column

    sdata[tid]=local_sum;//get one element of one column
    __syncthreads();

    
    for (int stride=blk/2; stride > 0; stride /= 2){
        if (tid < stride){
            sdata[tid]+=sdata[tid+stride];
        }
        __syncthreads();
    }

    if (tid == 0){
        v[bidx] = b[bidx] / (sdata[0]+1e-12f);
    }
}


// Update one log-domain Sinkhorn scaling vector.
//
// Each block owns one row i and computes:
//   log_u[i] = log_a[i] - logsumexp_j(-reg * M[i, j] + log_v[j])
// The logsumexp is evaluated with a two-pass block reduction:
// first max(z_j), then sum(exp(z_j - max)).
__global__ void u_update_log(
    const float* log_a,
    const float* M,
    float* log_u,
    const float* log_v,
    int N,
    float reg
){
    //sinkhorn for log domain iteration
    int blk = blockDim.x;
    int bidx = blockIdx.x;
    int tid = threadIdx.x;
    int total = N;
    float local_max = -INFINITY;

    if(bidx >= total){
        return;
    }

    extern __shared__ float sdata[];

    //generate each log_kv
    for (int j = tid; j < N; j += blk){
        float log_k = - reg * M[bidx*N+j];
        if(local_max <= (log_k + log_v[j])){
            local_max = log_k + log_v[j];
        }     
    }

    sdata[tid] = local_max;
    __syncthreads();

    //first reduction to find max(log_k+log_v)
    for (int stride=blk/2; stride > 0; stride /= 2){
        if (tid < stride){
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);//get larger num
        }
        __syncthreads();
    }

    float row_max = sdata[0];
    __syncthreads();
    float local_sum = 0.0f;

    //log-domain iteration log_u = log_a - LSE(log_k+log_v)
    for (int j = tid; j < N; j += blk){
        float log_k = - reg * M[bidx*N+j];
        local_sum += expf(log_k + log_v[j] - row_max);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride=blk/2; stride > 0; stride /= 2){
        if (tid < stride){
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0){
        log_u[bidx] = log_a[bidx] - row_max - logf(sdata[0]);
    }

}

__global__ void v_update_log(
    const float* log_b,
    const float* M,
    const float* log_u,
    float* log_v,
    int N,
    float reg
){
    //sinkhorn for log domain iteration
    int blk = blockDim.x;
    int bidx = blockIdx.x;
    int tid = threadIdx.x;
    int total = N;
    float local_max = -INFINITY;

    if(bidx >= total){
        return;
    }

    extern __shared__ float sdata[];

    //generate each log_kTu
    for (int j = tid; j < N; j += blk){
        float log_k = - reg * M[bidx+j*N];
        if(local_max <= (log_k + log_u[j])){
            local_max = log_k + log_u[j];
        }     
    }

    sdata[tid] = local_max;
    __syncthreads();

    //first reduction to find max(log_k+log_v)
    for (int stride=blk/2; stride > 0; stride /= 2){
        if (tid < stride){
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);//get larger num
        }
        __syncthreads();
    }

    float row_max = sdata[0];
    __syncthreads();
    float local_sum = 0.0f;

    //log-domain iteration log_v = log_b - LSE(log_k+log_u)
    for (int j = tid; j < N; j += blk){
        float log_k = - reg * M[bidx+j*N];
        local_sum += expf(log_k + log_u[j] - row_max);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride=blk/2; stride > 0; stride /= 2){
        if (tid < stride){
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0){
        log_v[bidx] = log_b[bidx] - row_max - logf(sdata[0]);
    }

}

std::tuple<torch::Tensor, torch::Tensor> sinkhorn_fastiter(
    const torch::Tensor source,
    const torch::Tensor target,
    const torch::Tensor k,
    torch::Tensor u,
    torch::Tensor v,
    int itrstep,
    int N
){
    int threads = 256;
    int blocks = N;
    int shared_memory = threads * sizeof(float);

    for (int i = 0; i < itrstep; i++){
        u_update<<<blocks, threads, shared_memory>>>(
            source.data_ptr<float>(),
            target.data_ptr<float>(),
            k.data_ptr<float>(),
            u.data_ptr<float>(),
            v.data_ptr<float>(),
            N
        );
        v_update<<<blocks, threads, shared_memory>>>(
            source.data_ptr<float>(),
            target.data_ptr<float>(),
            k.data_ptr<float>(),
            u.data_ptr<float>(),
            v.data_ptr<float>(),
            N
        );
    }
return std::make_tuple(u, v);
}

std::tuple<torch::Tensor, torch::Tensor> sinkhorn_logiter(
    const torch::Tensor log_a,
    const torch::Tensor log_b,
    const torch::Tensor M,
    torch::Tensor log_u,
    torch::Tensor log_v,
    int itrstep,
    int N,
    float reg
){
    int threads = 256;
    int blocks = N;
    int shared_memory = threads * sizeof(float);

    for (int i = 0; i < itrstep; i++){
        u_update_log<<<blocks, threads, shared_memory>>>(
            log_a.data_ptr<float>(),
            M.data_ptr<float>(),
            log_u.data_ptr<float>(),
            log_v.data_ptr<float>(),
            N,
            reg
        );
        v_update_log<<<blocks, threads, shared_memory>>>(
            log_b.data_ptr<float>(),
            M.data_ptr<float>(),
            log_u.data_ptr<float>(),
            log_v.data_ptr<float>(),
            N,
            reg
        );
    }
return std::make_tuple(log_u, log_v);
}
