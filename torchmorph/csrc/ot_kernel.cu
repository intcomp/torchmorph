#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <cmath>

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

__global__ void sinkhorn_large(
    const float* a,
    const float* b,
    const float* K,
    float* u,
    float* v,
    int N
){
    //sinkhorn for large scale matrix
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