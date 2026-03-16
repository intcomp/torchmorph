#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cmath>
#include <tuple>

#define BFDT_INF_VAL 1e20f
#define BFDT_MAX_NDIM 8
#define BFDT_BLOCK_SIZE 256 // Shared memory tile size

// ============================================================================
// High-performance CUDA Kernels (Templated to eliminate branch overhead)
// MetricType: 0=Euclidean, 1=Taxicab(L1), 2=Chessboard(L-inf)
// ============================================================================

template <int MetricType, int NDIM>
__global__ void bfdt_kernel(
    const int* __restrict__ foreground_coords,  // [num_foreground, NDIM]
    const int* __restrict__ background_coords,  // [num_background, NDIM]
    float* __restrict__ dist_out,               // [num_foreground]
    int* __restrict__ indices_out,              // [NDIM, num_foreground]
    int num_foreground,
    int num_background,
    const float* __restrict__ sampling,
    bool return_distances,
    bool return_indices
) {
    int f_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // 1. Register pre-computation: Load the current foreground point's coordinates,
    // apply sampling, and convert to float upfront to avoid repeating this in the loop.
    float my_coords_f[NDIM];
    if (f_idx < num_foreground) {
        #pragma unroll
        for (int d = 0; d < NDIM; d++) {
            my_coords_f[d] = (float)foreground_coords[f_idx * NDIM + d] * sampling[d];
        }
    }

    float min_dist = BFDT_INF_VAL;
    int best_bg_idx = -1;

    // 2. Allocate shared memory to cache a tile of background coordinates
    extern __shared__ float shared_bg[];

    // 3. Shared Memory Tiling: Iterate over background points in blocks
    for (int tile = 0; tile < num_background; tile += BFDT_BLOCK_SIZE) {
        
        // Collaborative loading: Each thread in the block loads one background point,
        // applies sampling, and stores it into shared memory.
        int bg_load_idx = tile + tid;
        if (bg_load_idx < num_background) {
            #pragma unroll
            for (int d = 0; d < NDIM; d++) {
                shared_bg[tid * NDIM + d] = (float)background_coords[bg_load_idx * NDIM + d] * sampling[d];
            }
        }
        
        // Sync! Ensure all background points for the current tile are fully loaded.
        __syncthreads();

        // Only valid foreground threads perform the distance calculations
        if (f_idx < num_foreground) {
            int limit = min(BFDT_BLOCK_SIZE, num_background - tile);
            
            // 4. Highly optimized inner loop
            for (int i = 0; i < limit; i++) {
                float dist = 0.0f;
                
                // The compiler resolves this branching at compile time based on the template,
                // resulting in zero runtime 'if' overhead.
                if (MetricType == 0) { // Euclidean (Compare squared distances, no sqrt here)
                    #pragma unroll
                    for (int d = 0; d < NDIM; d++) {
                        float diff = my_coords_f[d] - shared_bg[i * NDIM + d];
                        dist += diff * diff; // FMA instruction
                    }
                } else if (MetricType == 1) { // Taxicab (L1)
                    #pragma unroll
                    for (int d = 0; d < NDIM; d++) {
                        dist += fabsf(my_coords_f[d] - shared_bg[i * NDIM + d]);
                    }
                } else if (MetricType == 2) { // Chessboard (L-inf)
                    #pragma unroll
                    for (int d = 0; d < NDIM; d++) {
                        dist = fmaxf(dist, fabsf(my_coords_f[d] - shared_bg[i * NDIM + d]));
                    }
                }

                // Update minimum distance and track the global background index
                if (dist < min_dist) {
                    min_dist = dist;
                    best_bg_idx = tile + i;
                }
            }
        }
        
        // Sync! Ensure all threads are done with the current shared memory tile 
        // before the next iteration overwrites it.
        __syncthreads();
    }

    // 5. Write final results back to Global Memory
    if (f_idx < num_foreground) {
        if (return_distances) {
            if (MetricType == 0) {
                dist_out[f_idx] = sqrtf(min_dist); // Perform sqrt only once at the very end
            } else {
                dist_out[f_idx] = min_dist;
            }
        }

        if (return_indices && best_bg_idx != -1) {
            #pragma unroll
            for (int d = 0; d < NDIM; d++) {
                // Fetch the original integer background coordinates directly from global memory
                indices_out[d * num_foreground + f_idx] = background_coords[best_bg_idx * NDIM + d];
            }
        }
    }
}


// ============================================================================
// define dispatch ndim macro
// ============================================================================

#define DISPATCH_NDIM_AND_METRIC(METRIC_TYPE, NDIM_VAL, ...) \
    switch (NDIM_VAL) { \
        case 1: bfdt_kernel<METRIC_TYPE, 1><<<blocks, threads, shared_mem_bytes>>>(__VA_ARGS__); break; \
        case 2: bfdt_kernel<METRIC_TYPE, 2><<<blocks, threads, shared_mem_bytes>>>(__VA_ARGS__); break; \
        case 3: bfdt_kernel<METRIC_TYPE, 3><<<blocks, threads, shared_mem_bytes>>>(__VA_ARGS__); break; \
        case 4: bfdt_kernel<METRIC_TYPE, 4><<<blocks, threads, shared_mem_bytes>>>(__VA_ARGS__); break; \
        case 5: bfdt_kernel<METRIC_TYPE, 5><<<blocks, threads, shared_mem_bytes>>>(__VA_ARGS__); break; \
        case 6: bfdt_kernel<METRIC_TYPE, 6><<<blocks, threads, shared_mem_bytes>>>(__VA_ARGS__); break; \
        case 7: bfdt_kernel<METRIC_TYPE, 7><<<blocks, threads, shared_mem_bytes>>>(__VA_ARGS__); break; \
        case 8: bfdt_kernel<METRIC_TYPE, 8><<<blocks, threads, shared_mem_bytes>>>(__VA_ARGS__); break; \
        default: TORCH_CHECK(false, "Unsupported number of dimensions: ", NDIM_VAL); \
    }


// ============================================================================
// Host Implementation
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor> bfdt_cuda(
    torch::Tensor input,
    const std::string& metric,
    std::vector<float> sampling,
    bool return_distances,
    bool return_indices
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    
    input = input.contiguous();
    auto shape = input.sizes();
    int ndim = input.dim() - 2; // Extract spatial dimensions (excluding Batch and Channel)
    int64_t batch_size = shape[0] * shape[1];
    
    int metric_type = 0;
    if (metric == "euclidean") metric_type = 0;
    else if (metric == "taxicab") metric_type = 1;
    else if (metric == "chessboard") metric_type = 2;
    else TORCH_CHECK(false, "Unsupported metric type. Use 'euclidean', 'taxicab', or 'chessboard'.");

    auto sampling_tensor = torch::tensor(sampling, input.options());
    
    auto dist_out = torch::full_like(input, BFDT_INF_VAL);
    torch::Tensor indices_out;
    if (return_indices) {
        // Output shape matches PyTorch convention: [ndim, Batch, Channel, Spatial...]
        std::vector<int64_t> idx_shape = {ndim};
        for (int i = 0; i < input.dim(); i++) idx_shape.push_back(shape[i]);
        indices_out = torch::full(idx_shape, -1, input.options().dtype(torch::kInt32));
    }

    int64_t spatial_size = 1;
    for (int i = 2; i < input.dim(); i++) spatial_size *= shape[i];

    // Process each batch/channel independently
    for (int b = 0; b < batch_size; b++) {
        auto input_batch = input.view({batch_size, spatial_size})[b];
        
        auto foreground_mask = (input_batch != 0.0f);
        auto background_mask = (input_batch == 0.0f);
        
        auto fg_indices_flat = torch::nonzero(foreground_mask).view(-1);
        auto bg_indices_flat = torch::nonzero(background_mask).view(-1);
        
        int num_fg = fg_indices_flat.size(0);
        int num_bg = bg_indices_flat.size(0);
        
        if (num_fg == 0) continue;
        if (num_bg == 0) continue; // No background: distances remain initialized to INF

        auto spatial_shape = input.sizes().slice(2);
        auto fg_coords = torch::empty({num_fg, ndim}, input.options().dtype(torch::kInt32));
        auto bg_coords = torch::empty({num_bg, ndim}, input.options().dtype(torch::kInt32));
        
        // Convert flat 1D indices back to N-dimensional spatial coordinates
        auto get_coords = [&](torch::Tensor flat_indices, torch::Tensor coords_out) {
            auto rem = flat_indices.clone();
            for (int d = ndim - 1; d >= 0; d--) {
                coords_out.select(1, d).copy_(rem % spatial_shape[d]);
                rem.div_(spatial_shape[d], "trunc");
            }
        };
        
        get_coords(fg_indices_flat, fg_coords);
        get_coords(bg_indices_flat, bg_coords);

        auto batch_dist = return_distance ? 
        torch::empty({num_fg}, input.options()) : torch::Tensor();
        auto batch_indices = return_indices ? torch::empty({ndim, num_fg},
        input.options().dtype(torch::kInt32)) : torch::Tensor();

        int threads = BFDT_BLOCK_SIZE;
        int blocks = (num_fg + threads - 1) / threads;
        size_t shared_mem_bytes = BFDT_BLOCK_SIZE * ndim * sizeof(float);

        // Dispatch to different template instantiations based on metric type 
        // to guarantee zero-overhead branching inside the inner loops.
        if (metric_type == 0) {
            DISPATCH_NDIM_AND_METRIC(0, ndim,
                fg_coords.data_ptr<int>(), bg_coords.data_ptr<int>(),
                batch_dist.data_ptr<float>(), batch_indices.data_ptr<int>(),
                num_fg, num_bg, sampling_tensor.data_ptr<float>(),
                return_distances, return_indices
            );
        } else if (metric_type == 1) {
            DISPATCH_NDIM_AND_METRIC(1, ndim,
                fg_coords.data_ptr<int>(), bg_coords.data_ptr<int>(),
                batch_dist.data_ptr<float>(), batch_indices.data_ptr<int>(),
                num_fg, num_bg, sampling_tensor.data_ptr<float>(),
                return_distances, return_indices
            );
        } else if (metric_type == 2) {
           DISPATCH_NDIM_AND_METRIC(2, ndim,
                fg_coords.data_ptr<int>(), bg_coords.data_ptr<int>(),
                batch_dist.data_ptr<float>(), batch_indices.data_ptr<int>(),
                num_fg, num_bg, sampling_tensor.data_ptr<float>(),
                return_distances, return_indices
            );
        }

        // Scatter the computed results back to their spatial positions
        if (return_distances) {
            auto dist_batch_flat = dist_out.view({batch_size, spatial_size})[b];
            dist_batch_flat.scatter_(0, fg_indices_flat, batch_dist);
            auto bg_dist = torch::zeros({num_bg}, input.options());
            dist_batch_flat.scatter_(0, bg_indices_flat, bg_dist);
        }

        if (return_indices) {
            for (int d = 0; d < ndim; d++) {
                auto idx_slice = indices_out.select(0, d).view({batch_size, spatial_size})[b];
                idx_slice.scatter_(0, fg_indices_flat, batch_indices.select(0, d));
                auto bg_coord_d = bg_coords.select(1, d);
                idx_slice.scatter_(0, bg_indices_flat, bg_coord_d);
            }
        }
    }

    return std::make_tuple(dist_out, indices_out);
}