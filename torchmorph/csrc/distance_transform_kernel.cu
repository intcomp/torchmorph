#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// ============================================================================
// 1D Euclidean Distance Transform - Optimized Warp-Level Parallel
// ============================================================================
// 
// Based on the paper's algorithm using:
// - __ballot_sync() for feature point voting
// - __shfl_sync() for warp-level communication (NO shared memory within warp)
// - Parallel reduction tree for cross-warp propagation
// - Time complexity: O(log32(n)) with O(n) total work
//
// ============================================================================

#define WARP_SIZE 32
#define INF_VAL 1e9f

__device__ __forceinline__ float sqr(float x) { return x * x; }

// ============================================================================
// Device Function: Find nearest feature to the LEFT using warp operations
// ============================================================================
// 
// Algorithm (as described in the paper, Figure 7a):
// 1. Each thread votes if it holds a feature point -> ballot() creates bitmask
// 2. Mask high (warpSize - lane - 1) bits with 0
// 3. Use clz() to count leading zeros
// 4. Nearest thread lane = (warpSize - clz() - 1)
// 5. Use __shfl_sync() to get the feature index from that thread
//
// Returns: index of nearest feature to the left, or -1 if none exists
// ============================================================================

__device__ __forceinline__ int find_nearest_left_in_warp(
    int lane,
    int my_index,
    unsigned int feature_mask
) {
    // Mask high bits: only keep features to the LEFT of current lane
    unsigned int left_mask = feature_mask & ((1U << lane) - 1);
    
    // We must execute __shfl_sync for ALL threads in the warp
    // Calculate nearest_lane if valid, otherwise use 0 (safe default)
    int nearest_lane = 0;
    if (left_mask != 0) {
        nearest_lane = 31 - __clz(left_mask);
    }
    
    // Perform shuffle for ALL threads
    int nearest_index = __shfl_sync(0xFFFFFFFF, my_index, nearest_lane);
    
    // Only return valid result if we actually found a feature
    return (left_mask != 0) ? nearest_index : -1;
}

// ============================================================================
// Device Function: Find nearest feature to the RIGHT using warp operations
// ============================================================================

__device__ __forceinline__ int find_nearest_right_in_warp(
    int lane,
    int my_index,
    unsigned int feature_mask
) {
    // Mask low bits: only keep features to the RIGHT of current lane
    unsigned int right_mask = feature_mask & ~((1U << (lane + 1)) - 1);
    
    // Calculate nearest_lane if valid, otherwise use 0 (safe default)
    int nearest_lane = 0;
    if (right_mask != 0) {
        nearest_lane = __ffs(right_mask) - 1;
    }
    
    // Perform shuffle for ALL threads
    int nearest_index = __shfl_sync(0xFFFFFFFF, my_index, nearest_lane);
    
    return (right_mask != 0) ? nearest_index : -1;
}

// ============================================================================
// Warp Scan Helpers
// ============================================================================

// Inclusive Max Scan for positive integers (returns max seen so far)
__device__ __forceinline__ int warp_scan_inclusive_max(int val, int width) {
    // Hillis-Steele Scan (O(log N))
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        int neighbor_val = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x % 32 >= offset) {
            // Logic: take the max of current and neighbor
            // Handle -1 (invalid) carefully: max behavior handles -1 naturally if features are >= 0
            if (neighbor_val > val) val = neighbor_val;
        }
    }
    return val;
}

// Inclusive Min Scan (returns min seen so far from right)
// Note: We use __shfl_down_sync for suffix scan
__device__ __forceinline__ int warp_scan_suffix_min(int val, int width) {
    // Suffix Scan (Right to Left)
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        int neighbor_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
        // If we have a neighbor to the right
        if ((threadIdx.x % 32) + offset < width) {
            // Logic: take min. If current is -1 (invalid), take neighbor.
            // If neighbor is -1, ignore it.
            if (val == -1) val = neighbor_val;
            else if (neighbor_val != -1) {
                if (neighbor_val < val) val = neighbor_val;
            }
        }
    }
    return val;
}


// ============================================================================
// Kernel: Optimized 1D EDT using Two-Level Tree reduction
// ============================================================================

__global__ void edt_1d_warp_optimized_kernel(
    const float* __restrict__ d_input,
    float* __restrict__ d_dist,
    int32_t* __restrict__ d_indices,
    int width,
    int height
) {
    int row = blockIdx.x;
    if (row >= height) return;
    
    const float* row_input = d_input + row * width;
    float* row_dist = d_dist + row * width;
    int32_t* row_indices = d_indices + row * width;
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    // Shared memory for Inter-Warp Scan
    // We use one buffer for the reduction result
    __shared__ int s_warp_boundary[32]; 
    
    // ========================================================================
    // PASS 1: Find nearest feature to the LEFT (Prefix Max Scan)
    // ========================================================================
    
    int global_left_feature = -1;
    
    for (int base = 0; base < width; base += blockDim.x) {
        int i = base + tid;
        bool is_valid = (i < width);
        bool is_feature = is_valid && (row_input[i] > 0.5f);
        
        // 1. Warp-Level: Find local nearest
        unsigned int feature_mask = __ballot_sync(0xFFFFFFFF, is_feature);
        int my_index = is_feature ? i : -1;
        int warp_left_feature = find_nearest_left_in_warp(lane, my_index, feature_mask);
        
        // 2. Prepare for Block-Level Scan: Write rightmost feature of this warp
        int rightmost_lane = (feature_mask != 0) ? (31 - __clz(feature_mask)) : 0;
        int rightmost_index = __shfl_sync(0xFFFFFFFF, my_index, rightmost_lane);
        
        if (lane == 0) {
            s_warp_boundary[warp_id] = (feature_mask != 0) ? rightmost_index : -1;
        }
        __syncthreads();
        
        // 3. Block-Level: Warp 0 performs parallel prefix scan over warp boundaries
        // This is the "Tree" part for inter-warp communication
        if (warp_id == 0) {
            // Load boundary from shared memory (only if valid warp)
            int val = (lane < num_warps) ? s_warp_boundary[lane] : -1;
            
            // Perform inclusive max scan
            int scan_res = warp_scan_inclusive_max(val, num_warps);
            
            // Write back inclusive scan result
            if (lane < num_warps) {
                s_warp_boundary[lane] = scan_res;
            }
        }
        __syncthreads();
        
        // 4. Combine Results
        if (is_valid) {
            int left_feature = -1;
            
            if (is_feature) left_feature = i;
            else if (warp_left_feature != -1) left_feature = warp_left_feature;
            else {
                // Look at the scan result from the PREVIOUS warp
                if (warp_id > 0) {
                    left_feature = s_warp_boundary[warp_id - 1];
                }
                
                // If still -1, fallback to global history
                if (left_feature == -1) left_feature = global_left_feature;
            }
            
            // Store result
            if (left_feature >= 0) {
                row_dist[i] = sqr((float)(i - left_feature));
                row_indices[i] = left_feature;
            } else {
                row_dist[i] = INF_VAL;
                row_indices[i] = -1;
            }
        }
        
        // 5. Update Global History
        // The last warp's scan result contains the max index for the whole tile
        int tile_max = s_warp_boundary[num_warps - 1];
        if (tile_max != -1) global_left_feature = tile_max;
        
        __syncthreads();
    }
    
    // ========================================================================
    // PASS 2: Find nearest feature to the RIGHT (Suffix Min Scan)
    // ========================================================================
    
    int global_right_feature = -1;
    int num_tiles = (width + blockDim.x - 1) / blockDim.x;
    
    for (int tile = num_tiles - 1; tile >= 0; --tile) {
        int base = tile * blockDim.x;
        int i = base + tid;
        bool is_valid = (i < width);
        bool is_feature = is_valid && (row_input[i] > 0.5f);
        
        // 1. Warp-Level
        unsigned int feature_mask = __ballot_sync(0xFFFFFFFF, is_feature);
        int my_index = is_feature ? i : -1;
        int warp_right_feature = find_nearest_right_in_warp(lane, my_index, feature_mask);
        
        // 2. Prepare: Write leftmost feature of this warp
        int leftmost_lane = (feature_mask != 0) ? (__ffs(feature_mask) - 1) : 0;
        int leftmost_index = __shfl_sync(0xFFFFFFFF, my_index, leftmost_lane);
        
        if (lane == 0) {
            s_warp_boundary[warp_id] = (feature_mask != 0) ? leftmost_index : -1;
        }
        __syncthreads();
        
        // 3. Block-Level: Warp 0 performs parallel suffix scan (Right-to-Left tree)
        if (warp_id == 0) {
            int val = (lane < num_warps) ? s_warp_boundary[lane] : -1;
            
            // Perform suffix min scan
            int scan_res = warp_scan_suffix_min(val, num_warps);
            
            if (lane < num_warps) {
                s_warp_boundary[lane] = scan_res;
            }
        }
        __syncthreads();
        
        // 4. Combine Results
        if (is_valid) {
            int right_feature = -1;
            
            if (is_feature) right_feature = i;
            else if (warp_right_feature != -1) right_feature = warp_right_feature;
            else {
                // Look at scan result from NEXT warp
                if (warp_id < num_warps - 1) {
                    right_feature = s_warp_boundary[warp_id + 1];
                }
                
                if (right_feature == -1) right_feature = global_right_feature;
            }
            
            // Update Min Distance
            if (right_feature >= 0) {
                float d = sqr((float)(right_feature - i));
                if (d < row_dist[i]) {
                    row_dist[i] = d;
                    row_indices[i] = right_feature;
                }
            }
        }
        
        // 5. Update Global History
        // First warp's scan result contains min index for whole tile
        int tile_min = s_warp_boundary[0];
        if (tile_min != -1) global_right_feature = tile_min;
        
        __syncthreads();
    }
}

// ============================================================================
// PyTorch Wrapper Function
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor> distance_transform_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() >= 1, "Input must be at least 1D");
    
    // Get dimensions
    int64_t ndim = input.dim();
    int64_t width = input.size(-1);
    int64_t height = 1;
    
    if (ndim >= 2) {
        height = input.size(-2);
    }
    
    int64_t batch_size = input.numel() / (width * height);
    
    // Flatten to [batch * height, width]
    auto input_flat = input.view({batch_size * height, width});
    
    // Create output tensors
    auto dist_map = torch::empty_like(input);
    auto dist_flat = dist_map.view({batch_size * height, width});
    
    // Index map: same shape as input + last dimension for coordinate
    auto idx_shape = input.sizes().vec();
    idx_shape.push_back(1);
    auto idx_map = torch::empty(idx_shape, input.options().dtype(torch::kInt32));
    auto idx_flat = idx_map.view({batch_size * height, width, 1});
    
    // Launch kernel
    // Use 256 threads per block (8 warps) for good occupancy
    int threads_per_block = 256;
    int num_rows = batch_size * height;
    
    dim3 block(threads_per_block);
    dim3 grid(num_rows);
    
    edt_1d_warp_optimized_kernel<<<grid, block>>>(
        input_flat.data_ptr<float>(),
        dist_flat.data_ptr<float>(),
        idx_flat.data_ptr<int32_t>(),
        width,
        num_rows
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    // Take square root to get actual distance (not squared)
    dist_map = torch::sqrt(dist_map);
    
    return std::make_tuple(dist_map, idx_map);
}
