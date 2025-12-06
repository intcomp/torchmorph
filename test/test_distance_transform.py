import torch
import pytest
import numpy as np
from scipy.ndimage import distance_transform_edt as scipy_edt
import torchmorph as tm 

# ======================================================================
# 辅助函数
# ======================================================================
def batch_scipy_edt_with_indices(batch_numpy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist_results, indices_results = [], []
    
    # 保证 batch_numpy 至少是 (Batch, ...)
    # 如果进来的是 (H, W)，我们在外面已经处理成 (1, H, W) 了
    if batch_numpy.ndim == 1:
        batch_numpy = batch_numpy[np.newaxis, ...]

    for sample in batch_numpy:
        dist, indices = scipy_edt(sample, return_indices=True, return_distances=True)
        dist_results.append(dist)
        indices_results.append(indices)
        
    output_dist = np.stack(dist_results, axis=0)
    output_indices = np.stack(indices_results, axis=0)
    output_indices = np.moveaxis(output_indices, 1, -1) 
    
    return output_dist, output_indices

# ======================================================================
# 测试数据
# ======================================================================
case_batch_1d = np.array([[1, 1, 0, 1, 0, 1], [0, 1, 1, 1, 1, 0]], dtype=np.float32)

case_batch_2d = np.array([[[0., 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 0]],
                          [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]], dtype=np.float32)

# 这里定义为 (4, 4)，意图是单张 2D 图
case_single_2d = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.float32)
case_explicit_batch_one = case_single_2d[np.newaxis, ...]

_case_3d_s1 = np.ones((4, 5, 6), dtype=np.float32); _case_3d_s1[1, 1, 1] = 0.0; _case_3d_s1[2, 3, 4] = 0.0
_case_3d_s2 = np.ones((4, 5, 6), dtype=np.float32); _case_3d_s2[0, 0, 0] = 0.0
case_batch_3d = np.stack([_case_3d_s1, _case_3d_s2], axis=0)

case_dim_one = np.ones((2, 5, 1), dtype=np.float32); case_dim_one[0, 2, 0] = 0.0; case_dim_one[1, 4, 0] = 0.0

# 4D Case
_case_4d_s1 = np.ones((3, 3, 3, 3), dtype=np.float32); _case_4d_s1[0, 0, 0, 0] = 0.0
_case_4d_s2 = np.ones((3, 3, 3, 3), dtype=np.float32); _case_4d_s2[1, 1, 1, 1] = 0.0
case_batch_4d_spatial = np.stack([_case_4d_s1, _case_4d_s2], axis=0)

# 5D Case
case_batch_5d_spatial = np.ones((1, 2, 2, 2, 2, 2), dtype=np.float32)
case_batch_5d_spatial[0, 0, 0, 0, 0, 0] = 0.0; case_batch_5d_spatial[0, 1, 1, 1, 1, 1] = 0.0

# ======================================================================
# 测试逻辑
# ======================================================================
@pytest.mark.parametrize(
    "input_numpy, has_batch_dim",
    [
        pytest.param(case_batch_1d, True, id="1D_Batch"),
        pytest.param(case_batch_2d, True, id="2D_Batch"),
        pytest.param(case_single_2d, False, id="2D_Single_NoBatch"),
        pytest.param(case_explicit_batch_one, True, id="2D_Single_ExplicitBatch"),
        pytest.param(case_batch_3d, True, id="3D_Batch"),
        pytest.param(case_dim_one, True, id="2D_UnitDim_Batch"),
        pytest.param(case_batch_4d_spatial, True, id="4D_Spatial_Batch"),
        pytest.param(case_batch_5d_spatial, True, id="5D_Spatial_Batch"),
    ],
)
def test_distance_transform_and_indices(input_numpy: np.ndarray, has_batch_dim: bool, request: pytest.FixtureRequest):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # 1. 准备 Numpy 数据
    x_numpy_contiguous = np.ascontiguousarray(input_numpy)
    
    # 2. 准备 SciPy 输入
    # 如果意图是单样本 (has_batch_dim=False)，我们手动增加 Batch 维，
    # 这样 scipy 辅助函数就会把它当做一张图来处理，而不是 N 张 1D 图
    if not has_batch_dim:
        scipy_input = x_numpy_contiguous[np.newaxis, ...]
    else:
        scipy_input = x_numpy_contiguous

    # 3. 准备 CUDA 输入
    # 关键修复: 
    # 如果 has_batch_dim=False，说明这是单张 (H, W)，我们要测 2D EDT。
    # C++ API 默认第一维是 Batch，所以我们必须 unsqueeze(0) 变成 (1, H, W)。
    # 否则 C++ 会把它当做 (Batch=H, Len=W) 做 1D EDT。
    x_cuda = torch.from_numpy(x_numpy_contiguous).cuda()
    if not has_batch_dim:
        x_cuda = x_cuda.unsqueeze(0)

    print(f"\n\n--- 运行测试: {request.node.callspec.id} ---")
    print(f"CUDA 输入形状: {x_cuda.shape}")

    # 4. 运行 CUDA EDT
    dist_cuda, idx_cuda = tm.distance_transform(x_cuda.clone())

    # 5. 运行 SciPy (Ground Truth)
    dist_ref_numpy, idx_ref_numpy = batch_scipy_edt_with_indices(scipy_input)
    dist_ref = torch.from_numpy(dist_ref_numpy).to(torch.float32).cuda()

    # 6. 验证距离
    # 此时 dist_cuda 是 (1, H, W)，dist_ref 也是 (1, H, W)
    # 如果原意是 NoBatch，我们可以把 Batch 维 squeeze 掉再比，或者直接比
    print(f"CUDA Out Shape: {dist_cuda.shape}, Ref Shape: {dist_ref.shape}")
    assert dist_cuda.shape == dist_ref.shape, f"Shape mismatch: {dist_cuda.shape} vs {dist_ref.shape}"
    torch.testing.assert_close(dist_cuda, dist_ref, atol=1e-3, rtol=1e-3)
    print(">> 距离验证通过。")

    # 7. 验证索引
    # idx_cuda: (1, H, W, 2)
    # 构造 Grid
    spatial_shape = x_cuda.shape[1:] # (H, W)
    coords = [torch.arange(s, device='cuda') for s in spatial_shape]
    grid = torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1) # (H, W, 2)
    grid = grid.unsqueeze(0) # (1, H, W, 2)

    diff = grid.float() - idx_cuda.float()
    dist_sq_calculated = torch.sum(diff * diff, dim=-1)
    dist_sq_output = dist_cuda * dist_cuda
    
    torch.testing.assert_close(dist_sq_calculated, dist_sq_output, atol=1e-3, rtol=1e-3)
    print(">> 索引验证通过。")