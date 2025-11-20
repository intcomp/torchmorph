import torch
import pytest
from scipy.ndimage import distance_transform_edt as scipy_edt
import numpy as np
import torchmorph as tm

# 辅助函数：调用 SciPy 并处理格式
def batch_scipy_edt_with_indices(batch_numpy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    
    input_is_1d_batch = (batch_numpy.ndim == 2) 
    input_is_single_sample_no_batch = (batch_numpy.ndim == 1)

    if input_is_single_sample_no_batch:
        batch_numpy = batch_numpy[np.newaxis, ...] # (L) -> (1, L)

    
    dist_results, indices_results = [], []
    for sample in batch_numpy:
        dist, indices = scipy_edt(sample, return_indices=True, return_distances=True)
        dist_results.append(dist)
        indices_results.append(indices)
        
    output_dist = np.stack(dist_results, axis=0)
    output_indices = np.stack(indices_results, axis=0)
    
    # indices shape fix: (N, ndim_sample, ...) -> (N, ..., ndim_sample)
    # 对于 1D: (N, 1, L) -> (N, L, 1)
    output_indices = np.moveaxis(output_indices, 1, -1)
    
    if input_is_single_sample_no_batch:
        output_dist = output_dist.squeeze(0)
        output_indices = output_indices.squeeze(0)
        
    return output_dist, output_indices

# 用例定义 
case_batch_2d = np.array([[[0., 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 0]],[[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]], dtype=np.float32)
_case_3d_s1 = np.ones((4, 5, 6), dtype=np.float32); _case_3d_s1[1, 1, 1] = 0.0; _case_3d_s1[2, 3, 4] = 0.0
_case_3d_s2 = np.ones((4, 5, 6), dtype=np.float32); _case_3d_s2[0, 0, 0] = 0.0
case_batch_3d = np.stack([_case_3d_s1, _case_3d_s2], axis=0)
case_single_2d = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.float32)
case_explicit_batch_one = case_single_2d[np.newaxis, ...]
case_dim_one = np.ones((2, 5, 1), dtype=np.float32); case_dim_one[0, 2, 0] = 0.0; case_dim_one[1, 4, 0] = 0.0
case_batch_1d = np.array([[1, 1, 0, 1, 0, 1], [0, 1, 1, 1, 1, 0]], dtype=np.float32)

@pytest.mark.parametrize(
    "input_numpy",
    [
        pytest.param(case_batch_2d, id="批处理2D图像"),
        pytest.param(case_batch_3d, id="批处理3D图像"),
        pytest.param(case_single_2d, id="单张2D图像(隐式批处理)"),
        pytest.param(case_explicit_batch_one, id="单张2D图像(显式批处理)"),
        pytest.param(case_dim_one, id="含幺元维度的批处理"),
        pytest.param(case_batch_1d, id="批处理1D数据"),
    ],
)
def test_distance_transform_and_indices(input_numpy: np.ndarray, request: pytest.FixtureRequest):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    x_numpy_contiguous = np.ascontiguousarray(input_numpy)
    x_cuda = torch.from_numpy(x_numpy_contiguous).cuda()

    print(f"\n\n--- 正在运行测试: {request.node.callspec.id} ---")
    print(f"输入张量形状: {x_cuda.shape}")

    # 调用您的 Python 包装函数
    dist_cuda, idx_cuda = tm.distance_transform(x_cuda.clone())

    print(f"CUDA 距离输出形状: {dist_cuda.shape}")
    print(f"CUDA 坐标输出形状: {idx_cuda.shape}")

    # 调用 SciPy 作为参考基准
    dist_ref_numpy, idx_ref_numpy = batch_scipy_edt_with_indices(x_numpy_contiguous)
    dist_ref = torch.from_numpy(dist_ref_numpy).to(torch.float32).cuda()
    
    print(f"SciPy 距离输出形状: {dist_ref.shape}")

    # 断言验证
    print("\n--- 正在验证距离... ---")
    assert dist_cuda.shape == dist_ref.shape
    torch.testing.assert_close(dist_cuda, dist_ref, atol=1e-3, rtol=1e-3)
    print("距离断言通过 (形状和数值接近)。")

    print("\n--- 正在验证坐标... ---")
    
    # 鲁棒的坐标验证逻辑
    had_no_batch_dim = (x_numpy_contiguous.ndim <= idx_cuda.shape[-1])
    spatial_shape = x_cuda.shape if had_no_batch_dim else x_cuda.shape[1:]
    coords = [torch.arange(s, device='cuda') for s in spatial_shape]
    grid = torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1)
    
    if not had_no_batch_dim:
        grid = grid.unsqueeze(0)
        
    diff = grid.float() - idx_cuda.float()
    dist_sq_from_indices = torch.sum(diff * diff, dim=-1)
    
    torch.testing.assert_close(dist_sq_from_indices, dist_cuda * dist_cuda, atol=1e-3, rtol=1e-3)
    print("坐标正确性断言通过 (计算出的距离与返回距离匹配)。")
        
    print("--- 测试通过 ---")