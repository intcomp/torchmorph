import torch
import pytest
from scipy.ndimage import distance_transform_edt as dte
import torchmorph as tm
import numpy as np


def batch_distance_transform_edt(batch_numpy):

    is_single_sample = batch_numpy.ndim <= 2
    # (H, W) -> (1, H, W)
    if is_single_sample:
        batch_numpy = batch_numpy[np.newaxis, ...]
        
    results = [dte(sample) for sample in batch_numpy]
    output = np.stack(results, axis=0)  
    # (1, H, W) -> (H, W)
    if is_single_sample:
        output = output.squeeze(0)
        
    return output

# 用例 1: 批处理的 2D 图像
case_batch_2d = np.array([
    # 第 1 张图
    [[0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 0]],
    # 第 2 张图
    [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
], dtype=np.float32)


# 用例 2: 批处理的 3D 图像
case_3d_sample1 = np.ones((4, 5, 6), dtype=np.float32); case_3d_sample1[1, 1, 1] = 0.0; case_3d_sample1[2, 3, 4] = 0.0
case_3d_sample2 = np.ones((4, 5, 6), dtype=np.float32); case_3d_sample2[0, 0, 0] = 0.0
case_batch_3d = np.stack([case_3d_sample1, case_3d_sample2], axis=0)

# 用例 3: 单张 2D 图像 (隐式批处理)
case_single_2d = np.array([
    [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0],
], dtype=np.float32)


# 用例 4: 单张 2D 图像 (显式批处理)
case_explicit_batch_one = case_single_2d[np.newaxis, ...]

# 用例 5: 含幺元维度的批处理
case_dim_one = np.ones((2, 5, 1), dtype=np.float32) # 两张 5x1 的图片
case_dim_one[0, 2, 0] = 0.0
case_dim_one[1, 4, 0] = 0.0

# 用例 6: 1D 张量的批处理
case_batch_1d = np.array([
    [1, 1, 0, 1, 0, 1],
    [0, 1, 1, 1, 1, 0]
], dtype=np.float32)

@pytest.mark.parametrize(
    "input_numpy",
    [
        pytest.param(case_batch_2d, id="批处理2D图像"),
        pytest.param(case_batch_3d, id="批处理3D图像"),
        pytest.param(case_single_2d, id="单张2D图像(隐式批处理)"),
        pytest.param(case_explicit_batch_one, id="单张2D图像(显式批处理)"),
        pytest.param(case_dim_one, id="含幺元维度的批处理"),
        pytest.param(case_batch_1d, id="批处理1D数据"),
    ]
)
def test_batch_processing(input_numpy, request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x_numpy_contiguous = np.ascontiguousarray(input_numpy)
    x = torch.from_numpy(x_numpy_contiguous).cuda()

    print(f"\n\n--- 正在运行测试: {request.node.callspec.id} ---")
    print(f"输入张量形状: {x.shape}")
    y_cuda = tm.distance_transform(x.clone())
    
    y_ref_numpy = batch_distance_transform_edt(x_numpy_contiguous)
    y_ref = torch.from_numpy(y_ref_numpy).to(torch.float32).cuda()
    
    assert y_cuda.shape == y_ref.shape, f"形状不匹配! CUDA输出: {y_cuda.shape}, SciPy应为: {y_ref.shape}"
    print("CUDA 和 SciPy 输出形状匹配。")
    
    torch.testing.assert_close(y_cuda, y_ref, atol=1e-3, rtol=1e-3)
    print("--- 断言通过 (数值接近) ---")