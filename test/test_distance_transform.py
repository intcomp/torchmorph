import torch
import pytest
from scipy.ndimage import distance_transform_edt as dte
import torchmorph as tm
import numpy as np

# --- 我们在这里定义所有的测试用例 ---

# 用例 1: 我们之前成功的那个标准例子
case_standard = np.array([
    [0, 1, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 0, 0]
], dtype=np.float32)

# 用例 2: 全是背景 (0)，输出应该全是 0
case_all_background = np.zeros((5, 5), dtype=np.float32)

# 用例 3: 全是前景 (1)，输出应该也全是 0 (因为前景点到背景的距离未定义，SciPy默认输出0)
case_all_foreground = np.ones((5, 5), dtype=np.float32)

# 用例 4: 只有一个背景点 (0) 在中间
case_single_background = np.ones((5, 5), dtype=np.float32)
case_single_background[2, 2] = 0

# 用例 5: 只有一个前景点 (1) 在中间
case_single_foreground = np.zeros((5, 5), dtype=np.float32)
case_single_foreground[2, 2] = 1

# 用例 6: 非正方形的矩阵 (高 > 宽)
case_tall_matrix = np.array([
    [1, 0, 1],
    [1, 1, 1],
    [1, 1, 1],
    [0, 1, 0],
    [1, 1, 1],
], dtype=np.float32)

# 用例 7: 非正方形的矩阵 (宽 > 高)
case_wide_matrix = np.array([
    [1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1],
], dtype=np.float32)

# 用例 8: 棋盘格，考验对角线距离的计算
case_checkerboard = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
], dtype=np.float32)

# --- 使用 pytest.mark.parametrize 来自动运行所有测试用例 ---

@pytest.mark.parametrize(
    "input_numpy",
    [
        pytest.param(case_standard, id="Standard Case"),
        pytest.param(case_all_background, id="All Background"),
        pytest.param(case_all_foreground, id="All Foreground"),
        pytest.param(case_single_background, id="Single Background Pixel"),
        pytest.param(case_single_foreground, id="Single Foreground Pixel"),
        pytest.param(case_tall_matrix, id="Tall Matrix (H>W)"),
        pytest.param(case_wide_matrix, id="Wide Matrix (W>H)"),
        pytest.param(case_checkerboard, id="Checkerboard"),
    ]
)
def test_distance_transform_comprehensive(input_numpy, request):
    """
    一个统一的测试函数，用来验证所有不同的输入情况。
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # 准备输入数据
    x = torch.from_numpy(input_numpy).cuda()

    # 1. 运行你的 CUDA 实现
    y_cuda = tm.distance_transform(x)

    # 2. 运行 SciPy 官方实现
    y_ref_numpy = dte(input_numpy)
    y_ref = torch.from_numpy(y_ref_numpy).to(torch.float32).cuda()

    # 打印结果用于直观对比
    print(f"\n\n--- Running Test: {request.node.callspec.id} ---")
    print("Input Array:\n", input_numpy)
    print("\nYour CUDA Implementation Output:\n", y_cuda.cpu().numpy())
    print("\nSciPy Reference Output:\n", y_ref.cpu().numpy())
    if request.node.callspec.id == "All Foreground":
        # 对于这个特殊情况，我们不与 SciPy 比较。
        # 我们验证我们自己的逻辑：输出值是否都非常大 (代表无穷远)。
        print("\nSciPy has different behavior for this edge case. Verifying CUDA output is ~inf.")
        # 断言所有元素都大于一个很大的阈值
        assert torch.all(y_cuda > 1e4)
    else:
        # 对于所有其他正常情况，我们与 SciPy 的黄金标准进行比较。
        y_ref_numpy = dte(input_numpy)
        y_ref = torch.from_numpy(y_ref_numpy).to(torch.float32).cuda()
        print("\nSciPy Reference Output:\n", y_ref.cpu().numpy())
        torch.testing.assert_close(y_cuda, y_ref, atol=1e-3, rtol=1e-3)
    print("--- Test End ---")
