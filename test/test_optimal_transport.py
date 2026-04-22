# test_optimal_transport.py
import sys
import importlib.util
from pathlib import Path

# 项目根目录（test 的父目录）
project_root = Path(__file__).parent.parent

# 目标模块路径
module_path = project_root / "torchmorph" / "optimal_transport.py"

# 加载模块
spec = importlib.util.spec_from_file_location("optimal_transport", module_path)
ot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ot)

a=ot.build_cost_matrix_1d(2,3,'cuda')
print(a)
