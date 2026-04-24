import importlib.util
from pathlib import Path

import ot
import torch
from torch.utils import benchmark

# Project root directory (parent of test)
project_root = Path(__file__).parent.parent

# Target module path and Load module
module_path = project_root / "torchmorph" / "optimal_transport.py"
spec = importlib.util.spec_from_file_location("optimal_transport", module_path)
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

torch.manual_seed(42)


def run_sinkhorn_balanced_benchmark():
    # Set parameters
    B, C, H, W = 1, 1, 32, 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Prepare input data
    source = torch.rand(B, C, H, W, device=device, dtype=dtype)
    target = torch.rand(B, C, H, W, device=device, dtype=dtype)
    source = source / source.sum()
    target = target / target.sum()

    globals_dict = {
        "tr": tr,
        "source": source,
        "target": target,
    }

    stmt = "tr.sinkhorn_balanced(source, target, reg=10.0, itrstep=100)"

    timer = benchmark.Timer(stmt=stmt, globals=globals_dict)
    # result_sinkhorn_balanced = timer.blocked_autorange(min_run_time=1)
    number_of_runs = 50
    result_sinkhorn_balanced = timer.timeit(number_of_runs)
    return result_sinkhorn_balanced


def run_ot_sinkhorn_benchmark():
    # Set parameters
    H, W = 32, 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Prepare input data
    source = torch.rand(H * W, device=device, dtype=dtype)
    target = torch.rand(H * W, device=device, dtype=dtype)
    source = source / source.sum()
    target = target / target.sum()
    M = tr.build_cost_matrix((H, W), device=device, p=2)

    globals_dict = {
        "ot": ot,
        "source": source,
        "target": target,
        "M": M,
    }

    stmt = "ot.sinkhorn(source, target, M, reg=10.0, numItermax=100)"

    timer = benchmark.Timer(stmt=stmt, globals=globals_dict)
    # timer.blocked_autorange(min_run_time=1)
    number_of_runs = 50
    result_ot = timer.timeit(number_of_runs)
    return result_ot


def large_scale_sinkhorn_balanced_benchmark():
    # Set parameters
    B, C, H, W = 1, 1, 128, 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Prepare input data
    source = torch.rand(B, C, H, W, device=device, dtype=dtype)
    target = torch.rand(B, C, H, W, device=device, dtype=dtype)
    source = source / source.sum()
    target = target / target.sum()

    globals_dict = {
        "tr": tr,
        "source": source,
        "target": target,
    }

    stmt = "tr.sinkhorn_balanced(source, target, reg=10.0, itrstep=100)"

    timer = benchmark.Timer(stmt=stmt, globals=globals_dict)
    # result_sinkhorn_balanced = timer.blocked_autorange(min_run_time=1)
    number_of_runs = 10
    result_sinkhorn_balanced = timer.timeit(number_of_runs)
    return result_sinkhorn_balanced


def batch_channel_sinkhorn_balanced_benchmark():
    # Set parameters
    B, C, H, W = 32, 32, 32, 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Prepare input data
    source = torch.rand(B, C, H, W, device=device, dtype=dtype)
    target = torch.rand(B, C, H, W, device=device, dtype=dtype)
    source = source / source.sum()
    target = target / target.sum()

    globals_dict = {
        "tr": tr,
        "source": source,
        "target": target,
    }

    stmt = "tr.sinkhorn_balanced(source, target, reg=10.0, itrstep=100)"

    timer = benchmark.Timer(stmt=stmt, globals=globals_dict)
    # result_sinkhorn_balanced = timer.blocked_autorange(min_run_time=1)
    number_of_runs = 10
    result_sinkhorn_balanced = timer.timeit(number_of_runs)
    return result_sinkhorn_balanced


if __name__ == "__main__":
    result_sinkhorn_balanced = run_sinkhorn_balanced_benchmark()
    result_ot = run_ot_sinkhorn_benchmark()
    result_large_scale = large_scale_sinkhorn_balanced_benchmark()
    result_batch_channel = batch_channel_sinkhorn_balanced_benchmark()

print(result_sinkhorn_balanced)
print(result_ot)
print(result_large_scale)
print(result_batch_channel)

speedup = result_ot.mean / result_sinkhorn_balanced.mean
print(f"My sinkhorn to OT sinkhorn Speedup: {speedup:.1f}x")
