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


def run_sinkhorn_balanced_benchmark():
    # Set parameters
    B, C, H, W = 1, 1, 32, 32
    myreg = 0.5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    torch.manual_seed(42)

    # Prepare input data
    source = torch.rand(B, C, H, W, device=device, dtype=dtype)
    target = torch.rand(B, C, H, W, device=device, dtype=dtype)
    source = source / source.sum()
    target = target / target.sum()
    cost_matrix = tr.build_cost_matrix((H, W), device=device, p=2)

    source, target, cost_matrix = tr.data_preprocess(
        source, target, cost_matrix=cost_matrix, p=2, device=device
    )

    globals_dict = {
        "tr": tr,
        "source": source,
        "target": target,
        "myreg": myreg,
        "cost_matrix": cost_matrix,
    }

    stmt = (
        "tr.sinkhorn_balanced(source, target, cost_matrix=cost_matrix, "
        "reg=myreg, itrstep=100, threshold=1e-5)"
    )

    timer = benchmark.Timer(stmt=stmt, globals=globals_dict)
    # result_sinkhorn_balanced = timer.blocked_autorange(min_run_time=1)
    number_of_runs = 100
    result_sinkhorn_balanced = timer.timeit(number_of_runs)
    return result_sinkhorn_balanced


def run_ot_sinkhorn_benchmark():
    # Set parameters
    H, W = 32, 32
    myreg = 0.5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    torch.manual_seed(42)

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
        "myreg": myreg,
    }

    stmt = "ot.sinkhorn(source, target, M, reg=1/myreg, numItermax=100, stopThr=1e-5)"

    timer = benchmark.Timer(stmt=stmt, globals=globals_dict)
    # timer.blocked_autorange(min_run_time=1)
    number_of_runs = 100
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

    stmt = "tr.sinkhorn_balanced_full(source, target, reg=25.0, itrstep=100, threshold=1e-5)"

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

    stmt = "tr.sinkhorn_balanced_full(source, target, reg=25.0, itrstep=100, threshold=1e-5)"

    timer = benchmark.Timer(stmt=stmt, globals=globals_dict)
    # result_sinkhorn_balanced = timer.blocked_autorange(min_run_time=1)
    number_of_runs = 10
    result_sinkhorn_balanced = timer.timeit(number_of_runs)
    return result_sinkhorn_balanced


if __name__ == "__main__":
    result_sinkhorn_balanced = run_sinkhorn_balanced_benchmark()
    result_ot = run_ot_sinkhorn_benchmark()
    # result_large_scale = large_scale_sinkhorn_balanced_benchmark()
    # result_batch_channel = batch_channel_sinkhorn_balanced_benchmark()

    print(result_sinkhorn_balanced)
    print(result_ot)
    speedup = result_ot.mean / result_sinkhorn_balanced.mean
    print(f"My sinkhorn to OT sinkhorn Speedup: {speedup:.1f}x")
    # print(result_large_scale)
    # print(result_batch_channel)
