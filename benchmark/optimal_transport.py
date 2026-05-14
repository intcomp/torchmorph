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


def run_sinkhorn_balanced_benchmark(
    B: int = 1,
    C: int = 1,
    H: int = 32,
    W: int = 32,
    myreg: float = 0.02,
    itrstep: int = 0,
    threshold: float = 1e-5,
):
    # Set parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # Prepare input data
    source = torch.rand(B, C, H, W, device=device, dtype=torch.float32)
    target = torch.rand(B, C, H, W, device=device, dtype=torch.float32)
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
        "itrstep": itrstep,
        "threshold": threshold,
    }

    stmt = (
        "tr.sinkhorn_balanced_batch(source, target, cost_matrix=cost_matrix, "
        "reg=myreg, itrstep=itrstep, threshold=threshold)"
    )

    timer = benchmark.Timer(stmt=stmt, globals=globals_dict)
    # result_sinkhorn_balanced = timer.blocked_autorange(min_run_time=1)
    # number_of_runs = 100
    result_sinkhorn_balanced = timer.blocked_autorange(min_run_time=2)
    return result_sinkhorn_balanced


def run_ot_sinkhorn_benchmark(H: int = 32, W: int = 32, myreg: float = 0.02):
    # Set parameters

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
    # number_of_runs = 100
    result_ot = timer.blocked_autorange(min_run_time=2)
    return result_ot


@torch.no_grad()
def run_sinkhorn_relative_error(H: int = 32, W: int = 32, myreg: float = 0.02):
    # Set parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    torch.manual_seed(42)

    # Prepare the same input data for both implementations.
    source = torch.rand(1, 1, H, W, device=device, dtype=dtype)
    target = torch.rand(1, 1, H, W, device=device, dtype=dtype)
    source = source / source.sum()
    target = target / target.sum()
    cost_matrix = tr.build_cost_matrix((H, W), device=device, p=2)

    source_preprocessed, target_preprocessed, cost_matrix = tr.data_preprocess(
        source, target, cost_matrix=cost_matrix, p=2, device=device
    )

    sinkhorn_plan = (
        tr.sinkhorn_balanced_batch(
            source_preprocessed,
            target_preprocessed,
            cost_matrix=cost_matrix,
            reg=myreg,
            itrstep=100,
            threshold=1e-5,
        )["plan"]
        .squeeze(0)
        .squeeze(0)
    )

    ot_plan = ot.sinkhorn(
        source_preprocessed.squeeze(0).squeeze(0),
        target_preprocessed.squeeze(0).squeeze(0),
        cost_matrix,
        reg=1 / myreg,
        numItermax=100,
        stopThr=1e-5,
    )

    return torch.linalg.norm(sinkhorn_plan - ot_plan) / torch.clamp(
        torch.linalg.norm(ot_plan), min=1e-12
    )


def large_scale_sinkhorn_balanced_benchmark(B: int = 1, C: int = 1, H: int = 128, W: int = 128):
    # Set parameters
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
    print_size_benchmark = True
    print_lambda_benchmark = False

    if print_size_benchmark:
        # sizes = [16, 32, 64, 100]
        sizes = [100]
        result_sinkhorn_balanced = []
        result_ot = []
        result_lambda = []
        result_relative_error = []

        for size in sizes:
            sinkhorn_result = run_sinkhorn_balanced_benchmark(B=1, C=1, H=size, W=size)
            ot_result = run_ot_sinkhorn_benchmark(H=size, W=size)
            lambda_result = run_sinkhorn_balanced_benchmark(B=1, C=1, H=size, W=size, myreg=0.01)
            relative_error = run_sinkhorn_relative_error(H=size, W=size)
            result_sinkhorn_balanced.append(sinkhorn_result)
            result_ot.append(ot_result)
            result_lambda.append(lambda_result)
            result_relative_error.append(relative_error)

        print(
            "size,my_sinkhorn_ms,ot_sinkhorn_ms,lambda_100_sinkhorn_ms,"
            "ot_speedup,lambda_100_vs_lambda_50,plan_relative_error"
        )
        for size, sinkhorn_result, ot_result, lambda_result, relative_error in zip(
            sizes, result_sinkhorn_balanced, result_ot, result_lambda, result_relative_error
        ):
            speedup = ot_result.mean / sinkhorn_result.mean
            lambda_vs_default = lambda_result.mean / sinkhorn_result.mean
            print(
                f"{size}x{size},"
                f"{sinkhorn_result.mean * 1000:.4f},"
                f"{ot_result.mean * 1000:.4f},"
                f"{lambda_result.mean * 1000:.4f},"
                f"{speedup:.2f}x,"
                f"{lambda_vs_default:.2f}x,"
                f"{relative_error.item():.4e}"
            )

    if print_lambda_benchmark:
        H, W = 64, 64
        lambdas = [1, 1.5, 2, 5, 10, 25, 50, 100]
        result_sinkhorn_balanced = []

        for lambda_value in lambdas:
            myreg = 1 / lambda_value
            sinkhorn_result = run_sinkhorn_balanced_benchmark(
                B=1, C=1, H=H, W=W, myreg=myreg, threshold=1e-5
            )
            result_sinkhorn_balanced.append(sinkhorn_result)

        lambda_50_index = lambdas.index(50)
        lambda_50_mean = result_sinkhorn_balanced[lambda_50_index].mean

        print("lambda,myreg,size,my_sinkhorn_ms,vs_lambda_50")
        for lambda_value, sinkhorn_result in zip(lambdas, result_sinkhorn_balanced):
            vs_lambda_50 = sinkhorn_result.mean / lambda_50_mean
            print(
                f"{lambda_value},"
                f"{1 / lambda_value:.4f},"
                f"{H}x{W},"
                f"{sinkhorn_result.mean * 1000:.4f},"
                f"{vs_lambda_50:.2f}x"
            )
