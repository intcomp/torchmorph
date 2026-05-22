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
    itrstep: int = 100,
    threshold: float = 0,
):
    torch.cuda.reset_peak_memory_stats()
    # Set parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # Prepare input data
    source = torch.rand(B, C, H, W, device=device, dtype=torch.float32)
    target = torch.rand(B, C, H, W, device=device, dtype=torch.float32)
    source = source / source.sum()
    target = target / target.sum()
    cost_matrix = tr.build_cost_matrix((H, W), device=device, p=2)

    solver = tr.SinkhornSolver(
        reg=myreg,
        itrstep=itrstep,
        threshold=threshold,
        p=2,
        device=device,
    )
    source, target, cost_matrix, _ = solver.data_preprocess(source, target, cost_matrix=cost_matrix)

    globals_dict = {
        "solver": solver,
        "source": source,
        "target": target,
        "cost_matrix": cost_matrix,
    }

    stmt = "solver.sinkhorn_batch(source, target, cost_matrix=cost_matrix)"

    timer = benchmark.Timer(stmt=stmt, globals=globals_dict)
    # result_sinkhorn_balanced = timer.blocked_autorange(min_run_time=1)
    # number_of_runs = 100
    result_sinkhorn_balanced = timer.blocked_autorange(min_run_time=3)

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    print(f"peak allocated: {peak_allocated / 1024**2:.2f} MB")
    print(f"peak reserved:   {peak_reserved / 1024**2:.2f} MB")

    return result_sinkhorn_balanced


def run_ot_sinkhorn_benchmark(H: int = 32, W: int = 32, myreg: float = 0.02):
    # Set parameters
    import ot

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

    solver = tr.SinkhornSolver(
        reg=myreg,
        itrstep=100,
        threshold=1e-5,
        p=2,
        device=device,
    )
    source_preprocessed, target_preprocessed, cost_matrix, _ = solver.data_preprocess(
        source, target, cost_matrix=cost_matrix
    )

    sinkhorn_plan = (
        solver.sinkhorn_batch(
            source_preprocessed,
            target_preprocessed,
            cost_matrix=cost_matrix,
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
    solver = tr.SinkhornSolver(
        reg=25.0,
        itrstep=100,
        threshold=1e-5,
        device=device,
    )

    globals_dict = {
        "solver": solver,
        "source": source,
        "target": target,
    }

    stmt = "solver.sinkhorn(source, target)"

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
    solver = tr.SinkhornSolver(
        reg=25.0,
        itrstep=100,
        threshold=1e-5,
        device=device,
    )

    globals_dict = {
        "solver": solver,
        "source": source,
        "target": target,
    }

    stmt = "solver.sinkhorn(source, target)"

    timer = benchmark.Timer(stmt=stmt, globals=globals_dict)
    # result_sinkhorn_balanced = timer.blocked_autorange(min_run_time=1)
    number_of_runs = 10
    result_sinkhorn_balanced = timer.timeit(number_of_runs)
    return result_sinkhorn_balanced


def run_sinkhorn_log_benchmark(
    H: int = 32,
    W: int = 32,
    myreg: float = 0.02,
    itrstep: int = 100,
):
    torch.cuda.reset_peak_memory_stats()
    # Set parameters
    if not torch.cuda.is_available():
        raise ValueError("CUDA device not available.")

    device = "cuda"
    torch.manual_seed(42)

    # Prepare input data
    source = torch.rand(H, W, device=device, dtype=torch.float32)
    target = torch.rand(H, W, device=device, dtype=torch.float32)
    cost_matrix = tr.build_cost_matrix((H, W), device=device, p=2)

    solver = tr.SinkhornSolver(reg=myreg, itrstep=itrstep, p=2, device=device)
    source, target, cost_matrix, cost_matrix_T = solver.data_preprocess(
        source, target, cost_matrix=cost_matrix
    )

    globals_dict = {
        "solver": solver,
        "source": source,
        "target": target,
        "cost_matrix": cost_matrix,
        "cost_matrix_T": cost_matrix_T,
    }

    stmt = "solver.sinkhorn_log_cuda(source, target, cost_matrix, cost_matrix_T)"

    timer = benchmark.Timer(stmt=stmt, globals=globals_dict)
    result_sinkhorn_log = timer.blocked_autorange(min_run_time=3)

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    print(f"peak allocated: {peak_allocated / 1024**2:.2f} MB")
    print(f"peak reserved:   {peak_reserved / 1024**2:.2f} MB")

    return result_sinkhorn_log


def run_sinkhorn_cuda_benchmark(
    H: int = 32,
    W: int = 32,
    myreg: float = 0.02,
    itrstep: int = 100,
):
    torch.cuda.reset_peak_memory_stats()
    # Set parameters
    if not torch.cuda.is_available():
        raise ValueError("CUDA device not available.")

    device = "cuda"
    torch.manual_seed(42)

    # Prepare input data
    source = torch.rand(H, W, device=device, dtype=torch.float32)
    target = torch.rand(H, W, device=device, dtype=torch.float32)
    cost_matrix = tr.build_cost_matrix((H, W), device=device, p=2)

    solver = tr.SinkhornSolver(reg=myreg, itrstep=itrstep, p=2, device=device)
    source, target, cost_matrix, _ = solver.data_preprocess(source, target, cost_matrix=cost_matrix)

    globals_dict = {
        "solver": solver,
        "source": source,
        "target": target,
        "cost_matrix": cost_matrix,
    }

    stmt = "solver.sinkhorn_cuda(source, target, cost_matrix=cost_matrix)"

    timer = benchmark.Timer(stmt=stmt, globals=globals_dict)
    result_sinkhorn_cuda = timer.blocked_autorange(min_run_time=3)

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    print(f"peak allocated: {peak_allocated / 1024**2:.2f} MB")
    print(f"peak reserved:   {peak_reserved / 1024**2:.2f} MB")

    return result_sinkhorn_cuda


def reference_error_check(
    size: int = 32, myreg: float = 0.1, itrstep: int = 200, verbose: bool = False
):
    if not torch.cuda.is_available():
        raise ValueError("CUDA device not available.")

    device = "cuda"
    torch.manual_seed(2)

    # Prepare input data
    source = torch.rand(size, size, device=device, dtype=torch.float32)
    target = torch.rand(size, size, device=device, dtype=torch.float32)
    cost_matrix = tr.build_cost_matrix((size, size), device=device, p=2)

    solver = tr.SinkhornSolver(reg=myreg, itrstep=itrstep, p=2, device=device)
    source, target, cost_matrix, cost_matrix_T = solver.data_preprocess(
        source, target, cost_matrix=cost_matrix
    )

    _, log = ot.bregman.sinkhorn_log(
        source,
        target,
        cost_matrix,
        reg=1 / myreg,
        numItermax=itrstep,
        stopThr=0,
        log=True,
    )

    u, v = solver.sinkhorn_log_cuda(
        source,
        target,
        cost_matrix,
        cost_matrix_T,
    )

    grad_f, grad_g = solver.gradient(u, v)

    log_u = log["log_u"]
    log_v = log["log_v"]

    f = log_u / myreg
    g = log_v / myreg
    f = f - f.mean()
    g = g - g.mean()

    if verbose:
        print(f)
        print(g)
        print(grad_f)
        print(grad_g)

    f_abs_error = (f - grad_f).abs()
    g_abs_error = (g - grad_g).abs()
    f_rel_l2 = torch.linalg.norm(f - grad_f) / torch.clamp(torch.linalg.norm(f), min=1e-12)
    g_rel_l2 = torch.linalg.norm(g - grad_g) / torch.clamp(torch.linalg.norm(g), min=1e-12)

    return {
        "size": size,
        "points": size * size,
        "reg": myreg,
        "itrstep": itrstep,
        "f_grad_max_abs": f_abs_error.max().item(),
        "f_grad_mean_abs": f_abs_error.mean().item(),
        "f_grad_rel_l2": f_rel_l2.item(),
        "g_grad_max_abs": g_abs_error.max().item(),
        "g_grad_mean_abs": g_abs_error.mean().item(),
        "g_grad_rel_l2": g_rel_l2.item(),
        "f_allclose": torch.allclose(f, grad_f, rtol=1e-4, atol=1e-4),
        "g_allclose": torch.allclose(g, grad_g, rtol=1e-4, atol=1e-4),
    }


def print_reference_error_table(
    sizes=(2, 4, 8, 16, 32, 64),
    itrsteps=(100, 200, 500),
    myreg: float = 0.1,
):
    """Print a size and iteration sweep for POT-vs-CUDA gradient error."""
    rows = [
        reference_error_check(size=size, myreg=myreg, itrstep=itrstep)
        for size in sizes
        for itrstep in itrsteps
    ]

    headers = (
        "size",
        "N",
        "reg",
        "itr",
        "f_max",
        "f_mean",
        "f_rel_l2",
        "g_max",
        "g_mean",
        "g_rel_l2",
        "close",
    )
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        close = f"{row['f_allclose']}/{row['g_allclose']}"
        print(
            f"| {row['size']}x{row['size']} "
            f"| {row['points']} "
            f"| {row['reg']:.3g} "
            f"| {row['itrstep']} "
            f"| {row['f_grad_max_abs']:.3e} "
            f"| {row['f_grad_mean_abs']:.3e} "
            f"| {row['f_grad_rel_l2']:.3e} "
            f"| {row['g_grad_max_abs']:.3e} "
            f"| {row['g_grad_mean_abs']:.3e} "
            f"| {row['g_grad_rel_l2']:.3e} "
            f"| {close} |"
        )

    return rows


if __name__ == "__main__":
    print_reference_error_table()
