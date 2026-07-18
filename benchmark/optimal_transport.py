import ot
import torch
from torch.utils import benchmark

from torchmorph import SinkhornSolver, build_cost_matrix


def _grid_problem(n, H, W, epsilon, max_iter, log_space=False, seed=42):
    """Random (n, H*W) marginals with a 2-D grid cost matrix."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    source = torch.rand(n, H * W, device=device)
    target = torch.rand(n, H * W, device=device)
    cost_matrix = build_cost_matrix((H, W), device=device)
    solver = SinkhornSolver(epsilon=epsilon, max_iter=max_iter, log_space=log_space)
    return solver, source, target, cost_matrix


def run_forward_benchmark(n=1, H=32, W=32, epsilon=1.0, max_iter=100, log_space=False):
    solver, source, target, cost_matrix = _grid_problem(n, H, W, epsilon, max_iter, log_space)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    timer = benchmark.Timer(
        stmt="solver(source, target, cost_matrix)",
        globals={
            "solver": solver,
            "source": source,
            "target": target,
            "cost_matrix": cost_matrix,
        },
    )
    result = timer.blocked_autorange(min_run_time=3)
    if torch.cuda.is_available():
        print(f"peak allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        print(f"peak reserved:  {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")
    return result


def run_pot_sinkhorn_benchmark(H=32, W=32, epsilon=1.0, max_iter=100):
    solver, source, target, cost_matrix = _grid_problem(1, H, W, epsilon, max_iter)
    source, target, cost_matrix = solver.data_preprocess(source, target, cost_matrix)
    timer = benchmark.Timer(
        stmt="ot.sinkhorn(source, target, M, reg=epsilon, numItermax=max_iter, stopThr=1e-5)",
        globals={
            "ot": ot,
            "source": source[0],
            "target": target[0],
            "M": cost_matrix,
            "epsilon": epsilon,
            "max_iter": max_iter,
        },
    )
    return timer.blocked_autorange(min_run_time=2)


@torch.no_grad()
def run_sinkhorn_relative_error(H=32, W=32, epsilon=1.0, max_iter=100):
    solver, source, target, cost_matrix = _grid_problem(1, H, W, epsilon, max_iter)
    plan = solver.plan(source, target, cost_matrix)
    a, b, cost_matrix = solver.data_preprocess(source, target, cost_matrix)
    ot_plan = ot.sinkhorn(a[0], b[0], cost_matrix, reg=epsilon, numItermax=max_iter, stopThr=1e-5)
    return torch.linalg.norm(plan[0] - ot_plan) / torch.clamp(torch.linalg.norm(ot_plan), min=1e-12)


def reference_error_check(size=32, epsilon=10.0, max_iter=200, verbose=False):
    """Compare log-space fused-kernel potentials against POT's sinkhorn_log."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available.")
    solver, source, target, cost_matrix = _grid_problem(
        1, size, size, epsilon, max_iter, log_space=True, seed=2
    )

    grad_f, grad_g = solver.potentials(source, target, cost_matrix)
    grad_f, grad_g = grad_f[0], grad_g[0]

    a, b, cost_matrix = solver.data_preprocess(source, target, cost_matrix)
    _, log = ot.bregman.sinkhorn_log(
        a[0],
        b[0],
        cost_matrix,
        reg=epsilon,
        numItermax=max_iter,
        stopThr=0,
        log=True,
    )
    f = epsilon * log["log_u"]
    g = epsilon * log["log_v"]
    f = f - f.mean()
    g = g - g.mean()

    if verbose:
        print(f, g, grad_f, grad_g, sep="\n")

    f_rel_l2 = torch.linalg.norm(f - grad_f) / torch.clamp(torch.linalg.norm(f), min=1e-12)
    g_rel_l2 = torch.linalg.norm(g - grad_g) / torch.clamp(torch.linalg.norm(g), min=1e-12)
    return {
        "size": size,
        "points": size * size,
        "epsilon": epsilon,
        "max_iter": max_iter,
        "f_grad_max_abs": (f - grad_f).abs().max().item(),
        "f_grad_rel_l2": f_rel_l2.item(),
        "g_grad_max_abs": (g - grad_g).abs().max().item(),
        "g_grad_rel_l2": g_rel_l2.item(),
        "f_allclose": torch.allclose(f, grad_f, rtol=1e-4, atol=1e-4),
        "g_allclose": torch.allclose(g, grad_g, rtol=1e-4, atol=1e-4),
    }


def print_reference_error_table(
    sizes=(2, 4, 8, 16, 32, 64), max_iters=(100, 200, 500), epsilon=10.0
):
    """Print a size and iteration sweep for POT-vs-CUDA gradient error."""
    headers = ("size", "N", "eps", "itr", "f_max", "f_rel_l2", "g_max", "g_rel_l2", "close")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for size in sizes:
        for max_iter in max_iters:
            row = reference_error_check(size=size, epsilon=epsilon, max_iter=max_iter)
            print(
                f"| {row['size']}x{row['size']} "
                f"| {row['points']} "
                f"| {row['epsilon']:.3g} "
                f"| {row['max_iter']} "
                f"| {row['f_grad_max_abs']:.3e} "
                f"| {row['f_grad_rel_l2']:.3e} "
                f"| {row['g_grad_max_abs']:.3e} "
                f"| {row['g_grad_rel_l2']:.3e} "
                f"| {row['f_allclose']}/{row['g_allclose']} |"
            )


if __name__ == "__main__":
    print_reference_error_table()
