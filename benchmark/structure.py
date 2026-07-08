import argparse

import scipy.ndimage as ndi
import torch
import torch.utils.benchmark as benchmark
from prettytable import PrettyTable

import torchmorph as tm

GENERATE_CASES = [(1, 1), (2, 1), (4, 1), (8, 1), (10, 1), (10, 4), (10, 10), (12, 1)]
ITERATE_CASES = [
    (1, 1, 3),
    (2, 1, 2),
    (2, 2, 3),
    (3, 1, 2),
    (3, 2, 3),
    (4, 1, 2),
    (7, 2, 2),
]
MIN_RUN_TIME = 1.0


def bench_generate_binary_structure(min_run_time):
    print("\n========================================")
    print(" Benchmark: generate binary structure ")
    print("========================================")

    table = PrettyTable()
    table.field_names = ["Rank", "Connectivity", "SciPy(ms)", "Torch(ms)", "Speedup"]
    for column in table.field_names:
        table.align[column] = "r"

    for rank, connectivity in GENERATE_CASES:
        t_scipy = benchmark.Timer(
            stmt="generate_binary_structure(rank, connectivity)",
            globals={
                "generate_binary_structure": ndi.generate_binary_structure,
                "rank": rank,
                "connectivity": connectivity,
            },
        ).blocked_autorange(min_run_time=min_run_time)

        t_torch = benchmark.Timer(
            stmt="generate_binary_structure(rank, connectivity)",
            globals={
                "generate_binary_structure": tm.generate_binary_structure,
                "rank": rank,
                "connectivity": connectivity,
            },
        ).blocked_autorange(min_run_time=min_run_time)

        scipy_ms = t_scipy.median * 1e3
        torch_ms = t_torch.median * 1e3

        table.add_row(
            [
                rank,
                connectivity,
                f"{scipy_ms:.3f}",
                f"{torch_ms:.3f}",
                f"{scipy_ms / torch_ms:.1f}x",
            ]
        )

    print(table)


def bench_iterate_structure(min_run_time):
    print("\n========================================")
    print(" Benchmark: iterate structure ")
    print("========================================")

    table = PrettyTable()
    table.field_names = [
        "Rank",
        "Connectivity",
        "Iterations",
        "Output",
        "True",
        "SciPy(ms)",
        "Torch(ms)",
        "Speedup",
    ]
    for column in table.field_names:
        table.align[column] = "r"

    for rank, connectivity, iterations in ITERATE_CASES:
        scipy_structure = ndi.generate_binary_structure(rank, connectivity)
        torch_structure = torch.as_tensor(scipy_structure)
        expected = ndi.iterate_structure(scipy_structure, iterations)

        t_scipy = benchmark.Timer(
            stmt="iterate_structure(structure, iterations)",
            globals={
                "iterate_structure": ndi.iterate_structure,
                "structure": scipy_structure,
                "iterations": iterations,
            },
        ).blocked_autorange(min_run_time=min_run_time)

        t_torch = benchmark.Timer(
            stmt="iterate_structure(structure, iterations)",
            globals={
                "iterate_structure": tm.iterate_structure,
                "structure": torch_structure,
                "iterations": iterations,
            },
        ).blocked_autorange(min_run_time=min_run_time)

        scipy_ms = t_scipy.median * 1e3
        torch_ms = t_torch.median * 1e3

        table.add_row(
            [
                rank,
                connectivity,
                iterations,
                "x".join(str(size) for size in expected.shape),
                int(expected.sum()),
                f"{scipy_ms:.3f}",
                f"{torch_ms:.3f}",
                f"{scipy_ms / torch_ms:.1f}x",
            ]
        )

    print(table)


def main():
    parser = argparse.ArgumentParser(description="Benchmark binary structure helpers.")
    parser.add_argument(
        "operation",
        choices=("generate", "iterate", "all"),
        nargs="?",
        default="all",
        help="Structure helper to benchmark (default: all).",
    )
    parser.add_argument(
        "--min-run-time",
        type=float,
        default=MIN_RUN_TIME,
        help="Minimum benchmark time per timer in seconds (default: 1.0).",
    )
    args = parser.parse_args()

    print("Load from:", tm.__file__)
    if args.operation in ("generate", "all"):
        bench_generate_binary_structure(args.min_run_time)
    if args.operation in ("iterate", "all"):
        bench_iterate_structure(args.min_run_time)


if __name__ == "__main__":
    main()
