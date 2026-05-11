import scipy.ndimage as ndi
import torch.utils.benchmark as benchmark
from prettytable import PrettyTable

import torchmorph as tm

DATA_CASES = [(1, 1), (2, 1), (4, 1), (8, 1), (10, 1), (10, 4), (10, 10), (12, 1)]
MIN_RUN = 1.0


def bench_generate_binary_structure():
    print("\n========================================")
    print("\n Benchmark: generate binary structure ")
    print("\n========================================")

    table = PrettyTable()
    table.field_names = ["Rank", "Connectivity", "SciPy(ms)", "Torch(ms)", "Speedup"]
    for column in table.field_names:
        table.align[column] = "r"

    for rank, connectivity in DATA_CASES:
        t_scipy = benchmark.Timer(
            stmt="generate_binary_structure(rank, connectivity)",
            globals={
                "generate_binary_structure": ndi.generate_binary_structure,
                "rank": rank,
                "connectivity": connectivity,
            },
        ).blocked_autorange(min_run_time=MIN_RUN)

        t_torch = benchmark.Timer(
            stmt="generate_binary_structure(rank, connectivity)",
            globals={
                "generate_binary_structure": tm.generate_binary_structure,
                "rank": rank,
                "connectivity": connectivity,
            },
        ).blocked_autorange(min_run_time=MIN_RUN)

        scipy_ms = t_scipy.median * 1e3
        torch_ms = t_torch.median * 1e3
        speedup = scipy_ms / torch_ms

        table.add_row(
            [
                rank,
                connectivity,
                f"{scipy_ms:.3f}",
                f"{torch_ms:.3f}",
                f"{speedup:.1f}x",
            ]
        )

    print("Load from:", tm.__file__)
    print(table)


if __name__ == "__main__":
    bench_generate_binary_structure()
