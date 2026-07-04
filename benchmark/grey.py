import argparse

import scipy.ndimage as ndi
import torch
import torch.utils.benchmark as benchmark
from prettytable import PrettyTable

import torchmorph as tm

IMAGE_SIZES = [64, 128, 256, 1024]
BATCH_SIZES = [1, 2, 4, 16]
STRUCTURE_SIZE = 3
MIN_RUN_TIME = 1.0

GREY_OPERATORS = {
    "erosion": (ndi.grey_erosion, tm.grey_erosion),
    "dilation": (ndi.grey_dilation, tm.grey_dilation),
    "opening": (ndi.grey_opening, tm.grey_opening),
    "closing": (ndi.grey_closing, tm.grey_closing),
}


def run_cuda(torch_op, x):
    result = torch_op(x, size=STRUCTURE_SIZE)
    torch.cuda.synchronize()
    return result


def bench_grey_operator(operation):
    scipy_op, torch_op = GREY_OPERATORS[operation]

    print("\n============================================")
    print(f" Benchmark: grey {operation} ")
    print("============================================")

    for batch_size in BATCH_SIZES:
        table = PrettyTable()
        table.field_names = [
            "Size",
            "SciPy(ms)",
            "Torch 1x(ms)",
            "Torch batch(ms)",
            "Speedup 1x",
            "Speedup batch",
        ]
        for column in table.field_names:
            table.align[column] = "r"

        for image_size in IMAGE_SIZES:
            x = torch.randn(
                batch_size,
                1,
                image_size,
                image_size,
                device="cuda",
                dtype=torch.float32,
            )
            x_np_list = [x[i, 0].detach().cpu().numpy() for i in range(batch_size)]
            x_one = [x[i : i + 1] for i in range(batch_size)]

            scipy_time = benchmark.Timer(
                stmt="[scipy_op(data, size=structure_size) for data in inputs]",
                globals={
                    "scipy_op": scipy_op,
                    "structure_size": STRUCTURE_SIZE,
                    "inputs": x_np_list,
                },
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)

            for data in x_one:
                torch_op(data, size=STRUCTURE_SIZE)
            torch.cuda.synchronize()

            torch_single_time = benchmark.Timer(
                stmt="[run_cuda(torch_op, data) for data in inputs]",
                globals={
                    "run_cuda": run_cuda,
                    "torch_op": torch_op,
                    "inputs": x_one,
                },
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)

            torch_batch_time = benchmark.Timer(
                stmt="run_cuda(torch_op, x)",
                globals={
                    "run_cuda": run_cuda,
                    "torch_op": torch_op,
                    "x": x,
                },
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)

            scipy_ms = scipy_time.median * 1e3 / batch_size
            torch_single_ms = torch_single_time.median * 1e3 / batch_size
            torch_batch_ms = torch_batch_time.median * 1e3 / batch_size

            table.add_row(
                [
                    image_size,
                    f"{scipy_ms:.3f}",
                    f"{torch_single_ms:.3f}",
                    f"{torch_batch_ms:.3f}",
                    f"{scipy_ms / torch_single_ms:.1f}x",
                    f"{scipy_ms / torch_batch_ms:.1f}x",
                ]
            )

        print(f"\n=========== Batch size : {batch_size} ===========")
        print(table)


def main():
    choices = (*GREY_OPERATORS, "all")
    parser = argparse.ArgumentParser(description="Benchmark grey morphology operators.")
    parser.add_argument(
        "operation",
        choices=choices,
        nargs="?",
        default="all",
        help="Operator to benchmark (default: all).",
    )
    args = parser.parse_args()

    operations = GREY_OPERATORS if args.operation == "all" else (args.operation,)
    for operation in operations:
        bench_grey_operator(operation)


if __name__ == "__main__":
    main()
