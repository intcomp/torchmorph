import argparse

import scipy.ndimage as ndi
import torch
import torch.utils.benchmark as benchmark
from prettytable import PrettyTable

import torchmorph as tm

IMAGE_SIZES = [64, 128, 256, 1024]
BATCH_SIZES = [1, 2, 4, 16]
MIN_RUN_TIME = 1.0

BINARY_OPERATORS = {
    "erosion": (ndi.binary_erosion, tm.binary_erosion),
    "dilation": (ndi.binary_dilation, tm.binary_dilation),
    "fill_holes": (ndi.binary_fill_holes, tm.binary_fill_holes),
    "hit_or_miss": (ndi.binary_hit_or_miss, tm.binary_hit_or_miss),
    "opening": (ndi.binary_opening, tm.binary_opening),
    "closing": (ndi.binary_closing, tm.binary_closing),
    "propagation": (ndi.binary_propagation, tm.binary_propagation),
}


def run_cuda(torch_op, x):
    result = torch_op(x)
    torch.cuda.synchronize()
    return result


def bench_binary_operator(operation, image_sizes, batch_sizes, min_run_time):
    scipy_op, torch_op = BINARY_OPERATORS[operation]

    print("\n============================================")
    print(f" Benchmark: binary {operation} ")
    print("============================================")

    for batch_size in batch_sizes:
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

        for image_size in image_sizes:
            x = (torch.randn(batch_size, 1, image_size, image_size, device="cuda") > 0).to(
                torch.float32
            )
            x_np_list = [x[i, 0].detach().cpu().numpy() for i in range(batch_size)]
            x_one = [x[i : i + 1] for i in range(batch_size)]

            scipy_time = benchmark.Timer(
                stmt="[scipy_op(data) for data in inputs]",
                globals={
                    "scipy_op": scipy_op,
                    "inputs": x_np_list,
                },
            ).blocked_autorange(min_run_time=min_run_time)

            for data in x_one:
                torch_op(data)
            torch.cuda.synchronize()

            torch_single_time = benchmark.Timer(
                stmt="[run_cuda(torch_op, data) for data in inputs]",
                globals={
                    "run_cuda": run_cuda,
                    "torch_op": torch_op,
                    "inputs": x_one,
                },
            ).blocked_autorange(min_run_time=min_run_time)

            torch_batch_time = benchmark.Timer(
                stmt="run_cuda(torch_op, x)",
                globals={
                    "run_cuda": run_cuda,
                    "torch_op": torch_op,
                    "x": x,
                },
            ).blocked_autorange(min_run_time=min_run_time)

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
    choices = (*BINARY_OPERATORS, "all")
    parser = argparse.ArgumentParser(description="Benchmark binary morphology operators.")
    parser.add_argument(
        "operation",
        choices=choices,
        nargs="?",
        default="all",
        help="Operator to benchmark (default: all).",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=IMAGE_SIZES,
        help="Image sizes to benchmark (default: 64 128 256 1024).",
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=BATCH_SIZES,
        help="Batch sizes to benchmark (default: 1 2 4 16).",
    )
    parser.add_argument(
        "--min-run-time",
        type=float,
        default=MIN_RUN_TIME,
        help="Minimum benchmark time per timer in seconds (default: 1.0).",
    )
    args = parser.parse_args()

    operations = BINARY_OPERATORS if args.operation == "all" else (args.operation,)
    for operation in operations:
        bench_binary_operator(
            operation,
            image_sizes=args.sizes,
            batch_sizes=args.batches,
            min_run_time=args.min_run_time,
        )


if __name__ == "__main__":
    main()
