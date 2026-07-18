import argparse
from functools import partial

import scipy.ndimage as ndi
import torch
import torch.utils.benchmark as benchmark
from prettytable import PrettyTable

import torchmorph as tm

SIZES_2D = [64, 128, 256, 512, 1024]
SIZES_3D = [32, 64, 128, 256]
SIZES_BFDT = [32, 64, 128, 256]
BATCHES_2D = [1, 4, 8, 16]
BATCHES_3D = [1, 2, 4, 8]
MIN_RUN_TIME = 1.0


def run_scipy(operation, inputs):
    return [operation(input) for input in inputs]


def run_cuda(operation, input):
    result = operation(input)
    torch.cuda.synchronize()
    return result


def run_cuda_singles(operation, inputs):
    result = [operation(input) for input in inputs]
    torch.cuda.synchronize()
    return result


def measure_ms(statement, globals, batch_size, min_run_time):
    measurement = benchmark.Timer(
        stmt=statement,
        globals=globals,
        num_threads=torch.get_num_threads(),
    ).blocked_autorange(min_run_time=min_run_time)
    return measurement.median * 1e3 / batch_size


def make_inputs(batch_size, image_size, spatial_ndim):
    shape = (batch_size, 1, *([image_size] * spatial_ndim))
    input = (torch.randn(shape, device="cuda") > 0).float()
    scipy_inputs = [input[index, 0].cpu().numpy() for index in range(batch_size)]
    torch_inputs = [input[index : index + 1] for index in range(batch_size)]
    return input, scipy_inputs, torch_inputs


def benchmark_transform(
    title,
    scipy_operation,
    torch_operations,
    spatial_ndim,
    sizes,
    batch_sizes,
    min_run_time,
):
    print(f"\n=== {title} ===")
    for batch_size in batch_sizes:
        table = PrettyTable()
        fields = ["Size", "SciPy (ms/item)"]
        for name in torch_operations:
            fields.extend(
                [
                    f"{name} 1x (ms/item)",
                    f"{name} batch (ms/item)",
                    f"{name} speedup",
                ]
            )
        table.field_names = fields
        for field in fields:
            table.align[field] = "r"

        for image_size in sizes:
            if spatial_ndim == 3 and image_size >= 256 and batch_size >= 4:
                table.add_row([f"{image_size}^3", *(["OOM"] * (len(fields) - 1))])
                continue

            input, scipy_inputs, torch_inputs = make_inputs(batch_size, image_size, spatial_ndim)
            scipy_ms = measure_ms(
                "run_scipy(operation, inputs)",
                {
                    "run_scipy": run_scipy,
                    "operation": scipy_operation,
                    "inputs": scipy_inputs,
                },
                batch_size,
                min_run_time,
            )

            row = [f"{image_size}^{spatial_ndim}", f"{scipy_ms:.3f}"]
            for operation in torch_operations.values():
                single_ms = measure_ms(
                    "run_cuda_singles(operation, inputs)",
                    {
                        "run_cuda_singles": run_cuda_singles,
                        "operation": operation,
                        "inputs": torch_inputs,
                    },
                    batch_size,
                    min_run_time,
                )
                batch_ms = measure_ms(
                    "run_cuda(operation, input)",
                    {
                        "run_cuda": run_cuda,
                        "operation": operation,
                        "input": input,
                    },
                    batch_size,
                    min_run_time,
                )
                row.extend(
                    [
                        f"{single_ms:.3f}",
                        f"{batch_ms:.3f}",
                        f"{scipy_ms / batch_ms:.1f}x",
                    ]
                )
            table.add_row(row)

        print(f"\nBatch size: {batch_size}")
        print(table)


def bench_edt(args):
    operations = {"torchmorph": tm.euclidean_distance_transform}
    benchmark_transform(
        "EDT 2D",
        ndi.distance_transform_edt,
        operations,
        2,
        args.sizes_2d,
        args.batches_2d,
        args.min_run_time,
    )
    benchmark_transform(
        "EDT 3D",
        ndi.distance_transform_edt,
        operations,
        3,
        args.sizes_3d,
        args.batches_3d,
        args.min_run_time,
    )


def bench_cdt(args):
    for metric in ("chessboard", "taxicab"):
        benchmark_transform(
            f"CDT 2D ({metric})",
            partial(ndi.distance_transform_cdt, metric=metric),
            {
                metric: partial(
                    tm.chamfer_distance_transform,
                    metric=metric,
                )
            },
            2,
            args.sizes_2d,
            args.batches_2d,
            args.min_run_time,
        )


def bench_bfdt(args):
    for metric in ("euclidean", "taxicab", "chessboard"):
        benchmark_transform(
            f"BFDT 2D ({metric})",
            partial(ndi.distance_transform_bf, metric=metric),
            {
                metric: partial(
                    tm.brute_force_distance_transform,
                    metric=metric,
                )
            },
            2,
            args.sizes_bfdt,
            [1],
            min(args.min_run_time, 0.5),
        )


BENCHMARKS = {
    "edt": bench_edt,
    "cdt": bench_cdt,
    "bfdt": bench_bfdt,
}


def main():
    parser = argparse.ArgumentParser(description="Distance transform benchmarks.")
    parser.add_argument(
        "operation",
        nargs="?",
        choices=(*BENCHMARKS, "all"),
        default="all",
    )
    parser.add_argument("--sizes-2d", type=int, nargs="+", default=SIZES_2D)
    parser.add_argument("--sizes-3d", type=int, nargs="+", default=SIZES_3D)
    parser.add_argument("--sizes-bfdt", type=int, nargs="+", default=SIZES_BFDT)
    parser.add_argument("--batches-2d", type=int, nargs="+", default=BATCHES_2D)
    parser.add_argument("--batches-3d", type=int, nargs="+", default=BATCHES_3D)
    parser.add_argument("--min-run-time", type=float, default=MIN_RUN_TIME)
    args = parser.parse_args()

    operations = BENCHMARKS if args.operation == "all" else (args.operation,)
    for operation in operations:
        BENCHMARKS[operation](args)


if __name__ == "__main__":
    main()
