import scipy.ndimage as ndi
import torch
import torch.utils.benchmark as benchmark
from prettytable import PrettyTable

import torchmorph as tm

image_size = [64, 128, 256, 1024]
batch_size = [1, 2, 4, 16]
MIN_RUN_TIME = 1.0


def bench_binary_erosion():
    print("\n============================================")
    print(" Benchmark: binary erosion ")
    print("============================================")

    for batch in batch_size:
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

        for size in image_size:
            x = (torch.randn(batch, 1, size, size, device="cuda") > 0).to(torch.float32)
            # scipy data
            x_np_list = [x[i, 0].detach().cpu().numpy() for i in range(batch)]
            # torch 1x data
            x_one = [x[i : i + 1] for i in range(batch)]

            t_scipy = benchmark.Timer(
                stmt="[scipy_erosion(data) for data in x_np_list]",
                globals={
                    "scipy_erosion": ndi.binary_erosion,
                    "x_np_list": x_np_list,
                },
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)

            # warmup
            for data in x_one:
                tm.binary_erosion(data)
            torch.cuda.synchronize()

            t_torch_1x = benchmark.Timer(
                stmt="[tm_erosion(data) for data in x_one]",
                globals={
                    "tm_erosion": tm.binary_erosion,
                    "x_one": x_one,
                },
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)

            t_torch_batch = benchmark.Timer(
                stmt="tm_erosion(x)",
                globals={
                    "tm_erosion": tm.binary_erosion,
                    "x": x,
                },
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)

            scipy_ms = t_scipy.median * 1e3 / batch
            torch_1x_ms = t_torch_1x.median * 1e3 / batch
            torch_batch_ms = t_torch_batch.median * 1e3 / batch

            speedup1x = scipy_ms / torch_1x_ms
            speedupbatch = scipy_ms / torch_batch_ms

            table.add_row(
                [
                    size,
                    f"{scipy_ms:.3f}",
                    f"{torch_1x_ms:.3f}",
                    f"{torch_batch_ms:.3f}",
                    f"{speedup1x:.1f}x",
                    f"{speedupbatch:.1f}x",
                ]
            )
        print(f"\n=========== Batch size : {batch} ===========")
        print(table)


def bench_binary_dilation():
    print("\n============================================")
    print(" Benchmark: binary dilation ")
    print("============================================")

    for batch in batch_size:
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

        for size in image_size:
            x = (torch.randn(batch, 1, size, size, device="cuda") > 0).to(torch.float32)
            # scipy data
            x_np_list = [x[i, 0].detach().cpu().numpy() for i in range(batch)]
            # torch 1x data
            x_one = [x[i : i + 1] for i in range(batch)]

            t_scipy = benchmark.Timer(
                stmt="[scipy_dilation(data) for data in x_np_list]",
                globals={
                    "scipy_dilation": ndi.binary_dilation,
                    "x_np_list": x_np_list,
                },
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)

            # warmup
            for data in x_one:
                tm.binary_dilation(data)
            torch.cuda.synchronize()

            t_torch_1x = benchmark.Timer(
                stmt="[tm_dilation(data) for data in x_one]",
                globals={
                    "tm_dilation": tm.binary_dilation,
                    "x_one": x_one,
                },
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)

            t_torch_batch = benchmark.Timer(
                stmt="tm_dilation(x)",
                globals={
                    "tm_dilation": tm.binary_dilation,
                    "x": x,
                },
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)

            scipy_ms = t_scipy.median * 1e3 / batch
            torch_1x_ms = t_torch_1x.median * 1e3 / batch
            torch_batch_ms = t_torch_batch.median * 1e3 / batch

            speedup1x = scipy_ms / torch_1x_ms
            speedupbatch = scipy_ms / torch_batch_ms

            table.add_row(
                [
                    size,
                    f"{scipy_ms:.3f}",
                    f"{torch_1x_ms:.3f}",
                    f"{torch_batch_ms:.3f}",
                    f"{speedup1x:.1f}x",
                    f"{speedupbatch:.1f}x",
                ]
            )
        print(f"\n=========== Batch size : {batch} ===========")
        print(table)


def bench_binary_opening():
    print("\n============================================")
    print(" Benchmark: binary opening ")
    print("============================================")

    for batch in batch_size:
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

        for size in image_size:
            x = (torch.randn(batch, 1, size, size, device="cuda") > 0).to(torch.float32)
            # scipy data
            x_np_list = [x[i, 0].detach().cpu().numpy() for i in range(batch)]
            # torch 1x data
            x_one = [x[i : i + 1] for i in range(batch)]

            t_scipy = benchmark.Timer(
                stmt="[scipy_opening(data) for data in x_np_list]",
                globals={
                    "scipy_opening": ndi.binary_opening,
                    "x_np_list": x_np_list,
                },
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)

            # warmup
            for data in x_one:
                tm.binary_opening(data)

            torch.cuda.synchronize()
            t_torch_1x = benchmark.Timer(
                stmt="[tm_opening(data) for data in x_one]",
                globals={
                    "tm_opening": tm.binary_opening,
                    "x_one": x_one,
                },
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)

            t_torch_batch = benchmark.Timer(
                stmt="tm_opening(x)",
                globals={
                    "tm_opening": tm.binary_opening,
                    "x": x,
                },
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)

            scipy_ms = t_scipy.median * 1e3 / batch
            torch_1x_ms = t_torch_1x.median * 1e3 / batch
            torch_batch_ms = t_torch_batch.median * 1e3 / batch

            speedup1x = scipy_ms / torch_1x_ms
            speedupbatch = scipy_ms / torch_batch_ms

            table.add_row(
                [
                    size,
                    f"{scipy_ms:.3f}",
                    f"{torch_1x_ms:.3f}",
                    f"{torch_batch_ms:.3f}",
                    f"{speedup1x:.1f}x",
                    f"{speedupbatch:.1f}x",
                ]
            )
        print(f"\n=========== Batch size : {batch} ===========")
        print(table)


def bench_binary_closing():
    print("\n============================================")
    print(" Benchmark: binary closing ")
    print("============================================")

    for batch in batch_size:
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

        for size in image_size:
            x = (torch.randn(batch, 1, size, size, device="cuda") > 0).to(torch.float32)
            # scipy data
            x_np_list = [x[i, 0].detach().cpu().numpy() for i in range(batch)]
            # torch 1x data
            x_one = [x[i : i + 1] for i in range(batch)]

            t_scipy = benchmark.Timer(
                stmt="[scipy_closing(data) for data in x_np_list]",
                globals={
                    "scipy_closing": ndi.binary_closing,
                    "x_np_list": x_np_list,
                },
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)

            # warmup
            for data in x_one:
                tm.binary_closing(data)

            torch.cuda.synchronize()
            t_torch_1x = benchmark.Timer(
                stmt="[tm_closing(data) for data in x_one]",
                globals={
                    "tm_closing": tm.binary_closing,
                    "x_one": x_one,
                },
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)

            t_torch_batch = benchmark.Timer(
                stmt="tm_closing(x)",
                globals={
                    "tm_closing": tm.binary_closing,
                    "x": x,
                },
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)

            scipy_ms = t_scipy.median * 1e3 / batch
            torch_1x_ms = t_torch_1x.median * 1e3 / batch
            torch_batch_ms = t_torch_batch.median * 1e3 / batch

            speedup1x = scipy_ms / torch_1x_ms
            speedupbatch = scipy_ms / torch_batch_ms

            table.add_row(
                [
                    size,
                    f"{scipy_ms:.3f}",
                    f"{torch_1x_ms:.3f}",
                    f"{torch_batch_ms:.3f}",
                    f"{speedup1x:.1f}x",
                    f"{speedupbatch:.1f}x",
                ]
            )
        print(f"\n=========== Batch size : {batch} ===========")
        print(table)


if __name__ == "__main__":
    bench_binary_erosion()
    bench_binary_dilation()
    bench_binary_opening()
    bench_binary_closing()
