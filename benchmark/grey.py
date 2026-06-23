import scipy.ndimage as ndi
import torch
import torch.utils.benchmark as benchmark
from prettytable import PrettyTable

import torchmorph as tm

image_size = [64, 128, 256, 1024]
batch_size = [1, 2, 4, 16]
STRUCTURE_SIZE = 3
MIN_RUN_TIME = 1.0


def _sync_grey_erosion(x):
    out = tm.grey_erosion(x, size=STRUCTURE_SIZE)
    torch.cuda.synchronize()
    return out


def bench_grey_erosion():
    print("\n============================================")
    print(" Benchmark: grey erosion ")
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
            x = torch.randn(batch, 1, size, size, device="cuda", dtype=torch.float32)
            # scipy data
            x_np_list = [x[i, 0].detach().cpu().numpy() for i in range(batch)]
            # torch 1x data
            x_one = [x[i : i + 1] for i in range(batch)]

            t_scipy = benchmark.Timer(
                stmt="[scipy_erosion(data, size=structure_size) for data in x_np_list]",
                globals={
                    "scipy_erosion": ndi.grey_erosion,
                    "structure_size": STRUCTURE_SIZE,
                    "x_np_list": x_np_list,
                },
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)

            # warmup
            for data in x_one:
                tm.grey_erosion(data, size=STRUCTURE_SIZE)
            torch.cuda.synchronize()

            t_torch_1x = benchmark.Timer(
                stmt="[tm_erosion(data) for data in x_one]",
                globals={
                    "tm_erosion": _sync_grey_erosion,
                    "x_one": x_one,
                },
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)

            t_torch_batch = benchmark.Timer(
                stmt="tm_erosion(x)",
                globals={
                    "tm_erosion": _sync_grey_erosion,
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
    bench_grey_erosion()
