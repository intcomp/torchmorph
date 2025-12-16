import torch
import torch.utils.benchmark as benchmark
import scipy.ndimage as ndi
from prettytable import PrettyTable
import torchmorph as tm

sizes = [64, 128, 256, 512]
batches = [1, 4, 8, 16]
dtype = torch.float32
device = "cuda"
MIN_RUN = 1.0  # seconds per measurement

torch.set_num_threads(torch.get_num_threads())


def bench_single_op(op_name):
    """
    op_name: "dilation" or "erosion"
    """

    scipy_op = ndi.binary_dilation if op_name == "dilation" else ndi.binary_erosion
    torch_op = tm.binary_dilation if op_name == "dilation" else tm.binary_erosion

    print("\n==============================")
    print(f"   Benchmark: Binary {op_name}")
    print("==============================")

    for B in batches:
        table = PrettyTable()
        table.field_names = [
            "Size",
            "SciPy (ms/img)",
            "Torch 1× (ms/img)",
            "Torch batch (ms/img)",
            "Speedup 1×",
            "Speedup batch",
        ]
        for c in table.field_names:
            table.align[c] = "r"

        for s in sizes:
            # Generate binary input
            x = (torch.randn(B, 1, s, s, device=device) > 0).to(dtype)
            x_np_list = [x[i, 0].detach().cpu().numpy() for i in range(B)]
            x_imgs = [x[i:i+1] for i in range(B)]  # (1, 1, H, W)
            # SciPy (CPU, one-by-one)
            stmt_scipy = "out = [scipy_op(arr) for arr in x_np_list]"
            t_scipy = benchmark.Timer(
                stmt=stmt_scipy,
                globals={"x_np_list": x_np_list, "scipy_op": scipy_op},
            ).blocked_autorange(min_run_time=MIN_RUN)
            scipy_ms = (t_scipy.median * 1e3) / B

            # Torch CUDA (one-by-one)
            stmt_torch1 = "out = [torch_op(img) for img in x_imgs]"
            t_torch1 = benchmark.Timer(
                stmt=stmt_torch1,
                globals={"x_imgs": x_imgs, "torch_op": torch_op},
            ).blocked_autorange(min_run_time=MIN_RUN)
            torch1_ms = (t_torch1.median * 1e3) / B

            # Torch CUDA (batched)
            t_batch = benchmark.Timer(
                stmt="torch_op(x)",
                globals={"x": x, "torch_op": torch_op},
            ).blocked_autorange(min_run_time=MIN_RUN)
            torchB_ms = (t_batch.median * 1e3) / B

            # Speedups
            speed1 = scipy_ms / torch1_ms
            speedB = scipy_ms / torchB_ms

            table.add_row([
                s,
                f"{scipy_ms:.3f}",
                f"{torch1_ms:.3f}",
                f"{torchB_ms:.3f}",
                f"{speed1:.1f}×",
                f"{speedB:.1f}×",
            ])

        print(f"\n=== Batch Size: {B} ===")
        print(table)


print("Loaded from:", tm.__file__)
bench_single_op("dilation")
bench_single_op("erosion")
