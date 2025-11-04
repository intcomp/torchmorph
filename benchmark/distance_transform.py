import torch
import torch.utils.benchmark as benchmark
import scipy.ndimage as ndi
import numpy as np
from prettytable import PrettyTable
import torchmorph as tm

sizes = [64, 128, 256, 512, 1024]
batches = [1, 4, 8, 16]
dtype = torch.float32
device = "cuda"
MIN_RUN = 1.0  # seconds per measurement

torch.set_num_threads(torch.get_num_threads())

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
        # Inputs
        x = (torch.randn(B, 1, s, s, device=device) > 0).to(dtype)
        x_np_list = [x[i, 0].detach().cpu().numpy() for i in range(B)]
        x_imgs = [x[i:i+1] for i in range(B)]

        # SciPy (CPU, one-by-one)
        stmt_scipy = "out = [ndi.distance_transform_edt(arr) for arr in x_np_list]"
        t_scipy = benchmark.Timer(
            stmt=stmt_scipy,
            setup="from __main__ import x_np_list, ndi",
            num_threads=torch.get_num_threads(),
        ).blocked_autorange(min_run_time=MIN_RUN)
        scipy_per_img_ms = (t_scipy.median * 1e3) / B

        # Torch (CUDA, one-by-one)
        stmt_torch1 = """
for xi in x_imgs:
    tm.distance_transform(xi)
"""
        t_torch1 = benchmark.Timer(
            stmt=stmt_torch1,
            setup="from __main__ import x_imgs, tm",
            num_threads=torch.get_num_threads(),
        ).blocked_autorange(min_run_time=MIN_RUN)
        torch1_per_img_ms = (t_torch1.median * 1e3) / B

        # Torch (CUDA, batched)
        t_batch = benchmark.Timer(
            stmt="tm.distance_transform(x)",
            setup="from __main__ import x, tm",
            num_threads=torch.get_num_threads(),
        ).blocked_autorange(min_run_time=MIN_RUN)
        torchB_per_img_ms = (t_batch.median * 1e3) / B

        # Speedups
        speed1 = scipy_per_img_ms / torch1_per_img_ms
        speedB = scipy_per_img_ms / torchB_per_img_ms

        table.add_row([
            s,
            f"{scipy_per_img_ms:.3f}",
            f"{torch1_per_img_ms:.3f}",
            f"{torchB_per_img_ms:.3f}",
            f"{speed1:.1f}×",
            f"{speedB:.1f}×",
        ])

    print(f"\n=== Batch Size: {B} ===")
    print(table)

