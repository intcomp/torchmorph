import scipy.ndimage as ndi  # noqa: F401
import torch
import torch.utils.benchmark as benchmark
from prettytable import PrettyTable

import torchmorph as tm  # noqa: F401

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
        "Exact 1× (ms/img)",
        "Exact batch (ms/img)",
        "JFA 1× (ms/img)",
        "JFA batch (ms/img)",
        "Speedup Exact",
        "Speedup JFA",
    ]
    for c in table.field_names:
        table.align[c] = "r"

    for s in sizes:
        # Inputs: (B, C, H, W) format - C=1 for single channel
        x = (torch.randn(B, 1, s, s, device=device) > 0).to(dtype)
        # For scipy, we need (H, W) arrays
        x_np_list = [x[i, 0].detach().cpu().numpy() for i in range(B)]
        # For torch single image processing: each is (1, 1, H, W)
        x_imgs = [x[i : i + 1] for i in range(B)]

        # SciPy (CPU, one-by-one)
        stmt_scipy = "out = [ndi.distance_transform_edt(arr) for arr in x_np_list]"
        t_scipy = benchmark.Timer(
            stmt=stmt_scipy,
            setup="from __main__ import x_np_list, ndi",
            num_threads=torch.get_num_threads(),
        ).blocked_autorange(min_run_time=MIN_RUN)
        scipy_per_img_ms = (t_scipy.median * 1e3) / B

        # Torch Exact (CUDA, one-by-one)
        stmt_exact1 = """
for xi in x_imgs:
    tm.distance_transform_edt(xi, algorithm="exact")
"""
        t_exact1 = benchmark.Timer(
            stmt=stmt_exact1,
            setup="from __main__ import x_imgs, tm",
            num_threads=torch.get_num_threads(),
        ).blocked_autorange(min_run_time=MIN_RUN)
        exact1_per_img_ms = (t_exact1.median * 1e3) / B

        # Torch Exact (CUDA, batched)
        t_exact_batch = benchmark.Timer(
            stmt='tm.distance_transform_edt(x, algorithm="exact")',
            setup="from __main__ import x, tm",
            num_threads=torch.get_num_threads(),
        ).blocked_autorange(min_run_time=MIN_RUN)
        exactB_per_img_ms = (t_exact_batch.median * 1e3) / B

        # Torch JFA (CUDA, one-by-one)
        stmt_jfa1 = """
for xi in x_imgs:
    tm.distance_transform_edt(xi, algorithm="jfa")
"""
        t_jfa1 = benchmark.Timer(
            stmt=stmt_jfa1,
            setup="from __main__ import x_imgs, tm",
            num_threads=torch.get_num_threads(),
        ).blocked_autorange(min_run_time=MIN_RUN)
        jfa1_per_img_ms = (t_jfa1.median * 1e3) / B

        # Torch JFA (CUDA, batched)
        t_jfa_batch = benchmark.Timer(
            stmt='tm.distance_transform_edt(x, algorithm="jfa")',
            setup="from __main__ import x, tm",
            num_threads=torch.get_num_threads(),
        ).blocked_autorange(min_run_time=MIN_RUN)
        jfaB_per_img_ms = (t_jfa_batch.median * 1e3) / B

        # Speedups (batch mode vs scipy)
        speed_exact = scipy_per_img_ms / exactB_per_img_ms
        speed_jfa = scipy_per_img_ms / jfaB_per_img_ms

        table.add_row(
            [
                s,
                f"{scipy_per_img_ms:.3f}",
                f"{exact1_per_img_ms:.3f}",
                f"{exactB_per_img_ms:.3f}",
                f"{jfa1_per_img_ms:.3f}",
                f"{jfaB_per_img_ms:.3f}",
                f"{speed_exact:.1f}×",
                f"{speed_jfa:.1f}×",
            ]
        )

    print(f"\n=== Batch Size: {B} ===")
    print(table)
