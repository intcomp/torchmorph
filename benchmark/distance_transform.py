import argparse

import scipy.ndimage as ndi  # noqa: F401
import torch
import torch.utils.benchmark as benchmark
from prettytable import PrettyTable

import torchmorph as tm  # noqa: F401

# Config
sizes_2d = [64, 128, 256, 512, 1024]
sizes_3d = [32, 64, 128, 256]
batches_2d = [1, 4, 8, 16]
batches_3d = [1, 2, 4, 8]
dtype = torch.float32
device = "cuda"
MIN_RUN = 1.0  # seconds per measurement

torch.set_num_threads(torch.get_num_threads())


# ======================================================================
# Section 1: Euclidean Distance Transform (EDT) — 2D
# ======================================================================


def bench_edt_2d():
    for B in batches_2d:
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

        for s in sizes_2d:
            # Inputs: (B, C, H, W) format - C=1 for single channel
            x = (torch.randn(B, 1, s, s, device=device) > 0).to(dtype)
            # For scipy, we need (H, W) arrays
            x_np_list = [x[i, 0].detach().cpu().numpy() for i in range(B)]
            # For torch single image processing: each is (1, 1, H, W)
            x_imgs = [x[i : i + 1] for i in range(B)]

            # Expose locals to __main__ for benchmark.Timer
            globals().update(x=x, x_np_list=x_np_list, x_imgs=x_imgs)

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
    tm.euclidean_distance_transform(xi, algorithm="exact")
"""
            t_exact1 = benchmark.Timer(
                stmt=stmt_exact1,
                setup="from __main__ import x_imgs, tm",
                num_threads=torch.get_num_threads(),
            ).blocked_autorange(min_run_time=MIN_RUN)
            exact1_per_img_ms = (t_exact1.median * 1e3) / B

            # Torch Exact (CUDA, batched)
            t_exact_batch = benchmark.Timer(
                stmt='tm.euclidean_distance_transform(x, algorithm="exact")',
                setup="from __main__ import x, tm",
                num_threads=torch.get_num_threads(),
            ).blocked_autorange(min_run_time=MIN_RUN)
            exactB_per_img_ms = (t_exact_batch.median * 1e3) / B

            # Torch JFA (CUDA, one-by-one)
            stmt_jfa1 = """
for xi in x_imgs:
    tm.euclidean_distance_transform(xi, algorithm="jfa")
"""
            t_jfa1 = benchmark.Timer(
                stmt=stmt_jfa1,
                setup="from __main__ import x_imgs, tm",
                num_threads=torch.get_num_threads(),
            ).blocked_autorange(min_run_time=MIN_RUN)
            jfa1_per_img_ms = (t_jfa1.median * 1e3) / B

            # Torch JFA (CUDA, batched)
            t_jfa_batch = benchmark.Timer(
                stmt='tm.euclidean_distance_transform(x, algorithm="jfa")',
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

        print(f"\n=== EDT 2D | Batch Size: {B} ===")
        print(table)


# ======================================================================
# Section 2: Euclidean Distance Transform (EDT) — 3D
# ======================================================================


def bench_edt_3d():
    for B in batches_3d:
        table = PrettyTable()
        table.field_names = [
            "Size (D×H×W)",
            "SciPy (ms/vol)",
            "Exact 1× (ms/vol)",
            "Exact batch (ms/vol)",
            "JFA 1× (ms/vol)",
            "JFA batch (ms/vol)",
            "Speedup Exact",
            "Speedup JFA",
        ]
        for c in table.field_names:
            table.align[c] = "r"

        for s in sizes_3d:
            # Skip large sizes with large batches to avoid OOM
            if s >= 256 and B >= 4:
                table.add_row([f"{s}³", "OOM", "OOM", "OOM", "OOM", "OOM", "-", "-"])
                continue

            # Inputs: (B, D, H, W) format for 3D - no channel dimension for JFA 3D
            x = (torch.randn(B, s, s, s, device=device) > 0).to(dtype)
            # For scipy, we need (D, H, W) arrays
            x_np_list = [x[i].detach().cpu().numpy() for i in range(B)]
            # For torch single volume processing: each is (1, D, H, W)
            x_vols = [x[i : i + 1] for i in range(B)]

            # Expose locals to __main__ for benchmark.Timer
            globals().update(x=x, x_np_list=x_np_list, x_vols=x_vols)

            # SciPy (CPU, one-by-one)
            stmt_scipy = "out = [ndi.distance_transform_edt(arr) for arr in x_np_list]"
            t_scipy = benchmark.Timer(
                stmt=stmt_scipy,
                setup="from __main__ import x_np_list, ndi",
                num_threads=torch.get_num_threads(),
            ).blocked_autorange(min_run_time=MIN_RUN)
            scipy_per_vol_ms = (t_scipy.median * 1e3) / B

            # Torch Exact (CUDA, one-by-one)
            stmt_exact1 = """
for xi in x_vols:
    tm.euclidean_distance_transform(xi, algorithm="exact")
"""
            t_exact1 = benchmark.Timer(
                stmt=stmt_exact1,
                setup="from __main__ import x_vols, tm",
                num_threads=torch.get_num_threads(),
            ).blocked_autorange(min_run_time=MIN_RUN)
            exact1_per_vol_ms = (t_exact1.median * 1e3) / B

            # Torch Exact (CUDA, batched)
            t_exact_batch = benchmark.Timer(
                stmt='tm.euclidean_distance_transform(x, algorithm="exact")',
                setup="from __main__ import x, tm",
                num_threads=torch.get_num_threads(),
            ).blocked_autorange(min_run_time=MIN_RUN)
            exactB_per_vol_ms = (t_exact_batch.median * 1e3) / B

            # Torch JFA (CUDA, one-by-one)
            stmt_jfa1 = """
for xi in x_vols:
    tm.euclidean_distance_transform(xi, algorithm="jfa")
"""
            t_jfa1 = benchmark.Timer(
                stmt=stmt_jfa1,
                setup="from __main__ import x_vols, tm",
                num_threads=torch.get_num_threads(),
            ).blocked_autorange(min_run_time=MIN_RUN)
            jfa1_per_vol_ms = (t_jfa1.median * 1e3) / B

            # Torch JFA (CUDA, batched)
            t_jfa_batch = benchmark.Timer(
                stmt='tm.euclidean_distance_transform(x, algorithm="jfa")',
                setup="from __main__ import x, tm",
                num_threads=torch.get_num_threads(),
            ).blocked_autorange(min_run_time=MIN_RUN)
            jfaB_per_vol_ms = (t_jfa_batch.median * 1e3) / B

            # Speedups (batch mode vs scipy)
            speed_exact = scipy_per_vol_ms / exactB_per_vol_ms
            speed_jfa = scipy_per_vol_ms / jfaB_per_vol_ms

            table.add_row(
                [
                    f"{s}³",
                    f"{scipy_per_vol_ms:.3f}",
                    f"{exact1_per_vol_ms:.3f}",
                    f"{exactB_per_vol_ms:.3f}",
                    f"{jfa1_per_vol_ms:.3f}",
                    f"{jfaB_per_vol_ms:.3f}",
                    f"{speed_exact:.1f}×",
                    f"{speed_jfa:.1f}×",
                ]
            )

        print(f"\n=== EDT 3D | Batch Size: {B} ===")
        print(table)


# ======================================================================
# Section 3: Chamfer Distance Transform (CDT) — 2D
# ======================================================================


def bench_cdt_2d():
    for metric in ["chessboard", "taxicab"]:
        print(f"\n{'=' * 60}")
        print(f"  CDT Benchmark - Metric: {metric}")
        print(f"{'=' * 60}")

        for B in batches_2d:
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

            for s in sizes_2d:
                # Inputs: (B, C, H, W) format - C=1 for single channel
                x = (torch.randn(B, 1, s, s, device=device) > 0).to(dtype)
                # For scipy, we need (H, W) arrays
                x_np_list = [x[i, 0].detach().cpu().numpy() for i in range(B)]
                # For torch single image processing: each is (1, 1, H, W)
                x_imgs = [x[i : i + 1] for i in range(B)]

                # Expose locals to __main__ for benchmark.Timer
                globals().update(x=x, x_np_list=x_np_list, x_imgs=x_imgs)

                # SciPy (CPU, one-by-one)
                stmt_scipy = (
                    f"out = [ndi.distance_transform_cdt(arr,metric='{metric}')for arr in x_np_list]"
                )
                t_scipy = benchmark.Timer(
                    stmt=stmt_scipy,
                    setup="from __main__ import x_np_list, ndi",
                    num_threads=torch.get_num_threads(),
                ).blocked_autorange(min_run_time=MIN_RUN)
                scipy_per_img_ms = (t_scipy.median * 1e3) / B

                # Torch (CUDA, one-by-one)
                stmt_torch1 = f"""
for xi in x_imgs:
    tm.chamfer_distance_transform(xi, metric='{metric}')
"""
                t_torch1 = benchmark.Timer(
                    stmt=stmt_torch1,
                    setup="from __main__ import x_imgs, tm",
                    num_threads=torch.get_num_threads(),
                ).blocked_autorange(min_run_time=MIN_RUN)
                torch1_per_img_ms = (t_torch1.median * 1e3) / B

                # Torch (CUDA, batched)
                t_batch = benchmark.Timer(
                    stmt=f"tm.chamfer_distance_transform(x, metric='{metric}')",
                    setup="from __main__ import x, tm",
                    num_threads=torch.get_num_threads(),
                ).blocked_autorange(min_run_time=MIN_RUN)
                torchB_per_img_ms = (t_batch.median * 1e3) / B

                # Speedups
                speed1 = scipy_per_img_ms / torch1_per_img_ms
                speedB = scipy_per_img_ms / torchB_per_img_ms

                table.add_row(
                    [
                        s,
                        f"{scipy_per_img_ms:.3f}",
                        f"{torch1_per_img_ms:.3f}",
                        f"{torchB_per_img_ms:.3f}",
                        f"{speed1:.1f}×",
                        f"{speedB:.1f}×",
                    ]
                )

            print(f"\n=== CDT 2D | Metric: {metric}, Batch Size: {B} ===")
            print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distance Transform Benchmarks")
    parser.add_argument(
        "--section",
        nargs="*",
        choices=["edt-2d", "edt-3d", "cdt"],
        default=None,
        help="Sections to run (default: all)",
    )
    args = parser.parse_args()

    sections = args.section if args.section else ["edt-2d", "edt-3d", "cdt"]
    if "edt-2d" in sections:
        bench_edt_2d()
    if "edt-3d" in sections:
        bench_edt_3d()
    if "cdt" in sections:
        bench_cdt_2d()
