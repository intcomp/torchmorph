import torch
import torch.utils.benchmark as benchmark
import scipy.ndimage as ndi
import torchmorph as tm

for size in [64, 128, 256, 512, 1024, 2048]:
    x = (torch.randn(1, 1, size, size, device="cuda") > 0).to(torch.float32)

    # TorchMorph CUDA
    t1 = benchmark.Timer(
        stmt="tm.distance_transform(x)",
        setup="from __main__ import x, tm",
        num_threads=torch.get_num_threads()
    )
    # SciPy (CPU)
    import numpy as np
    x_np = x.cpu().squeeze().numpy()
    t2 = benchmark.Timer(
        stmt="ndi.distance_transform_edt(x_np)",
        setup="from __main__ import x_np, ndi"
    )

    print(f"Size {size}:\n", t1.blocked_autorange())
    print(f"Size {size}:\n", t2.blocked_autorange())
