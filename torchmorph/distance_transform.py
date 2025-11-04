import torch
from torchmorph import _C


def distance_transform(input: torch.Tensor) -> torch.Tensor:
    """Distance Transform in CUDA."""
    if not input.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    if input.ndim < 2 or input.numel() == 0:
        raise ValueError(f"Invalid input dimension: {input.shape}.")

    # binarize input
    input[input != 0] = 1

    return _C.distance_transform_cuda(input)
