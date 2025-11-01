import torch
from torchmorph import _C


def distance_transform(input: torch.Tensor) -> torch.Tensor:
    """Distance Transform in CUDA."""
    if not input.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    return _C.distance_transform(input)
