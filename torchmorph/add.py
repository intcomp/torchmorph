import torch
from torchmorph import _C


def add(input: torch.Tensor, scalar: float) -> torch.Tensor:
    """Add the input tensor by a scalar using CUDA."""
    if not input.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    return _C.add_cuda(input, scalar)
