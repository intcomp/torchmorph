import torch
from torch import Tensor


def generate_binary_structure(rank: int, connectivity: int) -> Tensor:
    """N-D generate binary structure"""
    if connectivity < 1 or connectivity > rank:
        raise ValueError(f"connectivity must be in [1, rank], got {connectivity}")

    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")

    axes = [torch.tensor([-1, 0, 1]) for _ in range(rank)]
    grids = torch.meshgrid(*axes, indexing="ij")
    offsets = torch.stack(grids, dim=0)
    return (offsets != 0).sum(dim=0) <= connectivity
