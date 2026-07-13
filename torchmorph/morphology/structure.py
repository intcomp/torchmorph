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


def _generated_structure_connectivity(structure: Tensor) -> int | None:
    if any(size != 3 for size in structure.shape):
        return None

    rank = structure.ndim
    axes = [torch.tensor([-1, 0, 1], device=structure.device) for _ in range(rank)]
    grids = torch.meshgrid(*axes, indexing="ij")
    offsets = torch.stack(grids, dim=0)
    distances = (offsets != 0).sum(dim=0)

    for connectivity in range(1, rank + 1):
        if torch.equal(structure, distances <= connectivity):
            return connectivity
    return None


def _iterate_generated_structure(rank: int, connectivity: int, iterations: int, device) -> Tensor:
    axes = [torch.arange(-iterations, iterations + 1, device=device) for _ in range(rank)]
    grids = torch.meshgrid(*axes, indexing="ij")
    offsets = torch.stack(grids, dim=0)
    return offsets.abs().sum(dim=0) <= connectivity * iterations


def _normalize_origin(origin: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    if isinstance(origin, int):
        return (origin,) * ndim

    origin_tuple = tuple(origin)
    if len(origin_tuple) != ndim:
        raise ValueError(f"origin dimension is not {ndim}, got {len(origin_tuple)}")
    return origin_tuple


def iterate_structure(
    structure: Tensor,
    iterations: int,
    origin: int | tuple[int, ...] | None = None,
) -> Tensor | tuple[Tensor, list[int]]:
    """Iterate a binary structure by dilating it with itself."""
    structure = structure != 0

    if iterations < 2:
        result = structure.clone()
    else:
        connectivity = _generated_structure_connectivity(structure)
        if connectivity is not None:
            result = _iterate_generated_structure(
                structure.ndim, connectivity, iterations, structure.device
            )
        else:
            coordinates = torch.nonzero(structure, as_tuple=False)
            if coordinates.numel() == 0:
                result_shape = tuple(iterations * (size - 1) + 1 for size in structure.shape)
                result = torch.zeros(result_shape, dtype=torch.bool, device=structure.device)
            else:
                current = coordinates
                for _ in range(iterations - 1):
                    current = (current[:, None, :] + coordinates[None, :, :]).reshape(
                        -1, structure.ndim
                    )
                    current = torch.unique(current, dim=0)

                result_shape = tuple(iterations * (size - 1) + 1 for size in structure.shape)
                result = torch.zeros(result_shape, dtype=torch.bool, device=structure.device)
                result[tuple(current.unbind(dim=1))] = True

    if origin is None:
        return result

    origin_tuple = _normalize_origin(origin, structure.ndim)
    return result, [value * iterations for value in origin_tuple]
