import torch
from torch import Tensor


def generate_binary_structure(rank: int, connectivity: int) -> Tensor:
    """Generate an N-dimensional binary structuring element

    The returned tensor has shape ``(3,) * rank``. Elements whose offset differs
    from the center along at most ``connectivity`` axes are ``True``; all other
    elements are ``False``. This matches SciPy's
    ``ndimage.generate_binary_structure`` connectivity convention.

    Args:
        rank (int): Number of spatial dimensions in the structuring element.
            Must be at least ``1``.
        connectivity (int): Neighborhood connectivity from ``1`` to ``rank``.
            ``1`` includes axis-adjacent neighbors; ``rank`` includes the full
            ``3 ** rank`` neighborhood.

    Returns:
        torch.Tensor: Boolean tensor with shape ``(3,) * rank``.

    Examples:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> tm.generate_binary_structure(2, 1).to(dtype=torch.int32)
        tensor([[0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]], dtype=torch.int32)
        >>> tm.generate_binary_structure(2, 2).to(dtype=torch.int32)
        tensor([[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]], dtype=torch.int32)
        ```
    """
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


def _validate_origin(origin: tuple[int, ...], structure: Tensor, name: str = "origin") -> None:
    for origin_value, structure_size in zip(origin, structure.shape):
        min_origin = -(structure_size // 2)
        max_origin = (structure_size - 1) // 2
        if not min_origin <= origin_value <= max_origin:
            raise ValueError(f"invalid {name}")


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
