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
