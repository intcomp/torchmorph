import torch
from torch import Tensor

from .. import _C

_MODE_MAP = {
    'constant': 0,
    'reflect': 1,
    'nearest': 2,
    'mirror': 3,
    'wrap': 4,
}


def grey_erosion(
    input: Tensor,
    size: int | tuple[int, ...] | None = None,
    footprint: Tensor | None = None,
    structure: Tensor | None = None,
    output: Tensor | None = None,
    mode: str = 'reflect',
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> Tensor:
    """N-dimensional grey erosion for (B, C, Spatial...) tensors.

    For a single image or volume, add batch and channel dimensions first.

    The structuring element is resolved with the following priority:
        1. `structure` (non-flat, values are subtracted)
        2. `footprint` (flat, only positions where footprint > 0 are used)
        3. `size` (flat rectangle shaped by `size`)
    Exactly one of these three parameters must be provided.

    Args:
        input: Input tensor in (B, C, Spatial...) format. Must be on CUDA.
        size: Shape of a flat rectangular structuring element. Mutually
            exclusive with footprint and structure.
        footprint: Binary structuring element. Non-zero positions define
            the neighborhood over which the minimum is computed. Mutually
            exclusive with size and structure.
        structure: Value structuring element. Values are subtracted from
            each pixel in the neighborhood. Mutually exclusive with size
            and footprint. When given, `footprint` may optionally be
            provided as a mask to restrict the neighborhood.
        output: Pre-allocated output tensor. If provided, the result is
            written in-place and returned.
        mode: Border mode. One of 'constant', 'reflect', 'nearest',
            'mirror', 'wrap'. Default is 'reflect'.
        cval: Constant fill value used when mode='constant'. Default is 0.0.
        origin: Origin offset. Can be an int (same for all axes) or tuple.

    Returns:
        Grey-eroded tensor with the same shape as input.
    """
    if not input.is_cuda:
        raise ValueError('Input tensor must be on CUDA device.')
    if input.ndim < 3:
        raise ValueError(
            f'Input must be (B, C, Spatial...) with at least 3 dimensions, '
            f'got {input.shape}.'
        )
    if input.numel() == 0:
        raise ValueError(
            f'Invalid input: empty tensor with shape {input.shape}.'
        )

    spatial_ndim = input.ndim - 2

    footprint_cpu = torch.empty(0, dtype=torch.bool)

    # Resolve structuring element (priority: structure > footprint > size)
    if structure is not None:
        struct = structure.to(torch.float32)
        if footprint is not None:
            mask = footprint.to(torch.bool)
            if tuple(mask.shape) != tuple(struct.shape):
                raise ValueError(
                    f'Footprint shape {tuple(mask.shape)} must match structure '
                    f'shape {tuple(struct.shape)}.'
                )
            if not mask.any():
                raise ValueError('All-zero footprint is not supported.')
            footprint_cpu = (
                mask.detach().to(device='cpu', dtype=torch.bool).contiguous()
            )
    elif footprint is not None:
        mask = footprint.to(torch.bool)
        if not mask.any():
            raise ValueError('All-zero footprint is not supported.')
        struct = torch.zeros(tuple(mask.shape), dtype=torch.float32)
        footprint_cpu = (
            mask.detach().to(device='cpu', dtype=torch.bool).contiguous()
        )
    elif size is not None:
        if isinstance(size, int):
            size = (size,) * spatial_ndim
        struct = torch.zeros(size, dtype=torch.float32)
    else:
        raise ValueError(
            'At least one of size, footprint, or structure must be specified.'
        )

    if struct.ndim != spatial_ndim:
        raise ValueError(
            f'Structure dimension {struct.ndim} must match spatial dimension '
            f'{spatial_ndim}.'
        )
    struct = struct.detach().to(device='cpu', dtype=torch.float32).contiguous()

    # Normalize origin
    if isinstance(origin, int):
        origin_list = [origin] * spatial_ndim
    else:
        origin_list = list(origin)
    if len(origin_list) != spatial_ndim:
        raise ValueError(
            f'Origin length {len(origin_list)} must match spatial dimension '
            f'{spatial_ndim}.'
        )

    # Map mode string to int
    if mode not in _MODE_MAP:
        raise ValueError(
            f"Unknown mode '{mode}'. Must be one of {list(_MODE_MAP.keys())}."
        )
    mode_int = _MODE_MAP[mode]

    # CUDA kernel call
    result = _C.grey_erosion_cuda(
        input.contiguous().float(),
        struct.contiguous(),
        footprint_cpu,
        origin_list,
        mode_int,
        cval,
    )

    if output is not None:
        output.copy_(result)
        return output
    return result
