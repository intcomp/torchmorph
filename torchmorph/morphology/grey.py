import torch
from torch import Tensor

from .. import _C


def _validate_grey_input(input: Tensor) -> int:
    if not input.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    if input.ndim < 3:
        raise ValueError(
            f"Input must be (B, C, Spatial...) with at least 3 dimensions, got {input.shape}."
        )
    spatial_ndim = input.ndim - 2
    if spatial_ndim > 8:
        raise ValueError(f"Input spatial dimensions must be in range 1 to 8, got {spatial_ndim}.")
    if input.numel() == 0:
        raise ValueError(f"Invalid input: empty tensor with shape {input.shape}.")
    return spatial_ndim


def _validate_grey_output(input: Tensor, output: Tensor | None) -> None:
    if output is None:
        return
    if output.shape != input.shape:
        raise ValueError(f"output shape {output.shape} must match input shape {input.shape}")
    if output.device != input.device:
        raise ValueError(
            f"output must be on the same device as input, got {output.device} and {input.device}"
        )


_MODE_MAP = {
    'constant': 0,
    'reflect': 1,
    'nearest': 2,
    'mirror': 3,
    'wrap': 4,
}


def _grey_morphology(
    input: Tensor,
    size: int | tuple[int, ...] | None = None,
    footprint: Tensor | None = None,
    structure: Tensor | None = None,
    output: Tensor | None = None,
    mode: str = 'reflect',
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
    *,
    operation: str,
) -> Tensor:
    """Shared argument normalization for grey erosion and dilation."""
    spatial_ndim = _validate_grey_input(input)
    _validate_grey_output(input, output)

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
            footprint_cpu = mask.detach().to(device='cpu', dtype=torch.bool).contiguous()
    elif footprint is not None:
        mask = footprint.to(torch.bool)
        if not mask.any():
            raise ValueError('All-zero footprint is not supported.')
        struct = torch.zeros(tuple(mask.shape), dtype=torch.float32)
        footprint_cpu = mask.detach().to(device='cpu', dtype=torch.bool).contiguous()
    elif size is not None:
        if isinstance(size, int):
            size = (size,) * spatial_ndim
        struct = torch.zeros(size, dtype=torch.float32)
    else:
        raise ValueError('At least one of size, footprint, or structure must be specified.')

    if struct.ndim != spatial_ndim:
        raise ValueError(
            f'Structure dimension {struct.ndim} must match spatial dimension ' f'{spatial_ndim}.'
        )
    struct = struct.detach().to(device='cpu', dtype=torch.float32).contiguous()

    # Normalize origin
    if isinstance(origin, int):
        origin_list = [origin] * spatial_ndim
    else:
        origin_list = list(origin)
    if len(origin_list) != spatial_ndim:
        raise ValueError(
            f'Origin length {len(origin_list)} must match spatial dimension ' f'{spatial_ndim}.'
        )

    for origin_value, structure_size in zip(origin_list, struct.shape):
        min_origin = -(structure_size // 2)
        max_origin = (structure_size - 1) // 2
        if not min_origin <= origin_value <= max_origin:
            raise ValueError("invalid origin")

    # Map mode string to int
    if mode not in _MODE_MAP:
        raise ValueError(f"Unknown mode '{mode}'. Must be one of {list(_MODE_MAP.keys())}.")
    mode_int = _MODE_MAP[mode]

    # CUDA kernel call
    kernel = _C.grey_erosion_cuda if operation == "erosion" else _C.grey_dilation_cuda
    result = kernel(
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


def grey_erosion(
    input: Tensor,
    size: int | tuple[int, ...] | None = None,
    footprint: Tensor | None = None,
    structure: Tensor | None = None,
    output: Tensor | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> Tensor:
    """N-dimensional grey erosion for ``(B, C, Spatial...)`` CUDA tensors.

    Computation uses float32. The result is float32 unless ``output`` is given.
    """
    return _grey_morphology(
        input,
        size,
        footprint,
        structure,
        output,
        mode,
        cval,
        origin,
        operation="erosion",
    )


def grey_dilation(
    input: Tensor,
    size: int | tuple[int, ...] | None = None,
    footprint: Tensor | None = None,
    structure: Tensor | None = None,
    output: Tensor | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> Tensor:
    """N-dimensional grey dilation for ``(B, C, Spatial...)`` CUDA tensors.

    Computation uses float32. The result is float32 unless ``output`` is given.
    """
    return _grey_morphology(
        input,
        size,
        footprint,
        structure,
        output,
        mode,
        cval,
        origin,
        operation="dilation",
    )


def grey_opening(
    input: Tensor,
    size: int | tuple[int, ...] | None = None,
    footprint: Tensor | None = None,
    structure: Tensor | None = None,
    output: Tensor | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> Tensor:
    """N-dimensional grey opening for ``(B, C, Spatial...)`` CUDA tensors.

    Computation uses float32. The result is float32 unless ``output`` is given.
    """
    eroded = grey_erosion(
        input,
        size=size,
        footprint=footprint,
        structure=structure,
        mode=mode,
        cval=cval,
        origin=origin,
    )
    return grey_dilation(
        eroded,
        size=size,
        footprint=footprint,
        structure=structure,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
    )


def grey_closing(
    input: Tensor,
    size: int | tuple[int, ...] | None = None,
    footprint: Tensor | None = None,
    structure: Tensor | None = None,
    output: Tensor | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> Tensor:
    """N-dimensional grey closing for ``(B, C, Spatial...)`` CUDA tensors.

    Computation uses float32. The result is float32 unless ``output`` is given.
    """
    dilated = grey_dilation(
        input,
        size=size,
        footprint=footprint,
        structure=structure,
        mode=mode,
        cval=cval,
        origin=origin,
    )
    return grey_erosion(
        dilated,
        size=size,
        footprint=footprint,
        structure=structure,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
    )


def morphological_gradient(
    input: Tensor,
    size: int | tuple[int, ...] | None = None,
    footprint: Tensor | None = None,
    structure: Tensor | None = None,
    output: Tensor | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> Tensor:
    """N-dimensional morphological gradient for ``(B, C, Spatial...)`` CUDA tensors."""
    _validate_grey_input(input)
    _validate_grey_output(input, output)

    dilated = grey_dilation(
        input,
        size=size,
        footprint=footprint,
        structure=structure,
        mode=mode,
        cval=cval,
        origin=origin,
    )
    eroded = grey_erosion(
        input,
        size=size,
        footprint=footprint,
        structure=structure,
        mode=mode,
        cval=cval,
        origin=origin,
    )
    result = dilated - eroded
    if output is not None:
        output.copy_(result)
        return output
    return result


def white_tophat(
    input: Tensor,
    size: int | tuple[int, ...] | None = None,
    footprint: Tensor | None = None,
    structure: Tensor | None = None,
    output: Tensor | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> Tensor:
    """N-dimensional white top-hat filter for ``(B, C, Spatial...)`` CUDA tensors."""
    _validate_grey_input(input)
    _validate_grey_output(input, output)

    opened = grey_opening(
        input,
        size=size,
        footprint=footprint,
        structure=structure,
        mode=mode,
        cval=cval,
        origin=origin,
    )
    result = input.float() - opened
    if output is not None:
        output.copy_(result)
        return output
    return result


def black_tophat(
    input: Tensor,
    size: int | tuple[int, ...] | None = None,
    footprint: Tensor | None = None,
    structure: Tensor | None = None,
    output: Tensor | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> Tensor:
    """N-dimensional black top-hat filter for ``(B, C, Spatial...)`` CUDA tensors."""
    _validate_grey_input(input)
    _validate_grey_output(input, output)

    closed = grey_closing(
        input,
        size=size,
        footprint=footprint,
        structure=structure,
        mode=mode,
        cval=cval,
        origin=origin,
    )
    result = closed - input.float()
    if output is not None:
        output.copy_(result)
        return output
    return result


def morphological_laplace(
    input: Tensor,
    size: int | tuple[int, ...] | None = None,
    footprint: Tensor | None = None,
    structure: Tensor | None = None,
    output: Tensor | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> Tensor:
    """N-dimensional morphological Laplace for ``(B, C, Spatial...)`` CUDA tensors."""
    _validate_grey_input(input)
    _validate_grey_output(input, output)

    dilated = grey_dilation(
        input,
        size=size,
        footprint=footprint,
        structure=structure,
        mode=mode,
        cval=cval,
        origin=origin,
    )
    eroded = grey_erosion(
        input,
        size=size,
        footprint=footprint,
        structure=structure,
        mode=mode,
        cval=cval,
        origin=origin,
    )
    result = dilated + eroded - 2 * input.float()
    if output is not None:
        output.copy_(result)
        return output
    return result
