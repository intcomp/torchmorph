import torch
from torch import Tensor

from .. import _C
from .._validation import validate_bcs_input, validate_output
from .structure import _normalize_origin, _validate_origin

_MODE_MAP = {
    'constant': 0,
    'reflect': 1,
    'nearest': 2,
    'mirror': 3,
    'wrap': 4,
}


def _element_kwargs(size, footprint, structure, mode, cval, origin) -> dict:
    """Pack the structuring-element arguments shared by every grey operator."""
    return dict(
        size=size,
        footprint=footprint,
        structure=structure,
        mode=mode,
        cval=cval,
        origin=origin,
    )


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
    spatial_ndim = validate_bcs_input(input)
    validate_output(input, output)

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
        if any(value <= 0 for value in size):
            raise ValueError("size values must be greater than zero")
        struct = torch.zeros(size, dtype=torch.float32)
    else:
        raise ValueError('At least one of size, footprint, or structure must be specified.')

    if struct.ndim != spatial_ndim:
        raise ValueError(
            f'Structure dimension {struct.ndim} must match spatial dimension ' f'{spatial_ndim}.'
        )
    struct = struct.detach().to(device='cpu', dtype=torch.float32).contiguous()

    # Normalize and validate origin
    origin_tuple = _normalize_origin(origin, spatial_ndim)
    _validate_origin(origin_tuple, struct)
    origin_list = list(origin_tuple)

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
    kwargs = _element_kwargs(size, footprint, structure, mode, cval, origin)
    eroded = grey_erosion(input, **kwargs)
    return grey_dilation(eroded, output=output, **kwargs)


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
    kwargs = _element_kwargs(size, footprint, structure, mode, cval, origin)
    dilated = grey_dilation(input, **kwargs)
    return grey_erosion(dilated, output=output, **kwargs)


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
    validate_bcs_input(input)
    validate_output(input, output)

    kwargs = _element_kwargs(size, footprint, structure, mode, cval, origin)
    dilated = grey_dilation(input, **kwargs)
    eroded = grey_erosion(input, **kwargs)
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
    validate_bcs_input(input)
    validate_output(input, output)

    opened = grey_opening(input, **_element_kwargs(size, footprint, structure, mode, cval, origin))
    result = input.detach().float() - opened
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
    validate_bcs_input(input)
    validate_output(input, output)

    closed = grey_closing(input, **_element_kwargs(size, footprint, structure, mode, cval, origin))
    result = closed - input.detach().float()
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
    validate_bcs_input(input)
    validate_output(input, output)

    kwargs = _element_kwargs(size, footprint, structure, mode, cval, origin)
    dilated = grey_dilation(input, **kwargs)
    eroded = grey_erosion(input, **kwargs)
    result = dilated + eroded - 2 * input.detach().float()
    if output is not None:
        output.copy_(result)
        return output
    return result
