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
    """Erode a batched grayscale CUDA tensor

    Computes the local minimum after subtracting the non-flat ``structure``.
    Computation uses ``float32`` and supports one to eight spatial dimensions.

    Args:
        input (torch.Tensor): CUDA tensor with shape ``(B, C, Spatial...)``.
        size (int or tuple[int, ...], optional): Shape of a flat, full
            structuring element. A scalar is broadcast to every spatial axis.
        footprint (torch.Tensor, optional): Boolean mask selecting participating
            positions. Must match ``structure`` when both are supplied.
        structure (torch.Tensor, optional): Non-flat additive structuring
            element. It takes precedence over ``footprint`` and ``size``.
        output (torch.Tensor, optional): Preallocated result tensor with the
            same shape and device as ``input``.
        mode (str): Boundary mode: ``"reflect"``, ``"constant"``, ``"nearest"``,
            ``"mirror"``, or ``"wrap"``.
        cval (float): Boundary value used when ``mode="constant"``.
        origin (int or tuple[int, ...]): Structuring-element anchor offset.

    Returns:
        torch.Tensor: Eroded tensor in ``float32``, or ``output`` when supplied.

    Example:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.arange(9, device="cuda").reshape(1, 1, 3, 3)
        >>> tm.grey_erosion(x, size=3).shape
        torch.Size([1, 1, 3, 3])
        ```
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
    """Dilate a batched grayscale CUDA tensor

    Computes the local maximum after adding the non-flat ``structure``.
    Computation uses ``float32`` and supports one to eight spatial dimensions.

    Args:
        input (torch.Tensor): CUDA tensor with shape ``(B, C, Spatial...)``.
        size (int or tuple[int, ...], optional): Shape of a flat, full
            structuring element. A scalar is broadcast to every spatial axis.
        footprint (torch.Tensor, optional): Boolean mask selecting participating
            positions. Must match ``structure`` when both are supplied.
        structure (torch.Tensor, optional): Non-flat additive structuring
            element. It takes precedence over ``footprint`` and ``size``.
        output (torch.Tensor, optional): Preallocated result tensor with the
            same shape and device as ``input``.
        mode (str): Boundary mode: ``"reflect"``, ``"constant"``, ``"nearest"``,
            ``"mirror"``, or ``"wrap"``.
        cval (float): Boundary value used when ``mode="constant"``.
        origin (int or tuple[int, ...]): Structuring-element anchor offset.

    Returns:
        torch.Tensor: Dilated tensor in ``float32``, or ``output`` when supplied.

    Example:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.arange(9, device="cuda").reshape(1, 1, 3, 3)
        >>> tm.grey_dilation(x, size=3).dtype
        torch.float32
        ```
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
    """Open a grayscale tensor by erosion followed by dilation

    Opening suppresses bright features smaller than the selected structuring
    element. Computation uses ``float32``.

    Args:
        input (torch.Tensor): CUDA tensor with shape ``(B, C, Spatial...)``.
        size (int or tuple[int, ...], optional): Shape of a flat, full
            structuring element.
        footprint (torch.Tensor, optional): Boolean mask selecting participating
            positions.
        structure (torch.Tensor, optional): Non-flat additive structuring
            element. It takes precedence over ``footprint`` and ``size``.
        output (torch.Tensor, optional): Preallocated result tensor with the
            same shape and device as ``input``.
        mode (str): Boundary mode: ``"reflect"``, ``"constant"``, ``"nearest"``,
            ``"mirror"``, or ``"wrap"``.
        cval (float): Boundary value used when ``mode="constant"``.
        origin (int or tuple[int, ...]): Structuring-element anchor offset.

    Returns:
        torch.Tensor: Opened tensor in ``float32``, or ``output`` when supplied.

    Example:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.zeros((1, 1, 5, 5), device="cuda")
        >>> x[0, 0, 2, 2] = 1
        >>> tm.grey_opening(x, size=3).max().item()
        0.0
        ```
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
    """Close a grayscale tensor by dilation followed by erosion

    Closing suppresses dark features smaller than the selected structuring
    element. Computation uses ``float32``.

    Args:
        input (torch.Tensor): CUDA tensor with shape ``(B, C, Spatial...)``.
        size (int or tuple[int, ...], optional): Shape of a flat, full
            structuring element.
        footprint (torch.Tensor, optional): Boolean mask selecting participating
            positions.
        structure (torch.Tensor, optional): Non-flat additive structuring
            element. It takes precedence over ``footprint`` and ``size``.
        output (torch.Tensor, optional): Preallocated result tensor with the
            same shape and device as ``input``.
        mode (str): Boundary mode: ``"reflect"``, ``"constant"``, ``"nearest"``,
            ``"mirror"``, or ``"wrap"``.
        cval (float): Boundary value used when ``mode="constant"``.
        origin (int or tuple[int, ...]): Structuring-element anchor offset.

    Returns:
        torch.Tensor: Closed tensor in ``float32``, or ``output`` when supplied.

    Example:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.ones((1, 1, 5, 5), device="cuda")
        >>> x[0, 0, 2, 2] = 0
        >>> tm.grey_closing(x, size=3).min().item()
        1.0
        ```
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
    """Compute the morphological gradient of a grayscale tensor

    The gradient is ``dilation(input) - erosion(input)`` and emphasizes local
    intensity transitions.

    Args:
        input (torch.Tensor): CUDA tensor with shape ``(B, C, Spatial...)``.
        size (int or tuple[int, ...], optional): Shape of a flat, full
            structuring element.
        footprint (torch.Tensor, optional): Boolean mask selecting participating
            positions.
        structure (torch.Tensor, optional): Non-flat additive structuring
            element. It takes precedence over ``footprint`` and ``size``.
        output (torch.Tensor, optional): Preallocated result tensor with the
            same shape and device as ``input``.
        mode (str): Boundary mode: ``"reflect"``, ``"constant"``, ``"nearest"``,
            ``"mirror"``, or ``"wrap"``.
        cval (float): Boundary value used when ``mode="constant"``.
        origin (int or tuple[int, ...]): Structuring-element anchor offset.

    Returns:
        torch.Tensor: Morphological gradient in ``float32``, or ``output``.

    Example:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.arange(25, device="cuda").reshape(1, 1, 5, 5)
        >>> tm.morphological_gradient(x, size=3).shape
        torch.Size([1, 1, 5, 5])
        ```
    """
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
    """Extract small bright features with a white top-hat filter

    Computes ``input - grey_opening(input)`` using ``float32`` arithmetic.

    Args:
        input (torch.Tensor): CUDA tensor with shape ``(B, C, Spatial...)``.
        size (int or tuple[int, ...], optional): Shape of a flat, full
            structuring element.
        footprint (torch.Tensor, optional): Boolean mask selecting participating
            positions.
        structure (torch.Tensor, optional): Non-flat additive structuring
            element. It takes precedence over ``footprint`` and ``size``.
        output (torch.Tensor, optional): Preallocated result tensor with the
            same shape and device as ``input``.
        mode (str): Boundary mode: ``"reflect"``, ``"constant"``, ``"nearest"``,
            ``"mirror"``, or ``"wrap"``.
        cval (float): Boundary value used when ``mode="constant"``.
        origin (int or tuple[int, ...]): Structuring-element anchor offset.

    Returns:
        torch.Tensor: White top-hat response in ``float32``, or ``output``.

    Example:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.zeros((1, 1, 5, 5), device="cuda")
        >>> x[0, 0, 2, 2] = 2
        >>> tm.white_tophat(x, size=3).max().item()
        2.0
        ```
    """
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
    """Extract small dark features with a black top-hat filter

    Computes ``grey_closing(input) - input`` using ``float32`` arithmetic.

    Args:
        input (torch.Tensor): CUDA tensor with shape ``(B, C, Spatial...)``.
        size (int or tuple[int, ...], optional): Shape of a flat, full
            structuring element.
        footprint (torch.Tensor, optional): Boolean mask selecting participating
            positions.
        structure (torch.Tensor, optional): Non-flat additive structuring
            element. It takes precedence over ``footprint`` and ``size``.
        output (torch.Tensor, optional): Preallocated result tensor with the
            same shape and device as ``input``.
        mode (str): Boundary mode: ``"reflect"``, ``"constant"``, ``"nearest"``,
            ``"mirror"``, or ``"wrap"``.
        cval (float): Boundary value used when ``mode="constant"``.
        origin (int or tuple[int, ...]): Structuring-element anchor offset.

    Returns:
        torch.Tensor: Black top-hat response in ``float32``, or ``output``.

    Example:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.ones((1, 1, 5, 5), device="cuda")
        >>> x[0, 0, 2, 2] = 0
        >>> tm.black_tophat(x, size=3).max().item()
        1.0
        ```
    """
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
    """Compute the morphological Laplacian of a grayscale tensor

    Computes ``dilation(input) + erosion(input) - 2 * input`` using ``float32``
    arithmetic.

    Args:
        input (torch.Tensor): CUDA tensor with shape ``(B, C, Spatial...)``.
        size (int or tuple[int, ...], optional): Shape of a flat, full
            structuring element.
        footprint (torch.Tensor, optional): Boolean mask selecting participating
            positions.
        structure (torch.Tensor, optional): Non-flat additive structuring
            element. It takes precedence over ``footprint`` and ``size``.
        output (torch.Tensor, optional): Preallocated result tensor with the
            same shape and device as ``input``.
        mode (str): Boundary mode: ``"reflect"``, ``"constant"``, ``"nearest"``,
            ``"mirror"``, or ``"wrap"``.
        cval (float): Boundary value used when ``mode="constant"``.
        origin (int or tuple[int, ...]): Structuring-element anchor offset.

    Returns:
        torch.Tensor: Morphological Laplacian in ``float32``, or ``output``.

    Example:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.arange(25, device="cuda").reshape(1, 1, 5, 5)
        >>> tm.morphological_laplace(x, size=3).dtype
        torch.float32
        ```
    """
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
