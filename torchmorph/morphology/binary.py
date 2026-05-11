import torch
import torch.nn.functional as F
from torch import Tensor

from ._convnd import conv_nd
from .structure import generate_binary_structure


def _prepare_origin(origin: int | tuple[int, ...], ndim=int) -> tuple[int, ...]:
    """change the origin into tuple"""
    if isinstance(origin, int):
        return (origin,) * ndim
    origin = tuple(origin)

    if (len(origin)) != ndim:
        raise ValueError(f"origin dimension is not {ndim}, got {len(origin)}")

    return origin


def _extend_pad(kernel_shape: torch.Size, origin: tuple[int, ...]) -> list[int]:
    """extend the padlist for kernel"""
    pad = []
    for dim in range(len(kernel_shape) - 1, -1, -1):
        center = kernel_shape[dim] // 2
        pad_before = center + origin[dim]
        pad_after = kernel_shape[dim] - 1 - pad_before
        pad.extend([pad_before, pad_after])
    return pad


def _flip_structure(structure: Tensor) -> Tensor:
    dim = tuple(range(structure.ndim))
    return torch.flip(structure, dim)


def _binary_morphology(
    input: Tensor,
    structure: Tensor | None,
    iterations: int,
    mask: Tensor | None,
    output: Tensor | None,
    border_value: bool,
    origin: int | tuple[int, ...],
    *,
    mode: str,
) -> Tensor:
    iterations_flag = iterations < 1

    spatial_ndim = input.ndim - 2

    if structure is None:
        structure = generate_binary_structure(spatial_ndim, 1)

    batch, channels = input.shape[:2]
    spatial_shape = input.shape[2:]

    x = (input != 0).to(dtype=torch.float32).reshape(batch * channels, 1, *spatial_shape)
    structure = (structure != 0).to(device=input.device, dtype=torch.float32)
    if mode == "dilation":
        structure = _flip_structure(structure)
    kernel = structure.unsqueeze(0).unsqueeze(0)
    kernel_sum = kernel.sum()

    origin = _prepare_origin(origin, spatial_ndim)
    if mode == "dilation":
        origin = tuple(-value for value in origin)
    pad = _extend_pad(structure.shape, origin)
    pad_value = float(bool(border_value))

    if mask is not None:
        mask_flat = mask.to(dtype=torch.bool).reshape(batch * channels, 1, *spatial_shape)
        input_flat = (
            (input != 0).to(dtype=torch.float32).reshape(batch * channels, 1, *spatial_shape)
        )
    else:
        mask_flat = None
        input_flat = None

    if iterations_flag:
        old = None
        while True:
            x_padded = F.pad(x, pad, value=pad_value)
            conv = conv_nd(x_padded, kernel)
            if mode == "erosion":
                x = (conv == kernel_sum).to(dtype=torch.float32)
            else:
                x = (conv > 0).to(dtype=torch.float32)
            if mask_flat is not None:
                x = torch.where(mask_flat, x, input_flat)

            if old is not None and torch.equal(x, old):
                break

            old = x.clone()
    else:
        for _ in range(iterations):
            x_padded = F.pad(x, pad, value=pad_value)
            conv = conv_nd(x_padded, kernel)
            if mode == "erosion":
                x = (conv == kernel_sum).to(dtype=torch.float32)
            else:
                x = (conv > 0).to(dtype=torch.float32)

            if mask_flat is not None:
                x = torch.where(mask_flat, x, input_flat)

    result = x.reshape(batch, channels, *spatial_shape).to(dtype=torch.bool)
    if output is not None:
        output.copy_(result)
        return output
    return result


def binary_erosion(
    input: Tensor,
    structure: Tensor | None = None,
    iterations: int = 1,
    mask: Tensor | None = None,
    output: Tensor | None = None,
    border_value: bool = False,
    origin: int | tuple[int, ...] = 0,
) -> Tensor:
    """Erode foreground regions in an N-dimensional binary tensor

    Treats non-zero input values as foreground and applies binary erosion to
    each ``(B, C)`` sample independently. A foreground value survives only when
    every active element of the structuring element overlaps foreground.

    Args:
        input (torch.Tensor): Tensor with shape ``(B, C, *spatial)``. Non-zero
            values are treated as foreground.
        structure (Optional[torch.Tensor]): Structuring element with one axis
            per spatial dimension. Non-zero values are active. If ``None``, a
            rank-matched connectivity-1 structure is generated.
        iterations (int): Number of erosion passes. Values less than ``1`` run
            until the result no longer changes.
        mask (Optional[torch.Tensor]): Boolean mask that restricts which output
            locations may change. Use the same ``(B, C, *spatial)`` layout as
            ``input``.
        output (Optional[torch.Tensor]): Optional tensor to fill in-place with
            the boolean result. The same tensor is returned.
        border_value (bool): Value assumed outside the spatial boundary.
        origin (int | tuple[int, ...]): Structuring-element origin offset. A
            scalar is applied to every spatial dimension; a tuple must match the
            number of spatial dimensions.

    Returns:
        torch.Tensor: Boolean tensor with the same shape as ``input``.

    Examples:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.ones(1, 1, 5, 5)
        >>> x[0, 0, 0, :] = 0
        >>> tm.binary_erosion(x).to(torch.int32)[0, 0]
        tensor([[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]], dtype=torch.int32)
        ```
    """
    return _binary_morphology(
        input=input,
        structure=structure,
        iterations=iterations,
        mask=mask,
        output=output,
        border_value=border_value,
        origin=origin,
        mode="erosion",
    )


def binary_dilation(
    input: Tensor,
    structure: Tensor | None = None,
    iterations: int = 1,
    mask: Tensor | None = None,
    output: Tensor | None = None,
    border_value: bool = False,
    origin: int | tuple[int, ...] = 0,
) -> Tensor:
    """Dilate foreground regions in an N-dimensional binary tensor

    Treats non-zero input values as foreground and applies binary dilation to
    each ``(B, C)`` sample independently. A location becomes foreground when any
    active element of the structuring element overlaps foreground.

    Args:
        input (torch.Tensor): Tensor with shape ``(B, C, *spatial)``. Non-zero
            values are treated as foreground.
        structure (Optional[torch.Tensor]): Structuring element with one axis
            per spatial dimension. Non-zero values are active. If ``None``, a
            rank-matched connectivity-1 structure is generated.
        iterations (int): Number of dilation passes. Values less than ``1`` run
            until the result no longer changes.
        mask (Optional[torch.Tensor]): Boolean mask that restricts which output
            locations may change. Use the same ``(B, C, *spatial)`` layout as
            ``input``.
        output (Optional[torch.Tensor]): Optional tensor to fill in-place with
            the boolean result. The same tensor is returned.
        border_value (bool): Value assumed outside the spatial boundary.
        origin (int | tuple[int, ...]): Structuring-element origin offset. A
            scalar is applied to every spatial dimension; a tuple must match the
            number of spatial dimensions.

    Returns:
        torch.Tensor: Boolean tensor with the same shape as ``input``.

    Examples:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.zeros(1, 1, 5, 5)
        >>> x[0, 0, 2, 2] = 1
        >>> tm.binary_dilation(x).to(torch.int32)[0, 0]
        tensor([[0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0]], dtype=torch.int32)
        ```
    """
    return _binary_morphology(
        input,
        structure,
        iterations,
        mask,
        output,
        border_value,
        origin,
        mode="dilation",
    )


def binary_opening(
    input: Tensor,
    structure: Tensor | None = None,
    iterations: int = 1,
    mask: Tensor | None = None,
    output: Tensor | None = None,
    border_value: bool = False,
    origin: int | tuple[int, ...] = 0,
) -> Tensor:
    """Remove small foreground components with binary opening

    Opening is erosion followed by dilation using the same structuring element.
    The operation is applied independently to each ``(B, C)`` sample of an
    ``(B, C, *spatial)`` tensor.

    Args:
        input (torch.Tensor): Tensor with shape ``(B, C, *spatial)``. Non-zero
            values are treated as foreground.
        structure (Optional[torch.Tensor]): Structuring element with one axis
            per spatial dimension. If ``None``, a rank-matched connectivity-1
            structure is generated.
        iterations (int): Number of erosions followed by the same number of
            dilations. Values less than ``1`` run each stage until convergence.
        mask (Optional[torch.Tensor]): Boolean mask in the same
            ``(B, C, *spatial)`` layout restricting which locations may change.
        output (Optional[torch.Tensor]): Optional tensor to fill in-place.
        border_value (bool): Value assumed outside the spatial boundary.
        origin (int | tuple[int, ...]): Structuring-element origin offset.

    Returns:
        torch.Tensor: Boolean tensor with the same shape as ``input``.

    Examples:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.zeros(1, 1, 5, 5)
        >>> x[0, 0, 1:4, 1:4] = 1
        >>> x[0, 0, 0, 0] = 1
        >>> tm.binary_opening(x).to(torch.int32)[0, 0]
        tensor([[0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0]], dtype=torch.int32)
        ```
    """
    x = binary_erosion(
        input,
        structure,
        iterations,
        mask,
        output,
        border_value,
        origin,
    )
    x = binary_dilation(
        x,
        structure,
        iterations,
        mask,
        output,
        border_value,
        origin,
    )
    return x


def binary_closing(
    input: Tensor,
    structure: Tensor | None = None,
    iterations: int = 1,
    mask: Tensor | None = None,
    output: Tensor | None = None,
    border_value: bool = False,
    origin: int | tuple[int, ...] = 0,
) -> Tensor:
    """Fill small background gaps with binary closing

    Closing is dilation followed by erosion using the same structuring element.
    The operation is applied independently to each ``(B, C)`` sample of an
    ``(B, C, *spatial)`` tensor.

    Args:
        input (torch.Tensor): Tensor with shape ``(B, C, *spatial)``. Non-zero
            values are treated as foreground.
        structure (Optional[torch.Tensor]): Structuring element with one axis
            per spatial dimension. If ``None``, a rank-matched connectivity-1
            structure is generated.
        iterations (int): Number of dilations followed by the same number of
            erosions. Values less than ``1`` run each stage until convergence.
        mask (Optional[torch.Tensor]): Boolean mask in the same
            ``(B, C, *spatial)`` layout restricting which locations may change.
        output (Optional[torch.Tensor]): Optional tensor to fill in-place.
        border_value (bool): Value assumed outside the spatial boundary.
        origin (int | tuple[int, ...]): Structuring-element origin offset.

    Returns:
        torch.Tensor: Boolean tensor with the same shape as ``input``.

    Examples:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.ones(1, 1, 5, 5)
        >>> x[0, 0, 2, 2] = 0
        >>> tm.binary_closing(x, border_value=True).to(torch.int32)[0, 0]
        tensor([[1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]], dtype=torch.int32)
        ```
    """
    x = binary_dilation(
        input,
        structure,
        iterations,
        mask,
        output,
        border_value,
        origin,
    )
    x = binary_erosion(
        x,
        structure,
        iterations,
        mask,
        output,
        border_value,
        origin,
    )
    return x
