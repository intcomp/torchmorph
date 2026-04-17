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
    """
    N-dimensional binary erosion for `(B, C, Spatial...)` tensors.

    For a single image or volume, add batch and channel dimensions first.
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
    """binary dilation for `(B, C, Spatial...)` tensors."""
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
    """binary opening for '(B, C, ...)' Tensors ."""
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
    """binary closing for (B, C, ...) Tensors."""
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
