import torch
import torch.nn.functional as F
from torch import Tensor

from .structure import generate_binary_structure


def _prepare_origin(origin: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    """Normalize origin to an ndim-length tuple."""
    if isinstance(origin, bool):
        raise TypeError("origin must be an int or tuple of ints")
    if isinstance(origin, int):
        return (origin,) * ndim

    origin = tuple(origin)
    if len(origin) != ndim:
        raise ValueError(f"origin dimension is not {ndim}, got {len(origin)}")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in origin):
        raise TypeError("origin must be an int or tuple of ints")
    return origin


def _prepare_structure(
    structure: Tensor | None,
    spatial_ndim: int,
    device: torch.device,
) -> Tensor:
    if structure is None:
        structure = generate_binary_structure(spatial_ndim, 1)

    structure_bool = (structure != 0).to(device=device, dtype=torch.bool)
    if structure_bool.ndim != spatial_ndim:
        raise ValueError(
            f"structure.ndim must equal spatial_ndim, got {structure_bool.ndim} and {spatial_ndim}"
        )
    if not torch.any(structure_bool):
        raise ValueError("structure must contain at least one active element")
    return structure_bool


def _normalize_for_operation(
    structure: Tensor,
    origin: tuple[int, ...],
    *,
    mode: str,
) -> tuple[Tensor, tuple[int, ...]]:
    if mode == "erosion":
        return structure, origin

    dims = tuple(range(structure.ndim))
    structure = torch.flip(structure, dims=dims)
    dilation_origin = []
    for dim, value in enumerate(origin):
        adjusted = -value
        if structure.shape[dim] % 2 == 0:
            adjusted -= 1
        dilation_origin.append(adjusted)
    return structure, tuple(dilation_origin)


def _compute_anchor(
    structure_shape: torch.Size,
    origin: tuple[int, ...],
) -> tuple[int, ...]:
    anchor = tuple(structure_shape[d] // 2 + origin[d] for d in range(len(structure_shape)))
    for dim, value in enumerate(anchor):
        if not 0 <= value < structure_shape[dim]:
            raise ValueError(f"bounds for dim {dim}: {value}must in[0, {structure_shape[dim]})")
    return anchor


def _structure_to_offsets(
    structure: Tensor,
    anchor: tuple[int, ...],
) -> list[tuple[int, ...]]:
    active_coords = torch.nonzero(structure, as_tuple=False)
    return [
        tuple(int(coord[d]) - anchor[d] for d in range(structure.ndim)) for coord in active_coords
    ]


def _offsets_to_padding(
    offsets: list[tuple[int, ...]],
) -> tuple[list[int], list[int], list[int]]:
    mins = [min(offset[d] for offset in offsets) for d in range(len(offsets[0]))]
    maxs = [max(offset[d] for offset in offsets) for d in range(len(offsets[0]))]

    pad_before = [max(0, -value) for value in mins]
    pad_after = [max(0, value) for value in maxs]

    pad = []
    for dim in range(len(offsets[0]) - 1, -1, -1):
        pad.extend([pad_before[dim], pad_after[dim]])
    return pad_before, pad_after, pad


def _gather_neighbors(
    x_padded: Tensor,
    spatial_shape: tuple[int, ...],
    pad_before: list[int],
    offsets: list[tuple[int, ...]],
) -> list[Tensor]:
    neighbors = []
    spatial_ndim = len(spatial_shape)
    for offset in offsets:
        slices = [slice(None), slice(None)]
        for dim in range(spatial_ndim):
            start = pad_before[dim] + offset[dim]
            end = start + spatial_shape[dim]
            slices.append(slice(start, end))
        neighbors.append(x_padded[tuple(slices)])
    return neighbors


def _binary_morphology_step(
    x: Tensor,
    structure: Tensor,
    origin: tuple[int, ...],
    border_value: bool,
    *,
    mode: str,
) -> Tensor:
    structure, origin = _normalize_for_operation(structure, origin, mode=mode)
    anchor = _compute_anchor(structure.shape, origin)
    offsets = _structure_to_offsets(structure, anchor)
    pad_before, _, pad = _offsets_to_padding(offsets)

    spatial_shape = tuple(x.shape[2:])
    x_padded = F.pad(x.to(dtype=torch.float32), pad, value=float(bool(border_value))).to(
        dtype=torch.bool
    )
    neighbors = _gather_neighbors(x_padded, spatial_shape, pad_before, offsets)

    result = neighbors[0].clone()
    if mode == "erosion":
        for neighbor in neighbors[1:]:
            result &= neighbor
    else:
        for neighbor in neighbors[1:]:
            result |= neighbor
    return result


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
    if input.ndim < 3:
        raise ValueError(f"input.ndim must be >= 3, got {input.ndim}")
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise TypeError("iterations must be an integer")

    spatial_ndim = input.ndim - 2
    origin_tuple = _prepare_origin(origin, spatial_ndim)
    structure_bool = _prepare_structure(structure, spatial_ndim, input.device)
    x = (input != 0).to(dtype=torch.bool)

    if mask is not None:
        if mask.shape != input.shape:
            raise ValueError("mask.shape must equal input.shape")
        mask_bool = (mask != 0).to(device=input.device, dtype=torch.bool)
    else:
        mask_bool = None

    if iterations < 1:
        while True:
            candidate = _binary_morphology_step(
                x,
                structure_bool,
                origin_tuple,
                border_value,
                mode=mode,
            )
            if mask_bool is not None:
                candidate = torch.where(mask_bool, candidate, x)
            if torch.equal(candidate, x):
                break
            x = candidate
    else:
        for _ in range(iterations):
            candidate = _binary_morphology_step(
                x,
                structure_bool,
                origin_tuple,
                border_value,
                mode=mode,
            )
            if mask_bool is not None:
                candidate = torch.where(mask_bool, candidate, x)
            x = candidate

    result = x.to(dtype=torch.bool)
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
