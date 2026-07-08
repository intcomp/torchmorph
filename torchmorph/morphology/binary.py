import torch
from torch import Tensor

from .. import _C
from .structure import _normalize_origin, generate_binary_structure


def _validate_binary_input(input: Tensor) -> int:
    if not input.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    if input.ndim < 3:
        raise ValueError(
            f"Input must be (B, C, Spatial...) with at least 3 dimensions, got {input.shape}."
        )
    if input.numel() == 0:
        raise ValueError(f"Invalid input: empty tensor with shape {input.shape}.")

    return input.ndim - 2


def _binary_morphology_cuda_step(
    input: Tensor,
    structure: Tensor,
    border_value: bool,
    origin: tuple[int, ...],
    *,
    mode: str,
) -> Tensor:
    x = (input != 0).contiguous()
    structure = (structure != 0).detach().to(device="cpu", dtype=torch.bool).contiguous()
    kernel = _C.binary_erosion_cuda if mode == "erosion" else _C.binary_dilation_cuda
    return kernel(x, structure, list(origin), bool(border_value))


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
    iterate_until_stable = iterations < 1
    spatial_ndim = _validate_binary_input(input)

    if structure is None:
        structure = generate_binary_structure(spatial_ndim, 1)

    origin = _normalize_origin(origin, spatial_ndim)
    x = input != 0
    input_bool = x
    mask_bool = mask.to(device=input.device, dtype=torch.bool) if mask is not None else None

    def step(value: Tensor) -> Tensor:
        return _binary_morphology_cuda_step(
            value,
            structure,
            border_value,
            origin,
            mode=mode,
        )

    if iterate_until_stable:
        old = None
        while True:
            x = step(x)
            if mask_bool is not None:
                x = torch.where(mask_bool, x, input_bool)

            if old is not None and torch.equal(x, old):
                break

            old = x.clone()
    else:
        for _ in range(iterations):
            x = step(x)
            if mask_bool is not None:
                x = torch.where(mask_bool, x, input_bool)

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
    N-dimensional binary erosion for `(B, C, Spatial...)` CUDA tensors.

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
    """N-dimensional binary dilation for `(B, C, Spatial...)` CUDA tensors."""
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


def binary_propagation(
    input: Tensor,
    structure: Tensor | None = None,
    mask: Tensor | None = None,
    output: Tensor | None = None,
    border_value: bool = False,
    origin: int | tuple[int, ...] = 0,
) -> Tensor:
    """N-dimensional binary propagation for `(B, C, Spatial...)` CUDA tensors."""
    return _binary_morphology(
        input,
        structure,
        -1,
        mask,
        output,
        border_value,
        origin,
        mode="dilation",
    )


def binary_fill_holes(
    input: Tensor,
    structure: Tensor | None = None,
    output: Tensor | None = None,
    origin: int | tuple[int, ...] = 0,
) -> Tensor:
    """Fill holes in binary objects for `(B, C, Spatial...)` CUDA tensors."""
    _validate_binary_input(input)

    mask = input == 0
    seed = torch.zeros_like(mask, dtype=torch.bool)
    background = binary_propagation(
        seed,
        structure=structure,
        mask=mask,
        output=None,
        border_value=True,
        origin=origin,
    )
    result = torch.logical_not(background)

    if output is not None:
        output.copy_(result)
        return output
    return result


def binary_hit_or_miss(
    input: Tensor,
    structure1: Tensor | None = None,
    structure2: Tensor | None = None,
    output: Tensor | None = None,
    origin1: int | tuple[int, ...] = 0,
    origin2: int | tuple[int, ...] | None = None,
) -> Tensor:
    """N-dimensional binary hit-or-miss transform for `(B, C, Spatial...)` CUDA tensors."""
    spatial_ndim = _validate_binary_input(input)
    origin1 = _normalize_origin(origin1, spatial_ndim)
    origin2 = origin1 if origin2 is None else _normalize_origin(origin2, spatial_ndim)

    if structure1 is None:
        structure1 = generate_binary_structure(spatial_ndim, 1)
    elif structure1.ndim != spatial_ndim:
        raise ValueError(f"structure1 dimension is not {spatial_ndim}, got {structure1.ndim}")

    structure1 = structure1.detach().to(device="cpu", dtype=torch.bool).contiguous()
    if structure2 is None:
        structure2 = torch.logical_not(structure1).contiguous()
    else:
        if structure2.ndim != spatial_ndim:
            raise ValueError(f"structure2 dimension is not {spatial_ndim}, got {structure2.ndim}")
        structure2 = structure2.detach().to(device="cpu", dtype=torch.bool).contiguous()

    input_bool = input != 0
    if structure1.any().item():
        hit = binary_erosion(input_bool, structure=structure1, origin=origin1)
    else:
        hit = torch.ones_like(input_bool, dtype=torch.bool)

    if structure2.any().item():
        miss = binary_erosion(input_bool == 0, structure=structure2, origin=origin2)
        result = torch.logical_and(hit, miss)
    else:
        result = hit

    if output is not None:
        output.copy_(result)
        return output
    return result


def binary_opening(
    input: Tensor,
    structure: Tensor | None = None,
    iterations: int = 1,
    mask: Tensor | None = None,
    output: Tensor | None = None,
    border_value: bool = False,
    origin: int | tuple[int, ...] = 0,
) -> Tensor:
    """N-dimensional binary opening for `(B, C, Spatial...)` CUDA tensors."""
    x = binary_erosion(
        input,
        structure,
        iterations,
        mask,
        None,
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
    """N-dimensional binary closing for `(B, C, Spatial...)` CUDA tensors."""
    x = binary_dilation(
        input,
        structure,
        iterations,
        mask,
        None,
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
