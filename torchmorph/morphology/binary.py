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
    spatial_ndim = input.ndim - 2
    if spatial_ndim > 8:
        raise ValueError(f"Input spatial dimensions must be in range 1 to 8, got {spatial_ndim}.")
    if input.numel() == 0:
        raise ValueError(f"Invalid input: empty tensor with shape {input.shape}.")
    return spatial_ndim


def _validate_binary_output(input: Tensor, output: Tensor | None) -> None:
    if output is None:
        return
    if output.shape != input.shape:
        raise ValueError(f"output shape {output.shape} must match input shape {input.shape}")
    if output.device != input.device:
        raise ValueError(
            f"output must be on the same device as input, got {output.device} and {input.device}"
        )


def _normalize_structure(structure: Tensor, spatial_ndim: int, name: str = "structure") -> Tensor:
    if structure.ndim != spatial_ndim:
        raise ValueError(f"{name} dimension is not {spatial_ndim}, got {structure.ndim}")
    return (structure != 0).detach().to(device="cpu", dtype=torch.bool).contiguous()


def _validate_origin(origin: tuple[int, ...], structure: Tensor, name: str = "origin") -> None:
    for origin_value, structure_size in zip(origin, structure.shape):
        min_origin = -(structure_size // 2)
        max_origin = (structure_size - 1) // 2
        if not min_origin <= origin_value <= max_origin:
            raise ValueError(f"invalid {name}")


def _binary_morphology_cuda_step(
    input: Tensor,
    structure: Tensor,
    border_value: bool,
    origin: tuple[int, ...],
    *,
    mode: str,
) -> Tensor:
    x = (input != 0).contiguous()
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
    _validate_binary_output(input, output)

    if structure is None:
        structure = generate_binary_structure(spatial_ndim, 1)
    structure = _normalize_structure(structure, spatial_ndim)

    origin = _normalize_origin(origin, spatial_ndim)
    _validate_origin(origin, structure)
    x = input != 0
    input_bool = x
    if mask is not None and mask.shape != input.shape:
        raise ValueError(f"mask shape {mask.shape} must match input shape {input.shape}")
    mask_bool = mask.to(device=input.device, dtype=torch.bool) if mask is not None else None
    structure_is_empty = not structure.any().item()

    def step(value: Tensor) -> Tensor:
        if structure_is_empty:
            return torch.full_like(value, mode == "erosion", dtype=torch.bool)
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
    _validate_binary_output(input, output)

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
    _validate_binary_output(input, output)
    origin1 = _normalize_origin(origin1, spatial_ndim)
    origin2 = origin1 if origin2 is None else _normalize_origin(origin2, spatial_ndim)

    if structure1 is None:
        structure1 = generate_binary_structure(spatial_ndim, 1)
    structure1 = _normalize_structure(structure1, spatial_ndim, "structure1")
    _validate_origin(origin1, structure1, "origin1")

    if structure2 is None:
        structure2 = torch.logical_not(structure1).contiguous()
    else:
        structure2 = _normalize_structure(structure2, spatial_ndim, "structure2")
    _validate_origin(origin2, structure2, "origin2")

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
