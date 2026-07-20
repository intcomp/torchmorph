import torch
from torch import Tensor

from .. import _C
from .._validation import validate_bcs_input, validate_output
from .structure import _normalize_origin, _validate_origin, generate_binary_structure


def _normalize_structure(structure: Tensor, spatial_ndim: int, name: str = "structure") -> Tensor:
    if structure.ndim != spatial_ndim:
        raise ValueError(f"{name} dimension is not {spatial_ndim}, got {structure.ndim}")
    return (structure != 0).detach().to(device="cpu", dtype=torch.bool).contiguous()


def _binary_morphology_cuda_step(
    input: Tensor,
    structure: Tensor,
    border_value: bool,
    origin: tuple[int, ...],
    *,
    mode: str,
) -> Tensor:
    x = input if input.dtype == torch.bool and input.is_contiguous() else (input != 0).contiguous()
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
    spatial_ndim = validate_bcs_input(input)
    validate_output(input, output)

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

            old = x
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
    """Erode binary objects in a batched CUDA tensor

    A value is foreground when it is nonzero. The operation supports one to
    eight spatial dimensions and always treats the first two dimensions as
    batch and channel. When ``iterations`` is less than one, erosion continues
    until the result no longer changes.

    Args:
        input (torch.Tensor): CUDA tensor with shape ``(B, C, Spatial...)``.
        structure (torch.Tensor, optional): Spatial structuring element. Nonzero
            entries participate in the neighborhood. If ``None``, uses
            :func:`generate_binary_structure` with connectivity ``1``.
        iterations (int): Number of sequential erosions. Values less than ``1``
            repeat the operation until convergence.
        mask (torch.Tensor, optional): Tensor with the same shape as ``input``.
            Only locations where the mask is nonzero may change.
        output (torch.Tensor, optional): Preallocated tensor with the same shape
            and device as ``input``. When supplied, receives and is returned as
            the result.
        border_value (bool): Value used outside the input boundary.
        origin (int or tuple[int, ...]): Offset of the structuring-element
            anchor. A scalar is applied to every spatial dimension.

    Returns:
        torch.Tensor: Boolean erosion result, or ``output`` when it is supplied.

    Example:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.ones((1, 1, 5, 5), device="cuda")
        >>> y = tm.binary_erosion(x, structure=torch.ones(3, 3))
        >>> y.dtype, y.sum().item()
        (torch.bool, 9)
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
    """Dilate binary objects in a batched CUDA tensor

    A value is foreground when it is nonzero. The operation supports one to
    eight spatial dimensions and always treats the first two dimensions as
    batch and channel. When ``iterations`` is less than one, dilation continues
    until the result no longer changes.

    Args:
        input (torch.Tensor): CUDA tensor with shape ``(B, C, Spatial...)``.
        structure (torch.Tensor, optional): Spatial structuring element. Nonzero
            entries participate in the neighborhood. If ``None``, uses
            :func:`generate_binary_structure` with connectivity ``1``.
        iterations (int): Number of sequential dilations. Values less than ``1``
            repeat the operation until convergence.
        mask (torch.Tensor, optional): Tensor with the same shape as ``input``.
            Only locations where the mask is nonzero may change.
        output (torch.Tensor, optional): Preallocated tensor with the same shape
            and device as ``input``. When supplied, receives and is returned as
            the result.
        border_value (bool): Value used outside the input boundary.
        origin (int or tuple[int, ...]): Offset of the structuring-element
            anchor. A scalar is applied to every spatial dimension.

    Returns:
        torch.Tensor: Boolean dilation result, or ``output`` when it is supplied.

    Example:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.zeros((1, 1, 5, 5), device="cuda")
        >>> x[0, 0, 2, 2] = 1
        >>> y = tm.binary_dilation(x, structure=torch.ones(3, 3))
        >>> y.dtype, y.sum().item()
        (torch.bool, 9)
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


def binary_propagation(
    input: Tensor,
    structure: Tensor | None = None,
    mask: Tensor | None = None,
    output: Tensor | None = None,
    border_value: bool = False,
    origin: int | tuple[int, ...] = 0,
) -> Tensor:
    """Propagate a binary seed through a mask until convergence

    This repeatedly dilates ``input`` while allowing changes only where
    ``mask`` is nonzero. It is useful for morphological reconstruction.

    Args:
        input (torch.Tensor): Binary seed CUDA tensor with shape
            ``(B, C, Spatial...)``; nonzero values are foreground.
        structure (torch.Tensor, optional): Spatial structuring element. If
            ``None``, uses connectivity ``1``.
        mask (torch.Tensor, optional): Tensor with the same shape as ``input``.
            Only nonzero mask locations may change. If ``None``, propagation is
            unconstrained.
        output (torch.Tensor, optional): Preallocated result tensor with the
            same shape and device as ``input``.
        border_value (bool): Value used outside the input boundary.
        origin (int or tuple[int, ...]): Structuring-element anchor offset.

    Returns:
        torch.Tensor: Boolean converged result, or ``output`` when supplied.

    Example:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> seed = torch.zeros((1, 1, 5, 5), device="cuda")
        >>> seed[0, 0, 2, 2] = 1
        >>> mask = torch.ones_like(seed)
        >>> tm.binary_propagation(seed, mask=mask).all().item()
        True
        ```
    """
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
    """Fill holes enclosed by binary objects

    Background connected to the tensor boundary is reconstructed and removed;
    enclosed background regions become foreground.

    Args:
        input (torch.Tensor): CUDA tensor with shape ``(B, C, Spatial...)``;
            nonzero values are foreground.
        structure (torch.Tensor, optional): Spatial structuring element used to
            determine background connectivity. If ``None``, uses connectivity
            ``1``.
        output (torch.Tensor, optional): Preallocated result tensor with the
            same shape and device as ``input``.
        origin (int or tuple[int, ...]): Structuring-element anchor offset.

    Returns:
        torch.Tensor: Boolean tensor with enclosed holes filled, or ``output``.

    Example:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.ones((1, 1, 5, 5), device="cuda")
        >>> x[0, 0, 2, 2] = 0
        >>> tm.binary_fill_holes(x)[0, 0, 2, 2].item()
        True
        ```
    """
    validate_bcs_input(input)
    validate_output(input, output)

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
    """Find binary configurations with the hit-or-miss transform

    A location matches when ``structure1`` fits the foreground and
    ``structure2`` fits the background at the same anchor.

    Args:
        input (torch.Tensor): CUDA tensor with shape ``(B, C, Spatial...)``;
            nonzero values are foreground.
        structure1 (torch.Tensor, optional): Foreground structuring element. If
            ``None``, uses connectivity ``1``.
        structure2 (torch.Tensor, optional): Background structuring element. If
            ``None``, uses the logical complement of ``structure1``.
        output (torch.Tensor, optional): Preallocated result tensor with the
            same shape and device as ``input``.
        origin1 (int or tuple[int, ...]): Anchor offset for ``structure1``.
        origin2 (int or tuple[int, ...], optional): Anchor offset for
            ``structure2``. If ``None``, uses ``origin1``.

    Returns:
        torch.Tensor: Boolean tensor marking matching locations, or ``output``.

    Example:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.zeros((1, 1, 3, 3), device="cuda")
        >>> x[0, 0, 1, 1] = 1
        >>> hit = tm.binary_hit_or_miss(x, structure1=torch.ones(1, 1))
        >>> hit[0, 0, 1, 1].item()
        True
        ```
    """
    spatial_ndim = validate_bcs_input(input)
    validate_output(input, output)
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
    """Open binary objects by erosion followed by dilation

    Opening removes foreground features that cannot contain the structuring
    element while largely preserving larger objects.

    Args:
        input (torch.Tensor): CUDA tensor with shape ``(B, C, Spatial...)``;
            nonzero values are foreground.
        structure (torch.Tensor, optional): Spatial structuring element. If
            ``None``, uses connectivity ``1``.
        iterations (int): Number of erosions followed by the same number of
            dilations. Values less than ``1`` iterate each stage to convergence.
        mask (torch.Tensor, optional): Same-shaped tensor whose nonzero entries
            identify locations that may change.
        output (torch.Tensor, optional): Preallocated result tensor with the
            same shape and device as ``input``.
        border_value (bool): Value used outside the input boundary.
        origin (int or tuple[int, ...]): Structuring-element anchor offset.

    Returns:
        torch.Tensor: Boolean opened tensor, or ``output`` when supplied.

    Example:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.zeros((1, 1, 5, 5), device="cuda")
        >>> x[0, 0, 1:4, 1:4] = 1
        >>> x[0, 0, 0, 0] = 1
        >>> tm.binary_opening(x).sum().item()
        5
        ```
    """
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
    """Close binary objects by dilation followed by erosion

    Closing fills small gaps and joins nearby foreground regions while largely
    preserving the extent of larger objects.

    Args:
        input (torch.Tensor): CUDA tensor with shape ``(B, C, Spatial...)``;
            nonzero values are foreground.
        structure (torch.Tensor, optional): Spatial structuring element. If
            ``None``, uses connectivity ``1``.
        iterations (int): Number of dilations followed by the same number of
            erosions. Values less than ``1`` iterate each stage to convergence.
        mask (torch.Tensor, optional): Same-shaped tensor whose nonzero entries
            identify locations that may change.
        output (torch.Tensor, optional): Preallocated result tensor with the
            same shape and device as ``input``.
        border_value (bool): Value used outside the input boundary.
        origin (int or tuple[int, ...]): Structuring-element anchor offset.

    Returns:
        torch.Tensor: Boolean closed tensor, or ``output`` when supplied.

    Example:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.zeros((1, 1, 5, 5), device="cuda")
        >>> x[0, 0, 2, 1] = x[0, 0, 2, 3] = 1
        >>> tm.binary_closing(x)[0, 0, 2, 2].item()
        True
        ```
    """
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
