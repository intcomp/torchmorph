from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


def _to_bool_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Convert an input value into a boolean PyTorch tensor.

    This helper function ensures that the input is represented as a
    `torch.bool` tensor, which is the internal format required by
    binary morphological operations (e.g., dilation/erosion).

    Behavior:
    - If `x` is not already a tensor, it is converted using `torch.tensor(x)`.
    - Non-zero values become `True`; zero values become `False`.

    Args:
        x (torch.Tensor or array-like):
            Input data. May be a Python list, scalar, NumPy array, or torch.Tensor.

    Returns:
        torch.Tensor (dtype=torch.bool):
            Boolean tensor where each element is `True` if corresponding input value
            is non-zero, otherwise `False`.

    Examples:
        >>> _to_bool_tensor([0, 1, 2])
        tensor([False, True, True])

        >>> _to_bool_tensor(torch.tensor([3.0, 0.0]))
        tensor([True, False])
    """
    # If x is not a tensor yet (e.g., list, numpy array, int, float), convert to tensor.
    if not torch.is_tensor(x):
        x = torch.tensor(x)

    # Convert input tensor into boolean by checking non-zero status.
    # Non-zero -> True, zero -> False.
    return x != 0


def _normalize_structure(structure: Optional[torch.Tensor], ndim: int) -> torch.Tensor:
    """
    Normalize a structuring element into a boolean tensor with the correct
    number of spatial dimensions.

    This utility function standardizes user-provided structuring elements
    for binary morphological operations (e.g., dilation and erosion).

    Behavior:
    1. If `structure` is None, a default full-connectivity structuring
       element of shape (3, 3, ..., 3) with `ndim` dimensions is created.
       This matches the default behavior of scipy.ndimage morphology.
    2. If `structure` is provided, it is converted into a boolean tensor,
       where non-zero values are treated as True.
    3. The dimensionality of the structuring element is strictly checked
       to ensure it matches the spatial dimensionality of the input.
       A mismatch indicates an invalid morphological definition and
       raises a ValueError.

    Args:
        structure (Optional[torch.Tensor]):
            Structuring element defining the neighborhood for morphology.
            If None, a full (3,) * ndim boolean structure is used.
        ndim (int):
            Number of spatial dimensions of the input (e.g., 2 for H×W,
            3 for D×H×W). Batch and channel dimensions are excluded.

    Returns:
        torch.Tensor (dtype=torch.bool):
            An `ndim`-dimensional boolean tensor representing the normalized
            structuring element.

    Raises:
        ValueError:
            If the provided structuring element does not have exactly
            `ndim` dimensions.

    Notes:
        - This function does not enforce any particular kernel size other
          than dimensionality; arbitrary shapes are allowed.
        - Channel and batch dimensions are intentionally not supported
          for structuring elements, as morphology is defined purely in
          spatial dimensions.

    Examples:
        >>> _normalize_structure(None, ndim=2)
        tensor([[True, True, True],
                [True, True, True],
                [True, True, True]])

        >>> _normalize_structure([[0, 1, 0],
        ...                       [1, 1, 1],
        ...                       [0, 1, 0]], ndim=2)
        tensor([[False, True, False],
                [ True, True, True],
                [False, True, False]])
    """
    # Case 1: No structuring element provided by the user.
    # Use a default full-connectivity neighborhood of size 3 in each
    # spatial dimension (e.g., 3×3 for 2D, 3×3×3 for 3D).
    if structure is None:
        shape = (3,) * ndim
        return torch.ones(shape, dtype=torch.bool)

    # Case 2: A structuring element is provided.
    # Convert it to a boolean tensor so that non-zero values indicate
    # active neighbors and zero values are ignored.
    st = _to_bool_tensor(structure)

    # Validate dimensionality: the structuring element must have the same
    # number of dimensions as the spatial dimensions of the input tensor.
    if st.ndim != ndim:
        raise ValueError(f"structure must be {ndim}-D (got {st.ndim}-D)")

    # Return the normalized boolean structuring element.
    return st


def _origin_to_tuple(
    origin: Union[int, Sequence[int], Tuple[int, ...]], ndim: int
) -> Tuple[int, ...]:
    """
    Normalize the `origin` argument into an ndim-length tuple.

    The origin defines the anchor point of the structuring element,
    consistent with SciPy's definition.

    Args:
        origin (int or sequence of int):
            If an int is given, it is broadcast to all spatial dimensions.
            If a sequence is given, its length must match `ndim`.
        ndim (int):
            Number of spatial dimensions.

    Returns:
        Tuple[int, ...]:
            Origin offset per spatial dimension.
    """
    # If a scalar is given, replicate it across all dimensions.
    if isinstance(origin, int):
        return tuple([origin] * ndim)

    # Otherwise, ensure it is a tuple with correct dimensionality.
    origin = tuple(origin)
    if len(origin) != ndim:
        raise ValueError("origin must match spatial ndim")

    return origin


def _pad_for_kernel(
    kernel_shape: Sequence[int], origin: Sequence[int]
) -> Tuple[Tuple[int, int], ...]:
    """
    Compute per-dimension padding sizes required to keep output shape
    identical to input shape after convolution.

    This takes into account the kernel size and the origin offset.

    Returns:
        Tuple of (pad_before, pad_after) for each spatial dimension.
    """
    pads = []
    for k, o in zip(kernel_shape, origin):
        # Default symmetric padding would be k//2,
        # but origin shifts the effective center.
        pad_before = k // 2 - o
        pad_after = k - 1 - pad_before

        # Padding must be non-negative.
        pad_before = max(pad_before, 0)
        pad_after = max(pad_after, 0)

        pads.append((pad_before, pad_after))
    return tuple(pads)


def _make_padding_tuple_for_Fpad(pads: Tuple[Tuple[int, int], ...]) -> Tuple[int, ...]:
    """
    Convert per-dimension padding into the flattened format required
    by torch.nn.functional.pad.

    PyTorch expects padding in reverse order:
        (pad_last_dim_left, pad_last_dim_right, ..., pad_first_dim_left, pad_first_dim_right)
    """
    flat = []
    for pb, pa in reversed(pads):
        flat.append(pb)
        flat.append(pa)
    return tuple(flat)


def _conv_nd(x: torch.Tensor, kernel: torch.Tensor, ndim: int) -> torch.Tensor:
    """
    Dispatch N-dimensional convolution based on spatial dimensionality.

    Args:
        x (torch.Tensor):
            Input tensor of shape (B*C, 1, *spatial_dims)
        kernel (torch.Tensor):
            Structuring element kernel.
        ndim (int):
            Number of spatial dimensions (1, 2, or 3).

    Returns:
        torch.Tensor:
            Convolution result.
    """
    # Convert kernel into convolution weight:
    # shape -> (out_channels=1, in_channels=1, *kernel_shape)
    weight = kernel.to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)

    if ndim == 1:
        return F.conv1d(x, weight)
    elif ndim == 2:
        return F.conv2d(x, weight)
    elif ndim == 3:
        return F.conv3d(x, weight)
    else:
        raise NotImplementedError("Only supports 1D/2D/3D")


def _morph_op(
    input_tensor: torch.Tensor,
    structure: Optional[torch.Tensor],
    iterations: int,
    origin: Union[int, Sequence[int]],
    border_value: int,
    mode: str,
) -> torch.Tensor:
    """
    Core implementation of binary dilation and erosion using convolution.

    This function supports batch and channel dimensions by flattening
    (B, C) into a single dimension and applying morphology independently
    per channel.

    Args:
        input_tensor (torch.Tensor):
            Input binary tensor.
        structure (Optional[torch.Tensor]):
            Structuring element.
        iterations (int):
            Number of times to apply the operation.
        origin:
            Origin offset of the structuring element.
        border_value (int):
            Value used for padding outside image boundaries.
        mode (str):
            Either 'dilation' or 'erosion'.

    Returns:
        torch.Tensor (dtype=torch.bool):
            Output binary tensor.
    """
    if mode not in ('dilation', 'erosion'):
        raise ValueError("mode must be 'dilation' or 'erosion'")

    x = input_tensor
    if not torch.is_tensor(x):
        x = torch.tensor(x)

    # Convert input to boolean (binary morphology).
    x_bool = x != 0
    # Supported input shapes:
    # (H,W), (C,H,W), (B,C,H,W), (B,C,D,H,W)
    full_ndim = x_bool.ndim

    if full_ndim < 2:
        raise NotImplementedError("Need at least 2D (H,W)")
    if full_ndim > 5:
        raise NotImplementedError("Only supports up to 5D (B,C,D,H,W)")
    spatial_ndim = full_ndim - 2  # remove (B,C)
    if not (1 <= spatial_ndim <= 3):
        raise NotImplementedError("Supports 1D/2D/3D spatial dims")

    B, C = x_bool.shape[0], x_bool.shape[1]
    spatial_shape = x_bool.shape[2:]
    st = _normalize_structure(structure, spatial_ndim)
    origin_t = _origin_to_tuple(origin, spatial_ndim)

    k_sum = st.sum().item()
    kernel = st.to(torch.float32)

    # Apply origin shift by rolling kernel.
    for axis, o in enumerate(origin_t):
        if o != 0:
            kernel = torch.roll(kernel, shifts=-o, dims=axis)
    pads = _pad_for_kernel(kernel.shape, origin_t)
    pad_tuple = _make_padding_tuple_for_Fpad(pads)

    cur = x_bool.to(torch.float32)

    # Flatten (B,C) -> (B*C,1)
    cur = cur.view(B * C, 1, *spatial_shape)
    for _ in range(max(1, iterations)):
        x_pad = F.pad(cur, pad_tuple, value=float(border_value))
        conv_res = _conv_nd(x_pad, kernel, spatial_ndim)

        if mode == 'dilation':
            # Any overlap -> True
            cur = (conv_res > 0).to(torch.float32)
        else:
            # Full overlap -> True
            if k_sum == 0:
                cur = torch.ones_like(cur)
            else:
                cur = (conv_res >= (k_sum - 1e-6)).to(torch.float32)
    out = cur.view(B, C, *spatial_shape)
    return out.to(torch.bool)


def binary_dilation(input_tensor, structure=None, iterations=1, origin=0, border_value=0):
    return _morph_op(input_tensor, structure, iterations, origin, border_value, mode="dilation")


def binary_erosion(input_tensor, structure=None, iterations=1, origin=0, border_value=0):
    return _morph_op(input_tensor, structure, iterations, origin, border_value, mode="erosion")
