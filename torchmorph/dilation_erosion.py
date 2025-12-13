import torch
import torch.nn.functional as F
from typing import Optional, Union, Sequence, Tuple

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
    return (x != 0)


def _normalize_structure(structure: Optional[torch.Tensor], ndim: int) -> torch.Tensor:
    if structure is None:
        shape = (3,) * ndim
        return torch.ones(shape, dtype=torch.bool)
    st = _to_bool_tensor(structure)
    if st.ndim != ndim:
        raise ValueError(f"structure must be {ndim}-D (got {st.ndim}-D)")
    return st

def _origin_to_tuple(origin: Union[int, Sequence[int], Tuple[int,...]], ndim: int) -> Tuple[int,...]:
    if isinstance(origin, int):
        return tuple([origin] * ndim)
    origin = tuple(origin)
    if len(origin) != ndim:
        raise ValueError("origin must match spatial ndim")
    return origin

def _pad_for_kernel(kernel_shape: Sequence[int], origin: Sequence[int]) -> Tuple[Tuple[int,int], ...]:
    pads = []
    for k, o in zip(kernel_shape, origin):
        pad_before = k//2 - o
        pad_after  = k - 1 - pad_before
        pad_before = max(pad_before, 0)
        pad_after = max(pad_after, 0)
        pads.append((pad_before, pad_after))
    return tuple(pads)

def _make_padding_tuple_for_Fpad(pads: Tuple[Tuple[int,int], ...]) -> Tuple[int,...]:
    flat = []
    for pb, pa in reversed(pads):
        flat.append(pb)
        flat.append(pa)
    return tuple(flat)

def _conv_nd(x: torch.Tensor, kernel: torch.Tensor, ndim: int) -> torch.Tensor:
    weight = (
        kernel.to(dtype=x.dtype, device=x.device)
        .unsqueeze(0).unsqueeze(0)
    )
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
    mode: str
) -> torch.Tensor:

    if mode not in ('dilation', 'erosion'):
        raise ValueError("mode must be 'dilation' or 'erosion'")

    x = input_tensor
    if not torch.is_tensor(x):
        x = torch.tensor(x)

    x_bool = (x != 0)

    #  Support: (H,W), (C,H,W), (B,C,H,W), (B,C,D,H,W)
    full_ndim = x_bool.ndim

    if full_ndim < 2:
        raise NotImplementedError("Need at least 2D (H,W)")
    if full_ndim > 5:
        raise NotImplementedError("Only supports up to 5D (B,C,D,H,W)")

    # Spatial dims = last 1~3 dims
    spatial_ndim = full_ndim - 2     # remove (B,C)
    if not (1 <= spatial_ndim <= 3):
        raise NotImplementedError("Supports 1D/2D/3D spatial dims")

    B, C = x_bool.shape[0], x_bool.shape[1]
    spatial_shape = x_bool.shape[2:]

    # structure must match spatial dims
    st = _normalize_structure(structure, spatial_ndim)
    origin_t = _origin_to_tuple(origin, spatial_ndim)

    k_sum = st.sum().item()
    kernel = st.to(torch.float32)

    # apply origin shift
    for axis, o in enumerate(origin_t):
        if o != 0:
            kernel = torch.roll(kernel, shifts=-o, dims=axis)

    pads = _pad_for_kernel(kernel.shape, origin_t)
    pad_tuple = _make_padding_tuple_for_Fpad(pads)

    # cast to float
    cur = x_bool.to(torch.float32)

    # Now do B*C loops, because conv2d can't do dilation per-channel independently
    cur = cur.view(B*C, 1, *spatial_shape)

    for _ in range(max(1, iterations)):
        x_pad = F.pad(cur, pad_tuple, value=float(border_value))
        conv_res = _conv_nd(x_pad, kernel, spatial_ndim)
        conv_res = conv_res  # shape unchanged: (BC,1,H,W) or (BC,1,D,H,W)

        if mode == 'dilation':
            cur = (conv_res > 0).to(torch.float32)
        else:  # erosion
            if k_sum == 0:
                cur = torch.ones_like(cur)
            else:
                cur = (conv_res >= (k_sum - 1e-6)).to(torch.float32)

    # reshape back
    out = cur.view(B, C, *spatial_shape)
    return out.to(torch.bool)

def binary_dilation(input_tensor, structure=None, iterations=1, origin=0, border_value=0):
    return _morph_op(input_tensor, structure, iterations, origin, border_value, mode="dilation")

def binary_erosion(input_tensor, structure=None, iterations=1, origin=0, border_value=0):
    return _morph_op(input_tensor, structure, iterations, origin, border_value, mode="erosion")
