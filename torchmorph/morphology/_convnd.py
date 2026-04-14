import torch
from torch import Tensor

_NATIVE_CONVS = {
    1: torch.nn.functional.conv1d,
    2: torch.nn.functional.conv2d,
    3: torch.nn.functional.conv3d,
}


def conv_nd(x: Tensor, weight: Tensor) -> Tensor:
    """Apply correlation on tensors shaped as ``(N, C, Spatial...)``.

    This is a minimal private helper for ``torchmorph.morphology.binary``.
    Inputs are expected to already be padded, use channel-first layout, and
    convolve over the trailing spatial dimensions with stride=1, dilation=1,
    and groups=1.
    """

    num_spatial = x.ndim - 2
    if num_spatial < 1:
        raise ValueError(f"expected at least 1 spatial dim, got input ndim={x.ndim}")
    if weight.ndim != x.ndim:
        raise ValueError(
            f"expected weight ndim {x.ndim} to match input ndim {x.ndim}, got {weight.ndim}"
        )

    return _conv_core(x, weight)


def _conv_core(x: Tensor, weight: Tensor) -> Tensor:
    num_spatial = x.ndim - 2
    if num_spatial in _NATIVE_CONVS:
        return _NATIVE_CONVS[num_spatial](x, weight, bias=None, stride=1, padding=0, dilation=1)
    return _conv_recursive(x, weight)


def _conv_recursive(x: Tensor, weight: Tensor) -> Tensor:
    kernel_size = weight.shape[2]
    input_size = x.shape[2]
    output_size = input_size - kernel_size + 1
    batch_size = x.shape[0]

    accumulated = None
    for kernel_index in range(kernel_size):
        x_slice = x[:, :, kernel_index : kernel_index + output_size]
        flattened = x_slice.moveaxis(2, 1).reshape(
            batch_size * output_size,
            x.shape[1],
            *x_slice.shape[3:],
        )

        partial = _conv_core(flattened, weight[:, :, kernel_index])
        partial = partial.reshape(batch_size, output_size, partial.shape[1], *partial.shape[2:])
        partial = partial.moveaxis(2, 1)
        accumulated = partial if accumulated is None else accumulated + partial

    if accumulated is None:
        raise RuntimeError("recursive convolution produced no output")
    return accumulated
