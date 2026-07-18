from collections.abc import Sequence

from torch import Tensor

MAX_SPATIAL_NDIM = 8


def validate_bcs_input(input: Tensor) -> int:
    """Validate a ``(B, C, Spatial...)`` CUDA input and return its spatial ndim."""
    if not input.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    if input.ndim < 3:
        raise ValueError(
            f"Input must be (B, C, Spatial...) with at least 3 dimensions, got {input.shape}."
        )
    spatial_ndim = input.ndim - 2
    if spatial_ndim > MAX_SPATIAL_NDIM:
        raise ValueError(f"Input spatial dimensions must be in range 1 to 8, got {spatial_ndim}.")
    if input.numel() == 0:
        raise ValueError(f"Invalid input: empty tensor with shape {input.shape}.")
    return spatial_ndim


def validate_output(
    input: Tensor,
    output: Tensor | None,
    expected_shape: Sequence[int] | None = None,
    name: str = "output",
) -> None:
    """Validate that an optional pre-allocated output matches shape and device."""
    if output is None:
        return
    expected = input.shape if expected_shape is None else expected_shape
    if tuple(output.shape) != tuple(expected):
        raise ValueError(
            f"{name} shape {tuple(output.shape)} must match expected shape {tuple(expected)}"
        )
    if output.device != input.device:
        raise ValueError(
            f"{name} must be on the same device as input, "
            f"got {output.device} and {input.device}"
        )
