import math
from collections.abc import Sequence

from torch import Tensor

from . import _C
from ._validation import validate_bcs_input, validate_output


def _normalize_sampling(
    sampling: float | Sequence[float] | None,
    spatial_ndim: int,
) -> list[float]:
    if sampling is None:
        values = [1.0] * spatial_ndim
    elif isinstance(sampling, (int, float)):
        values = [float(sampling)] * spatial_ndim
    else:
        values = [float(value) for value in sampling]
        if len(values) == 1:
            values *= spatial_ndim
        elif len(values) != spatial_ndim:
            raise ValueError(f"sampling must have length 1 or {spatial_ndim}, got {len(values)}")

    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("sampling values must be finite and greater than zero")
    return values


def _prepare_distance_transform(
    input: Tensor,
    return_distances: bool,
    return_indices: bool,
    distances: Tensor | None,
    indices: Tensor | None,
) -> tuple[int, bool, bool]:
    spatial_ndim = validate_bcs_input(input)
    validate_output(input, distances, name="distances")
    validate_output(input, indices, (spatial_ndim, *input.shape), name="indices")

    return_distances = return_distances or distances is not None
    return_indices = return_indices or indices is not None
    if not return_distances and not return_indices:
        raise ValueError("At least one distance transform output must be requested.")
    return spatial_ndim, return_distances, return_indices


def _finish_distance_transform(
    raw_distances: Tensor | None,
    raw_indices: Tensor | None,
    return_distances: bool,
    return_indices: bool,
    distances: Tensor | None,
    indices: Tensor | None,
) -> Tensor | tuple[Tensor, Tensor] | None:
    if distances is not None and raw_distances is not None:
        distances.copy_(raw_distances)
    if indices is not None and raw_indices is not None:
        indices.copy_(raw_indices)

    returned_distances = return_distances and distances is None
    returned_indices = return_indices and indices is None
    if returned_distances and returned_indices:
        return raw_distances, raw_indices
    if returned_distances:
        return raw_distances
    if returned_indices:
        return raw_indices
    return None


def euclidean_distance_transform(
    input: Tensor,
    sampling: float | Sequence[float] | None = None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances: Tensor | None = None,
    indices: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor] | None:
    """Euclidean distance transform for (B, C, Spatial...) CUDA tensors."""
    spatial_ndim, return_distances, return_indices = _prepare_distance_transform(
        input,
        return_distances,
        return_indices,
        distances,
        indices,
    )
    normalized_sampling = _normalize_sampling(sampling, spatial_ndim)
    raw_distances, raw_indices = _C.edt_cuda(
        input.float().contiguous(),
        normalized_sampling,
        return_distances,
        return_indices,
    )
    return _finish_distance_transform(
        raw_distances,
        raw_indices,
        return_distances,
        return_indices,
        distances,
        indices,
    )


def chamfer_distance_transform(
    input: Tensor,
    metric: str = "chessboard",
    return_distances: bool = True,
    return_indices: bool = False,
    distances: Tensor | None = None,
    indices: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor] | None:
    """Chamfer distance transform for (B, C, Spatial...) CUDA tensors."""
    _, return_distances, return_indices = _prepare_distance_transform(
        input,
        return_distances,
        return_indices,
        distances,
        indices,
    )
    metric = {"cityblock": "taxicab", "manhattan": "taxicab"}.get(metric, metric)
    if metric not in {"chessboard", "taxicab"}:
        raise ValueError("metric must be 'chessboard', 'taxicab', 'cityblock', or 'manhattan'.")

    raw_distances, raw_indices = _C.cdt_cuda(
        input.float().contiguous(),
        metric,
        return_distances,
        return_indices,
    )
    return _finish_distance_transform(
        raw_distances,
        raw_indices,
        return_distances,
        return_indices,
        distances,
        indices,
    )


def brute_force_distance_transform(
    input: Tensor,
    metric: str = "euclidean",
    sampling: float | Sequence[float] | None = None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances: Tensor | None = None,
    indices: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor] | None:
    """Brute-force distance transform for (B, C, Spatial...) CUDA tensors."""
    spatial_ndim, return_distances, return_indices = _prepare_distance_transform(
        input,
        return_distances,
        return_indices,
        distances,
        indices,
    )
    if metric not in {"euclidean", "taxicab", "chessboard"}:
        raise ValueError("metric must be 'euclidean', 'taxicab', or 'chessboard'.")

    raw_distances, raw_indices = _C.bfdt_cuda(
        input.float().contiguous(),
        metric,
        _normalize_sampling(sampling, spatial_ndim),
        return_distances,
        return_indices,
    )
    return _finish_distance_transform(
        raw_distances,
        raw_indices,
        return_distances,
        return_indices,
        distances,
        indices,
    )
