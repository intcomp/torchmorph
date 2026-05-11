from typing import Optional, Sequence, Tuple, Union

import torch

from torchmorph import _C


def euclidean_distance_transform(
    input: torch.Tensor,
    sampling: Optional[Union[float, Sequence[float]]] = None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    algorithm: str = "exact",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None]:
    """Compute the Euclidean distance transform of an N-dimensional binary tensor

    The exact algorithm treats ``input`` as ``(B, C, *spatial)`` and processes
    each batch/channel slice independently. Zero values are background sites;
    non-zero values are foreground sites whose distance to the nearest
    background site is computed.

    Args:
        input (torch.Tensor): CUDA tensor with shape ``(B, C, *spatial)`` and at
            least one spatial dimension. Add batch and channel dimensions for a
            single image or volume.
        sampling (Optional[float | Sequence[float]]): Physical spacing for the
            spatial dimensions. ``None`` uses unit spacing, a scalar is
            broadcast to all spatial dimensions, and a sequence must have either
            one value or one value per spatial dimension.
        return_distances (bool): Whether to compute distance values.
        return_indices (bool): Whether to compute nearest-background coordinates.
        distances (Optional[torch.Tensor]): Optional preallocated distance
            output with the same shape as ``input``. Provided tensors are filled
            in-place and omitted from the return value.
        indices (Optional[torch.Tensor]): Optional preallocated index output
            with shape ``(spatial_ndim, *input.shape)``. Provided tensors are
            filled in-place and omitted from the return value.
        algorithm (str): Algorithm selector. ``"exact"`` is the standard
            N-dimensional path. ``"jfa"`` requests the approximate Jump
            Flooding Algorithm for unit-spaced 2D/3D distance-only use and
            otherwise falls back to exact. ``"auto"`` may choose JFA for
            eligible 2D unit-spaced distance computations. Use ``"exact"``
            when requesting indices, using multi-channel inputs, or relying on
            documented ``(B, C, *spatial)`` batch/channel semantics.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None: Requested
        outputs that were not supplied through ``distances`` or ``indices``.
        For ``algorithm="exact"``, distances have shape ``input.shape`` and
        indices have shape ``(spatial_ndim, *input.shape)``. Returns ``None``
        when all requested outputs were preallocated.

    Examples:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.zeros(1, 1, 4, 4, 4, 4, device="cuda")
        >>> x[..., 1, 1, 1, 1] = 1
        >>> dist = tm.euclidean_distance_transform(x, algorithm="exact")
        >>> dist.shape
        torch.Size([1, 1, 4, 4, 4, 4])
        >>> dist, idx = tm.euclidean_distance_transform(x, algorithm="exact", return_indices=True)
        >>> idx.shape
        torch.Size([4, 1, 1, 4, 4, 4, 4])
        ```
    """
    if not input.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    if input.ndim < 3:
        raise ValueError(
            f"Input must be (B, C, *spatial) format with at least 3 dimensions, got {input.shape}. "
            "For single images, use unsqueeze to add batch and channel dims."
        )
    if input.numel() == 0:
        raise ValueError(f"Invalid input: empty tensor with shape {input.shape}.")

    # Validate pre-allocated output tensors
    if distances is not None:
        if distances.shape != input.shape:
            raise ValueError(
                f"distances shape {distances.shape} must match input shape {input.shape}"
            )
        if not distances.is_cuda:
            raise ValueError("distances tensor must be on CUDA device.")
        return_distances = True

    if indices is not None:
        if not indices.is_cuda:
            raise ValueError("indices tensor must be on CUDA device.")
        return_indices = True

    if not return_distances and not return_indices:
        raise ValueError(
            "At least one of return_distances or return_indices must be True, "
            "or output tensors must be provided."
        )

    input = input.float().contiguous()
    total_ndim = input.ndim
    spatial_ndim = total_ndim - 2  # Exclude B and C dimensions

    # Process sampling parameter for spatial dimensions only
    if sampling is None:
        # Unit spacing for all spatial dimensions
        sampling_list = [1.0] * spatial_ndim
    elif isinstance(sampling, (int, float)):
        # Single value: same spacing for all spatial dimensions
        sampling_list = [float(sampling)] * spatial_ndim
    else:
        # Sequence: convert to list
        sampling_list = [float(s) for s in sampling]
        if len(sampling_list) == 1:
            # Single element list: broadcast to all spatial dimensions
            sampling_list = sampling_list * spatial_ndim
        elif len(sampling_list) != spatial_ndim:
            raise ValueError(
                f"sampling has {len(sampling_list)} but input {spatial_ndim}  dimensions "
                f"(input shape: {input.shape}, format: (B, C, Spatial...))"
            )

    # Call CUDA kernel - it handles batch dimensions based on sampling size
    raw_distances, raw_indices = _C.edt_cuda(
        input, sampling_list, return_distances, return_indices, algorithm
    )

    # Copy to pre-allocated tensors if provided
    if distances is not None and raw_distances is not None:
        distances.copy_(raw_distances)

    if indices is not None and raw_indices is not None:
        indices.copy_(raw_indices)

    # Return based on scipy convention:
    # Only return tensors that were NOT provided by the user
    return_dist_tensor = return_distances and distances is None
    return_idx_tensor = return_indices and indices is None

    if return_dist_tensor and return_idx_tensor:
        return raw_distances, raw_indices
    elif return_dist_tensor:
        return raw_distances
    elif return_idx_tensor:
        return raw_indices
    else:
        return None


def chamfer_distance_transform(
    input: torch.Tensor,
    metric: str = "chessboard",
    return_distances: bool = True,
    return_indices: bool = False,
    distances: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None]:
    """Compute the chamfer distance transform of an N-dimensional binary tensor

    The input is treated as ``(B, C, *spatial)`` and each batch/channel slice is
    processed independently. Zero values are background sites and non-zero
    values are foreground sites. The CUDA implementation supports 1D through
    16D spatial inputs.

    Args:
        input (torch.Tensor): CUDA tensor with shape ``(B, C, *spatial)`` and at
            least one spatial dimension.
        metric (str): Chamfer metric. ``"chessboard"`` computes L-infinity /
            Chebyshev distance. ``"taxicab"`` computes L1 / Manhattan distance.
            ``"cityblock"`` and ``"manhattan"`` are aliases for ``"taxicab"``.
        return_distances (bool): Whether to compute distance values.
        return_indices (bool): Whether to compute nearest-background coordinates.
        distances (Optional[torch.Tensor]): Optional preallocated distance
            output with the same shape as ``input``.
        indices (Optional[torch.Tensor]): Optional preallocated index output
            with shape ``(spatial_ndim, *input.shape)``.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None: Requested
        outputs that were not supplied through ``distances`` or ``indices``.
        Distances have shape ``input.shape`` and indices have shape
        ``(spatial_ndim, *input.shape)``.

    Examples:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.zeros(1, 1, 5, 5, 5, device="cuda")
        >>> x[..., 2, 2, 2] = 1
        >>> dist = tm.chamfer_distance_transform(x, metric="chessboard")
        >>> dist.shape
        torch.Size([1, 1, 5, 5, 5])
        >>> dist, idx = tm.chamfer_distance_transform(x, metric="taxicab", return_indices=True)
        >>> idx.shape
        torch.Size([3, 1, 1, 5, 5, 5])
        ```
    """
    if not input.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    if input.ndim < 2 or input.numel() == 0:
        raise ValueError(f"Invalid input dimension: {input.shape}.")

    # Normalize metric aliases
    if metric in ("cityblock", "manhattan"):
        metric = "taxicab"

    if metric not in ("chessboard", "taxicab"):
        raise ValueError("metric must be 'chessboard', 'taxicab', 'cityblock', or 'manhattan'.")
    if not return_distances and not return_indices:
        if distances is None and indices is None:
            raise ValueError(
                "At least one of return_distances or return_indices must be True, "
                "or output tensors must be provided."
            )

    input = input.float().contiguous()

    # Validate pre-allocated output tensors
    if distances is not None:
        if distances.shape != input.shape:
            raise ValueError(
                f"distances shape {distances.shape} must match input shape {input.shape}"
            )
        if not distances.is_cuda:
            raise ValueError("distances tensor must be on CUDA device.")
        return_distances = True

    if indices is not None:
        if not indices.is_cuda:
            raise ValueError("indices tensor must be on CUDA device.")
        return_indices = True

    # Call CUDA kernel
    raw_distances, raw_indices = _C.cdt_cuda(input, metric, return_distances, return_indices)

    # Copy to pre-allocated tensors if provided
    if distances is not None and raw_distances is not None:
        distances.copy_(raw_distances)

    if indices is not None and raw_indices is not None:
        indices.copy_(raw_indices)

    # Return based on scipy convention:
    # Only return tensors that were NOT provided by the user
    return_dist_tensor = return_distances and distances is None
    return_idx_tensor = return_indices and indices is None

    if return_dist_tensor and return_idx_tensor:
        return raw_distances, raw_indices
    elif return_dist_tensor:
        return raw_distances
    elif return_idx_tensor:
        return raw_indices
    else:
        return None


def brute_force_distance_transform(
    input: torch.Tensor,
    metric: str = "euclidean",
    sampling: Optional[Union[float, Sequence[float]]] = None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None]:
    """Compute the brute-force distance transform of an N-dimensional binary tensor

    The input is treated as ``(B, C, *spatial)`` and each batch/channel slice is
    processed independently. For every foreground site, the kernel compares all
    background sites and writes the minimum distance. The CUDA template dispatch
    supports 1D through 8D spatial inputs.

    Args:
        input (torch.Tensor): CUDA tensor with shape ``(B, C, *spatial)`` and at
            least one spatial dimension.
        metric (str): Distance metric. ``"euclidean"`` computes L2 distance,
            ``"taxicab"`` computes L1 / Manhattan distance, and
            ``"chessboard"`` computes L-infinity / Chebyshev distance.
        sampling (Optional[float | Sequence[float]]): Physical spacing for the
            spatial dimensions. ``None`` uses unit spacing, a scalar is
            broadcast to all spatial dimensions, and a sequence must have either
            one value or one value per spatial dimension.
        return_distances (bool): Whether to compute distance values.
        return_indices (bool): Whether to compute nearest-background coordinates.
        distances (Optional[torch.Tensor]): Optional preallocated distance
            output with the same shape as ``input``.
        indices (Optional[torch.Tensor]): Optional preallocated index output
            with shape ``(spatial_ndim, *input.shape)``.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None: Requested
        outputs that were not supplied through ``distances`` or ``indices``.
        Distances have shape ``input.shape`` and indices have shape
        ``(spatial_ndim, *input.shape)``.

    Examples:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> x = torch.zeros(1, 1, 3, 3, 3, 3, device="cuda")
        >>> x[..., 1, 1, 1, 1] = 1
        >>> dist = tm.brute_force_distance_transform(x, metric="euclidean")
        >>> dist.shape
        torch.Size([1, 1, 3, 3, 3, 3])
        >>> dist, idx = tm.brute_force_distance_transform(x, return_indices=True)
        >>> idx.shape
        torch.Size([4, 1, 1, 3, 3, 3, 3])
        ```
    """
    if not input.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    if input.ndim < 3:
        raise ValueError(
            f"Input must be (B, C, *spatial) format with at least 3 dimensions, got {input.shape}."
        )
    if input.numel() == 0:
        raise ValueError(f"Invalid input: empty tensor with shape {input.shape}.")

    if metric not in ("euclidean", "taxicab", "chessboard"):
        raise ValueError("metric must be 'euclidean', 'taxicab', or 'chessboard'.")

    # Validate pre-allocated output tensors
    if distances is not None:
        if distances.shape != input.shape:
            raise ValueError(
                f"distances shape {distances.shape} must match input shape {input.shape}"
            )
        if not distances.is_cuda:
            raise ValueError("distances tensor must be on CUDA device.")
        return_distances = True

    if indices is not None:
        if not indices.is_cuda:
            raise ValueError("indices tensor must be on CUDA device.")
        return_indices = True

    if not return_distances and not return_indices:
        raise ValueError(
            "At least one of return_distances or return_indices must be True, "
            "or output tensors must be provided."
        )

    input = input.float().contiguous()
    spatial_ndim = input.ndim - 2

    # Process sampling parameter
    if sampling is None:
        sampling_list = [1.0] * spatial_ndim
    elif isinstance(sampling, (int, float)):
        sampling_list = [float(sampling)] * spatial_ndim
    else:
        sampling_list = [float(s) for s in sampling]
        if len(sampling_list) == 1:
            sampling_list = sampling_list * spatial_ndim
        elif len(sampling_list) != spatial_ndim:
            raise ValueError(
                f"sampling has {len(sampling_list)} but input {spatial_ndim} dimensions."
            )

    # Call CUDA kernel
    raw_distances, raw_indices = _C.bfdt_cuda(
        input, metric, sampling_list, return_distances, return_indices
    )

    # Copy to pre-allocated tensors if provided
    if distances is not None and raw_distances is not None:
        distances.copy_(raw_distances)

    if indices is not None and raw_indices is not None:
        indices.copy_(raw_indices)

    # Return based on scipy convention
    return_dist_tensor = return_distances and distances is None
    return_idx_tensor = return_indices and indices is None

    if return_dist_tensor and return_idx_tensor:
        return raw_distances, raw_indices
    elif return_dist_tensor:
        return raw_distances
    elif return_idx_tensor:
        return raw_indices
    else:
        return None
