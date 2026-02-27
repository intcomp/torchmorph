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
    """Exact Euclidean Distance Transform (EDT) using Felzenszwalb algorithm.

    Args:
        input: Binary input tensor (0 = background, non-zero = foreground).
               Must be in (B, C, Spatial...) format where Spatial can be 1D, 2D, or 3D.
               For single images, use unsqueeze to add batch and channel dims.
        sampling: Spacing of elements along each spatial dimension. If a single
                  number, the spacing is uniform in all spatial dimensions. If a
                  sequence, it must match the number of spatial dimensions.
                  Default is None (unit spacing for all spatial dimensions).
                  Note: When sampling is not unit spacing, only "exact" algorithm is used.
        return_distances: Whether to calculate the distance transform.
                          Default is True.
        return_indices: Whether to calculate the feature transform (indices
                        of closest background element). Default is False.
        distances: Optional output tensor for distances. If provided, must have
                   the same shape as input. If None and return_distances is True,
                   a new tensor will be created and returned.
        indices: Optional output tensor for indices. If provided, must have shape
                 (spatial_ndim, ...) where ... matches input shape. If None and
                 return_indices is True, a new tensor will be created and returned.
        algorithm: Algorithm to use for distance transform. Options:
                   - "exact": Use Felzenszwalb's exact algorithm (default).
                   - "jfa": Use Jump Flooding Algorithm (fast but approximate).
                            Only available for 2D/3D with unit sampling.
                   - "auto": Automatically choose based on input (uses JFA when
                            applicable, otherwise exact).

    Returns:
        Depending on return_distances, return_indices, and whether output tensors
        are provided:
            - Returns distance tensor only when return_distances=True and distances=None
            - Returns indices tensor only when return_indices=True and indices=None
            - Returns tuple of (distances, indices) when both conditions above are met
            - Returns None if output tensors are provided for all requested outputs

    Example:
        >>> import torchmorph as tm
        >>> # 2D image: (B, C, H, W)
        >>> x = torch.zeros(1, 1, 64, 64, device='cuda')
        >>> x[0, 0, 10:20, 10:20] = 1
        >>> dist = tm.euclidean_distance_transform(x)
        >>> dist, indices = tm.euclidean_distance_transform(x, return_indices=True)
        >>> dist = tm.euclidean_distance_transform(x, sampling=[0.5, 1.0])
        >>> # Using JFA algorithm (faster for large images)
        >>> dist = tm.euclidean_distance_transform(x, algorithm="jfa")
        >>> # Using pre-allocated output tensors
        >>> dist_out = torch.empty_like(x)
        >>> tm.euclidean_distance_transform(x, distances=dist_out)  # Returns None, fills dist_out
        >>> # 3D volume: (B, C, D, H, W)
        >>> x_3d = torch.zeros(2, 1, 32, 64, 64, device='cuda')
        >>> dist_3d = tm.euclidean_distance_transform(x_3d, sampling=[2.0, 1.0, 1.0])
    """
    if not input.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    if input.ndim < 3:
        raise ValueError(
            f"Input must be (B, C, ) format with at least 3 dimensions, got {input.shape}. "
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
    """Chamfer Distance Transform (CDT).

    Calculates the distance transform of the input using a chamfer metric.
    The input is treated as a binary image where non-zero values are foreground
    and zero values are background. Distances are computed from each foreground
    pixel to the nearest background pixel.

    Args:
        input: Binary input tensor (0 = background, non-zero = foreground).
               Must be in (B, C, H, W) or (B, C, D, H, W) format for batch processing,
               or (H, W) / (D, H, W) for single images.
        metric: Distance metric to use:
                - "chessboard": L-infinity norm (default). Also known as Chebyshev distance.
                - "taxicab": L1 norm. Also known as Manhattan or city-block distance.
                - "cityblock": Alias for "taxicab".
                - "manhattan": Alias for "taxicab".
        return_distances: Whether to calculate the distance transform. Default is True.
        return_indices: Whether to calculate the feature transform (indices of closest
                        background element). Default is False.
        distances: Optional output tensor for distances. If provided, must have
                   the same shape as input. If None and return_distances is True,
                   a new tensor will be created.
        indices: Optional output tensor for indices. If provided, must have shape
                 (..., ndim) where ... matches input shape. If None and return_indices
                 is True, a new tensor will be created.

    Returns:
        Depending on return_distances, return_indices, and whether output tensors
        are provided:
            - Returns distance tensor only when return_distances=True and distances=None
            - Returns indices tensor only when return_indices=True and indices=None
            - Returns tuple of (distances, indices) when both conditions above are met
            - Returns None if output tensors are provided for all requested outputs

    Example:
        >>> import torchmorph as tm
        >>> # 2D image with batch: (B, C, H, W)
        >>> x = torch.zeros(1, 1, 64, 64, device='cuda')
        >>> x[0, 0, 10:20, 10:20] = 1
        >>> dist = tm.chamfer_distance_transform(x)  # chessboard by default
        >>> dist = tm.chamfer_distance_transform(x, metric='taxicab')
        >>> dist, indices = tm.chamfer_distance_transform(x, return_indices=True)
        >>> # Using pre-allocated output tensors
        >>> dist_out = torch.empty_like(x)
        >>> tm.chamfer_distance_transform(x, distances=dist_out)  # Returns None, fills dist_out
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
