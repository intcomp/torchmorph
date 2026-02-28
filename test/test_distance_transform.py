import numpy as np  # noqa: F401
import pytest
import torch
from scipy.ndimage import distance_transform_cdt as scipy_cdt
from scipy.ndimage import distance_transform_edt as scipy_edt  # noqa: F401

import torchmorph as tm  # noqa: F401


# ======================================================================
# EDT Helper functions
# ======================================================================
def batch_scipy_edt_with_indices(
    batch_numpy: np.ndarray,
    spatial_ndim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SciPy EDT and indices for a batch of arrays.

    Args:
        batch_numpy: Input array with shape (batch..., *spatial_shape)
        spatial_ndim: Number of spatial dimensions
    """
    dist_results: list[np.ndarray] = []
    indices_results: list[np.ndarray] = []

    # Compute batch shape
    batch_shape = batch_numpy.shape[:-spatial_ndim] if spatial_ndim > 0 else ()
    spatial_shape = batch_numpy.shape[-spatial_ndim:] if spatial_ndim > 0 else batch_numpy.shape

    # Flatten batch dimensions
    if len(batch_shape) > 0:
        batch_size = int(np.prod(batch_shape))
        flat_input = batch_numpy.reshape(batch_size, *spatial_shape)
    else:
        batch_size = 1
        flat_input = batch_numpy[np.newaxis, ...]

    for sample in flat_input:
        dist, indices = scipy_edt(
            sample,
            return_indices=True,
            return_distances=True,
        )
        dist_results.append(dist)
        indices_results.append(indices)

    output_dist = np.stack(dist_results, axis=0)
    output_indices = np.stack(indices_results, axis=0)

    # Reshape back to original batch shape
    if len(batch_shape) > 0:
        output_dist = output_dist.reshape(*batch_shape, *spatial_shape)
        output_indices = output_indices.reshape(*batch_shape, spatial_ndim, *spatial_shape)
    else:
        output_dist = output_dist[0]
        output_indices = output_indices[0]

    return output_dist, output_indices


def batch_scipy_edt_with_sampling(
    batch_numpy: np.ndarray,
    spatial_ndim: int,
    sampling: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SciPy EDT with sampling for a batch of arrays.

    Args:
        batch_numpy: Input array with shape (batch..., *spatial_shape)
        spatial_ndim: Number of spatial dimensions
        sampling: Spacing for each spatial dimension
    """
    dist_results: list[np.ndarray] = []
    indices_results: list[np.ndarray] = []

    batch_shape = batch_numpy.shape[:-spatial_ndim] if spatial_ndim > 0 else ()
    spatial_shape = batch_numpy.shape[-spatial_ndim:] if spatial_ndim > 0 else batch_numpy.shape

    if len(batch_shape) > 0:
        batch_size = int(np.prod(batch_shape))
        flat_input = batch_numpy.reshape(batch_size, *spatial_shape)
    else:
        batch_size = 1
        flat_input = batch_numpy[np.newaxis, ...]

    for sample in flat_input:
        dist, indices = scipy_edt(
            sample,
            sampling=sampling,
            return_indices=True,
            return_distances=True,
        )
        dist_results.append(dist)
        indices_results.append(indices)

    output_dist = np.stack(dist_results, axis=0)
    output_indices = np.stack(indices_results, axis=0)

    if len(batch_shape) > 0:
        output_dist = output_dist.reshape(*batch_shape, *spatial_shape)
        output_indices = output_indices.reshape(*batch_shape, spatial_ndim, *spatial_shape)
    else:
        output_dist = output_dist[0]
        output_indices = output_indices[0]

    return output_dist, output_indices


# ======================================================================
# CDT Helper functions
# ======================================================================
def batch_scipy_cdt(
    batch_numpy: np.ndarray,
    metric: str = "chessboard",
    return_indices: bool = False,
    spatial_ndim: int = 2,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute SciPy CDT for a batch of arrays.

    Args:
        batch_numpy: Input array with shape (batch..., spatial...)
        metric: 'chessboard' or 'taxicab'
        return_indices: Whether to return indices
        spatial_ndim: Number of spatial dimensions (1, 2 or 3)
    """
    original_shape = batch_numpy.shape
    spatial_shape = original_shape[-spatial_ndim:]
    batch_shape = original_shape[:-spatial_ndim]

    if len(batch_shape) > 0:
        batch_size = int(np.prod(batch_shape))
        flat_input = batch_numpy.reshape(batch_size, *spatial_shape)
    else:
        batch_size = 1
        flat_input = batch_numpy[np.newaxis, ...]

    dist_results: list[np.ndarray] = []
    indices_results: list[np.ndarray] = []

    for sample in flat_input:
        if return_indices:
            dist, indices = scipy_cdt(
                sample,
                metric=metric,
                return_distances=True,
                return_indices=True,
            )
            dist_results.append(dist)
            indices_results.append(indices)
        else:
            dist = scipy_cdt(sample, metric=metric)
            dist_results.append(dist)

    output_dist = np.stack(dist_results, axis=0)

    # Reshape back
    if len(batch_shape) > 0:
        output_dist = output_dist.reshape(*batch_shape, *spatial_shape)
    else:
        output_dist = output_dist[0]

    if return_indices:
        output_indices = np.stack(indices_results, axis=0)
        if len(batch_shape) > 0:
            output_indices = output_indices.reshape(*batch_shape, spatial_ndim, *spatial_shape)
        else:
            output_indices = output_indices[0]
        return output_dist, output_indices

    return output_dist, None


# ======================================================================
# EDT Test data: (B, C, Spatial...) format
# ======================================================================
# 1D spatial: (B=2, C=1, W=6)
edt_case_1d = np.array(
    [[[1, 1, 0, 1, 0, 1]], [[0, 1, 1, 1, 1, 0]]],
    dtype=np.float32,
)

# 2D spatial: (B=2, C=1, H=3, W=4)
edt_case_2d = np.array(
    [
        [[[0.0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 0]]],
        [[[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]],
    ],
    dtype=np.float32,
)

# 2D spatial single batch: (B=1, C=1, H=4, W=4)
edt_case_2d_single = np.array(
    [
        [
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ]
        ]
    ],
    dtype=np.float32,
)

# 3D spatial: (B=2, C=1, D=4, H=5, W=6)
_edt_case_3d_s1 = np.ones((1, 4, 5, 6), dtype=np.float32)
_edt_case_3d_s1[0, 1, 1, 1] = 0.0
_edt_case_3d_s1[0, 2, 3, 4] = 0.0

_edt_case_3d_s2 = np.ones((1, 4, 5, 6), dtype=np.float32)
_edt_case_3d_s2[0, 0, 0, 0] = 0.0

edt_case_3d = np.stack([_edt_case_3d_s1, _edt_case_3d_s2], axis=0)  # (B=2, C=1, D=4, H=5, W=6)

# 2D with unit dimension: (B=2, C=1, H=5, W=1)
edt_case_2d_unit = np.ones((2, 1, 5, 1), dtype=np.float32)
edt_case_2d_unit[0, 0, 2, 0] = 0.0
edt_case_2d_unit[1, 0, 4, 0] = 0.0


# ======================================================================
# CDT Test data: (B, C, Spatial...) format
# ======================================================================
# 1D spatial: (B=2, C=1, W=9)
cdt_case_1d = np.array(
    [[[0, 1, 1, 1, 1, 0, 1, 1, 0]], [[1, 1, 0, 1, 1, 1, 1, 0, 1]]],
    dtype=np.float32,
)

# 2D spatial: (B=1, C=1, H=5, W=6)
cdt_case_2d_simple = np.array(
    [
        [
            [
                [0, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0],
            ]
        ]
    ],
    dtype=np.float32,
)

# 2D spatial batch: (B=2, C=1, H=4, W=5)
cdt_case_2d_batch = np.array(
    [
        [[[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]]],
        [[[1, 1, 0, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1]]],
    ],
    dtype=np.float32,
)

# 2D checkerboard: (B=1, C=1, H=4, W=4)
cdt_case_checkerboard = np.array(
    [
        [
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ]
        ]
    ],
    dtype=np.float32,
)

# 3D spatial: (B=1, C=1, D=5, H=5, W=5)
_cdt_case_3d_simple = np.zeros((1, 1, 5, 5, 5), dtype=np.float32)
_cdt_case_3d_simple[0, 0, 1:4, 1:4, 1:4] = 1  # 3x3x3 cube of foreground
cdt_case_3d_simple = _cdt_case_3d_simple

# 3D sphere: (B=1, C=1, D=7, H=7, W=7)
_cdt_case_3d_sphere = np.zeros((1, 1, 7, 7, 7), dtype=np.float32)
for z in range(7):
    for y in range(7):
        for x in range(7):
            if (z - 3) ** 2 + (y - 3) ** 2 + (x - 3) ** 2 <= 4:
                _cdt_case_3d_sphere[0, 0, z, y, x] = 1
cdt_case_3d_sphere = _cdt_case_3d_sphere

# 3D batch: (B=2, C=1, D=4, H=5, W=6)
_cdt_case_3d_batch_s1 = np.ones((1, 4, 5, 6), dtype=np.float32)
_cdt_case_3d_batch_s1[0, 1, 1, 1] = 0.0
_cdt_case_3d_batch_s1[0, 2, 3, 4] = 0.0

_cdt_case_3d_batch_s2 = np.ones((1, 4, 5, 6), dtype=np.float32)
_cdt_case_3d_batch_s2[0, 0, 0, 0] = 0.0

cdt_case_3d_batch = np.stack(
    [_cdt_case_3d_batch_s1, _cdt_case_3d_batch_s2], axis=0
)  # (B=2, C=1, D=4, H=5, W=6)


# ======================================================================
# EDT Tests
# ======================================================================
@pytest.mark.parametrize(
    "input_numpy, spatial_ndim",
    [
        pytest.param(edt_case_1d, 1, id="1D_B2C1"),
        pytest.param(edt_case_2d, 2, id="2D_B2C1"),
        pytest.param(edt_case_2d_single, 2, id="2D_B1C1"),
        pytest.param(edt_case_3d, 3, id="3D_B2C1"),
        pytest.param(edt_case_2d_unit, 2, id="2D_UnitDim_B2C1"),
    ],
)
def test_edt_distance_and_indices(
    input_numpy: np.ndarray,
    spatial_ndim: int,
    request: pytest.FixtureRequest,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # 1. Prepare data
    x_numpy_contiguous = np.ascontiguousarray(input_numpy)
    x_cuda = torch.from_numpy(x_numpy_contiguous).cuda()

    print(f"\n\n--- Running test: {request.node.callspec.id} ---")
    print(f"CUDA input shape: {x_cuda.shape}, spatial_ndim: {spatial_ndim}")

    # 2. Create sampling list to specify spatial dimensions
    sampling = [1.0] * spatial_ndim

    # 3. Run CUDA EDT
    dist_cuda, idx_cuda = tm.euclidean_distance_transform(
        x_cuda.clone(), sampling=sampling, return_indices=True
    )

    # 4. Run SciPy (ground truth)
    dist_ref_numpy, idx_ref_numpy = batch_scipy_edt_with_indices(x_numpy_contiguous, spatial_ndim)
    dist_ref = torch.from_numpy(dist_ref_numpy).to(torch.float32).cuda()

    # 5. Validate distances
    print(
        f"CUDA distance shape: {dist_cuda.shape}, reference shape: {dist_ref.shape}",
    )
    assert (
        dist_cuda.shape == dist_ref.shape
    ), f"Shape mismatch: {dist_cuda.shape} vs {dist_ref.shape}"

    # Debug: print actual values for small tensors
    if dist_cuda.numel() <= 30:
        print(f"Input:\n{x_cuda.cpu().numpy()}")
        print(f"CUDA result:\n{dist_cuda.cpu().numpy()}")
        print(f"SciPy reference:\n{dist_ref.cpu().numpy()}")

    torch.testing.assert_close(dist_cuda, dist_ref, atol=1e-5, rtol=1e-5)
    print(">> Distance validation passed.")

    # 6. Validate indices
    # idx_cuda shape: (spatial_ndim, *input_shape)
    # We need to verify that the indices point to the correct nearest background pixel
    spatial_shape = x_cuda.shape[-spatial_ndim:]
    batch_shape = x_cuda.shape[:-spatial_ndim]

    # Create coordinate grid for spatial dimensions
    coords = [torch.arange(s, device="cuda") for s in spatial_shape]
    grid = torch.stack(
        torch.meshgrid(*coords, indexing="ij"), dim=0
    )  # (spatial_ndim, *spatial_shape)

    # Expand grid for batch dimensions
    for _ in batch_shape:
        grid = grid.unsqueeze(1)  # (spatial_ndim, 1, ..., *spatial_shape)
    grid = grid.expand(
        spatial_ndim, *batch_shape, *spatial_shape
    )  # (spatial_ndim, *batch_shape, *spatial_shape)

    # Calculate distance from indices
    diff = grid.float() - idx_cuda.float()
    dist_sq_calculated = torch.sum(diff * diff, dim=0)
    dist_sq_output = dist_cuda * dist_cuda

    torch.testing.assert_close(
        dist_sq_calculated,
        dist_sq_output,
        atol=1e-5,
        rtol=1e-5,
    )
    print(">> Index validation passed.")


@pytest.mark.parametrize(
    "input_numpy, spatial_ndim, sampling",
    [
        # 2D with non-uniform sampling
        pytest.param(edt_case_2d_single, 2, [0.5, 1.0], id="2D_Sampling_0.5_1.0"),
        pytest.param(edt_case_2d_single, 2, [2.0, 0.5], id="2D_Sampling_2.0_0.5"),
        pytest.param(edt_case_2d_single, 2, [0.25, 0.25], id="2D_Sampling_0.25_0.25"),
        # 2D batch with sampling
        pytest.param(edt_case_2d, 2, [1.5, 0.75], id="2D_Batch_Sampling"),
        # 3D with sampling
        pytest.param(edt_case_3d, 3, [1.0, 2.0, 0.5], id="3D_Batch_Sampling"),
        # 1D with sampling
        pytest.param(edt_case_1d, 1, [0.5], id="1D_Batch_Sampling"),
        # Test single-element list broadcast
        pytest.param(edt_case_2d_single, 2, [0.5], id="2D_SingleElementList_Broadcast"),
        pytest.param(edt_case_3d, 3, [2.0], id="3D_SingleElementList_Broadcast"),
    ],
)
def test_edt_with_sampling(
    input_numpy: np.ndarray,
    spatial_ndim: int,
    sampling: list[float],
    request: pytest.FixtureRequest,
) -> None:
    """Test EDT with non-unit sampling (pixel spacing)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x_numpy_contiguous = np.ascontiguousarray(input_numpy)
    x_cuda = torch.from_numpy(x_numpy_contiguous).cuda()

    print(f"\n\n--- Running test: {request.node.callspec.id} ---")
    print(f"CUDA input shape: {x_cuda.shape}, spatial_ndim: {spatial_ndim}, sampling: {sampling}")

    # Run CUDA EDT with sampling
    dist_cuda, idx_cuda = tm.euclidean_distance_transform(
        x_cuda.clone(), sampling=sampling, return_indices=True
    )

    # Expand single-element list for SciPy (it doesn't support broadcast)
    scipy_sampling = sampling if len(sampling) == spatial_ndim else sampling * spatial_ndim

    # Run SciPy with sampling (ground truth)
    dist_ref_numpy, idx_ref_numpy = batch_scipy_edt_with_sampling(
        x_numpy_contiguous, spatial_ndim, scipy_sampling
    )
    dist_ref = torch.from_numpy(dist_ref_numpy).to(torch.float32).cuda()

    # Validate distances
    print(f"CUDA distance shape: {dist_cuda.shape}, reference shape: {dist_ref.shape}")
    assert (
        dist_cuda.shape == dist_ref.shape
    ), f"Shape mismatch: {dist_cuda.shape} vs {dist_ref.shape}"
    torch.testing.assert_close(dist_cuda, dist_ref, atol=1e-5, rtol=1e-5)
    print(">> Distance validation with sampling passed.")

    # Validate indices shape
    expected_idx_shape = (spatial_ndim, *x_cuda.shape)
    assert (
        idx_cuda.shape == expected_idx_shape
    ), f"Index shape mismatch: {idx_cuda.shape} vs {expected_idx_shape}"
    print(">> Index shape validation passed.")

    # Validate indices correctness using sampling
    spatial_shape = x_cuda.shape[-spatial_ndim:]
    batch_shape = x_cuda.shape[:-spatial_ndim]

    coords = [torch.arange(s, device="cuda") for s in spatial_shape]
    grid = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=0)

    for _ in batch_shape:
        grid = grid.unsqueeze(1)
    grid = grid.expand(spatial_ndim, *batch_shape, *spatial_shape)

    # Calculate distance with sampling (use expanded sampling for validation)
    sampling_expanded = sampling if len(sampling) == spatial_ndim else sampling * spatial_ndim
    sampling_tensor = torch.tensor(sampling_expanded, device="cuda", dtype=torch.float32)
    for _ in range(len(batch_shape) + len(spatial_shape)):
        sampling_tensor = sampling_tensor.unsqueeze(-1)
    sampling_tensor = sampling_tensor.expand(spatial_ndim, *batch_shape, *spatial_shape)

    diff = (grid.float() - idx_cuda.float()) * sampling_tensor
    dist_sq_calculated = torch.sum(diff * diff, dim=0)
    dist_sq_output = dist_cuda * dist_cuda

    torch.testing.assert_close(dist_sq_calculated, dist_sq_output, atol=1e-5, rtol=1e-5)
    print(">> Index validation with sampling passed.")


def test_edt_return_flags() -> None:
    """Test return_distances and return_indices flags."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # (B=1, C=1, H=2, W=3)
    x = torch.tensor([[[[1, 1, 0], [1, 0, 0]]]], dtype=torch.float32).cuda()

    # Only distances
    result = tm.euclidean_distance_transform(x, return_distances=True, return_indices=False)
    assert isinstance(
        result, torch.Tensor
    ), "Should return single tensor when only distances requested"
    assert result.shape == x.shape

    # Only indices
    result = tm.euclidean_distance_transform(x, return_distances=False, return_indices=True)
    assert isinstance(
        result, torch.Tensor
    ), "Should return single tensor when only indices requested"
    assert result.shape == (2, *x.shape)  # (spatial_ndim, B, C, H, W)

    # Both
    dist, idx = tm.euclidean_distance_transform(x, return_distances=True, return_indices=True)
    assert dist.shape == x.shape
    assert idx.shape == (2, *x.shape)

    print(">> Return flags test passed.")


def test_edt_single_float_sampling() -> None:
    """Test that a single float sampling value applies to all dimensions."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Use edt_case_2d_single which is (B=1, C=1, H=4, W=4) format
    x_numpy = edt_case_2d_single
    x_cuda = torch.from_numpy(x_numpy).cuda()

    # Single float should apply to all spatial dimensions
    dist_cuda = tm.euclidean_distance_transform(x_cuda, sampling=0.5)

    # Compare with scipy using [0.5, 0.5] - use batch helper for BCHW format
    spatial_ndim = 2
    dist_ref_numpy, _ = batch_scipy_edt_with_sampling(x_numpy, spatial_ndim, [0.5, 0.5])
    dist_ref = torch.from_numpy(dist_ref_numpy).to(torch.float32).cuda()

    torch.testing.assert_close(dist_cuda, dist_ref, atol=1e-5, rtol=1e-5)
    print(">> Single float sampling test passed.")


@pytest.mark.parametrize(
    "input_numpy, spatial_ndim, algorithm",
    [
        # 2D tests with different algorithms
        pytest.param(edt_case_2d, 2, "exact", id="2D_exact"),
        pytest.param(edt_case_2d, 2, "jfa", id="2D_jfa"),
        pytest.param(edt_case_2d, 2, "auto", id="2D_auto"),
        pytest.param(edt_case_2d_single, 2, "exact", id="2D_single_exact"),
        pytest.param(edt_case_2d_single, 2, "jfa", id="2D_single_jfa"),
        pytest.param(edt_case_2d_single, 2, "auto", id="2D_single_auto"),
        # 3D tests with different algorithms
        pytest.param(edt_case_3d, 3, "exact", id="3D_exact"),
        pytest.param(edt_case_3d, 3, "jfa", id="3D_jfa"),
        pytest.param(edt_case_3d, 3, "auto", id="3D_auto"),
    ],
)
def test_edt_algorithm(
    input_numpy: np.ndarray,
    spatial_ndim: int,
    algorithm: str,
    request: pytest.FixtureRequest,
) -> None:
    """Test EDT with different algorithm options (exact, jfa, auto)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x_numpy_contiguous = np.ascontiguousarray(input_numpy)
    x_cuda = torch.from_numpy(x_numpy_contiguous).cuda()

    print(f"\n\n--- Running test: {request.node.callspec.id} ---")
    print(f"CUDA input shape: {x_cuda.shape}, spatial_ndim: {spatial_ndim}, algorithm: {algorithm}")

    # Run CUDA EDT with specified algorithm
    dist_cuda = tm.euclidean_distance_transform(x_cuda.clone(), algorithm=algorithm)

    # Run SciPy (ground truth)
    dist_ref_numpy, _ = batch_scipy_edt_with_indices(x_numpy_contiguous, spatial_ndim)
    dist_ref = torch.from_numpy(dist_ref_numpy).to(torch.float32).cuda()

    # Validate distances
    print(f"CUDA distance shape: {dist_cuda.shape}, reference shape: {dist_ref.shape}")
    assert (
        dist_cuda.shape == dist_ref.shape
    ), f"Shape mismatch: {dist_cuda.shape} vs {dist_ref.shape}"

    torch.testing.assert_close(dist_cuda, dist_ref, rtol=1e-5, atol=1e-5)

    print(f">> Algorithm '{algorithm}' validation passed.")


def test_edt_algorithm_fallback_with_sampling() -> None:
    """Test that JFA falls back to exact when sampling is provided."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x_numpy = edt_case_2d_single
    x_cuda = torch.from_numpy(x_numpy).cuda()

    # With non-unit sampling, JFA should fall back to exact algorithm
    # Both should give same result
    dist_jfa = tm.euclidean_distance_transform(x_cuda.clone(), sampling=[0.5, 1.0], algorithm="jfa")
    dist_exact = tm.euclidean_distance_transform(
        x_cuda.clone(), sampling=[0.5, 1.0], algorithm="exact"
    )

    # Compare with scipy
    spatial_ndim = 2
    dist_ref_numpy, _ = batch_scipy_edt_with_sampling(x_numpy, spatial_ndim, [0.5, 1.0])
    dist_ref = torch.from_numpy(dist_ref_numpy).to(torch.float32).cuda()

    torch.testing.assert_close(dist_jfa, dist_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dist_exact, dist_ref, atol=1e-5, rtol=1e-5)

    print(">> Algorithm fallback with sampling test passed.")


def test_edt_jfa_vs_exact_consistency() -> None:
    """Test that JFA and exact produce similar results for unit sampling."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create a larger random test case
    torch.manual_seed(42)
    x = (torch.randn(2, 1, 64, 64, device="cuda") > 0).float()

    dist_exact = tm.euclidean_distance_transform(x, algorithm="exact")
    dist_jfa = tm.euclidean_distance_transform(x, algorithm="jfa")

    # JFA should be very close to exact for most pixels
    # Allow for small differences due to JFA's approximate nature
    diff = torch.abs(dist_exact - dist_jfa)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"JFA vs Exact - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    # Most pixels should be exact or very close
    assert mean_diff < 0.1, f"Mean difference too large: {mean_diff}"
    print(">> JFA vs Exact consistency test passed.")


# ======================================================================
# CDT Tests
# ======================================================================
@pytest.mark.parametrize(
    "input_numpy, spatial_ndim, metric",
    [
        pytest.param(cdt_case_1d, 1, "chessboard", id="1D_B2C1_chessboard"),
        pytest.param(cdt_case_1d, 1, "taxicab", id="1D_B2C1_taxicab"),
        pytest.param(cdt_case_2d_simple, 2, "chessboard", id="2D_B1C1_chessboard"),
        pytest.param(cdt_case_2d_simple, 2, "taxicab", id="2D_B1C1_taxicab"),
        pytest.param(cdt_case_2d_batch, 2, "chessboard", id="2D_B2C1_chessboard"),
        pytest.param(cdt_case_2d_batch, 2, "taxicab", id="2D_B2C1_taxicab"),
        pytest.param(cdt_case_checkerboard, 2, "chessboard", id="2D_checkerboard_chessboard"),
        pytest.param(cdt_case_checkerboard, 2, "taxicab", id="2D_checkerboard_taxicab"),
        pytest.param(cdt_case_3d_simple, 3, "chessboard", id="3D_B1C1_simple_chessboard"),
        pytest.param(cdt_case_3d_simple, 3, "taxicab", id="3D_B1C1_simple_taxicab"),
        pytest.param(cdt_case_3d_sphere, 3, "chessboard", id="3D_B1C1_sphere_chessboard"),
        pytest.param(cdt_case_3d_sphere, 3, "taxicab", id="3D_B1C1_sphere_taxicab"),
        pytest.param(cdt_case_3d_batch, 3, "chessboard", id="3D_B2C1_batch_chessboard"),
        pytest.param(cdt_case_3d_batch, 3, "taxicab", id="3D_B2C1_batch_taxicab"),
    ],
)
def test_cdt_basic(
    input_numpy: np.ndarray,
    spatial_ndim: int,
    metric: str,
    request: pytest.FixtureRequest,
) -> None:
    """Test CDT distance computation against scipy with BCHW format."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x_numpy_contiguous = np.ascontiguousarray(input_numpy)
    x_cuda = torch.from_numpy(x_numpy_contiguous).cuda()

    print(f"\n\n--- Running test: {request.node.callspec.id} ---")
    print(f"CUDA input shape: {x_cuda.shape}, spatial_ndim: {spatial_ndim}, metric: {metric}")

    # Run torchmorph CDT
    dist_cuda = tm.chamfer_distance_transform(x_cuda, metric=metric)

    # Run scipy CDT (ground truth)
    dist_scipy, _ = batch_scipy_cdt(x_numpy_contiguous, metric=metric, spatial_ndim=spatial_ndim)
    dist_ref = torch.from_numpy(dist_scipy).to(torch.float32).cuda()

    print(f"CUDA distance shape: {dist_cuda.shape}, reference shape: {dist_ref.shape}")
    assert (
        dist_cuda.shape == dist_ref.shape
    ), f"Shape mismatch: {dist_cuda.shape} vs {dist_ref.shape}"

    # Debug: print actual values for small tensors
    if dist_cuda.numel() <= 50:
        print(f"Input:\n{x_cuda.cpu().numpy()}")
        print(f"CUDA result:\n{dist_cuda.cpu().numpy()}")
        print(f"SciPy reference:\n{dist_ref.cpu().numpy()}")

    torch.testing.assert_close(dist_cuda, dist_ref, atol=1e-5, rtol=1e-5)
    print(">> Distance validation passed.")


@pytest.mark.parametrize(
    "alias, canonical",
    [
        pytest.param("cityblock", "taxicab", id="cityblock"),
        pytest.param("manhattan", "taxicab", id="manhattan"),
    ],
)
def test_cdt_metric_aliases(alias: str, canonical: str) -> None:
    """Test that metric aliases produce same results."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x_cuda = torch.from_numpy(cdt_case_2d_simple).cuda()

    dist_alias = tm.chamfer_distance_transform(x_cuda, metric=alias)
    dist_canonical = tm.chamfer_distance_transform(x_cuda, metric=canonical)

    torch.testing.assert_close(dist_alias, dist_canonical, atol=1e-5, rtol=1e-5)
    print(f">> Alias '{alias}' == '{canonical}' validation passed.")


def test_cdt_return_flags() -> None:
    """Test return_distances and return_indices flags with BCHW format."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # (B=1, C=1, H=5, W=6)
    x = torch.from_numpy(cdt_case_2d_simple).cuda()

    # Only distances (default)
    result = tm.chamfer_distance_transform(x, return_distances=True, return_indices=False)
    assert isinstance(result, torch.Tensor), "Should return single tensor"
    assert result.shape == x.shape

    # Only indices - spatial_ndim=2 for BCHW
    result = tm.chamfer_distance_transform(x, return_distances=False, return_indices=True)
    assert isinstance(result, torch.Tensor), "Should return single tensor"
    assert result.shape == (2, *x.shape)  # (spatial_ndim, B, C, H, W)

    # Both
    dist, idx = tm.chamfer_distance_transform(x, return_distances=True, return_indices=True)
    assert dist.shape == x.shape
    assert idx.shape == (2, *x.shape)

    print(">> Return flags test passed.")


def test_cdt_preallocated_output() -> None:
    """Test pre-allocated output tensors with scipy-style return convention."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # (B=1, C=1, H=5, W=6)
    x = torch.from_numpy(cdt_case_2d_simple).cuda()

    # Pre-allocate distances tensor
    dist_out = torch.empty_like(x)
    result = tm.chamfer_distance_transform(x, distances=dist_out)

    # Should return None (scipy convention)
    assert result is None, "Should return None when distances tensor is provided"

    # But dist_out should be filled
    dist_ref, _ = batch_scipy_cdt(cdt_case_2d_simple, metric="chessboard", spatial_ndim=2)
    dist_ref_tensor = torch.from_numpy(dist_ref).to(torch.float32).cuda()
    torch.testing.assert_close(dist_out, dist_ref_tensor, atol=1e-5, rtol=1e-5)

    print(">> Pre-allocated output test passed.")


def test_cdt_indices_correctness() -> None:
    """Test that indices point to correct nearest background pixel with BCHW format."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # (B=1, C=1, H=5, W=6)
    x = torch.from_numpy(cdt_case_2d_simple).cuda()

    dist, idx = tm.chamfer_distance_transform(x, metric="chessboard", return_indices=True)

    # For each foreground pixel, verify the index points to a background pixel
    # idx shape: (spatial_ndim=2, B=1, C=1, H=5, W=6)
    B, C, H, W = x.shape
    x_np = x.cpu().numpy()
    idx_np = idx.cpu().numpy()  # (2, B, C, H, W)
    dist_np = dist.cpu().numpy()

    for b in range(B):
        for c in range(C):
            for y in range(H):
                for x_coord in range(W):
                    if x_np[b, c, y, x_coord] != 0:  # Foreground
                        idx_y = idx_np[0, b, c, y, x_coord]
                        idx_x = idx_np[1, b, c, y, x_coord]
                        # The pointed pixel should be background
                        assert (
                            x_np[b, c, idx_y, idx_x] == 0
                        ), f"Index ({idx_y}, {idx_x}) should point to background"
                        # Chessboard distance should match
                        expected_dist = max(abs(y - idx_y), abs(x_coord - idx_x))
                        assert (
                            dist_np[b, c, y, x_coord] == expected_dist
                        ), f"Distance mismatch at ({b}, {c}, {y}, {x_coord})"

    print(">> Indices correctness test passed.")


def test_cdt_indices_correctness_3d() -> None:
    """Test that 3D indices point to correct nearest background pixel with BCDHW format."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # (B=1, C=1, D=5, H=5, W=5)
    x = torch.from_numpy(cdt_case_3d_simple).cuda()

    dist, idx = tm.chamfer_distance_transform(x, metric="chessboard", return_indices=True)

    # For each foreground pixel, verify the index points to a background pixel
    # idx shape: (spatial_ndim=3, B=1, C=1, D=5, H=5, W=5)
    B, C, D, H, W = x.shape
    x_np = x.cpu().numpy()
    idx_np = idx.cpu().numpy()  # (3, B, C, D, H, W)
    dist_np = dist.cpu().numpy()

    for b in range(B):
        for c in range(C):
            for z in range(D):
                for y in range(H):
                    for x_coord in range(W):
                        if x_np[b, c, z, y, x_coord] != 0:  # Foreground
                            idx_z = idx_np[0, b, c, z, y, x_coord]
                            idx_y = idx_np[1, b, c, z, y, x_coord]
                            idx_x = idx_np[2, b, c, z, y, x_coord]
                            # The pointed pixel should be background
                            assert (
                                x_np[b, c, idx_z, idx_y, idx_x] == 0
                            ), f"Index ({idx_z}, {idx_y}, {idx_x}) should point to background"
                            # Chessboard distance should match
                            expected_dist = max(
                                abs(z - idx_z), abs(y - idx_y), abs(x_coord - idx_x)
                            )
                            assert (
                                dist_np[b, c, z, y, x_coord] == expected_dist
                            ), f"Distance mismatch at ({b}, {c}, {z}, {y}, {x_coord})"

    print(">> 3D Indices correctness test passed.")


def test_cdt_invalid_metric() -> None:
    """Test that invalid metric raises error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.from_numpy(cdt_case_2d_simple).cuda()

    with pytest.raises(ValueError, match="metric must be"):
        tm.chamfer_distance_transform(x, metric="invalid")


def test_cdt_cpu_input_error() -> None:
    """Test that CPU input raises error."""
    x = torch.from_numpy(cdt_case_2d_simple)  # CPU tensor

    with pytest.raises(ValueError, match="CUDA"):
        tm.chamfer_distance_transform(x)


@pytest.mark.parametrize(
    "shape, spatial_ndim, metric",
    [
        # 1D spatial: (B, C, W)
        pytest.param((2, 1, 32), 1, "chessboard", id="1D_B2C1_32_chessboard"),
        pytest.param((2, 1, 32), 1, "taxicab", id="1D_B2C1_32_taxicab"),
        pytest.param((4, 2, 64), 1, "chessboard", id="1D_B4C2_64_chessboard"),
        # 2D spatial: (B, C, H, W)
        pytest.param((1, 1, 32, 32), 2, "chessboard", id="2D_B1C1_32x32_chessboard"),
        pytest.param((1, 1, 32, 32), 2, "taxicab", id="2D_B1C1_32x32_taxicab"),
        pytest.param((2, 1, 32, 32), 2, "chessboard", id="2D_B2C1_32x32_chessboard"),
        pytest.param((2, 1, 32, 32), 2, "taxicab", id="2D_B2C1_32x32_taxicab"),
        pytest.param((2, 3, 64, 48), 2, "chessboard", id="2D_B2C3_64x48_chessboard"),
        # 3D spatial: (B, C, D, H, W)
        pytest.param((1, 1, 8, 8, 8), 3, "chessboard", id="3D_B1C1_8x8x8_chessboard"),
        pytest.param((1, 1, 8, 8, 8), 3, "taxicab", id="3D_B1C1_8x8x8_taxicab"),
        pytest.param((2, 1, 16, 16, 16), 3, "chessboard", id="3D_B2C1_16x16x16_chessboard"),
        pytest.param((2, 2, 12, 10, 8), 3, "taxicab", id="3D_B2C2_12x10x8_taxicab"),
    ],
)
def test_cdt_random_data(shape: tuple, spatial_ndim: int, metric: str) -> None:
    """Test CDT with random binary data in BCHW format."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    np.random.seed(42)
    input_numpy = (np.random.rand(*shape) > 0.3).astype(np.float32)

    x_cuda = torch.from_numpy(input_numpy).cuda()

    # Run torchmorph CDT
    dist_cuda = tm.chamfer_distance_transform(x_cuda, metric=metric)

    # Run scipy CDT
    dist_scipy, _ = batch_scipy_cdt(input_numpy, metric=metric, spatial_ndim=spatial_ndim)
    dist_ref = torch.from_numpy(dist_scipy).to(torch.float32).cuda()

    torch.testing.assert_close(dist_cuda, dist_ref, atol=1e-5, rtol=1e-5)
    print(f">> Random data test ({shape}, spatial_ndim={spatial_ndim}, {metric}) passed.")


@pytest.mark.parametrize(
    "input_numpy, spatial_ndim, metric",
    [
        pytest.param(cdt_case_1d, 1, "chessboard", id="1D_indices_chessboard"),
        pytest.param(cdt_case_1d, 1, "taxicab", id="1D_indices_taxicab"),
        pytest.param(cdt_case_2d_batch, 2, "chessboard", id="2D_batch_indices_chessboard"),
        pytest.param(cdt_case_3d_batch, 3, "chessboard", id="3D_batch_indices_chessboard"),
    ],
)
def test_cdt_indices_validation(
    input_numpy: np.ndarray,
    spatial_ndim: int,
    metric: str,
    request: pytest.FixtureRequest,
) -> None:
    """Test that indices correctly point to nearest background pixels in BCHW format."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x_numpy_contiguous = np.ascontiguousarray(input_numpy)
    x_cuda = torch.from_numpy(x_numpy_contiguous).cuda()

    print(f"\n\n--- Running test: {request.node.callspec.id} ---")
    print(f"CUDA input shape: {x_cuda.shape}, spatial_ndim: {spatial_ndim}")

    # Run torchmorph CDT with indices
    dist_cuda, idx_cuda = tm.chamfer_distance_transform(x_cuda, metric=metric, return_indices=True)

    # Validate indices shape: (spatial_ndim, *input_shape)
    expected_idx_shape = (spatial_ndim, *x_cuda.shape)
    assert (
        idx_cuda.shape == expected_idx_shape
    ), f"Index shape mismatch: {idx_cuda.shape} vs {expected_idx_shape}"

    # Validate that indices point to background pixels and distance matches
    spatial_shape = x_cuda.shape[-spatial_ndim:]
    batch_shape = x_cuda.shape[:-spatial_ndim]

    # Create coordinate grid for spatial dimensions
    coords = [torch.arange(s, device="cuda") for s in spatial_shape]
    grid = torch.stack(
        torch.meshgrid(*coords, indexing="ij"), dim=0
    )  # (spatial_ndim, *spatial_shape)

    # Expand grid for batch dimensions
    for _ in batch_shape:
        grid = grid.unsqueeze(1)
    grid = grid.expand(spatial_ndim, *batch_shape, *spatial_shape)

    # Calculate distance from indices based on metric
    diff = grid.float() - idx_cuda.float()
    if metric in ("chessboard",):
        # Chessboard: max of absolute differences
        dist_calculated = torch.max(torch.abs(diff), dim=0).values
    else:
        # Taxicab: sum of absolute differences
        dist_calculated = torch.sum(torch.abs(diff), dim=0)

    torch.testing.assert_close(dist_calculated, dist_cuda, atol=1e-5, rtol=1e-5)
    print(">> Index validation passed.")
