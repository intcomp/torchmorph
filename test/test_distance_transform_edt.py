import numpy as np  # noqa: F401
import pytest
import torch
from scipy.ndimage import distance_transform_edt as scipy_edt  # noqa: F401

import torchmorph as tm  # noqa: F401


# ======================================================================
# Helper functions
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


# ======================================================================
# Test data: (B, C, Spatial...) format
# ======================================================================
# 1D spatial: (B=2, C=1, W=6)
case_1d = np.array(
    [[[1, 1, 0, 1, 0, 1]], [[0, 1, 1, 1, 1, 0]]],
    dtype=np.float32,
)

# 2D spatial: (B=2, C=1, H=3, W=4)
case_2d = np.array(
    [
        [[[0.0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 0]]],
        [[[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]],
    ],
    dtype=np.float32,
)

# 2D spatial single batch: (B=1, C=1, H=4, W=4)
case_2d_single = np.array(
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
_case_3d_s1 = np.ones((1, 4, 5, 6), dtype=np.float32)
_case_3d_s1[0, 1, 1, 1] = 0.0
_case_3d_s1[0, 2, 3, 4] = 0.0

_case_3d_s2 = np.ones((1, 4, 5, 6), dtype=np.float32)
_case_3d_s2[0, 0, 0, 0] = 0.0

case_3d = np.stack([_case_3d_s1, _case_3d_s2], axis=0)  # (B=2, C=1, D=4, H=5, W=6)

# 2D with unit dimension: (B=2, C=1, H=5, W=1)
case_2d_unit = np.ones((2, 1, 5, 1), dtype=np.float32)
case_2d_unit[0, 0, 2, 0] = 0.0
case_2d_unit[1, 0, 4, 0] = 0.0


# ======================================================================
# Test logic
# ======================================================================
@pytest.mark.parametrize(
    "input_numpy, spatial_ndim",
    [
        pytest.param(case_1d, 1, id="1D_B2C1"),
        pytest.param(case_2d, 2, id="2D_B2C1"),
        pytest.param(case_2d_single, 2, id="2D_B1C1"),
        pytest.param(case_3d, 3, id="3D_B2C1"),
        pytest.param(case_2d_unit, 2, id="2D_UnitDim_B2C1"),
    ],
)
def test_distance_transform_and_indices(
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
    dist_cuda, idx_cuda = tm.distance_transform(
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


# ======================================================================
# Helper for sampling tests
# ======================================================================
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
# Test sampling functionality
# ======================================================================
@pytest.mark.parametrize(
    "input_numpy, spatial_ndim, sampling",
    [
        # 2D with non-uniform sampling
        pytest.param(case_2d_single, 2, [0.5, 1.0], id="2D_Sampling_0.5_1.0"),
        pytest.param(case_2d_single, 2, [2.0, 0.5], id="2D_Sampling_2.0_0.5"),
        pytest.param(case_2d_single, 2, [0.25, 0.25], id="2D_Sampling_0.25_0.25"),
        # 2D batch with sampling
        pytest.param(case_2d, 2, [1.5, 0.75], id="2D_Batch_Sampling"),
        # 3D with sampling
        pytest.param(case_3d, 3, [1.0, 2.0, 0.5], id="3D_Batch_Sampling"),
        # 1D with sampling
        pytest.param(case_1d, 1, [0.5], id="1D_Batch_Sampling"),
        # Test single-element list broadcast
        pytest.param(case_2d_single, 2, [0.5], id="2D_SingleElementList_Broadcast"),
        pytest.param(case_3d, 3, [2.0], id="3D_SingleElementList_Broadcast"),
    ],
)
def test_distance_transform_with_sampling(
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
    dist_cuda, idx_cuda = tm.distance_transform_edt(
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


# ======================================================================
# Test return_distances and return_indices flags
# ======================================================================
def test_return_flags() -> None:
    """Test return_distances and return_indices flags."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # (B=1, C=1, H=2, W=3)
    x = torch.tensor([[[[1, 1, 0], [1, 0, 0]]]], dtype=torch.float32).cuda()

    # Only distances
    result = tm.distance_transform_edt(x, return_distances=True, return_indices=False)
    assert isinstance(
        result, torch.Tensor
    ), "Should return single tensor when only distances requested"
    assert result.shape == x.shape

    # Only indices
    result = tm.distance_transform_edt(x, return_distances=False, return_indices=True)
    assert isinstance(
        result, torch.Tensor
    ), "Should return single tensor when only indices requested"
    assert result.shape == (2, *x.shape)  # (spatial_ndim, B, C, H, W)

    # Both
    dist, idx = tm.distance_transform_edt(x, return_distances=True, return_indices=True)
    assert dist.shape == x.shape
    assert idx.shape == (2, *x.shape)

    print(">> Return flags test passed.")


# ======================================================================
# Test single float sampling
# ======================================================================
def test_single_float_sampling() -> None:
    """Test that a single float sampling value applies to all dimensions."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Use case_2d_single which is (B=1, C=1, H=4, W=4) format
    x_numpy = case_2d_single
    x_cuda = torch.from_numpy(x_numpy).cuda()

    # Single float should apply to all spatial dimensions
    dist_cuda = tm.distance_transform_edt(x_cuda, sampling=0.5)

    # Compare with scipy using [0.5, 0.5] - use batch helper for BCHW format
    spatial_ndim = 2
    dist_ref_numpy, _ = batch_scipy_edt_with_sampling(x_numpy, spatial_ndim, [0.5, 0.5])
    dist_ref = torch.from_numpy(dist_ref_numpy).to(torch.float32).cuda()

    torch.testing.assert_close(dist_cuda, dist_ref, atol=1e-5, rtol=1e-5)
    print(">> Single float sampling test passed.")


# ======================================================================
# Test algorithm parameter (JFA vs Exact)
# ======================================================================
@pytest.mark.parametrize(
    "input_numpy, spatial_ndim, algorithm",
    [
        # 2D tests with different algorithms
        pytest.param(case_2d, 2, "exact", id="2D_exact"),
        pytest.param(case_2d, 2, "jfa", id="2D_jfa"),
        pytest.param(case_2d, 2, "auto", id="2D_auto"),
        pytest.param(case_2d_single, 2, "exact", id="2D_single_exact"),
        pytest.param(case_2d_single, 2, "jfa", id="2D_single_jfa"),
        pytest.param(case_2d_single, 2, "auto", id="2D_single_auto"),
        # 3D tests with different algorithms
        pytest.param(case_3d, 3, "exact", id="3D_exact"),
        pytest.param(case_3d, 3, "jfa", id="3D_jfa"),
        pytest.param(case_3d, 3, "auto", id="3D_auto"),
    ],
)
def test_distance_transform_algorithm(
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
    dist_cuda = tm.distance_transform_edt(x_cuda.clone(), algorithm=algorithm)

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


def test_algorithm_fallback_with_sampling() -> None:
    """Test that JFA falls back to exact when sampling is provided."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x_numpy = case_2d_single
    x_cuda = torch.from_numpy(x_numpy).cuda()

    # With non-unit sampling, JFA should fall back to exact algorithm
    # Both should give same result
    dist_jfa = tm.distance_transform_edt(x_cuda.clone(), sampling=[0.5, 1.0], algorithm="jfa")
    dist_exact = tm.distance_transform_edt(x_cuda.clone(), sampling=[0.5, 1.0], algorithm="exact")

    # Compare with scipy
    spatial_ndim = 2
    dist_ref_numpy, _ = batch_scipy_edt_with_sampling(x_numpy, spatial_ndim, [0.5, 1.0])
    dist_ref = torch.from_numpy(dist_ref_numpy).to(torch.float32).cuda()

    torch.testing.assert_close(dist_jfa, dist_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dist_exact, dist_ref, atol=1e-5, rtol=1e-5)

    print(">> Algorithm fallback with sampling test passed.")


def test_jfa_vs_exact_consistency() -> None:
    """Test that JFA and exact produce similar results for unit sampling."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create a larger random test case
    torch.manual_seed(42)
    x = (torch.randn(2, 1, 64, 64, device="cuda") > 0).float()

    dist_exact = tm.distance_transform_edt(x, algorithm="exact")
    dist_jfa = tm.distance_transform_edt(x, algorithm="jfa")

    # JFA should be very close to exact for most pixels
    # Allow for small differences due to JFA's approximate nature
    diff = torch.abs(dist_exact - dist_jfa)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"JFA vs Exact - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    # Most pixels should be exact or very close
    assert mean_diff < 0.1, f"Mean difference too large: {mean_diff}"
    print(">> JFA vs Exact consistency test passed.")
