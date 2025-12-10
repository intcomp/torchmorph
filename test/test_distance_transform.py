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
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SciPy EDT and indices for a batch of arrays."""
    dist_results: list[np.ndarray] = []
    indices_results: list[np.ndarray] = []

    # Ensure batch_numpy has at least shape (Batch, ...)
    # If the input is (H, W), it is already converted to (1, H, W) outside.
    if batch_numpy.ndim == 1:
        batch_numpy = batch_numpy[np.newaxis, ...]

    for sample in batch_numpy:
        dist, indices = scipy_edt(
            sample,
            return_indices=True,
            return_distances=True,
        )
        dist_results.append(dist)
        indices_results.append(indices)

    output_dist = np.stack(dist_results, axis=0)
    output_indices = np.stack(indices_results, axis=0)
    output_indices = np.moveaxis(output_indices, 1, -1)

    return output_dist, output_indices


# ======================================================================
# Test data
# ======================================================================
case_batch_1d = np.array(
    [[1, 1, 0, 1, 0, 1], [0, 1, 1, 1, 1, 0]],
    dtype=np.float32,
)

case_batch_2d = np.array(
    [
        [[0.0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 0]],
        [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
    ],
    dtype=np.float32,
)

# This is a single 2D image with shape (4, 4)
case_single_2d = np.array(
    [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ],
    dtype=np.float32,
)
case_explicit_batch_one = case_single_2d[np.newaxis, ...]

_case_3d_s1 = np.ones((4, 5, 6), dtype=np.float32)
_case_3d_s1[1, 1, 1] = 0.0
_case_3d_s1[2, 3, 4] = 0.0

_case_3d_s2 = np.ones((4, 5, 6), dtype=np.float32)
_case_3d_s2[0, 0, 0] = 0.0

case_batch_3d = np.stack([_case_3d_s1, _case_3d_s2], axis=0)

case_dim_one = np.ones((2, 5, 1), dtype=np.float32)
case_dim_one[0, 2, 0] = 0.0
case_dim_one[1, 4, 0] = 0.0

# 4D spatial case
_case_4d_s1 = np.ones((3, 3, 3, 3), dtype=np.float32)
_case_4d_s1[0, 0, 0, 0] = 0.0

_case_4d_s2 = np.ones((3, 3, 3, 3), dtype=np.float32)
_case_4d_s2[1, 1, 1, 1] = 0.0

case_batch_4d_spatial = np.stack([_case_4d_s1, _case_4d_s2], axis=0)

# 5D spatial case
case_batch_5d_spatial = np.ones((1, 2, 2, 2, 2, 2), dtype=np.float32)
case_batch_5d_spatial[0, 0, 0, 0, 0, 0] = 0.0
case_batch_5d_spatial[0, 1, 1, 1, 1, 1] = 0.0


# ======================================================================
# Test logic
# ======================================================================
@pytest.mark.parametrize(
    "input_numpy, has_batch_dim",
    [
        pytest.param(case_batch_1d, True, id="1D_Batch"),
        pytest.param(case_batch_2d, True, id="2D_Batch"),
        pytest.param(case_single_2d, False, id="2D_Single_NoBatch"),
        pytest.param(
            case_explicit_batch_one,
            True,
            id="2D_Single_ExplicitBatch",
        ),
        pytest.param(case_batch_3d, True, id="3D_Batch"),
        pytest.param(case_dim_one, True, id="2D_UnitDim_Batch"),
        pytest.param(case_batch_4d_spatial, True, id="4D_Spatial_Batch"),
        pytest.param(case_batch_5d_spatial, True, id="5D_Spatial_Batch"),
    ],
)
def test_distance_transform_and_indices(
    input_numpy: np.ndarray,
    has_batch_dim: bool,
    request: pytest.FixtureRequest,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # 1. Prepare NumPy data
    x_numpy_contiguous = np.ascontiguousarray(input_numpy)

    # 2. Prepare SciPy input.
    # If this is a single sample (has_batch_dim=False), manually add a
    # batch dimension so SciPy treats it as one image instead of N 1D
    # signals.
    if not has_batch_dim:
        scipy_input = x_numpy_contiguous[np.newaxis, ...]
    else:
        scipy_input = x_numpy_contiguous

    # 3. Prepare CUDA input.
    # If has_batch_dim=False, the input is (H, W) and we want 2D EDT.
    # The C++ API assumes the first dimension is batch, so we must
    # unsqueeze(0) to get shape (1, H, W). Otherwise, it will be
    # interpreted as (Batch=H, Length=W) and run 1D EDT.
    x_cuda = torch.from_numpy(x_numpy_contiguous).cuda()
    if not has_batch_dim:
        x_cuda = x_cuda.unsqueeze(0)

    print(f"\n\n--- Running test: {request.node.callspec.id} ---")
    print(f"CUDA input shape: {x_cuda.shape}")

    # 4. Run CUDA EDT
    dist_cuda, idx_cuda = tm.distance_transform(x_cuda.clone())

    # 5. Run SciPy (ground truth)
    dist_ref_numpy, idx_ref_numpy = batch_scipy_edt_with_indices(scipy_input)
    dist_ref = torch.from_numpy(dist_ref_numpy).to(torch.float32).cuda()

    # 6. Validate distances
    print(
        f"CUDA distance shape: {dist_cuda.shape}, " f"reference shape: {dist_ref.shape}",
    )
    assert (
        dist_cuda.shape == dist_ref.shape
    ), f"Shape mismatch: {dist_cuda.shape} vs {dist_ref.shape}"
    torch.testing.assert_close(dist_cuda, dist_ref, atol=1e-3, rtol=1e-3)
    print(">> Distance validation passed.")

    # 7. Validate indices
    # idx_cuda: (B, H, W, D)
    spatial_shape = x_cuda.shape[1:]
    coords = [torch.arange(s, device="cuda") for s in spatial_shape]
    grid = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)
    grid = grid.unsqueeze(0)  # (1, H, W, D)

    diff = grid.float() - idx_cuda.float()
    dist_sq_calculated = torch.sum(diff * diff, dim=-1)
    dist_sq_output = dist_cuda * dist_cuda

    torch.testing.assert_close(
        dist_sq_calculated,
        dist_sq_output,
        atol=1e-3,
        rtol=1e-3,
    )
    print(">> Index validation passed.")
