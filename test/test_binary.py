import numpy as np  # noqa: F401
import pytest
import torch
from scipy.ndimage import binary_erosion, generate_binary_structure

import torchmorph as tm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for torchmorph tests"
)


def apply_scipy_to_batch(np_input, scipy_func, **kwargs):
    """
    Apply a SciPy operator to each (B, C) sample independently.
    The input is expected to have shape (B, C, Spatial...). The batch and
    channel dimensions are flattened, `scipy_func` is applied to each spatial
    sample separately, and the results are reshaped back to the original
    layout.
    Args:
        np_input: Input array in (B, C, Spatial...) format.
        scipy_func: SciPy function applied to each spatial sample.
        **kwargs: Additional keyword arguments forwarded to `scipy_func`.
    Returns:
        np.ndarray: Output array with the same shape as `np_input`.
    """
    input_shape = np_input.shape
    batch_shape = input_shape[0:2]
    spatial_shape = input_shape[2:]

    batch_size = int(np.prod(batch_shape))
    flatten_input = np_input.reshape(batch_size, *spatial_shape)

    results = []

    for sample in flatten_input:
        result = scipy_func(sample, **kwargs)
        results.append(result)

    output = np.stack(results, axis=0).reshape(*batch_shape, *spatial_shape)
    return output


def batch_scipy_erosion(
    np_input: np.ndarray,
    structure: np.ndarray | None = None,
    iterations: int = 1,
    mask: np.ndarray | None = None,
    output: np.ndarray | None = None,
    border_value: bool = False,
    origin: int | tuple[int, ...] = 0,
) -> np.ndarray:
    return apply_scipy_to_batch(
        np_input,
        binary_erosion,
        structure=structure,
        iterations=iterations,
        mask=mask,
        output=output,
        border_value=border_value,
        origin=origin,
    )


case_2d = np.array(
    [
        [
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ]
        ]
    ],
    dtype=np.float32,
)
case_structure_2d = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

case_3d = np.arange(24).reshape(2, 1, 2, 3, 2)
case_4d = np.arange(48).reshape(2, 1, 2, 2, 3, 2)
case_structure_3d_1 = generate_binary_structure(rank=3, connectivity=2)
case_structure_3d_2 = generate_binary_structure(rank=3, connectivity=3)
case_structure_4d = generate_binary_structure(rank=4, connectivity=4)


@pytest.mark.parametrize(
    ("np_input, structure, iterations, origin, border_value"),
    [
        pytest.param(case_2d, None, 1, 0, False, id="2D_basic"),
        pytest.param(case_2d, case_structure_2d, 1, 0, False, id="2D_structure"),
        pytest.param(case_2d, None, 2, 0, False, id="2D_1iterations"),
        pytest.param(case_2d, None, -1, 0, False, id="2D_-1iterations"),
        pytest.param(case_2d, None, 1, 1, False, id="2D_1origin"),
        pytest.param(case_2d, None, 1, 1, True, id="2D_1origin"),
        pytest.param(case_3d, None, 1, 0, False, id="3D_basic"),
        pytest.param(case_3d, case_structure_3d_1, 1, 0, False, id="3D_2Dstructure"),
        pytest.param(case_3d, case_structure_3d_2, 1, 0, False, id="3D_3Dstructure"),
        pytest.param(case_3d, None, 2, 0, False, id="3D_1iterations"),
        pytest.param(case_3d, None, -1, 0, False, id="3D_-1iterations"),
        pytest.param(case_3d, None, 1, 1, False, id="3D_1origin"),
        pytest.param(case_3d, None, 1, 1, True, id="3D_1origin"),
        pytest.param(case_4d, None, 1, 0, False, id="4D_basic"),
        pytest.param(case_4d, case_structure_4d, 1, 0, False, id="4D_4Dstructure"),
        pytest.param(case_4d, case_structure_4d, 1, 0, False, id="4D_4Dstructure"),
        pytest.param(case_4d, None, 2, 0, False, id="4D_1iterations"),
        pytest.param(case_4d, None, -1, 0, False, id="4D_-1iterations"),
        pytest.param(case_4d, None, 1, 1, False, id="4D_1origin"),
        pytest.param(case_4d, None, 1, 1, True, id="4D_1origin"),
    ],
)
def test_binary_erosion_basic(
    np_input,
    structure,
    iterations,
    origin,
    border_value,
):
    x = torch.as_tensor(np_input, dtype=torch.float32)
    if structure is not None:
        structure_cuda = torch.as_tensor(structure, dtype=torch.float32)
    else:
        structure_cuda = None

    actual = tm.binary_erosion(
        x,
        structure=structure_cuda,
        iterations=iterations,
        origin=origin,
        border_value=border_value,
    )
    expected_np = batch_scipy_erosion(
        np_input,
        structure=structure,
        iterations=iterations,
        origin=origin,
        border_value=border_value,
    )
    expected = torch.as_tensor(expected_np)
    torch.testing.assert_close(actual, expected)
