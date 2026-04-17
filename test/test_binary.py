import numpy as np  # noqa: F401
import pytest
import torch
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    generate_binary_structure,
)

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


def batch_scipy(
    np_input: np.ndarray,
    scipy_func: callable,
    structure: np.ndarray | None = None,
    iterations: int = 1,
    mask: np.ndarray | None = None,
    output: np.ndarray | None = None,
    border_value: bool = False,
    origin: int | tuple[int, ...] = 0,
) -> np.ndarray:
    return apply_scipy_to_batch(
        np_input,
        scipy_func,
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
    ("np_input, scipy_func, structure, iterations, origin, border_value"),
    [
        pytest.param(case_2d, binary_erosion, None, 1, 0, False, id="erosion_2D_basic"),
        pytest.param(
            case_2d, binary_erosion, case_structure_2d, 1, 0, False, id="erosion_2D_structure"
        ),
        pytest.param(case_2d, binary_erosion, None, 2, 0, False, id="erosion_2D_2iterations"),
        pytest.param(case_2d, binary_erosion, None, -1, 0, False, id="erosion_2D_-1iterations"),
        pytest.param(
            case_2d, binary_erosion, None, 1, 1, False, id="erosion_2D_1origin_border_value=False"
        ),
        pytest.param(
            case_2d, binary_erosion, None, 1, 1, True, id="erosion_2D_1origin_border_value=True"
        ),
        pytest.param(case_3d, binary_erosion, None, 1, 0, False, id="erosion_3D_basic"),
        pytest.param(
            case_3d, binary_erosion, case_structure_3d_1, 1, 0, False, id="erosion_3D_2Dstructure1"
        ),
        pytest.param(
            case_3d, binary_erosion, case_structure_3d_2, 1, 0, False, id="erosion_3D_3Dstructure2"
        ),
        pytest.param(case_3d, binary_erosion, None, 2, 0, False, id="erosion_3D_2iterations"),
        pytest.param(case_3d, binary_erosion, None, -1, 0, False, id="erosion_3D_-1iterations"),
        pytest.param(
            case_3d, binary_erosion, None, 1, 1, False, id="erosion_3D_1origin_border_value=False"
        ),
        pytest.param(
            case_3d, binary_erosion, None, 1, 1, True, id="erosion_3D_1origin_border_value=True"
        ),
        pytest.param(case_4d, binary_erosion, None, 1, 0, False, id="erosion_4D_basic"),
        pytest.param(
            case_4d, binary_erosion, case_structure_4d, 1, 0, False, id="erosion_4D_4Dstructure"
        ),
        pytest.param(case_4d, binary_erosion, None, 2, 0, False, id="erosion_4D_2iterations"),
        pytest.param(case_4d, binary_erosion, None, -1, 0, False, id="erosion_4D_-1iterations"),
        pytest.param(
            case_4d, binary_erosion, None, 1, 1, False, id="erosion_4D_1origin_border_value=False"
        ),
        pytest.param(
            case_4d, binary_erosion, None, 1, 1, True, id="erosion_4D_1origin_border_value=True"
        ),
        pytest.param(case_2d, binary_dilation, None, 1, 0, False, id="dilation_2D_basic"),
        pytest.param(
            case_2d, binary_dilation, case_structure_2d, 1, 0, False, id="dilation_2D_structure"
        ),
        pytest.param(case_2d, binary_dilation, None, 2, 0, False, id="dilation_2D_2iterations"),
        pytest.param(case_2d, binary_dilation, None, -1, 0, False, id="dilation_2D_-1iterations"),
        pytest.param(
            case_2d, binary_dilation, None, 1, 1, False, id="dilation_2D_1origin_border_value=False"
        ),
        pytest.param(
            case_2d, binary_dilation, None, 1, 1, True, id="dilation_2D_1origin_border_value=True"
        ),
        pytest.param(case_3d, binary_dilation, None, 1, 0, False, id="dilation_3D_basic"),
        pytest.param(
            case_3d,
            binary_dilation,
            case_structure_3d_1,
            1,
            0,
            False,
            id="dilation_3D_2Dstructure1",
        ),
        pytest.param(
            case_3d,
            binary_dilation,
            case_structure_3d_2,
            1,
            0,
            False,
            id="dilation_3D_3Dstructure2",
        ),
        pytest.param(case_3d, binary_dilation, None, 2, 0, False, id="dilation_3D_2iterations"),
        pytest.param(case_3d, binary_dilation, None, -1, 0, False, id="dilation_3D_-1iterations"),
        pytest.param(
            case_3d, binary_dilation, None, 1, 1, False, id="dilation_3D_1origin_border_value=False"
        ),
        pytest.param(
            case_3d, binary_dilation, None, 1, 1, True, id="dilation_3D_1origin_border_value=True"
        ),
        pytest.param(case_4d, binary_dilation, None, 1, 0, False, id="dilation_4D_basic"),
        pytest.param(
            case_4d, binary_dilation, case_structure_4d, 1, 0, False, id="dilation_4D_4Dstructure"
        ),
        pytest.param(case_4d, binary_dilation, None, 2, 0, False, id="dilation_4D_2iterations"),
        pytest.param(case_4d, binary_dilation, None, -1, 0, False, id="dilation_4D_-1iterations"),
        pytest.param(
            case_4d, binary_dilation, None, 1, 1, False, id="dilation_4D_1origin_border_value=False"
        ),
        pytest.param(
            case_4d, binary_dilation, None, 1, 1, True, id="dilation_4D_1origin_border_value=True"
        ),
        pytest.param(case_2d, binary_opening, None, 1, 0, False, id="opening_2D_basic"),
        pytest.param(
            case_2d, binary_opening, case_structure_2d, 1, 0, False, id="opening_2D_structure"
        ),
        pytest.param(case_2d, binary_opening, None, 2, 0, False, id="opening_2D_2iterations"),
        pytest.param(case_2d, binary_opening, None, -1, 0, False, id="opening_2D_-1iterations"),
        pytest.param(
            case_2d, binary_opening, None, 1, 1, False, id="opening_2D_1origin_border_value=False"
        ),
        pytest.param(
            case_2d, binary_opening, None, 1, 1, True, id="opening_2D_1origin_border_value=True"
        ),
        pytest.param(case_3d, binary_opening, None, 1, 0, False, id="opening_3D_basic"),
        pytest.param(
            case_3d, binary_opening, case_structure_3d_1, 1, 0, False, id="opening_3D_2Dstructure1"
        ),
        pytest.param(
            case_3d, binary_opening, case_structure_3d_2, 1, 0, False, id="opening_3D_3Dstructure2"
        ),
        pytest.param(case_3d, binary_opening, None, 2, 0, False, id="opening_3D_2iterations"),
        pytest.param(case_3d, binary_opening, None, -1, 0, False, id="opening_3D_-1iterations"),
        pytest.param(
            case_3d, binary_opening, None, 1, 1, False, id="opening_3D_1origin_border_value=False"
        ),
        pytest.param(
            case_3d, binary_opening, None, 1, 1, True, id="opening_3D_1origin_border_value=True"
        ),
        pytest.param(case_4d, binary_opening, None, 1, 0, False, id="opening_4D_basic"),
        pytest.param(
            case_4d, binary_opening, case_structure_4d, 1, 0, False, id="opening_4D_4Dstructure"
        ),
        pytest.param(case_4d, binary_opening, None, 2, 0, False, id="opening_4D_2iterations"),
        pytest.param(case_4d, binary_opening, None, -1, 0, False, id="opening_4D_-1iterations"),
        pytest.param(
            case_4d, binary_opening, None, 1, 1, False, id="opening_4D_1origin_border_value=False"
        ),
        pytest.param(
            case_4d, binary_opening, None, 1, 1, True, id="opening_4D_1origin_border_value=True"
        ),
        pytest.param(case_2d, binary_closing, None, 1, 0, False, id="opening_2D_basic"),
        pytest.param(
            case_2d, binary_closing, case_structure_2d, 1, 0, False, id="opening_2D_structure"
        ),
        pytest.param(case_2d, binary_closing, None, 2, 0, False, id="opening_2D_2iterations"),
        pytest.param(case_2d, binary_closing, None, -1, 0, False, id="opening_2D_-1iterations"),
        pytest.param(
            case_2d, binary_closing, None, 1, 1, False, id="opening_2D_1origin_border_value=False"
        ),
        pytest.param(
            case_2d, binary_closing, None, 1, 1, True, id="opening_2D_1origin_border_value=True"
        ),
        pytest.param(case_3d, binary_closing, None, 1, 0, False, id="opening_3D_basic"),
        pytest.param(
            case_3d, binary_closing, case_structure_3d_1, 1, 0, False, id="opening_3D_2Dstructure1"
        ),
        pytest.param(
            case_3d, binary_closing, case_structure_3d_2, 1, 0, False, id="opening_3D_3Dstructure2"
        ),
        pytest.param(case_3d, binary_closing, None, 2, 0, False, id="opening_3D_2iterations"),
        pytest.param(case_3d, binary_closing, None, -1, 0, False, id="opening_3D_-1iterations"),
        pytest.param(
            case_3d, binary_closing, None, 1, 1, False, id="opening_3D_1origin_border_value=False"
        ),
        pytest.param(
            case_3d, binary_closing, None, 1, 1, True, id="opening_3D_1origin_border_value=True"
        ),
        pytest.param(case_4d, binary_closing, None, 1, 0, False, id="opening_4D_basic"),
        pytest.param(
            case_4d, binary_closing, case_structure_4d, 1, 0, False, id="opening_4D_4Dstructure"
        ),
        pytest.param(case_4d, binary_closing, None, 2, 0, False, id="opening_4D_2iterations"),
        pytest.param(case_4d, binary_closing, None, -1, 0, False, id="opening_4D_-1iterations"),
        pytest.param(
            case_4d, binary_closing, None, 1, 1, False, id="opening_4D_1origin_border_value=False"
        ),
        pytest.param(
            case_4d, binary_closing, None, 1, 1, True, id="opening_4D_1origin_border_value=True"
        ),
    ],
)
def test_binary_basic(
    np_input,
    scipy_func,
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
    tm_func = getattr(tm, scipy_func.__name__)
    actual = tm_func(
        x,
        structure=structure_cuda,
        iterations=iterations,
        origin=origin,
        border_value=border_value,
    )
    expected_np = batch_scipy(
        np_input,
        scipy_func,
        structure=structure,
        iterations=iterations,
        origin=origin,
        border_value=border_value,
    )
    expected = torch.as_tensor(expected_np)
    torch.testing.assert_close(actual, expected)
