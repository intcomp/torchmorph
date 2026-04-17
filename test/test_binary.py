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
    batch_shape = input_shape[:2]
    spatial_shape = input_shape[2:]
    batch_size = int(np.prod(batch_shape))
    flatten_input = np_input.reshape(batch_size, *spatial_shape)

    mask = kwargs.pop("mask", None)
    output = kwargs.pop("output", None)
    if mask is not None and mask.shape != spatial_shape:
        raise ValueError(f"mask shape must be {spatial_shape}, got {mask.shape}")
    if output is not None and output.shape != spatial_shape:
        raise ValueError(f"output shape must be {spatial_shape}, got {output.shape}")

    results = []
    for sample in flatten_input:
        sample_kwargs = dict(kwargs)
        if mask is not None:
            sample_kwargs["mask"] = mask
        if output is not None:
            sample_output = np.empty_like(output)
            sample_kwargs["output"] = sample_output
        result = scipy_func(sample, **sample_kwargs)
        results.append(np.asarray(result).copy())
    return np.stack(results, axis=0).reshape(*batch_shape, *spatial_shape)


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


# case
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
case_3d = np.zeros((2, 1, 5, 5, 5), dtype=bool)
case_3d[0, 0, 2:3, 2:4, 1:2] = True
case_4d = np.zeros((2, 1, 4, 4, 4, 4), dtype=bool)
case_4d[0, 0, 1:3, 1:2, 2:3] = True

# structure
structure_2d = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

structure_3d_1 = generate_binary_structure(rank=3, connectivity=2)
structure_3d_2 = generate_binary_structure(rank=3, connectivity=3)

structure_4d = generate_binary_structure(rank=4, connectivity=4)

# mask
mask_2d_np = np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=bool)
mask_2d_tm = torch.tensor([[[[1, 1, 0], [1, 1, 1], [1, 1, 0]]]], dtype=bool)

mask_3d_np = np.zeros((5, 5, 5), dtype=bool)
mask_3d_np[0:2, 0:1, 0:3] = True
mask_3d_tm = torch.zeros((2, 1, 5, 5, 5), dtype=bool)
mask_3d_tm[:, :, 0:2, 0:1, 0:3] = True

mask_4d_np = np.zeros((4, 4, 4, 4), dtype=bool)
mask_4d_np[1:2, 0:1, 1:3] = True
mask_4d_tm = torch.zeros((2, 1, 4, 4, 4, 4), dtype=bool)
mask_4d_tm[:, :, 1:2, 0:1, 1:3] = True

# output
output_2d_np = np.empty([3, 3])
output_2d_tm = torch.empty([1, 1, 3, 3])

output_3d_np = np.empty([5, 5, 5])
output_3d_tm = torch.empty([2, 1, 5, 5, 5])

output_4d_np = np.empty([4, 4, 4, 4])
output_4d_tm = torch.empty([2, 1, 4, 4, 4, 4])


@pytest.mark.parametrize(
    ("np_input, scipy_func, structure, iterations, origin, border_value"),
    [
        pytest.param(case_2d, binary_erosion, None, 1, 0, False, id="er_2D_basic"),
        pytest.param(case_2d, binary_erosion, structure_2d, 1, 0, False, id="er_2D_structure"),
        pytest.param(case_2d, binary_erosion, None, 2, 0, False, id="er_2D_2iterations"),
        pytest.param(case_2d, binary_erosion, None, -1, 0, False, id="er_2D_-1iterations"),
        pytest.param(case_2d, binary_erosion, None, 1, 1, False, id="er_2D_1origin_False"),
        pytest.param(case_2d, binary_erosion, None, 1, 1, True, id="er_2D_1origin_True"),
        pytest.param(case_3d, binary_erosion, None, 1, 0, False, id="er_3D_basic"),
        pytest.param(case_3d, binary_erosion, structure_3d_1, 1, 0, False, id="er_3D_2Dstructure1"),
        pytest.param(case_3d, binary_erosion, structure_3d_2, 1, 0, False, id="er_3D_3Dstructure2"),
        pytest.param(case_3d, binary_erosion, None, 2, 0, False, id="er_3D_2iterations"),
        pytest.param(case_3d, binary_erosion, None, -1, 0, False, id="er_3D_-1iterations"),
        pytest.param(case_3d, binary_erosion, None, 1, 1, False, id="er_3D_1origin_False"),
        pytest.param(case_3d, binary_erosion, None, 1, 1, True, id="er_3D_1origin_True"),
        pytest.param(case_4d, binary_erosion, None, 1, 0, False, id="er_4D_basic"),
        pytest.param(case_4d, binary_erosion, structure_4d, 1, 0, False, id="er_4D_4Dstructure"),
        pytest.param(case_4d, binary_erosion, None, 2, 0, False, id="er_4D_2iterations"),
        pytest.param(case_4d, binary_erosion, None, -1, 0, False, id="er_4D_-1iterations"),
        pytest.param(case_4d, binary_erosion, None, 1, 1, False, id="er_4D_1origin_False"),
        pytest.param(case_4d, binary_erosion, None, 1, 1, True, id="er_4D_1origin_True"),
        pytest.param(case_2d, binary_dilation, None, 1, 0, False, id="di_2D_basic"),
        pytest.param(case_2d, binary_dilation, structure_2d, 1, 0, False, id="di_2D_structure"),
        pytest.param(case_2d, binary_dilation, None, 2, 0, False, id="di_2D_2iterations"),
        pytest.param(case_2d, binary_dilation, None, -1, 0, False, id="di_2D_-1iterations"),
        pytest.param(case_2d, binary_dilation, None, 1, 1, False, id="di_2D_1origin_False"),
        pytest.param(case_2d, binary_dilation, None, 1, 1, True, id="di_2D_1origin_True"),
        pytest.param(case_3d, binary_dilation, None, 1, 0, False, id="di_3D_basic"),
        pytest.param(
            case_3d, binary_dilation, structure_3d_1, 1, 0, False, id="di_3D_2Dstructure1"
        ),
        pytest.param(
            case_3d, binary_dilation, structure_3d_2, 1, 0, False, id="di_3D_3Dstructure2"
        ),
        pytest.param(case_3d, binary_dilation, None, 2, 0, False, id="di_3D_2iterations"),
        pytest.param(case_3d, binary_dilation, None, -1, 0, False, id="di_3D_-1iterations"),
        pytest.param(case_3d, binary_dilation, None, 1, 1, False, id="di_3D_1origin_False"),
        pytest.param(case_3d, binary_dilation, None, 1, 1, True, id="di_3D_1origin_True"),
        pytest.param(case_4d, binary_dilation, None, 1, 0, False, id="di_4D_basic"),
        pytest.param(case_4d, binary_dilation, structure_4d, 1, 0, False, id="di_4D_4Dstructure"),
        pytest.param(case_4d, binary_dilation, None, 2, 0, False, id="di_4D_2iterations"),
        pytest.param(case_4d, binary_dilation, None, -1, 0, False, id="di_4D_-1iterations"),
        pytest.param(case_4d, binary_dilation, None, 1, 1, False, id="di_4D_1origin_False"),
        pytest.param(case_4d, binary_dilation, None, 1, 1, True, id="di_4D_1origin_True"),
        pytest.param(case_2d, binary_opening, None, 1, 0, False, id="op_2D_basic"),
        pytest.param(case_2d, binary_opening, structure_2d, 1, 0, False, id="op_2D_structure"),
        pytest.param(case_2d, binary_opening, None, 2, 0, False, id="op_2D_2iterations"),
        pytest.param(case_2d, binary_opening, None, -1, 0, False, id="op_2D_-1iterations"),
        pytest.param(case_2d, binary_opening, None, 1, 1, False, id="op_2D_1origin_False"),
        pytest.param(case_2d, binary_opening, None, 1, 1, True, id="op_2D_1origin_True"),
        pytest.param(case_3d, binary_opening, None, 1, 0, False, id="op_3D_basic"),
        pytest.param(case_3d, binary_opening, structure_3d_1, 1, 0, False, id="op_3D_2Dstructure1"),
        pytest.param(case_3d, binary_opening, structure_3d_2, 1, 0, False, id="op_3D_3Dstructure2"),
        pytest.param(case_3d, binary_opening, None, 2, 0, False, id="op_3D_2iterations"),
        pytest.param(case_3d, binary_opening, None, -1, 0, False, id="op_3D_-1iterations"),
        pytest.param(case_3d, binary_opening, None, 1, 1, False, id="op_3D_1origin_False"),
        pytest.param(case_3d, binary_opening, None, 1, 1, True, id="op_3D_1origin_True"),
        pytest.param(case_4d, binary_opening, None, 1, 0, False, id="op_4D_basic"),
        pytest.param(case_4d, binary_opening, structure_4d, 1, 0, False, id="op_4D_4Dstructure"),
        pytest.param(case_4d, binary_opening, None, 2, 0, False, id="op_4D_2iterations"),
        pytest.param(case_4d, binary_opening, None, -1, 0, False, id="op_4D_-1iterations"),
        pytest.param(case_4d, binary_opening, None, 1, 1, False, id="op_4D_1origin_False"),
        pytest.param(case_4d, binary_opening, None, 1, 1, True, id="op_4D_1origin_True"),
        pytest.param(case_2d, binary_closing, None, 1, 0, False, id="cl_2D_basic"),
        pytest.param(case_2d, binary_closing, structure_2d, 1, 0, False, id="cl_2D_structure"),
        pytest.param(case_2d, binary_closing, None, 2, 0, False, id="cl_2D_2iterations"),
        pytest.param(case_2d, binary_closing, None, -1, 0, False, id="cl_2D_-1iterations"),
        pytest.param(case_2d, binary_closing, None, 1, 1, False, id="cl_2D_1origin_False"),
        pytest.param(case_2d, binary_closing, None, 1, 1, True, id="cl_2D_1origin_True"),
        pytest.param(case_3d, binary_closing, None, 1, 0, False, id="cl_3D_basic"),
        pytest.param(case_3d, binary_closing, structure_3d_1, 1, 0, False, id="cl_3D_2Dstructure1"),
        pytest.param(case_3d, binary_closing, structure_3d_2, 1, 0, False, id="cl_3D_3Dstructure2"),
        pytest.param(case_3d, binary_closing, None, 2, 0, False, id="cl_3D_2iterations"),
        pytest.param(case_3d, binary_closing, None, -1, 0, False, id="cl_3D_-1iterations"),
        pytest.param(case_3d, binary_closing, None, 1, 1, False, id="cl_3D_1origin_False"),
        pytest.param(case_3d, binary_closing, None, 1, 1, True, id="cl_3D_1origin_True"),
        pytest.param(case_4d, binary_closing, None, 1, 0, False, id="cl_4D_basic"),
        pytest.param(case_4d, binary_closing, structure_4d, 1, 0, False, id="cl_4D_4Dstructure"),
        pytest.param(case_4d, binary_closing, None, 2, 0, False, id="cl_4D_2iterations"),
        pytest.param(case_4d, binary_closing, None, -1, 0, False, id="cl_4D_-1iterations"),
        pytest.param(case_4d, binary_closing, None, 1, 1, False, id="cl_4D_1origin_False"),
        pytest.param(case_4d, binary_closing, None, 1, 1, True, id="cl_4D_1origin_True"),
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


@pytest.mark.parametrize(
    ("input_np, scipy_func, mask_np, mask_tm"),
    [
        pytest.param(case_2d, binary_erosion, mask_2d_np, mask_2d_tm, id="er_2d_mask"),
        pytest.param(case_2d, binary_dilation, mask_2d_np, mask_2d_tm, id="di_2d_mask"),
        pytest.param(case_2d, binary_opening, mask_2d_np, mask_2d_tm, id="op_2d_mask"),
        pytest.param(case_2d, binary_closing, mask_2d_np, mask_2d_tm, id="cl_2d_mask"),
        pytest.param(case_3d, binary_erosion, mask_3d_np, mask_3d_tm, id="er_3d_mask"),
        pytest.param(case_3d, binary_dilation, mask_3d_np, mask_3d_tm, id="di_3d_mask"),
        pytest.param(case_3d, binary_opening, mask_3d_np, mask_3d_tm, id="op_3d_mask"),
        pytest.param(case_3d, binary_closing, mask_3d_np, mask_3d_tm, id="cl_3d_mask"),
        pytest.param(case_4d, binary_erosion, mask_4d_np, mask_4d_tm, id="er_4d_mask"),
        pytest.param(case_4d, binary_dilation, mask_4d_np, mask_4d_tm, id="di_4d_mask"),
        pytest.param(case_4d, binary_opening, mask_4d_np, mask_4d_tm, id="op_4d_mask"),
        pytest.param(case_4d, binary_closing, mask_4d_np, mask_4d_tm, id="cl_4d_mask"),
    ],
)
def test_binary_mask(
    input_np,
    scipy_func,
    mask_np,
    mask_tm,
):
    x = torch.as_tensor(input_np, dtype=torch.float32)
    tm_func = getattr(tm, scipy_func.__name__)
    actual = tm_func(
        x,
        mask=mask_tm,
    )

    expected_np = batch_scipy(
        input_np,
        scipy_func,
        mask=mask_np,
    )
    expected = torch.as_tensor(expected_np)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "input_np,scipy_func, output_np, output_tm",
    [
        pytest.param(case_2d, binary_erosion, output_2d_np, output_2d_tm, id='er_output_2d'),
        pytest.param(case_2d, binary_dilation, output_2d_np, output_2d_tm, id='di_output_2d'),
        pytest.param(case_2d, binary_opening, output_2d_np, output_2d_tm, id='op_output_2d'),
        pytest.param(case_2d, binary_closing, output_2d_np, output_2d_tm, id='cl_output_2d'),
        pytest.param(case_3d, binary_erosion, output_3d_np, output_3d_tm, id='er_output_3d'),
        pytest.param(case_3d, binary_dilation, output_3d_np, output_3d_tm, id='di_output_3d'),
        pytest.param(case_3d, binary_opening, output_3d_np, output_3d_tm, id='op_output_3d'),
        pytest.param(case_3d, binary_closing, output_3d_np, output_3d_tm, id='cl_output_3d'),
        pytest.param(case_4d, binary_erosion, output_4d_np, output_4d_tm, id='er_output_4d'),
        pytest.param(case_4d, binary_dilation, output_4d_np, output_4d_tm, id='di_output_4d'),
        pytest.param(case_4d, binary_opening, output_4d_np, output_4d_tm, id='op_output_4d'),
        pytest.param(case_4d, binary_closing, output_4d_np, output_4d_tm, id='cl_output_4d'),
    ],
)
def test_binary_output(
    input_np,
    scipy_func,
    output_np,
    output_tm,
):
    x = torch.as_tensor(input_np, dtype=torch.float32)
    tm_func = getattr(tm, scipy_func.__name__)

    actual = tm_func(
        x,
        output=output_tm,
    )

    expected_np = batch_scipy(
        input_np,
        scipy_func,
        output=output_np,
    )
    expected = torch.as_tensor(expected_np, dtype=torch.float32)
    torch.testing.assert_close(actual, expected)
