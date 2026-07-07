import numpy as np
import pytest
import torch
from scipy.ndimage import binary_closing as scipy_binary_closing
from scipy.ndimage import binary_dilation as scipy_binary_dilation
from scipy.ndimage import binary_erosion as scipy_binary_erosion
from scipy.ndimage import binary_opening as scipy_binary_opening
from scipy.ndimage import generate_binary_structure

import torchmorph as tm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for torchmorph tests"
)

BINARY_OPERATORS = [
    pytest.param(tm.binary_erosion, scipy_binary_erosion, id="erosion"),
    pytest.param(tm.binary_dilation, scipy_binary_dilation, id="dilation"),
    pytest.param(tm.binary_opening, scipy_binary_opening, id="opening"),
    pytest.param(tm.binary_closing, scipy_binary_closing, id="closing"),
]

TORCH_OPERATORS = [
    pytest.param(tm.binary_erosion, id="erosion"),
    pytest.param(tm.binary_dilation, id="dilation"),
    pytest.param(tm.binary_opening, id="opening"),
    pytest.param(tm.binary_closing, id="closing"),
]

CASE_2D = np.array(
    [[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]],
    dtype=np.float32,
)
CASE_3D = np.zeros((2, 1, 5, 5, 5), dtype=bool)
CASE_3D[0, 0, 2:3, 2:4, 1:2] = True
CASE_4D = np.zeros((2, 1, 4, 4, 4, 4), dtype=bool)
CASE_4D[0, 0, 1:3, 1:2, 2:3] = True

STRUCTURE_2D = generate_binary_structure(rank=2, connectivity=1)
STRUCTURE_3D_1 = generate_binary_structure(rank=3, connectivity=2)
STRUCTURE_3D_2 = generate_binary_structure(rank=3, connectivity=3)
STRUCTURE_4D = generate_binary_structure(rank=4, connectivity=4)

MASK_2D = np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=bool)
MASK_3D = np.zeros((5, 5, 5), dtype=bool)
MASK_3D[0:2, 0:1, 0:3] = True
MASK_4D = np.zeros((4, 4, 4, 4), dtype=bool)
MASK_4D[1:2, 0:1, 1:3] = True


def apply_scipy_to_batch(np_input, scipy_op, **kwargs):
    batch_shape = np_input.shape[:2]
    spatial_shape = np_input.shape[2:]
    samples = np_input.reshape(-1, *spatial_shape)

    mask = kwargs.pop("mask", None)
    output = kwargs.pop("output", None)
    results = []
    for sample in samples:
        sample_kwargs = dict(kwargs)
        if mask is not None:
            sample_kwargs["mask"] = mask
        if output is not None:
            sample_kwargs["output"] = np.empty_like(output)
        result = scipy_op(sample, **sample_kwargs)
        results.append(np.asarray(result).copy())
    return np.stack(results).reshape(*batch_shape, *spatial_shape)


def optional_cuda_tensor(value, dtype):
    if value is None:
        return None
    return torch.as_tensor(value, dtype=dtype, device="cuda")


def cuda_mask(mask, np_input):
    return torch.as_tensor(mask, dtype=torch.bool, device="cuda").expand(np_input.shape)


@pytest.mark.parametrize(("torch_op", "scipy_op"), BINARY_OPERATORS)
@pytest.mark.parametrize(
    ("np_input", "structure", "iterations", "origin", "border_value"),
    [
        pytest.param(CASE_2D, None, 1, 0, False, id="2d_basic"),
        pytest.param(CASE_2D, STRUCTURE_2D, 1, 0, False, id="2d_structure"),
        pytest.param(CASE_2D, None, 2, 0, False, id="2d_iterations"),
        pytest.param(CASE_2D, None, -1, 0, False, id="2d_until_stable"),
        pytest.param(CASE_2D, None, 1, 1, True, id="2d_origin_border"),
        pytest.param(CASE_3D, None, 1, 0, False, id="3d_basic"),
        pytest.param(CASE_3D, STRUCTURE_3D_1, 1, 0, False, id="3d_structure_1"),
        pytest.param(CASE_3D, STRUCTURE_3D_2, 1, 1, True, id="3d_origin_border"),
        pytest.param(CASE_4D, STRUCTURE_4D, 1, 0, False, id="4d_structure"),
        pytest.param(CASE_4D, None, 2, 0, False, id="4d_iterations"),
    ],
)
def test_binary_morphology_matches_scipy(
    torch_op, scipy_op, np_input, structure, iterations, origin, border_value
):
    result = torch_op(
        torch.as_tensor(np_input, dtype=torch.float32, device="cuda"),
        structure=optional_cuda_tensor(structure, torch.bool),
        iterations=iterations,
        origin=origin,
        border_value=border_value,
    )
    expected = apply_scipy_to_batch(
        np_input,
        scipy_op,
        structure=structure,
        iterations=iterations,
        origin=origin,
        border_value=border_value,
    )
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected))


@pytest.mark.parametrize(("torch_op", "scipy_op"), BINARY_OPERATORS)
@pytest.mark.parametrize(
    ("np_input", "mask"),
    [
        pytest.param(CASE_2D, MASK_2D, id="2d_mask"),
        pytest.param(CASE_3D, MASK_3D, id="3d_mask"),
        pytest.param(CASE_4D, MASK_4D, id="4d_mask"),
    ],
)
def test_binary_morphology_mask(torch_op, scipy_op, np_input, mask):
    result = torch_op(
        torch.as_tensor(np_input, dtype=torch.float32, device="cuda"),
        mask=cuda_mask(mask, np_input),
    )
    expected = apply_scipy_to_batch(np_input, scipy_op, mask=mask)
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected))


@pytest.mark.parametrize(("torch_op", "scipy_op"), BINARY_OPERATORS)
@pytest.mark.parametrize(
    ("np_input", "output_shape"),
    [
        pytest.param(CASE_2D, (3, 3), id="2d_output"),
        pytest.param(CASE_3D, (5, 5, 5), id="3d_output"),
        pytest.param(CASE_4D, (4, 4, 4, 4), id="4d_output"),
    ],
)
def test_binary_morphology_output(torch_op, scipy_op, np_input, output_shape):
    x = torch.as_tensor(np_input, dtype=torch.float32, device="cuda")
    output = torch.empty_like(x, dtype=torch.bool)

    result = torch_op(x, output=output)

    assert result is output
    expected = apply_scipy_to_batch(np_input, scipy_op, output=np.empty(output_shape, dtype=bool))
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected))


@pytest.mark.parametrize("torch_op", TORCH_OPERATORS)
def test_binary_morphology_requires_cuda(torch_op):
    x = torch.as_tensor(CASE_2D, dtype=torch.float32)
    with pytest.raises(ValueError, match="CUDA"):
        torch_op(x)


@pytest.mark.parametrize("torch_op", TORCH_OPERATORS)
def test_binary_morphology_rejects_invalid_origin_dimension(torch_op):
    x = torch.as_tensor(CASE_2D, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="origin dimension"):
        torch_op(x, origin=(0, 0, 0))


@pytest.mark.parametrize("torch_op", TORCH_OPERATORS)
def test_binary_morphology_uses_current_cuda_stream(torch_op):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.as_tensor(CASE_2D, dtype=torch.float32, device="cuda")
        result = torch_op(x)
    stream.synchronize()

    assert result.device == x.device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("torch_op", TORCH_OPERATORS)
def test_binary_morphology_uses_input_cuda_device(torch_op):
    with torch.cuda.device(0):
        x = torch.as_tensor(CASE_2D, dtype=torch.float32, device="cuda:1")
        result = torch_op(x)
    assert result.device == x.device
