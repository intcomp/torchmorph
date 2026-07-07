import numpy as np
import pytest
import torch
from scipy.ndimage import black_tophat as scipy_black_tophat
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import grey_closing as scipy_grey_closing
from scipy.ndimage import grey_dilation as scipy_grey_dilation
from scipy.ndimage import grey_erosion as scipy_grey_erosion
from scipy.ndimage import grey_opening as scipy_grey_opening
from scipy.ndimage import morphological_gradient as scipy_morphological_gradient
from scipy.ndimage import morphological_laplace as scipy_morphological_laplace
from scipy.ndimage import white_tophat as scipy_white_tophat

import torchmorph as tm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for torchmorph tests"
)

GREY_OPERATORS = [
    pytest.param(tm.grey_erosion, scipy_grey_erosion, id="erosion"),
    pytest.param(tm.grey_dilation, scipy_grey_dilation, id="dilation"),
    pytest.param(tm.grey_opening, scipy_grey_opening, id="opening"),
    pytest.param(tm.grey_closing, scipy_grey_closing, id="closing"),
    pytest.param(tm.morphological_gradient, scipy_morphological_gradient, id="gradient"),
    pytest.param(tm.morphological_laplace, scipy_morphological_laplace, id="laplace"),
    pytest.param(tm.white_tophat, scipy_white_tophat, id="white_tophat"),
    pytest.param(tm.black_tophat, scipy_black_tophat, id="black_tophat"),
]

TORCH_OPERATORS = [
    pytest.param(tm.grey_erosion, id="erosion"),
    pytest.param(tm.grey_dilation, id="dilation"),
    pytest.param(tm.grey_opening, id="opening"),
    pytest.param(tm.grey_closing, id="closing"),
    pytest.param(tm.morphological_gradient, id="gradient"),
    pytest.param(tm.morphological_laplace, id="laplace"),
    pytest.param(tm.white_tophat, id="white_tophat"),
    pytest.param(tm.black_tophat, id="black_tophat"),
]

CASE_2D = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
CASE_3D = np.arange(2 * 1 * 4 * 4 * 4, dtype=np.float32).reshape(2, 1, 4, 4, 4)

FOOTPRINT_CROSS = generate_binary_structure(rank=2, connectivity=1).astype(bool)
STRUCTURE_CROSS = FOOTPRINT_CROSS.astype(np.float32)
ASYMMETRIC_STRUCTURE = np.arange(9, dtype=np.float32).reshape(3, 3)


def apply_scipy_to_batch(np_input, scipy_op, **kwargs):
    batch_shape = np_input.shape[:2]
    spatial_shape = np_input.shape[2:]
    samples = np_input.reshape(-1, *spatial_shape)
    results = [scipy_op(sample, **kwargs) for sample in samples]
    return np.stack(results).reshape(*batch_shape, *spatial_shape)


def optional_cuda_tensor(value, dtype):
    if value is None:
        return None
    return torch.as_tensor(value, dtype=dtype, device="cuda")


@pytest.mark.parametrize(("torch_op", "scipy_op"), GREY_OPERATORS)
@pytest.mark.parametrize(
    ("np_input", "size", "footprint", "structure", "origin"),
    [
        pytest.param(CASE_2D, 3, None, None, 0, id="2d_size_int"),
        pytest.param(CASE_2D, (2, 3), None, None, 0, id="2d_even_size"),
        pytest.param(CASE_2D, None, FOOTPRINT_CROSS, None, 0, id="2d_footprint"),
        pytest.param(
            CASE_2D,
            None,
            None,
            ASYMMETRIC_STRUCTURE,
            1,
            id="2d_asymmetric_structure_origin",
        ),
        pytest.param(
            CASE_2D,
            None,
            FOOTPRINT_CROSS,
            STRUCTURE_CROSS,
            0,
            id="2d_structure_and_footprint",
        ),
        pytest.param(CASE_3D, 3, None, None, 0, id="3d_size"),
    ],
)
def test_grey_morphology_matches_scipy(
    torch_op, scipy_op, np_input, size, footprint, structure, origin
):
    result = torch_op(
        torch.as_tensor(np_input, device="cuda"),
        size=size,
        footprint=optional_cuda_tensor(footprint, torch.bool),
        structure=optional_cuda_tensor(structure, torch.float32),
        origin=origin,
    )
    expected = apply_scipy_to_batch(
        np_input,
        scipy_op,
        size=size,
        footprint=footprint,
        structure=structure,
        origin=origin,
    )
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected))


@pytest.mark.parametrize(("torch_op", "scipy_op"), GREY_OPERATORS)
@pytest.mark.parametrize(
    ("mode", "cval"),
    [
        pytest.param("reflect", 0.0, id="reflect"),
        pytest.param("constant", 0.0, id="constant_zero"),
        pytest.param("constant", 99.0, id="constant_positive"),
        pytest.param("constant", -99.0, id="constant_negative"),
        pytest.param("nearest", 0.0, id="nearest"),
        pytest.param("mirror", 0.0, id="mirror"),
        pytest.param("wrap", 0.0, id="wrap"),
    ],
)
def test_grey_morphology_border_modes(torch_op, scipy_op, mode, cval):
    x = torch.as_tensor(CASE_2D, device="cuda")
    result = torch_op(x, size=3, mode=mode, cval=cval)
    expected = apply_scipy_to_batch(CASE_2D, scipy_op, size=3, mode=mode, cval=cval)
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected))


@pytest.mark.parametrize(
    ("torch_op", "scipy_op", "value"),
    [
        pytest.param(tm.grey_erosion, scipy_grey_erosion, np.inf, id="erosion_positive"),
        pytest.param(tm.grey_dilation, scipy_grey_dilation, -np.inf, id="dilation_negative"),
    ],
)
def test_grey_morphology_preserves_infinite_extrema(torch_op, scipy_op, value):
    np_input = np.full((1, 1, 5, 5), value, dtype=np.float32)
    result = torch_op(torch.as_tensor(np_input, device="cuda"), size=3)
    expected = apply_scipy_to_batch(np_input, scipy_op, size=3)
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected))


@pytest.mark.parametrize(("torch_op", "scipy_op"), GREY_OPERATORS)
def test_grey_morphology_output(torch_op, scipy_op):
    x = torch.as_tensor(CASE_2D, device="cuda")
    output = torch.empty_like(x)

    result = torch_op(x, size=3, output=output)

    assert result is output
    expected = apply_scipy_to_batch(CASE_2D, scipy_op, size=3)
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected))


@pytest.mark.parametrize("torch_op", TORCH_OPERATORS)
def test_grey_morphology_requires_structuring_element(torch_op):
    x = torch.as_tensor(CASE_2D, device="cuda")
    with pytest.raises(ValueError, match="At least one"):
        torch_op(x)


@pytest.mark.parametrize("torch_op", TORCH_OPERATORS)
def test_grey_morphology_requires_cuda(torch_op):
    x = torch.as_tensor(CASE_2D)
    with pytest.raises(ValueError, match="CUDA"):
        torch_op(x, size=3)


@pytest.mark.parametrize("torch_op", TORCH_OPERATORS)
def test_grey_morphology_requires_spatial_dimensions(torch_op):
    x = torch.tensor([1.0, 2.0], device="cuda")
    with pytest.raises(ValueError, match="at least 3 dimensions"):
        torch_op(x, size=3)


@pytest.mark.parametrize("torch_op", TORCH_OPERATORS)
def test_grey_morphology_rejects_unknown_mode(torch_op):
    x = torch.as_tensor(CASE_2D, device="cuda")
    with pytest.raises(ValueError, match="Unknown mode"):
        torch_op(x, size=3, mode="invalid")


@pytest.mark.parametrize("torch_op", TORCH_OPERATORS)
@pytest.mark.parametrize("origin", [-2, 2])
def test_grey_morphology_rejects_invalid_origin(torch_op, origin):
    x = torch.as_tensor(CASE_2D, device="cuda")
    with pytest.raises(ValueError, match="invalid origin"):
        torch_op(x, size=3, origin=origin)


@pytest.mark.parametrize(("torch_op", "scipy_op"), GREY_OPERATORS)
def test_grey_morphology_uses_current_cuda_stream(torch_op, scipy_op):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.arange(25, dtype=torch.float32, device="cuda").reshape(1, 1, 5, 5)
        result = torch_op(x, size=3)
    stream.synchronize()

    expected = apply_scipy_to_batch(x.cpu().numpy(), scipy_op, size=3)
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("torch_op", TORCH_OPERATORS)
def test_grey_morphology_uses_input_cuda_device(torch_op):
    with torch.cuda.device(0):
        x = torch.arange(25, dtype=torch.float32, device="cuda:1").reshape(1, 1, 5, 5)
        result = torch_op(x, size=3)
    assert result.device == x.device
