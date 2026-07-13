import importlib

import numpy as np
import pytest
import torch
from scipy.ndimage import binary_closing as scipy_binary_closing
from scipy.ndimage import binary_dilation as scipy_binary_dilation
from scipy.ndimage import binary_erosion as scipy_binary_erosion
from scipy.ndimage import binary_fill_holes as scipy_binary_fill_holes
from scipy.ndimage import binary_hit_or_miss as scipy_binary_hit_or_miss
from scipy.ndimage import binary_opening as scipy_binary_opening
from scipy.ndimage import binary_propagation as scipy_binary_propagation
from scipy.ndimage import generate_binary_structure

import torchmorph as tm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for torchmorph tests"
)

OPERATOR_CASES = [
    ("erosion", tm.binary_erosion, scipy_binary_erosion, {"binary", "torch", "masked", "all"}),
    (
        "dilation",
        tm.binary_dilation,
        scipy_binary_dilation,
        {"binary", "torch", "masked", "all"},
    ),
    ("opening", tm.binary_opening, scipy_binary_opening, {"binary", "torch", "masked", "all"}),
    ("closing", tm.binary_closing, scipy_binary_closing, {"binary", "torch", "masked", "all"}),
    (
        "propagation",
        tm.binary_propagation,
        scipy_binary_propagation,
        {"propagation", "torch", "masked", "all"},
    ),
    ("fill_holes", tm.binary_fill_holes, scipy_binary_fill_holes, {"fill_holes", "torch", "all"}),
    ("hit_or_miss", tm.binary_hit_or_miss, scipy_binary_hit_or_miss, {"hit_or_miss", "all"}),
]


def operator_params(group, *, include_scipy=False):
    params = []
    for name, torch_op, scipy_op, groups in OPERATOR_CASES:
        if group not in groups:
            continue
        args = (torch_op, scipy_op) if include_scipy else (torch_op,)
        params.append(pytest.param(*args, id=name))
    return params


BINARY_OPERATORS = operator_params("binary", include_scipy=True)
PROPAGATION_OPERATORS = operator_params("propagation", include_scipy=True)
FILL_HOLES_OPERATORS = operator_params("fill_holes", include_scipy=True)
HIT_OR_MISS_OPERATORS = operator_params("hit_or_miss", include_scipy=True)
TORCH_OPERATORS = operator_params("torch")
ALL_TORCH_OPERATORS = operator_params("all")
MASKED_TORCH_OPERATORS = operator_params("masked")

CASE_2D = np.array(
    [[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]],
    dtype=np.float32,
)
CASE_3D = np.zeros((2, 1, 5, 5, 5), dtype=bool)
CASE_3D[0, 0, 2:3, 2:4, 1:2] = True
CASE_4D = np.zeros((2, 1, 4, 4, 4, 4), dtype=bool)
CASE_4D[0, 0, 1:3, 1:2, 2:3] = True

CASE_HOLES_2D = np.zeros((1, 1, 5, 5), dtype=bool)
CASE_HOLES_2D[0, 0, 1:4, 1:4] = True
CASE_HOLES_2D[0, 0, 2, 2] = False

CASE_HOLES_3D = np.zeros((1, 1, 5, 5, 5), dtype=bool)
CASE_HOLES_3D[0, 0, 1:4, 1:4, 1:4] = True
CASE_HOLES_3D[0, 0, 2, 2, 2] = False

CASE_HIT_3D = generate_binary_structure(rank=3, connectivity=1)[None, None, ...]

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
        sample_output = None
        if mask is not None:
            sample_kwargs["mask"] = mask
        if output is not None:
            sample_output = np.empty_like(output)
            sample_kwargs["output"] = sample_output
        result = scipy_op(sample, **sample_kwargs)
        if result is None:
            result = sample_output
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


@pytest.mark.parametrize(("torch_op", "scipy_op"), PROPAGATION_OPERATORS)
@pytest.mark.parametrize(
    ("np_input", "structure", "origin", "border_value"),
    [
        pytest.param(CASE_2D, None, 0, False, id="2d_basic"),
        pytest.param(CASE_2D, STRUCTURE_2D, 1, True, id="2d_origin_border"),
        pytest.param(CASE_3D, STRUCTURE_3D_1, 0, False, id="3d_structure_1"),
        pytest.param(CASE_3D, STRUCTURE_3D_2, 1, True, id="3d_origin_border"),
        pytest.param(CASE_4D, STRUCTURE_4D, 0, False, id="4d_structure"),
    ],
)
def test_binary_propagation_matches_scipy(
    torch_op, scipy_op, np_input, structure, origin, border_value
):
    result = torch_op(
        torch.as_tensor(np_input, dtype=torch.float32, device="cuda"),
        structure=optional_cuda_tensor(structure, torch.bool),
        origin=origin,
        border_value=border_value,
    )
    expected = apply_scipy_to_batch(
        np_input,
        scipy_op,
        structure=structure,
        origin=origin,
        border_value=border_value,
    )
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected))


@pytest.mark.parametrize(("torch_op", "scipy_op"), PROPAGATION_OPERATORS)
@pytest.mark.parametrize(
    ("np_input", "mask"),
    [
        pytest.param(CASE_2D, MASK_2D, id="2d_mask"),
        pytest.param(CASE_3D, MASK_3D, id="3d_mask"),
        pytest.param(CASE_4D, MASK_4D, id="4d_mask"),
    ],
)
def test_binary_propagation_mask(torch_op, scipy_op, np_input, mask):
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


@pytest.mark.parametrize(("torch_op", "scipy_op"), PROPAGATION_OPERATORS)
@pytest.mark.parametrize(
    ("np_input", "output_shape"),
    [
        pytest.param(CASE_2D, (3, 3), id="2d_output"),
        pytest.param(CASE_3D, (5, 5, 5), id="3d_output"),
        pytest.param(CASE_4D, (4, 4, 4, 4), id="4d_output"),
    ],
)
def test_binary_propagation_output(torch_op, scipy_op, np_input, output_shape):
    x = torch.as_tensor(np_input, dtype=torch.float32, device="cuda")
    output = torch.empty_like(x, dtype=torch.bool)

    result = torch_op(x, output=output)

    assert result is output
    expected = apply_scipy_to_batch(np_input, scipy_op, output=np.empty(output_shape, dtype=bool))
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected))


@pytest.mark.parametrize(("torch_op", "scipy_op"), FILL_HOLES_OPERATORS)
@pytest.mark.parametrize(
    ("np_input", "structure", "origin"),
    [
        pytest.param(CASE_HOLES_2D, None, 0, id="2d_default"),
        pytest.param(CASE_HOLES_2D, STRUCTURE_2D, 0, id="2d_structure"),
        pytest.param(CASE_HOLES_3D, None, 0, id="3d_default"),
    ],
)
def test_binary_fill_holes_matches_scipy(torch_op, scipy_op, np_input, structure, origin):
    result = torch_op(
        torch.as_tensor(np_input, dtype=torch.float32, device="cuda"),
        structure=optional_cuda_tensor(structure, torch.bool),
        origin=origin,
    )
    expected = apply_scipy_to_batch(
        np_input,
        scipy_op,
        structure=structure,
        origin=origin,
    )
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected))


@pytest.mark.parametrize(("torch_op", "scipy_op"), FILL_HOLES_OPERATORS)
@pytest.mark.parametrize(
    ("np_input", "output_shape"),
    [
        pytest.param(CASE_HOLES_2D, (5, 5), id="2d_output"),
        pytest.param(CASE_HOLES_3D, (5, 5, 5), id="3d_output"),
    ],
)
def test_binary_fill_holes_output(torch_op, scipy_op, np_input, output_shape):
    x = torch.as_tensor(np_input, dtype=torch.float32, device="cuda")
    output = torch.empty_like(x, dtype=torch.bool)

    result = torch_op(x, output=output)

    assert result is output
    expected = apply_scipy_to_batch(np_input, scipy_op, output=np.empty(output_shape, dtype=bool))
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected))


@pytest.mark.parametrize(("torch_op", "scipy_op"), HIT_OR_MISS_OPERATORS)
@pytest.mark.parametrize(
    ("np_input", "structure1", "structure2", "origin1", "origin2"),
    [
        pytest.param(CASE_2D, None, None, 0, None, id="2d_default"),
        pytest.param(
            CASE_2D,
            STRUCTURE_2D,
            np.logical_not(STRUCTURE_2D),
            0,
            None,
            id="2d_structures",
        ),
        pytest.param(CASE_HIT_3D, None, None, 0, None, id="3d_default"),
    ],
)
def test_binary_hit_or_miss_matches_scipy(
    torch_op, scipy_op, np_input, structure1, structure2, origin1, origin2
):
    result = torch_op(
        torch.as_tensor(np_input, dtype=torch.float32, device="cuda"),
        structure1=optional_cuda_tensor(structure1, torch.bool),
        structure2=optional_cuda_tensor(structure2, torch.bool),
        origin1=origin1,
        origin2=origin2,
    )
    expected = apply_scipy_to_batch(
        np_input,
        scipy_op,
        structure1=structure1,
        structure2=structure2,
        origin1=origin1,
        origin2=origin2,
    )
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected))


@pytest.mark.parametrize(("torch_op", "scipy_op"), HIT_OR_MISS_OPERATORS)
def test_binary_hit_or_miss_output(torch_op, scipy_op):
    x = torch.as_tensor(CASE_2D, dtype=torch.float32, device="cuda")
    output = torch.empty_like(x, dtype=torch.bool)

    result = torch_op(x, output=output)

    assert result is output
    expected = apply_scipy_to_batch(CASE_2D, scipy_op, output=np.empty((3, 3), dtype=bool))
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected))


def test_binary_hit_or_miss_allows_empty_miss_structure():
    x = torch.ones((1, 1, 3, 3), dtype=torch.bool, device="cuda")
    structure1 = torch.ones((3, 3), dtype=torch.bool, device="cuda")
    structure2 = torch.zeros((3, 3), dtype=torch.bool, device="cuda")

    result = tm.binary_hit_or_miss(x, structure1=structure1, structure2=structure2)
    expected = tm.binary_erosion(x, structure=structure1)

    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    ("torch_op", "expected_value"),
    [
        pytest.param(tm.binary_erosion, True, id="erosion"),
        pytest.param(tm.binary_dilation, False, id="dilation"),
    ],
)
def test_binary_morphology_supports_empty_structure(torch_op, expected_value):
    x = torch.as_tensor(CASE_2D, dtype=torch.float32, device="cuda")
    structure = torch.zeros((3, 3), dtype=torch.bool, device="cuda")

    result = torch_op(x, structure=structure)

    expected = torch.full_like(x, expected_value, dtype=torch.bool)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    ("torch_op", "scipy_op"),
    [
        pytest.param(tm.binary_opening, scipy_binary_opening, id="opening"),
        pytest.param(tm.binary_closing, scipy_binary_closing, id="closing"),
    ],
)
def test_binary_composite_output_contains_only_final_result(torch_op, scipy_op):
    x = torch.as_tensor(CASE_2D, dtype=torch.float32, device="cuda")
    output = torch.ones_like(x, dtype=torch.bool)

    result = torch_op(x, output=output)

    assert result is output
    expected = apply_scipy_to_batch(CASE_2D, scipy_op, output=np.empty((3, 3), dtype=bool))
    torch.testing.assert_close(output.cpu(), torch.as_tensor(expected))


@pytest.mark.parametrize("torch_op", ALL_TORCH_OPERATORS)
def test_binary_morphology_requires_cuda(torch_op):
    x = torch.as_tensor(CASE_2D, dtype=torch.float32)
    with pytest.raises(ValueError, match="CUDA"):
        torch_op(x)


@pytest.mark.parametrize("torch_op", TORCH_OPERATORS)
def test_binary_morphology_rejects_invalid_origin_dimension(torch_op):
    x = torch.as_tensor(CASE_2D, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="origin dimension"):
        torch_op(x, origin=(0, 0, 0))


def test_binary_hit_or_miss_rejects_invalid_origin1_dimension():
    x = torch.as_tensor(CASE_2D, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="origin dimension"):
        tm.binary_hit_or_miss(x, origin1=(0, 0, 0))


def test_binary_hit_or_miss_rejects_invalid_origin2_dimension():
    x = torch.as_tensor(CASE_2D, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="origin dimension"):
        tm.binary_hit_or_miss(x, origin2=(0, 0, 0))


@pytest.mark.parametrize("torch_op", ALL_TORCH_OPERATORS)
def test_binary_morphology_uses_current_cuda_stream(torch_op):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.as_tensor(CASE_2D, dtype=torch.float32, device="cuda")
        result = torch_op(x)
    stream.synchronize()

    assert result.device == x.device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("torch_op", ALL_TORCH_OPERATORS)
def test_binary_morphology_uses_input_cuda_device(torch_op):
    with torch.cuda.device(0):
        x = torch.as_tensor(CASE_2D, dtype=torch.float32, device="cuda:1")
        result = torch_op(x)
    assert result.device == x.device


@pytest.mark.parametrize("torch_op", TORCH_OPERATORS)
@pytest.mark.parametrize("origin", [-2, 2])
def test_binary_morphology_rejects_invalid_origin_value(torch_op, origin):
    x = torch.as_tensor(CASE_2D, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="invalid origin"):
        torch_op(x, origin=origin)


@pytest.mark.parametrize("torch_op", MASKED_TORCH_OPERATORS)
def test_binary_morphology_requires_mask_shape_to_match_input(torch_op):
    x = torch.as_tensor(CASE_2D, dtype=torch.float32, device="cuda")
    mask = torch.ones(x.shape[-2:], dtype=torch.bool, device="cuda")

    with pytest.raises(ValueError, match="mask shape"):
        torch_op(x, mask=mask)


@pytest.mark.parametrize("torch_op", ALL_TORCH_OPERATORS)
def test_binary_morphology_requires_output_shape_to_match_input(torch_op):
    x = torch.as_tensor(CASE_2D, dtype=torch.float32, device="cuda")
    output = torch.empty((1, 1, 2, 2), dtype=torch.bool, device="cuda")

    with pytest.raises(ValueError, match="output shape"):
        torch_op(x, output=output)


@pytest.mark.parametrize("torch_op", ALL_TORCH_OPERATORS)
def test_binary_morphology_requires_output_on_input_device(torch_op):
    x = torch.as_tensor(CASE_2D, dtype=torch.float32, device="cuda")
    output = torch.empty_like(x, dtype=torch.bool, device="cpu")

    with pytest.raises(ValueError, match="same device"):
        torch_op(x, output=output)


def test_binary_morphology_normalizes_structure_before_iterations(monkeypatch):
    binary_module = importlib.import_module("torchmorph.morphology.binary")
    original_step = binary_module._binary_morphology_cuda_step
    observed_devices = []

    def recording_step(input, structure, border_value, origin, *, mode):
        observed_devices.append(structure.device.type)
        return original_step(input, structure, border_value, origin, mode=mode)

    monkeypatch.setattr(binary_module, "_binary_morphology_cuda_step", recording_step)
    x = torch.as_tensor(CASE_2D, dtype=torch.float32, device="cuda")
    structure = torch.ones((3, 3), dtype=torch.bool, device="cuda")

    tm.binary_dilation(x, structure=structure, iterations=2)

    assert observed_devices == ["cpu", "cpu"]


@pytest.mark.parametrize("spatial_ndim", [1, 8])
def test_binary_morphology_supports_dimension_range(spatial_ndim):
    spatial_shape = (2,) * spatial_ndim
    np_input = np.zeros((1, 1, *spatial_shape), dtype=bool)
    np_input[(0, 0, *((0,) * spatial_ndim))] = True
    x = torch.as_tensor(np_input, device="cuda")

    result = tm.binary_dilation(x)

    expected = apply_scipy_to_batch(np_input, scipy_binary_dilation)
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected))


def test_binary_morphology_rejects_more_than_eight_spatial_dimensions():
    x = torch.zeros((1, 1, *((1,) * 9)), dtype=torch.bool, device="cuda")

    with pytest.raises(ValueError, match="1 to 8"):
        tm.binary_dilation(x)
