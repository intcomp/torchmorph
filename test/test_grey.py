import numpy as np
import pytest
import torch
from scipy.ndimage import generate_binary_structure, grey_erosion

import torchmorph as tm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for torchmorph tests"
)


def apply_scipy_to_batch(np_input, scipy_func, **kwargs):
    input_shape = np_input.shape
    batch_shape = input_shape[:2]
    spatial_shape = input_shape[2:]
    batch_size = int(np.prod(batch_shape))
    flatten_input = np_input.reshape(batch_size, *spatial_shape)

    output = kwargs.pop("output", None)
    if output is not None and output.shape != spatial_shape:
        raise ValueError(f"output shape must be {spatial_shape}, got {output.shape}")

    results = []
    for sample in flatten_input:
        sample_kwargs = dict(kwargs)
        if output is not None:
            sample_output = np.empty_like(output)
            sample_kwargs["output"] = sample_output
        result = scipy_func(sample, **sample_kwargs)
        results.append(np.asarray(result).copy())
    return np.stack(results, axis=0).reshape(*batch_shape, *spatial_shape)


# test cases
case_2d = np.array(
    [
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ]
    ],
    dtype=np.float32,
)

case_3d = np.arange(2 * 1 * 4 * 4 * 4, dtype=np.float32).reshape(2, 1, 4, 4, 4)

# structures
footprint_cross = generate_binary_structure(rank=2, connectivity=1).astype(bool)
structure_cross = footprint_cross.astype(np.float32)

footprint_3d = generate_binary_structure(rank=3, connectivity=2).astype(bool)

# output tensors
output_2d_tm = torch.empty(1, 1, 3, 3)


@pytest.mark.parametrize(
    ("np_input", "size", "footprint", "structure", "origin"),
    [
        pytest.param(case_2d, 3, None, None, 0, id="2d_size_int"),
        pytest.param(case_2d, (3, 3), None, None, 0, id="2d_size_tuple"),
        pytest.param(case_2d, None, footprint_cross, None, 0, id="2d_footprint"),
        pytest.param(case_2d, None, None, structure_cross, 0, id="2d_structure"),
        pytest.param(
            case_2d, None, footprint_cross, structure_cross, 0,
            id="2d_structure_and_footprint",
        ),
        pytest.param(case_2d, 3, None, None, 1, id="2d_size_origin_1"),
        pytest.param(case_3d, 3, None, None, 0, id="3d_size"),
    ]
    )
    

def test_grey_erosion_basic(np_input, size, footprint, structure, origin):
    x = torch.as_tensor(np_input, dtype=torch.float32).cuda()
    ft = (
        torch.as_tensor(footprint, dtype=torch.bool).cuda()
        if footprint is not None
        else None
    )
    st = (
        torch.as_tensor(structure, dtype=torch.float32).cuda()
        if structure is not None
        else None
    )
    result = tm.grey_erosion(
        x,
        size=size,
        footprint=ft,
        structure=st,
        origin=origin,
    )
    expected_np = apply_scipy_to_batch(
        np_input,
        grey_erosion,
        size=size,
        footprint=footprint,
        structure=structure,
        origin=origin,
    )
    expected = torch.as_tensor(expected_np)
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.parametrize(
    ("mode", "cval"),
    [
        pytest.param("reflect", 0.0, id="mode_reflect"),
        pytest.param("constant", 0.0, id="mode_constant"),
        pytest.param("constant", 99.0, id="mode_constant_cval99"),
        pytest.param("nearest", 0.0, id="mode_nearest"),
        pytest.param("mirror", 0.0, id="mode_mirror"),
        pytest.param("wrap", 0.0, id="mode_wrap"),
    ],
)
def test_grey_erosion_modes(mode, cval):
    x = torch.as_tensor(case_2d, dtype=torch.float32).cuda()
    result = tm.grey_erosion(x, size=3, mode=mode, cval=cval)
    expected_np = apply_scipy_to_batch(
        case_2d, grey_erosion, size=3, mode=mode, cval=cval
    )
    expected = torch.as_tensor(expected_np)
    torch.testing.assert_close(result.cpu(), expected)


def test_grey_erosion_output():
    x = torch.as_tensor(case_2d, dtype=torch.float32).cuda()
    out = output_2d_tm.cuda()
    result = tm.grey_erosion(x, size=3, output=out)
    assert result is out
    expected_np = apply_scipy_to_batch(case_2d, grey_erosion, size=3)
    expected = torch.as_tensor(expected_np)
    torch.testing.assert_close(result.cpu(), expected)


def test_grey_erosion_no_params():
    x = torch.as_tensor(case_2d, dtype=torch.float32).cuda()
    with pytest.raises(ValueError, match="At least one"):
        tm.grey_erosion(x)


def test_grey_erosion_invalid_device():
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    with pytest.raises(ValueError, match="CUDA"):
        tm.grey_erosion(x, size=3)


def test_grey_erosion_invalid_ndim():
    x = torch.tensor([1.0, 2.0]).cuda()
    with pytest.raises(ValueError, match="at least 3 dimensions"):
        tm.grey_erosion(x, size=3)


def test_grey_erosion_invalid_mode():
    x = torch.as_tensor(case_2d, dtype=torch.float32).cuda()
    with pytest.raises(ValueError, match="Unknown mode"):
        tm.grey_erosion(x, size=3, mode="invalid")
