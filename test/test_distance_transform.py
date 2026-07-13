import numpy as np
import pytest
import torch
from scipy import ndimage as ndi

import torchmorph as tm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for torchmorph tests"
)


def make_case(shape):
    values = np.ones(shape, dtype=np.float32)
    slices = [slice(None, None, 2) for _ in shape[2:]]
    values[(slice(None), slice(None), *slices)] = 0
    return values


CASES = {
    1: make_case((2, 1, 7)),
    2: make_case((2, 1, 5, 6)),
    3: make_case((1, 2, 4, 5, 6)),
}

DISTANCE_OPERATORS = [
    pytest.param(tm.euclidean_distance_transform, id="edt"),
    pytest.param(tm.chamfer_distance_transform, id="cdt"),
    pytest.param(tm.brute_force_distance_transform, id="bfdt"),
]

SCIPY_CASES = [
    pytest.param(tm.euclidean_distance_transform, ndi.distance_transform_edt, {}, {}, id="edt"),
    pytest.param(
        tm.chamfer_distance_transform,
        ndi.distance_transform_cdt,
        {"metric": "chessboard"},
        {"metric": "chessboard"},
        id="cdt-chessboard",
    ),
    pytest.param(
        tm.chamfer_distance_transform,
        ndi.distance_transform_cdt,
        {"metric": "taxicab"},
        {"metric": "taxicab"},
        id="cdt-taxicab",
    ),
    pytest.param(
        tm.brute_force_distance_transform,
        ndi.distance_transform_bf,
        {"metric": "euclidean"},
        {"metric": "euclidean"},
        id="bfdt-euclidean",
    ),
    pytest.param(
        tm.brute_force_distance_transform,
        ndi.distance_transform_bf,
        {"metric": "taxicab"},
        {"metric": "taxicab"},
        id="bfdt-taxicab",
    ),
    pytest.param(
        tm.brute_force_distance_transform,
        ndi.distance_transform_bf,
        {"metric": "chessboard"},
        {"metric": "chessboard"},
        id="bfdt-chessboard",
    ),
]


def scipy_batch(input_array, scipy_op, **kwargs):
    spatial_shape = input_array.shape[2:]
    samples = input_array.reshape(-1, *spatial_shape)
    results = [scipy_op(sample, **kwargs) for sample in samples]
    return np.stack(results).reshape(input_array.shape)


def cuda(array):
    return torch.as_tensor(np.ascontiguousarray(array), device="cuda")


def assert_indices_describe_distances(input, distances, indices, metric, sampling=None):
    spatial_ndim = input.ndim - 2
    assert indices.shape == (spatial_ndim, *input.shape)
    sampling = sampling or [1.0] * spatial_ndim

    for batch in range(input.shape[0]):
        for channel in range(input.shape[1]):
            sample = input[batch, channel]
            sample_indices = indices[:, batch, channel].long()
            assert torch.all(sample[tuple(sample_indices)] == 0)
            axes = torch.meshgrid(
                *[torch.arange(size, device=input.device) for size in input.shape[2:]],
                indexing="ij",
            )
            deltas = [
                (axis - sample_indices[dim]).abs() * sampling[dim] for dim, axis in enumerate(axes)
            ]
            if metric == "euclidean":
                expected = torch.sqrt(sum(delta.square() for delta in deltas))
            elif metric == "taxicab":
                expected = sum(deltas)
            else:
                expected = torch.stack(deltas).amax(dim=0)
            torch.testing.assert_close(distances[batch, channel], expected.float())


@pytest.mark.parametrize(("torch_op", "scipy_op", "torch_kwargs", "scipy_kwargs"), SCIPY_CASES)
@pytest.mark.parametrize("spatial_ndim", [1, 2, 3])
def test_distance_transform_matches_scipy(
    torch_op, scipy_op, torch_kwargs, scipy_kwargs, spatial_ndim
):
    input_array = CASES[spatial_ndim]
    result = torch_op(cuda(input_array), **torch_kwargs)
    expected = scipy_batch(input_array, scipy_op, **scipy_kwargs)
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected).float())


@pytest.mark.parametrize(
    ("torch_op", "kwargs", "metric"),
    [
        pytest.param(tm.euclidean_distance_transform, {}, "euclidean", id="edt"),
        pytest.param(
            tm.chamfer_distance_transform,
            {"metric": "chessboard"},
            "chessboard",
            id="cdt-chessboard",
        ),
        pytest.param(
            tm.chamfer_distance_transform,
            {"metric": "taxicab"},
            "taxicab",
            id="cdt-taxicab",
        ),
        pytest.param(
            tm.brute_force_distance_transform,
            {"metric": "euclidean"},
            "euclidean",
            id="bfdt-euclidean",
        ),
    ],
)
@pytest.mark.parametrize("spatial_ndim", [1, 2, 3])
def test_indices_point_to_nearest_background(torch_op, kwargs, metric, spatial_ndim):
    input = cuda(CASES[spatial_ndim])
    distances, indices = torch_op(input, return_indices=True, **kwargs)
    assert_indices_describe_distances(input, distances, indices, metric)


@pytest.mark.parametrize("distance_op", DISTANCE_OPERATORS)
def test_return_flags(distance_op):
    input = cuda(CASES[2])
    distances = distance_op(input)
    indices = distance_op(input, return_distances=False, return_indices=True)
    both = distance_op(input, return_indices=True)

    assert distances.shape == input.shape
    assert indices.shape == (2, *input.shape)
    assert both[0].shape == input.shape
    assert both[1].shape == (2, *input.shape)
    with pytest.raises(ValueError, match="At least one"):
        distance_op(input, return_distances=False, return_indices=False)


@pytest.mark.parametrize("distance_op", DISTANCE_OPERATORS)
def test_preallocated_outputs_are_filled_and_not_returned(distance_op):
    input = cuda(CASES[2])
    expected_distances, expected_indices = distance_op(input, return_indices=True)
    distances = torch.empty_like(input, dtype=torch.float64)
    indices = torch.empty((2, *input.shape), device="cuda", dtype=torch.int32)

    result = distance_op(
        input,
        return_distances=False,
        return_indices=False,
        distances=distances,
        indices=indices,
    )

    assert result is None
    torch.testing.assert_close(distances, expected_distances.double())
    torch.testing.assert_close(indices, expected_indices.int())


@pytest.mark.parametrize("distance_op", DISTANCE_OPERATORS)
def test_provided_output_auto_enables_only_that_output(distance_op):
    input = cuda(CASES[2])
    distances = torch.empty_like(input)
    result = distance_op(
        input,
        return_distances=False,
        return_indices=True,
        distances=distances,
    )
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, *input.shape)


@pytest.mark.parametrize("distance_op", DISTANCE_OPERATORS)
def test_distance_transform_validates_input(distance_op):
    with pytest.raises(ValueError, match="CUDA"):
        distance_op(torch.zeros((1, 1, 3)))
    with pytest.raises(ValueError, match="at least 3 dimensions"):
        distance_op(torch.zeros((1, 3), device="cuda"))
    with pytest.raises(ValueError, match="empty tensor"):
        distance_op(torch.zeros((1, 1, 0), device="cuda"))
    with pytest.raises(ValueError, match="spatial dimensions"):
        distance_op(torch.zeros((1, 1, *([1] * 9)), device="cuda"))


@pytest.mark.parametrize("distance_op", DISTANCE_OPERATORS)
def test_distance_transform_validates_outputs(distance_op):
    input = cuda(CASES[2])
    with pytest.raises(ValueError, match="distances shape"):
        distance_op(input, distances=torch.empty((1, 1, 2, 2), device="cuda"))
    with pytest.raises(ValueError, match="same device"):
        distance_op(input, distances=torch.empty_like(input, device="cpu"))
    with pytest.raises(ValueError, match="indices shape"):
        distance_op(input, indices=torch.empty((2, 1, 1, 2, 2), device="cuda"))
    with pytest.raises(ValueError, match="same device"):
        distance_op(
            input,
            indices=torch.empty((2, *input.shape), device="cpu", dtype=torch.int64),
        )


SAMPLING_OPERATORS = [
    pytest.param(tm.euclidean_distance_transform, ndi.distance_transform_edt, {}, id="edt"),
    pytest.param(
        tm.brute_force_distance_transform,
        ndi.distance_transform_bf,
        {"metric": "euclidean"},
        id="bfdt",
    ),
]


@pytest.mark.parametrize(("torch_op", "scipy_op", "kwargs"), SAMPLING_OPERATORS)
@pytest.mark.parametrize(
    ("sampling", "normalized"),
    [
        pytest.param(0.5, [0.5, 0.5], id="scalar"),
        pytest.param([0.5], [0.5, 0.5], id="singleton"),
        pytest.param([0.5, 2.0], [0.5, 2.0], id="per-axis"),
    ],
)
def test_sampling_matches_scipy(torch_op, scipy_op, kwargs, sampling, normalized):
    input_array = CASES[2]
    result = torch_op(cuda(input_array), sampling=sampling, **kwargs)
    expected = scipy_batch(input_array, scipy_op, sampling=normalized, **kwargs)
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected).float())


@pytest.mark.parametrize(("distance_op", "_", "kwargs"), SAMPLING_OPERATORS)
@pytest.mark.parametrize("sampling", [[1.0, 2.0, 3.0], 0.0, -1.0, float("inf"), float("nan")])
def test_sampling_validation(distance_op, _, kwargs, sampling):
    with pytest.raises(ValueError, match="sampling"):
        distance_op(cuda(CASES[2]), sampling=sampling, **kwargs)


@pytest.mark.parametrize("algorithm", ["exact", "jfa", "auto"])
@pytest.mark.parametrize("spatial_ndim", [2, 3])
def test_edt_algorithms_match_scipy(algorithm, spatial_ndim):
    input_array = CASES[spatial_ndim]
    result = tm.euclidean_distance_transform(cuda(input_array), algorithm=algorithm)
    expected = scipy_batch(input_array, ndi.distance_transform_edt)
    torch.testing.assert_close(result.cpu(), torch.as_tensor(expected).float())


def test_edt_jfa_falls_back_for_nonunit_sampling():
    input = cuda(CASES[2])
    kwargs = {"sampling": [0.5, 2.0]}
    exact = tm.euclidean_distance_transform(input, algorithm="exact", **kwargs)
    jfa = tm.euclidean_distance_transform(input, algorithm="jfa", **kwargs)
    torch.testing.assert_close(jfa, exact)


@pytest.mark.parametrize(
    ("alias", "canonical"), [("cityblock", "taxicab"), ("manhattan", "taxicab")]
)
def test_cdt_metric_aliases(alias, canonical):
    input = cuda(CASES[2])
    result = tm.chamfer_distance_transform(input, metric=alias)
    expected = tm.chamfer_distance_transform(input, metric=canonical)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    ("distance_op", "kwargs"),
    [
        pytest.param(tm.chamfer_distance_transform, {"metric": "invalid"}, id="cdt"),
        pytest.param(tm.brute_force_distance_transform, {"metric": "invalid"}, id="bfdt"),
    ],
)
def test_invalid_metric(distance_op, kwargs):
    with pytest.raises(ValueError, match="metric must be"):
        distance_op(cuda(CASES[2]), **kwargs)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("distance_op", DISTANCE_OPERATORS)
def test_distance_transform_uses_input_device(distance_op):
    input = torch.as_tensor(CASES[2], device="cuda:1")
    result = distance_op(input)
    assert result.device == input.device
