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


def scipy_batch_with_indices(input_array, scipy_op, **kwargs):
    spatial_shape = input_array.shape[2:]
    samples = input_array.reshape(-1, *spatial_shape)
    results = [scipy_op(sample, return_indices=True, **kwargs) for sample in samples]
    distances = np.stack([result[0] for result in results]).reshape(input_array.shape)
    indices = np.stack([result[1] for result in results], axis=1).reshape(
        len(spatial_shape), *input_array.shape
    )
    return distances, indices


def cuda(array):
    return torch.as_tensor(np.ascontiguousarray(array), device="cuda")


def own_indices(input):
    spatial_ndim = input.ndim - 2
    coordinates = torch.stack(
        torch.meshgrid(
            *[torch.arange(size, device=input.device) for size in input.shape[2:]],
            indexing="ij",
        )
    ).to(torch.int32)
    return coordinates.view(spatial_ndim, 1, 1, *input.shape[2:]).expand(spatial_ndim, *input.shape)


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


def test_edt_rejects_removed_algorithm_argument():
    with pytest.raises(TypeError, match="algorithm"):
        tm.euclidean_distance_transform(cuda(CASES[2]), algorithm="exact")


@pytest.mark.parametrize(("torch_op", "scipy_op", "torch_kwargs", "scipy_kwargs"), SCIPY_CASES)
@pytest.mark.parametrize("value", [0.0, 1.0], ids=["all-background", "all-foreground"])
def test_uniform_inputs_match_scipy(torch_op, scipy_op, torch_kwargs, scipy_kwargs, value):
    input_array = np.full((2, 2, 3, 4), value, dtype=np.float32)
    distances, indices = torch_op(cuda(input_array), return_indices=True, **torch_kwargs)
    expected_distances, expected_indices = scipy_batch_with_indices(
        input_array, scipy_op, **scipy_kwargs
    )

    torch.testing.assert_close(distances.cpu(), torch.as_tensor(expected_distances).float())
    torch.testing.assert_close(indices.cpu(), torch.as_tensor(expected_indices).int())


@pytest.mark.parametrize(("torch_op", "scipy_op", "torch_kwargs", "scipy_kwargs"), SCIPY_CASES)
@pytest.mark.parametrize("spatial_ndim", [4, 5, 6, 7, 8])
def test_high_dimensional_inputs_match_scipy(
    torch_op, scipy_op, torch_kwargs, scipy_kwargs, spatial_ndim
):
    rng = np.random.default_rng(42)
    input_array = rng.integers(0, 2, size=(2, 2, *([2] * spatial_ndim))).astype(np.float32)
    distances, indices = torch_op(cuda(input_array), return_indices=True, **torch_kwargs)
    expected_distances, expected_indices = scipy_batch_with_indices(
        input_array, scipy_op, **scipy_kwargs
    )
    torch.testing.assert_close(distances.cpu(), torch.as_tensor(expected_distances).float())
    assert indices.dtype == torch.int32
    assert_indices_describe_distances(
        cuda(input_array),
        distances,
        indices,
        torch_kwargs.get("metric", "euclidean"),
    )


@pytest.mark.parametrize(("torch_op", "scipy_op", "torch_kwargs", "scipy_kwargs"), SCIPY_CASES)
def test_noncontiguous_batches_and_channels_match_scipy(
    torch_op, scipy_op, torch_kwargs, scipy_kwargs
):
    input_array = np.ones((2, 2, 5, 6), dtype=np.float32)
    input_array[0, 0] = 0
    input_array[0, 1, ::2, ::2] = 0
    input_array[1, 0, 1::2, ::2] = 0
    input_array[1, 1, 2, 3] = 0
    input = cuda(input_array).transpose(-1, -2)
    assert not input.is_contiguous()

    transposed_array = input.cpu().numpy()
    distances, indices = torch_op(input, return_indices=True, **torch_kwargs)
    expected_distances, expected_indices = scipy_batch_with_indices(
        transposed_array, scipy_op, **scipy_kwargs
    )
    torch.testing.assert_close(distances.cpu(), torch.as_tensor(expected_distances).float())
    assert indices.dtype == torch.int32
    assert_indices_describe_distances(
        input,
        distances,
        indices,
        torch_kwargs.get("metric", "euclidean"),
    )


@pytest.mark.parametrize("distance_op", DISTANCE_OPERATORS)
def test_distance_transform_is_not_differentiable(distance_op):
    input = cuda(CASES[2]).requires_grad_()
    result = distance_op(input)
    assert not result.requires_grad


@pytest.mark.parametrize("distance_op", DISTANCE_OPERATORS)
@pytest.mark.parametrize("spatial_shape", [(16, 16), (2, 2, 2, 2)], ids=["2d", "4d"])
@pytest.mark.parametrize("output_mode", ["distances", "indices", "preallocated"])
def test_distance_transform_uses_current_cuda_stream(distance_op, spatial_shape, output_mode):
    input = torch.ones((1, 1, *spatial_shape), device="cuda")
    stream = torch.cuda.Stream()
    distances = None
    indices = None

    with torch.cuda.stream(stream):
        torch.cuda._sleep(5_000_000)
        input.zero_()
        if output_mode == "distances":
            distances = distance_op(input)
        elif output_mode == "indices":
            indices = distance_op(input, return_distances=False, return_indices=True)
        else:
            distances = torch.full_like(input, float("nan"))
            indices = torch.full(
                (len(spatial_shape), *input.shape),
                -2,
                dtype=torch.int32,
                device=input.device,
            )
            result = distance_op(
                input,
                return_distances=False,
                return_indices=False,
                distances=distances,
                indices=indices,
            )
            assert result is None

    stream.synchronize()
    if distances is not None:
        torch.testing.assert_close(distances, torch.zeros_like(distances))
    if indices is not None:
        torch.testing.assert_close(indices, own_indices(input))


@pytest.mark.parametrize("spatial_ndim", [1, 2, 4])
def test_edt_all_foreground_sampling_matches_scipy(spatial_ndim):
    input_array = np.ones((2, 2, *([3] * spatial_ndim)), dtype=np.float32)
    sampling = [0.5 + dim for dim in range(spatial_ndim)]
    distances, indices = tm.euclidean_distance_transform(
        cuda(input_array), sampling=sampling, return_indices=True
    )
    expected_distances, expected_indices = scipy_batch_with_indices(
        input_array, ndi.distance_transform_edt, sampling=sampling
    )
    torch.testing.assert_close(distances.cpu(), torch.as_tensor(expected_distances).float())
    torch.testing.assert_close(indices.cpu(), torch.as_tensor(expected_indices).int())


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
@pytest.mark.parametrize("output_mode", ["distances", "indices", "preallocated"])
def test_distance_transform_uses_input_device(distance_op, output_mode):
    input = torch.as_tensor(CASES[2], device="cuda:1")
    with torch.cuda.device(0):
        if output_mode == "distances":
            outputs = (distance_op(input),)
        elif output_mode == "indices":
            outputs = (distance_op(input, return_distances=False, return_indices=True),)
        else:
            distances = torch.empty_like(input)
            indices = torch.empty((2, *input.shape), dtype=torch.int32, device=input.device)
            result = distance_op(
                input,
                return_distances=False,
                return_indices=False,
                distances=distances,
                indices=indices,
            )
            assert result is None
            outputs = (distances, indices)

    assert all(output.device == input.device for output in outputs)
