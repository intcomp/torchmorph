import numpy as np  # noqa: F401
import pytest
import torch
from scipy.ndimage import generate_binary_structure as scipy_generate_binary_structure
from scipy.ndimage import iterate_structure as scipy_iterate_structure

import torchmorph as tm  # noqa: F401


@pytest.mark.parametrize(
    "rank, connectivity",
    [
        pytest.param(1, 1, id="1D"),
        pytest.param(2, 1, id="2D1C"),
        pytest.param(2, 2, id="2D2C"),
        pytest.param(3, 1, id="3D1C"),
        pytest.param(3, 3, id="3D3C"),
        pytest.param(4, 4, id="4D4C"),
        pytest.param(7, 2, id="7D2C"),
    ],
)
def test_generate_binary_structure(rank, connectivity):
    expected = scipy_generate_binary_structure(rank, connectivity)
    actual = tm.generate_binary_structure(rank, connectivity)

    np.testing.assert_array_equal(actual, expected)

    expected_tensor = torch.as_tensor(expected)
    torch.testing.assert_close(actual, expected_tensor)


@pytest.mark.parametrize(
    "structure, iterations",
    [
        pytest.param(scipy_generate_binary_structure(1, 1), 3, id="1d_cross_3"),
        pytest.param(scipy_generate_binary_structure(2, 1), 1, id="2d_cross_1"),
        pytest.param(scipy_generate_binary_structure(2, 1), 2, id="2d_cross_2"),
        pytest.param(scipy_generate_binary_structure(2, 2), 3, id="2d_full_3"),
        pytest.param(scipy_generate_binary_structure(3, 1), 2, id="3d_cross_2"),
        pytest.param(np.array([[False, True], [True, True]], dtype=bool), 3, id="asymmetric"),
        pytest.param(np.zeros((3, 3), dtype=bool), 2, id="empty"),
    ],
)
def test_iterate_structure(structure, iterations):
    expected = scipy_iterate_structure(structure, iterations)
    actual = tm.iterate_structure(torch.as_tensor(structure), iterations)

    np.testing.assert_array_equal(actual, expected)
    torch.testing.assert_close(actual, torch.as_tensor(expected))


@pytest.mark.parametrize(
    "origin",
    [
        pytest.param(0, id="scalar_zero"),
        pytest.param((1, 0), id="tuple_positive"),
        pytest.param((-1, 1), id="tuple_mixed"),
    ],
)
def test_iterate_structure_with_origin(origin):
    structure = scipy_generate_binary_structure(2, 1)

    expected_structure, expected_origin = scipy_iterate_structure(structure, 3, origin=origin)
    actual_structure, actual_origin = tm.iterate_structure(
        torch.as_tensor(structure), 3, origin=origin
    )

    np.testing.assert_array_equal(actual_structure, expected_structure)
    torch.testing.assert_close(actual_structure, torch.as_tensor(expected_structure))
    assert actual_origin == expected_origin


def test_iterate_structure_rejects_invalid_origin_dimension():
    structure = torch.ones((3, 3), dtype=torch.bool)

    with pytest.raises(ValueError, match="origin dimension"):
        tm.iterate_structure(structure, 2, origin=(0, 0, 0))
