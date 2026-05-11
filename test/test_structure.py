import numpy as np  # noqa: F401
import pytest
import torch
from scipy.ndimage import generate_binary_structure as scipy_generate_binary_structure

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
