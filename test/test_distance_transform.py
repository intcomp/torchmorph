import torch
import pytest
import torchmorph as tm
from scipy.ndimage import distance_transform_edt as dte


@pytest.mark.cuda
def test_distance_transform():
    """Test that tm.foo doubles all tensor elements."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3)
    y = tm.distance_transform(x)

    expected = x * 2
    torch.testing.assert_close(y, expected)
    assert y.device.type == "cuda"
    assert y.shape == x.shape
    print("tm.foo test passed ✅")
