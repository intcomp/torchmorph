import torch
import pytest
import torchmorph as tm


@pytest.mark.cuda
def test_add():
    """Test that tm.add adds a tensor by a scalar."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    scalar = 3.5
    y = tm.add(x, scalar)

    expected = x + scalar
    torch.testing.assert_close(y, expected)
    assert y.device.type == "cuda"
    assert y.shape == x.shape
    print("tm.bar test passed ✅")
