import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

try:
    import ot
except ImportError:  # pragma: no cover - exercised only when POT is absent
    ot = None

# Project root directory (parent of test)
project_root = Path(__file__).parent.parent

# Target module path
module_path = project_root / "torchmorph" / "optimal_transport.py"

# Load module
spec = importlib.util.spec_from_file_location("optimal_transport", module_path)
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Sinkhorn tests")


def _require_pot():
    if ot is None:
        pytest.skip("POT is required for baseline Sinkhorn tests")


def _make_positive(shape, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(shape, generator=generator, device=device) + 0.1


def _expected_batch_channel_n(shape):
    if len(shape) <= 2:
        n = int(np.prod(shape))
        return 1, 1, n
    b, c = shape[:2]
    n = int(np.prod(shape[2:]))
    return b, c, n


def _pot_plan(source, target, cost_matrix, solver):
    source_np = source.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    cost_np = cost_matrix.detach().cpu().numpy()
    return ot.sinkhorn(
        source_np,
        target_np,
        cost_np,
        reg=1.0 / solver.reg,
        numItermax=solver.itrstep,
        stopThr=1e-9,
    )


def test_build_cost_matrix():
    """Simple test for the correctness of the build_cost_matrix function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test 1D shape
    shape_1d = (5,)
    cost_1d = tr.build_cost_matrix(shape_1d, device=device, p=2)
    N_1d = shape_1d[0]
    assert cost_1d.shape == (
        N_1d,
        N_1d,
    ), f"Expected shape {(N_1d, N_1d)}, got {cost_1d.shape}"
    # Diagonal should be 0
    assert torch.allclose(cost_1d.diag(), torch.zeros(N_1d, device=device)), "Diagonal not zero"
    # Symmetry
    assert torch.allclose(cost_1d, cost_1d.T), "Matrix not symmetric"
    # Distance between adjacent elements should be 1
    assert torch.allclose(
        cost_1d[0, 1], torch.tensor(1.0, device=device)
    ), "Adjacent distance should be 1"

    # ---------- 2. Test 2D shape ----------
    shape_2d = (3, 4)  # H=3, W=4
    cost_2d = tr.build_cost_matrix(shape_2d, device=device, p=2)
    N_2d = shape_2d[0] * shape_2d[1]
    assert cost_2d.shape == (N_2d, N_2d), f"Expected shape {(N_2d, N_2d)}, got {cost_2d.shape}"
    # Diagonal should be 0
    assert torch.allclose(cost_2d.diag(), torch.zeros(N_2d, device=device)), "Diagonal not zero"
    # Symmetry
    assert torch.allclose(cost_2d, cost_2d.T), "Matrix not symmetric"

    idx_00 = 0 * shape_2d[1] + 0  # Row 0, Column 0
    idx_01 = 0 * shape_2d[1] + 1  # Row 0, Column 1
    assert torch.allclose(
        cost_2d[idx_00, idx_01], torch.tensor(1.0, device=device)
    ), f"Expected distance 1.0 between adjacent columns, got {cost_2d[idx_00, idx_01]}"

    # ---------- 3. Test 3D shape ----------
    shape_3d = (2, 2, 2)  # A small 2x2x2 volume
    cost_3d = tr.build_cost_matrix(shape_3d, device=device, p=2)
    N_3d = 8
    assert cost_3d.shape == (N_3d, N_3d), f"Expected shape {(N_3d, N_3d)}, got {cost_3d.shape}"
    assert torch.allclose(cost_3d.diag(), torch.zeros(N_3d, device=device))
    assert torch.allclose(cost_3d, cost_3d.T)

    # ---------- 4. Test different p values (Manhattan distance) ----------
    shape = (1, 1, 1, 3)  # Correctly define shape before usage
    print(f"Input shape: {shape}")
    cost_l1 = tr.build_cost_matrix(shape, device=device, p=1)
    cost_l2 = tr.build_cost_matrix(shape, device=device, p=2)

    expected = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], device=device)
    assert torch.allclose(cost_l1, expected), "L1 distance matrix incorrect"
    assert torch.allclose(cost_l2, expected), "L2 distance matrix incorrect"


@pytest.mark.parametrize(
    "shape",
    [
        (4,),
        (2, 3),
        (2, 2, 2),
        (1, 1, 2, 3),
        (2, 3, 2, 2),
        (1, 2, 2, 2, 2),
    ],
)
def test_run_once_accepts_multiple_raw_dimensions(shape):
    _require_cuda()
    device = "cuda"
    solver = tr.SinkhornSolver(reg=0.7, itrstep=300, threshold=0, device=device)

    source = _make_positive(shape, device=device, seed=11)
    target = _make_positive(shape, device=device, seed=23)
    out = solver.run_once(
        source,
        target,
        return_plan=True,
        return_uv=True,
        return_distance=True,
    )

    batch, channel, n = _expected_batch_channel_n(shape)
    assert out["source"].shape == (batch, channel, n)
    assert out["target"].shape == (batch, channel, n)
    assert out["cost_matrix"].shape == (n, n)
    assert out["plan"].shape == (batch, channel, n, n)
    assert out["u"].shape == (batch, channel, n)
    assert out["v"].shape == (batch, channel, n)
    assert out["distance"].shape == (batch, channel)

    distance_from_plan = (out["plan"] * out["cost_matrix"]).sum(dim=(-2, -1))
    assert torch.allclose(out["distance"], distance_from_plan, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "shape",
    [
        (4,),
        (2, 3),
        (2, 2, 2),
        (2, 2, 2, 2),
    ],
)
def test_run_once_distance_matches_pot_sinkhorn(shape):
    _require_cuda()
    _require_pot()
    device = "cuda"
    solver = tr.SinkhornSolver(reg=0.7, itrstep=1000, threshold=0, device=device)

    source = _make_positive(shape, device=device, seed=101)
    target = _make_positive(shape, device=device, seed=211)
    out = solver.run_once(source, target, return_plan=True, return_distance=True)

    expected_distances = []
    for b in range(out["source"].shape[0]):
        channel_distances = []
        for c in range(out["source"].shape[1]):
            pot_transport = _pot_plan(
                out["source"][b, c],
                out["target"][b, c],
                out["cost_matrix"],
                solver,
            )
            pot_distance = np.sum(pot_transport * out["cost_matrix"].detach().cpu().numpy())
            channel_distances.append(pot_distance)
        expected_distances.append(channel_distances)

    expected = torch.tensor(expected_distances, device=device, dtype=out["distance"].dtype)
    assert torch.allclose(out["distance"], expected, rtol=5e-3, atol=1e-5)


def test_run_once_plan_matches_pot_sinkhorn_small():
    _require_cuda()
    _require_pot()
    device = "cuda"
    solver = tr.SinkhornSolver(reg=0.7, itrstep=1000, threshold=0, device=device)

    source = _make_positive((2, 2), device=device, seed=307)
    target = _make_positive((2, 2), device=device, seed=401)
    out = solver.run_once(source, target, return_plan=True)

    pot_transport = _pot_plan(out["source"][0, 0], out["target"][0, 0], out["cost_matrix"], solver)
    expected = torch.tensor(pot_transport, device=device, dtype=out["plan"].dtype)
    assert torch.allclose(out["plan"][0, 0], expected, rtol=5e-3, atol=1e-5)


def test_run_once_cuda_matches_torch_backend_small():
    _require_cuda()
    device = "cuda"
    solver = tr.SinkhornSolver(reg=0.7, itrstep=1000, threshold=0, device=device)

    source = _make_positive((2, 2), device=device, seed=503)
    target = _make_positive((2, 2), device=device, seed=607)

    torch_out = solver.run_once(source, target, return_plan=True, return_distance=True)
    cuda_out = solver.run_once(
        source,
        target,
        use_cuda=True,
        return_plan=True,
        return_distance=True,
    )

    assert cuda_out["backend"] == "cuda"
    assert torch.allclose(cuda_out["plan"], torch_out["plan"], rtol=1e-4, atol=1e-5)
    assert torch.allclose(cuda_out["distance"], torch_out["distance"], rtol=1e-4, atol=1e-5)


def test_run_once_log_domain_cuda_matches_torch_backend_small():
    _require_cuda()
    device = "cuda"
    solver = tr.SinkhornSolver(reg=0.7, itrstep=1000, threshold=0, device=device)

    source = _make_positive((2, 2), device=device, seed=701)
    target = _make_positive((2, 2), device=device, seed=809)

    torch_out = solver.run_once(source, target, return_plan=True, return_distance=True)
    log_out = solver.run_once(
        source,
        target,
        log_domain=True,
        return_plan=True,
        return_distance=True,
    )

    assert log_out["backend"] == "cuda_log"
    assert torch.allclose(log_out["plan"], torch_out["plan"], rtol=1e-4, atol=1e-5)
    assert torch.allclose(log_out["distance"], torch_out["distance"], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("kwargs", [{"use_cuda": True}, {"log_domain": True}])
def test_cuda_backends_reject_batched_inputs(kwargs):
    _require_cuda()
    device = "cuda"
    solver = tr.SinkhornSolver(reg=0.7, itrstep=300, threshold=0, device=device)

    source = _make_positive((2, 1, 2, 2), device=device, seed=911)
    target = _make_positive((2, 1, 2, 2), device=device, seed=919)

    with pytest.raises(ValueError, match="B=C=1"):
        solver.run_once(source, target, **kwargs)


@pytest.mark.parametrize(
    "shape",
    [
        (4,),
        (2, 3),
        (2, 2, 2),
        (1, 1, 2, 3),
        (2, 2, 2, 2),
    ],
)
def test_gradient_matches_numeric_directional_derivative(shape):
    _require_cuda()
    device = "cuda"
    solver = tr.SinkhornSolver(reg=0.7, itrstep=3000, threshold=0, device=device)

    source_raw = _make_positive(shape, device=device, seed=1009)
    target_raw = _make_positive(shape, device=device, seed=2003)
    source, target, cost_matrix, _ = solver.data_preprocess(
        source_raw,
        target_raw,
        force_batched=True,
        dtype=torch.float64,
    )

    direction = _make_positive(source.shape, device=device, seed=3001).double() - 0.6
    direction = direction - direction.mean(dim=-1, keepdim=True)
    eps = 1e-4

    def regularized_objective(x):
        out = solver.run_once(
            x,
            target,
            cost_matrix=cost_matrix,
            return_plan=True,
            return_distance=True,
            dtype=torch.float64,
        )
        plan = out["plan"]
        entropy_term = (plan * torch.log(torch.clamp(plan, min=1e-12))).sum()
        return out["distance"].sum() + entropy_term / solver.reg

    out = solver.run_once(
        source,
        target,
        cost_matrix=cost_matrix,
        return_plan=False,
        return_uv=True,
        return_grad=True,
        return_distance=True,
        dtype=torch.float64,
    )
    analytic_grad = out["grad_source"].reshape(-1)
    analytic_directional = torch.sum(analytic_grad * direction.reshape(-1))

    plus = source + eps * direction
    minus = source - eps * direction
    loss_plus = regularized_objective(plus)
    loss_minus = regularized_objective(minus)
    numeric_directional = (loss_plus - loss_minus) / (2 * eps)

    assert torch.allclose(
        analytic_directional,
        numeric_directional,
        rtol=1e-7,
        atol=1e-7,
    )


def loss_test():
    if not torch.cuda.is_available():
        raise ValueError("CUDA not available, skipping loss test.")

    device = "cuda"
    torch.manual_seed(42)

    reg = 5
    itrstep = 100
    solver = tr.SinkhornSolver(reg=reg, itrstep=itrstep, device=device)

    source = torch.nn.Parameter(torch.randn(4, device=device).abs())
    optimizer = torch.optim.Adam([source], lr=1e-2)

    target = torch.rand(4, device=device).abs()

    for step in range(500):
        optimizer.zero_grad()

        source_prob, target, cost_matrix, cost_matrix_T = solver.data_preprocess(source, target)

        with torch.no_grad():
            u, v = solver.sinkhorn_log_cuda(
                source=source_prob.detach(),
                target=target.detach(),
                cost_matrix=cost_matrix,
                cost_matrix_T=cost_matrix_T,
            )

            K = torch.exp(-cost_matrix * reg)
            P = u.unsqueeze(-1) * K * v.unsqueeze(-2)
            loss_value = (P * cost_matrix).sum()

            grad_source, _ = solver.gradient(u, v)

        source_prob.backward(grad_source)

        optimizer.step()

        print(f"step {step}, loss = {loss_value.item():.6f}")
        print("source:", source.detach())
        print("target:", target)


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-k",
            "not loss_test",
            "-q",
        ]
    )
