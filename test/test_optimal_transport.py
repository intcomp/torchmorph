import numpy as np
import pytest
import torch

from torchmorph import SinkhornSolver, build_cost_matrix

try:
    import ot
except ImportError:  # pragma: no cover - exercised only when POT is absent
    ot = None


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the CUDA Sinkhorn backends")


def _require_pot():
    if ot is None:
        pytest.skip("POT is required for baseline Sinkhorn tests")


def _make_positive(shape, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(shape, generator=generator, device=device) + 0.1


def _pot_plan(source, target, cost_matrix, solver):
    return ot.sinkhorn(
        source.detach().cpu().numpy().astype(np.float64),
        target.detach().cpu().numpy().astype(np.float64),
        cost_matrix.detach().cpu().numpy().astype(np.float64),
        reg=solver.epsilon,
        numItermax=solver.max_iter,
        stopThr=1e-9,
    )


def test_build_cost_matrix():
    # 1-D grid: zero diagonal, symmetry, unit spacing
    cost_1d = build_cost_matrix((5,))
    assert cost_1d.shape == (5, 5)
    assert torch.allclose(cost_1d.diag(), torch.zeros(5))
    assert torch.allclose(cost_1d, cost_1d.T)
    assert cost_1d[0, 1] == pytest.approx(1.0)

    # 2-D grid: adjacent columns are distance 1 apart
    cost_2d = build_cost_matrix((3, 4))
    assert cost_2d.shape == (12, 12)
    assert torch.allclose(cost_2d, cost_2d.T)
    assert cost_2d[0, 1] == pytest.approx(1.0)

    # p-norms differ on diagonal moves: grid points (0, 0) and (1, 1)
    assert build_cost_matrix((2, 2), p=1)[0, 3] == pytest.approx(2.0)
    assert build_cost_matrix((2, 2), p=2)[0, 3] == pytest.approx(2.0**0.5)


@pytest.mark.parametrize("n,d", [(1, 4), (3, 6), (2, 16)])
def test_solve_shapes(n, d):
    device = _device()
    solver = SinkhornSolver(epsilon=1.5, max_iter=300, device=device)

    source = _make_positive((n, d), device=device, seed=11)
    target = _make_positive((n, d), device=device, seed=23)
    out = solver.solve(
        source,
        target,
        return_plan=True,
        return_potentials=True,
        return_distance=True,
    )

    assert out["source"].shape == (n, d)
    assert out["target"].shape == (n, d)
    assert out["cost_matrix"].shape == (d, d)
    assert out["plan"].shape == (n, d, d)
    assert out["log_u"].shape == (n, d)
    assert out["log_v"].shape == (n, d)
    assert out["distance"].shape == (n,)

    distance_from_plan = (out["plan"] * out["cost_matrix"]).sum(dim=(-2, -1))
    assert torch.allclose(out["distance"], distance_from_plan, rtol=1e-5, atol=1e-6)
    # rows sum to the normalized marginals
    assert torch.allclose(out["source"].sum(dim=-1), torch.ones(n, device=device))


@pytest.mark.parametrize("shape", [(4,), (2, 2, 2)])
def test_solve_rejects_non_2d_inputs(shape):
    solver = SinkhornSolver(epsilon=1.5, max_iter=10)
    source = _make_positive(shape, device="cpu", seed=911)
    target = _make_positive(shape, device="cpu", seed=919)

    with pytest.raises(ValueError, match=r"\(n, d\)"):
        solver.solve(source, target)


def test_solve_rejects_mismatched_cost_matrix():
    solver = SinkhornSolver(epsilon=1.5, max_iter=10)
    source = _make_positive((2, 4), device="cpu", seed=5)
    target = _make_positive((2, 4), device="cpu", seed=7)

    with pytest.raises(ValueError, match="cost_matrix"):
        solver.solve(source, target, cost_matrix=torch.zeros(3, 3))


def test_solve_distance_matches_pot_sinkhorn():
    _require_pot()
    device = _device()
    solver = SinkhornSolver(epsilon=1.5, max_iter=1000, device=device)

    source = _make_positive((3, 8), device=device, seed=101)
    target = _make_positive((3, 8), device=device, seed=211)
    out = solver.solve(source, target, return_distance=True)

    cost_np = out["cost_matrix"].detach().cpu().numpy()
    expected = torch.tensor(
        [
            float(np.sum(_pot_plan(out["source"][i], out["target"][i], out["cost_matrix"], solver) * cost_np))
            for i in range(out["source"].shape[0])
        ],
        device=device,
        dtype=out["distance"].dtype,
    )
    assert torch.allclose(out["distance"], expected, rtol=5e-3, atol=1e-5)


def test_solve_plan_matches_pot_sinkhorn_grid_cost():
    """A flattened 2-D image with an explicit grid cost matrix matches POT."""
    _require_pot()
    device = _device()
    solver = SinkhornSolver(epsilon=1.5, max_iter=1000, device=device)
    cost_matrix = build_cost_matrix((2, 3), device=device)

    source = _make_positive((1, 6), device=device, seed=307)
    target = _make_positive((1, 6), device=device, seed=401)
    out = solver.solve(source, target, cost_matrix=cost_matrix, return_plan=True)

    pot_transport = _pot_plan(out["source"][0], out["target"][0], out["cost_matrix"], solver)
    expected = torch.tensor(pot_transport, device=device, dtype=out["plan"].dtype)
    assert torch.allclose(out["plan"][0], expected, rtol=5e-3, atol=1e-5)


@pytest.mark.parametrize("backend", ["cuda", "cuda_log"])
def test_cuda_backends_match_torch_backend_batched(backend):
    _require_cuda()
    solver = SinkhornSolver(epsilon=1.5, max_iter=1000, device="cuda")

    source = _make_positive((3, 6), device="cuda", seed=503)
    target = _make_positive((3, 6), device="cuda", seed=607)

    torch_out = solver.solve(source, target, return_plan=True, return_distance=True)
    cuda_out = solver.solve(source, target, backend=backend, return_plan=True, return_distance=True)

    assert cuda_out["backend"] == backend
    assert torch.allclose(cuda_out["plan"], torch_out["plan"], rtol=1e-4, atol=1e-5)
    assert torch.allclose(cuda_out["distance"], torch_out["distance"], rtol=1e-4, atol=1e-5)


def test_log_domain_survives_small_epsilon():
    """At this epsilon exp(-C/eps) underflows float32, so only the log backend is usable."""
    _require_cuda()
    _require_pot()
    solver = SinkhornSolver(epsilon=0.1, max_iter=5000, device="cuda")
    cost_matrix = build_cost_matrix((8, 8), device="cuda")

    source = _make_positive((1, 64), device="cuda", seed=31)
    target = _make_positive((1, 64), device="cuda", seed=37)
    out = solver.solve(
        source,
        target,
        cost_matrix=cost_matrix,
        backend="cuda_log",
        return_plan=True,
        return_potentials=True,
        return_distance=True,
    )

    assert torch.isfinite(out["log_u"]).all()
    assert torch.isfinite(out["log_v"]).all()
    plan = out["plan"][0]
    assert torch.isfinite(plan).all()
    assert (plan >= 0).all()
    assert torch.allclose(plan.sum(dim=0), out["target"][0], atol=1e-5)
    assert torch.allclose(plan.sum(dim=1), out["source"][0], atol=5e-3)

    cost_np = out["cost_matrix"].cpu().numpy().astype(np.float64)
    pot_plan = ot.bregman.sinkhorn_log(
        out["source"][0].cpu().numpy().astype(np.float64),
        out["target"][0].cpu().numpy().astype(np.float64),
        cost_np,
        reg=solver.epsilon,
        numItermax=solver.max_iter,
        stopThr=1e-12,
    )
    pot_distance = float(np.sum(pot_plan * cost_np))
    assert out["distance"][0].item() == pytest.approx(pot_distance, rel=2e-2)


@pytest.mark.parametrize("n,d", [(1, 4), (3, 8)])
def test_gradient_matches_numeric_directional_derivative(n, d):
    device = _device()
    solver = SinkhornSolver(epsilon=1.5, max_iter=3000, device=device)

    source_raw = _make_positive((n, d), device=device, seed=1009)
    target_raw = _make_positive((n, d), device=device, seed=2003)
    source, target, cost_matrix = solver.data_preprocess(
        source_raw, target_raw, dtype=torch.float64
    )

    direction = _make_positive((n, d), device=device, seed=3001).double() - 0.6
    direction = direction - direction.mean(dim=-1, keepdim=True)
    eps_fd = 1e-4

    def regularized_cost(x):
        out = solver.solve(
            x, target, cost_matrix=cost_matrix, return_distance=True, dtype=torch.float64
        )
        entropy = (out["plan"] * torch.log(out["plan"].clamp(min=1e-300))).sum()
        return out["distance"].sum() + solver.epsilon * entropy

    out = solver.solve(
        source,
        target,
        cost_matrix=cost_matrix,
        return_plan=False,
        return_grad=True,
        dtype=torch.float64,
    )
    analytic_directional = (out["grad_source"] * direction).sum()

    loss_plus = regularized_cost(source + eps_fd * direction)
    loss_minus = regularized_cost(source - eps_fd * direction)
    numeric_directional = (loss_plus - loss_minus) / (2 * eps_fd)

    assert torch.allclose(analytic_directional, numeric_directional, rtol=1e-5, atol=1e-7)
