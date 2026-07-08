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
        pytest.skip("CUDA is required for the fused Sinkhorn kernels")


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


@pytest.mark.parametrize("log_space", [False, True])
@pytest.mark.parametrize("n,d", [(1, 4), (3, 6), (2, 16)])
def test_forward_shapes_and_marginals(n, d, log_space):
    device = _device()
    solver = SinkhornSolver(epsilon=1.5, max_iter=300, log_space=log_space)

    source = _make_positive((n, d), device=device, seed=11)
    target = _make_positive((n, d), device=device, seed=23)

    distance = solver(source, target)
    plan = solver.plan(source, target)
    f, g = solver.potentials(source, target)
    assert distance.shape == (n,)
    assert plan.shape == (n, d, d)
    assert f.shape == (n, d) and g.shape == (n, d)

    a, b, cost_matrix = solver.data_preprocess(source, target)
    assert torch.allclose(distance, (plan * cost_matrix).sum(dim=(-2, -1)), rtol=1e-5, atol=1e-6)
    assert torch.allclose(plan.sum(dim=-1), a, atol=1e-4)
    assert torch.allclose(plan.sum(dim=-2), b, atol=1e-4)


@pytest.mark.parametrize("shape", [(4,), (2, 2, 2)])
def test_forward_rejects_non_2d_inputs(shape):
    solver = SinkhornSolver(epsilon=1.5, max_iter=10)
    source = _make_positive(shape, device="cpu", seed=911)
    target = _make_positive(shape, device="cpu", seed=919)

    with pytest.raises(ValueError, match=r"\(n, d\)"):
        solver(source, target)


def test_forward_rejects_mismatched_cost_matrix():
    solver = SinkhornSolver(epsilon=1.5, max_iter=10)
    source = _make_positive((2, 4), device="cpu", seed=5)
    target = _make_positive((2, 4), device="cpu", seed=7)

    with pytest.raises(ValueError, match="cost_matrix"):
        solver(source, target, cost_matrix=torch.zeros(3, 3))


def test_distance_matches_pot_sinkhorn():
    _require_pot()
    device = _device()
    solver = SinkhornSolver(epsilon=1.5, max_iter=1000)

    source = _make_positive((3, 8), device=device, seed=101)
    target = _make_positive((3, 8), device=device, seed=211)
    distance = solver(source, target)

    a, b, cost_matrix = solver.data_preprocess(source, target)
    cost_np = cost_matrix.cpu().numpy()
    expected = torch.tensor(
        [
            float(np.sum(_pot_plan(a[i], b[i], cost_matrix, solver) * cost_np))
            for i in range(a.shape[0])
        ],
        device=device,
        dtype=distance.dtype,
    )
    assert torch.allclose(distance, expected, rtol=5e-3, atol=1e-5)


def test_plan_matches_pot_sinkhorn_grid_cost():
    """A flattened 2-D image with an explicit grid cost matrix matches POT."""
    _require_pot()
    device = _device()
    solver = SinkhornSolver(epsilon=1.5, max_iter=1000)
    cost_matrix = build_cost_matrix((2, 3), device=device)

    source = _make_positive((1, 6), device=device, seed=307)
    target = _make_positive((1, 6), device=device, seed=401)
    plan = solver.plan(source, target, cost_matrix)

    a, b, _ = solver.data_preprocess(source, target, cost_matrix)
    expected = torch.tensor(
        _pot_plan(a[0], b[0], cost_matrix, solver), device=device, dtype=plan.dtype
    )
    assert torch.allclose(plan[0], expected, rtol=5e-3, atol=1e-5)


@pytest.mark.parametrize("log_space", [False, True])
def test_fused_cuda_kernels_match_cpu(log_space):
    """CUDA float32 inputs dispatch to the fused kernels; results must match CPU."""
    _require_cuda()
    solver = SinkhornSolver(epsilon=1.5, max_iter=1000, log_space=log_space)

    source = _make_positive((3, 6), device="cpu", seed=503)
    target = _make_positive((3, 6), device="cpu", seed=607)

    cpu_distance = solver(source, target)
    cpu_plan = solver.plan(source, target)
    cuda_distance = solver(source.cuda(), target.cuda())
    cuda_plan = solver.plan(source.cuda(), target.cuda())

    assert torch.allclose(cuda_distance.cpu(), cpu_distance, rtol=1e-4, atol=1e-5)
    assert torch.allclose(cuda_plan.cpu(), cpu_plan, rtol=1e-4, atol=1e-5)


def test_log_space_survives_small_epsilon():
    """At this epsilon exp(-C/eps) underflows float32, so only log_space=True is usable."""
    _require_cuda()
    _require_pot()
    solver = SinkhornSolver(epsilon=0.1, max_iter=5000, log_space=True)
    cost_matrix = build_cost_matrix((8, 8), device="cuda")

    source = _make_positive((1, 64), device="cuda", seed=31)
    target = _make_positive((1, 64), device="cuda", seed=37)

    f, g = solver.potentials(source, target, cost_matrix)
    assert torch.isfinite(f).all()
    assert torch.isfinite(g).all()

    plan = solver.plan(source, target, cost_matrix)[0]
    a, b, _ = solver.data_preprocess(source, target, cost_matrix)
    assert torch.isfinite(plan).all()
    assert (plan >= 0).all()
    assert torch.allclose(plan.sum(dim=0), b[0], atol=1e-5)
    assert torch.allclose(plan.sum(dim=1), a[0], atol=5e-3)

    cost_np = cost_matrix.cpu().numpy().astype(np.float64)
    pot_plan = ot.bregman.sinkhorn_log(
        a[0].cpu().numpy().astype(np.float64),
        b[0].cpu().numpy().astype(np.float64),
        cost_np,
        reg=solver.epsilon,
        numItermax=solver.max_iter,
        stopThr=1e-12,
    )
    pot_distance = float(np.sum(pot_plan * cost_np))
    distance = solver(source, target, cost_matrix)
    assert distance[0].item() == pytest.approx(pot_distance, rel=2e-2)


@pytest.mark.parametrize("log_space", [False, True])
@pytest.mark.parametrize("n,d", [(1, 4), (3, 8)])
def test_backward_matches_numeric_directional_derivative(n, d, log_space):
    """Autograd through forward() must match a numeric derivative of the entropic cost."""
    device = _device()
    solver = SinkhornSolver(epsilon=1.5, max_iter=3000, log_space=log_space)
    cost_matrix = build_cost_matrix((d,), device=device).double()

    source = _make_positive((n, d), device=device, seed=1009).double().requires_grad_(True)
    target = _make_positive((n, d), device=device, seed=2003).double()
    direction = _make_positive((n, d), device=device, seed=3001).double() - 0.6

    solver(source, target, cost_matrix).sum().backward()
    analytic = (source.grad * direction).sum()

    def entropic_cost(x):
        plan = solver.plan(x, target, cost_matrix)
        entropy = (plan * plan.clamp(min=1e-300).log()).sum()
        return (plan * cost_matrix).sum() + solver.epsilon * entropy

    eps_fd = 1e-4
    raw = source.detach()
    numeric = (
        entropic_cost(raw + eps_fd * direction) - entropic_cost(raw - eps_fd * direction)
    ) / (2 * eps_fd)
    assert torch.allclose(analytic, numeric, rtol=1e-5, atol=1e-7)
