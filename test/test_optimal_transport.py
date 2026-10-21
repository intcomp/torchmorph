from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from torchmorph import SinkhornSolver, build_cost_matrix
import torchmorph.optimal_transport as ot_impl

try:
    import ot
except ImportError:  # pragma: no cover - exercised only when POT is absent
    ot = None

# Every test parametrized over DEVICES runs on CPU and, when available, CUDA,
# so both the pure-torch and the fused-kernel implementations are covered.
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    ),
]
LOG_SPACE = [False, True]


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


@pytest.mark.parametrize("device", DEVICES)
def test_build_cost_matrix(device):
    # 1-D grid: zero diagonal, symmetry, unit spacing
    cost_1d = build_cost_matrix((5,), device=device)
    assert cost_1d.shape == (5, 5)
    assert cost_1d.device.type == device
    assert torch.allclose(cost_1d.diag(), torch.zeros(5, device=device))
    assert torch.allclose(cost_1d, cost_1d.T)
    assert cost_1d[0, 1].item() == pytest.approx(1.0)

    # 2-D grid: adjacent columns are distance 1 apart
    cost_2d = build_cost_matrix((3, 4), device=device)
    assert cost_2d.shape == (12, 12)
    assert torch.allclose(cost_2d, cost_2d.T)
    assert cost_2d[0, 1].item() == pytest.approx(1.0)

    # p-norms differ on diagonal moves: grid points (0, 0) and (1, 1)
    assert build_cost_matrix((2, 2), p=1, device=device)[0, 3].item() == pytest.approx(2.0)
    assert build_cost_matrix((2, 2), p=2, device=device)[0, 3].item() == pytest.approx(2.0**0.5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log_space", LOG_SPACE)
@pytest.mark.parametrize("n,d", [(1, 4), (3, 6), (2, 16)])
def test_forward_shapes_and_marginals(n, d, log_space, device):
    solver = SinkhornSolver(epsilon=1.5, max_iter=300, log_space=log_space)

    source = _make_positive((n, d), device=device, seed=11)
    target = _make_positive((n, d), device=device, seed=23)

    distance = solver(source, target)
    plan = solver.plan(source, target)
    f, g = solver.potentials(source, target)
    assert distance.shape == (n,)
    assert distance.device.type == device
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log_space", LOG_SPACE)
def test_distance_matches_pot_sinkhorn(log_space, device):
    _require_pot()
    solver = SinkhornSolver(epsilon=1.5, max_iter=1000, log_space=log_space)

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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log_space", LOG_SPACE)
def test_plan_matches_pot_sinkhorn_grid_cost(log_space, device):
    """A flattened 2-D image with an explicit grid cost matrix matches POT."""
    _require_pot()
    solver = SinkhornSolver(epsilon=1.5, max_iter=1000, log_space=log_space)
    cost_matrix = build_cost_matrix((2, 3), device=device)

    source = _make_positive((1, 6), device=device, seed=307)
    target = _make_positive((1, 6), device=device, seed=401)
    plan = solver.plan(source, target, cost_matrix)

    a, b, _ = solver.data_preprocess(source, target, cost_matrix)
    expected = torch.tensor(
        _pot_plan(a[0], b[0], cost_matrix, solver), device=device, dtype=plan.dtype
    )
    assert torch.allclose(plan[0], expected, rtol=5e-3, atol=1e-5)


@pytest.mark.parametrize("log_space", LOG_SPACE)
@pytest.mark.parametrize("n", [1, 3, 20])  # single-item, partial-tile, and multi-tile batches
def test_fused_cuda_kernels_match_cpu(log_space, n):
    """CUDA float32 inputs dispatch to the fused kernels; results must match CPU."""
    _require_cuda()
    solver = SinkhornSolver(epsilon=1.5, max_iter=1000, log_space=log_space)

    source = _make_positive((n, 6), device="cpu", seed=503)
    target = _make_positive((n, 6), device="cpu", seed=607)

    cpu_distance = solver(source, target)
    cpu_plan = solver.plan(source, target)
    cpu_f, cpu_g = solver.potentials(source, target)
    cuda_distance = solver(source.cuda(), target.cuda())
    cuda_plan = solver.plan(source.cuda(), target.cuda())
    cuda_f, cuda_g = solver.potentials(source.cuda(), target.cuda())

    assert torch.allclose(cuda_distance.cpu(), cpu_distance, rtol=1e-4, atol=1e-5)
    assert torch.allclose(cuda_plan.cpu(), cpu_plan, rtol=1e-4, atol=1e-5)
    assert torch.allclose(cuda_f.cpu(), cpu_f, rtol=1e-4, atol=1e-4)
    assert torch.allclose(cuda_g.cpu(), cpu_g, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("log_space", LOG_SPACE)
def test_fused_short_run_without_cuda_graph_matches_cpu(log_space):
    """max_iter below the CUDA-graph threshold takes the plain-launch fused path."""
    _require_cuda()
    solver = SinkhornSolver(epsilon=1.5, max_iter=60, log_space=log_space)

    source = _make_positive((2, 6), device="cpu", seed=1409)
    target = _make_positive((2, 6), device="cpu", seed=1423)

    cpu_distance = solver(source, target)
    cuda_distance = solver(source.cuda(), target.cuda())
    assert torch.allclose(cuda_distance.cpu(), cpu_distance, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("log_space", LOG_SPACE)
@pytest.mark.parametrize("method", ["forward", "plan", "potentials"])
def test_graph_reuses_buffers_with_new_inputs(log_space, method, monkeypatch):
    """New tensors, costs and budgets must not replay stale data or recapture."""
    _require_cuda()
    monkeypatch.setattr(ot_impl, "_GRAPH_CACHE", SimpleNamespace())
    graph = Mock(wraps=torch.cuda.CUDAGraph)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", graph)
    solver = SinkhornSolver(epsilon=1.5, log_space=log_space)
    solve = getattr(solver, method)

    for seed, max_iter in enumerate((100, 101, 124, 125, 200), start=1500):
        solver.max_iter = max_iter
        source = _make_positive((3, 6), device="cpu", seed=seed)
        target = _make_positive((3, 6), device="cpu", seed=seed + 100)
        cost = _make_positive((6, 6), device="cpu", seed=seed + 200)
        expected = solve(source, target, cost)
        actual = solve(source.cuda(), target.cuda(), cost.cuda())
        actual_parts = actual if isinstance(actual, tuple) else (actual,)
        expected_parts = expected if isinstance(expected, tuple) else (expected,)
        for cuda_value, cpu_value in zip(actual_parts, expected_parts):
            assert torch.allclose(cuda_value.cpu(), cpu_value, rtol=1e-4, atol=1e-4)
        assert graph.call_count == 1


def test_graph_iteration_count_and_output_lifetime(monkeypatch):
    """Warmup and previous replays must not affect this call or earlier outputs."""
    _require_cuda()
    monkeypatch.setattr(ot_impl, "_GRAPH_CACHE", SimpleNamespace())
    saved = []

    def launch(inputs, outputs, k):
        outputs[0].add_(inputs[0], alpha=k)

    for value, max_iter in enumerate((99, 100, 101, 124, 125, 200), start=1):
        inputs = (torch.full((2,), float(value), device="cuda"),)
        outputs = (torch.zeros_like(inputs[0]),)
        ot_impl._run_fused(launch, max_iter, inputs[0].device, inputs, outputs, "count")
        saved.append((outputs[0], value * max_iter))
    for result, expected in saved:
        assert torch.equal(result, torch.full_like(result, expected))


def test_graph_rebuilds_for_configuration_and_stream(monkeypatch):
    _require_cuda()
    monkeypatch.setattr(ot_impl, "_GRAPH_CACHE", SimpleNamespace())
    graph = Mock(wraps=torch.cuda.CUDAGraph)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", graph)

    def launch(inputs, outputs, k):
        outputs[0].add_(inputs[0], alpha=k)

    def run(size, signature):
        inputs = (torch.ones(size, device="cuda"),)
        outputs = (torch.zeros_like(inputs[0]),)
        ot_impl._run_fused(launch, 100, inputs[0].device, inputs, outputs, signature)
        assert torch.equal(outputs[0], torch.full_like(outputs[0], 100))

    run(2, ("log", 1.5))
    run(3, ("log", 1.5))
    run(3, ("log", 2.0))
    run(3, "scaling")
    assert graph.call_count == 4
    with torch.cuda.stream(torch.cuda.Stream()):
        run(3, "scaling")
    assert graph.call_count == 5


@pytest.mark.parametrize("stage", ["warmup", "capture", "replay"])
def test_graph_errors_propagate_without_retry(stage, monkeypatch):
    """An error in any graph stage must never trigger a full-budget retry."""
    monkeypatch.setattr(ot_impl, "_GRAPH_CACHE", SimpleNamespace())
    stream = SimpleNamespace(cuda_stream=1, wait_stream=lambda _: None)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    graph = SimpleNamespace(replay=Mock(side_effect=RuntimeError("replay failed")))
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(torch.cuda, "graph", lambda *args, **kwargs: nullcontext())
    error = RuntimeError(f"{stage} failed")
    effects = {"warmup": [error], "capture": [None, error], "replay": [None, None]}
    launch = Mock(side_effect=effects[stage])
    inputs, outputs = (torch.ones(2),), (torch.zeros(2),)
    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        ot_impl._run_fused(launch, 101, "cuda:0", inputs, outputs, "error")
    assert [call.args[2] for call in launch.call_args_list] == [25] * len(effects[stage])


@pytest.mark.parametrize("log_space", LOG_SPACE)
def test_dtype_device_combinations_agree(log_space):
    """float32/float64 on CPU/CUDA must all reach the same solution.

    float64 CUDA inputs exercise the torch fallback on the GPU, float32 CUDA
    inputs the fused kernels, and CPU inputs the pure-torch paths.
    """
    solver = SinkhornSolver(epsilon=1.5, max_iter=2000, log_space=log_space)
    source = _make_positive((2, 8), device="cpu", seed=41)
    target = _make_positive((2, 8), device="cpu", seed=43)
    reference = solver(source.double(), target.double())

    combos = [(source, target)]  # cpu float32
    if torch.cuda.is_available():
        combos += [
            (source.cuda(), target.cuda()),  # cuda float32: fused kernels
            (source.cuda().double(), target.cuda().double()),  # cuda float64: torch ops
        ]
    for src, tgt in combos:
        distance = solver(src, tgt)
        tol = 1e-4 if distance.dtype == torch.float32 else 1e-8
        assert torch.allclose(distance.cpu().double(), reference, rtol=tol, atol=tol)


@pytest.mark.parametrize("device", DEVICES)
def test_log_space_survives_small_epsilon(device):
    """At this epsilon exp(-C/eps) underflows float32, so only log_space=True is usable.

    On CPU this exercises the torch logsumexp path, on CUDA the fused log kernel.
    """
    _require_pot()
    solver = SinkhornSolver(epsilon=0.1, max_iter=5000, log_space=True)
    cost_matrix = build_cost_matrix((8, 8), device=device)

    source = _make_positive((1, 64), device=device, seed=31)
    target = _make_positive((1, 64), device=device, seed=37)

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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log_space", LOG_SPACE)
def test_backward_matches_numeric_directional_derivative(log_space, device):
    """Autograd gradients w.r.t. BOTH source and target must match numeric
    derivatives of the entropic cost.

    Runs in float64, i.e. through the torch paths on both CPU and CUDA.
    """
    n, d = 3, 8
    solver = SinkhornSolver(epsilon=1.5, max_iter=3000, log_space=log_space)
    cost_matrix = build_cost_matrix((d,), device=device).double()

    source = _make_positive((n, d), device=device, seed=1009).double().requires_grad_(True)
    target = _make_positive((n, d), device=device, seed=2003).double().requires_grad_(True)
    direction = _make_positive((n, d), device=device, seed=3001).double() - 0.6

    solver(source, target, cost_matrix).sum().backward()
    assert source.grad is not None and target.grad is not None

    def entropic_cost(src, tgt):
        plan = solver.plan(src, tgt, cost_matrix)
        entropy = (plan * plan.clamp(min=1e-300).log()).sum()
        return (plan * cost_matrix).sum() + solver.epsilon * entropy

    eps_fd = 1e-4
    src, tgt = source.detach(), target.detach()
    for grad, plus, minus in [
        (source.grad, (src + eps_fd * direction, tgt), (src - eps_fd * direction, tgt)),
        (target.grad, (src, tgt + eps_fd * direction), (src, tgt - eps_fd * direction)),
    ]:
        analytic = (grad * direction).sum()
        numeric = (entropic_cost(*plus) - entropic_cost(*minus)) / (2 * eps_fd)
        assert torch.allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("log_space", LOG_SPACE)
def test_backward_through_fused_kernels_matches_cpu(log_space):
    """float32 CUDA backward runs through the fused kernels; grads must match CPU."""
    _require_cuda()
    solver = SinkhornSolver(epsilon=1.5, max_iter=1000, log_space=log_space)

    source = _make_positive((3, 6), device="cpu", seed=1213)
    target = _make_positive((3, 6), device="cpu", seed=1217)

    cpu_source = source.clone().requires_grad_(True)
    cpu_target = target.clone().requires_grad_(True)
    solver(cpu_source, cpu_target).sum().backward()
    cuda_source = source.cuda().requires_grad_(True)
    cuda_target = target.cuda().requires_grad_(True)
    solver(cuda_source, cuda_target).sum().backward()

    for cuda_grad, cpu_grad in [
        (cuda_source.grad, cpu_source.grad),
        (cuda_target.grad, cpu_target.grad),
    ]:
        assert torch.isfinite(cuda_grad).all()
        assert torch.allclose(cuda_grad.cpu(), cpu_grad, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log_space", LOG_SPACE)
def test_threshold_early_stopping_reaches_same_solution(log_space, device):
    """threshold > 0 stops the torch paths early without changing the solution."""
    source = _make_positive((2, 8), device=device, seed=1301)
    target = _make_positive((2, 8), device=device, seed=1303)
    source, target = source.double(), target.double()  # force the torch paths

    full = SinkhornSolver(epsilon=1.5, max_iter=5000, log_space=log_space)
    early = SinkhornSolver(epsilon=1.5, max_iter=5000, threshold=1e-12, log_space=log_space)

    assert torch.allclose(early(source, target), full(source, target), rtol=1e-8, atol=1e-10)
    assert torch.allclose(
        early.plan(source, target), full.plan(source, target), rtol=1e-6, atol=1e-10
    )
