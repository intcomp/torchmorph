import importlib.util
from pathlib import Path

import torch

# Project root directory (parent of test)
project_root = Path(__file__).parent.parent

# Target module path
module_path = project_root / "torchmorph" / "optimal_transport.py"

# Load module
spec = importlib.util.spec_from_file_location("optimal_transport", module_path)
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)


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


def test_sinkhorn_balanced_print():

    print("=" * 50)
    print("Testing sinkhorn_balanced function")
    print("Input shape: ")

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise ValueError("Warning: CUDA not available, tests may fail.")
    else:
        device = 'cuda'
        print("CUDA available, running on GPU.")

    # Create two random distributions
    source = torch.rand(1, 1, 2, 2, device=device)
    target = torch.rand(1, 1, 2, 2, device=device)

    print("\n--- Input source ---")
    print(source)
    print("\n--- Input target ---")
    print(target)

    print("\n--- Calling sinkhorn_balanced ---")
    try:
        solver = tr.SinkhornSolver(
            reg=0.1,
            itrstep=100,
            threshold=1e-5,
            device=device,
            p=2,
        )
        P = solver.sinkhorn(
            source,
            target,
            cost_matrix=None,
            returngrad=False,
        )
        print("\n--- Output P (transport plan) shape ---")
        print(P.shape)
        print("\n--- Output P content (partial) ---")
        print(P)
    except Exception as e:
        print("\n--- Error occurred ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback

        traceback.print_exc()


def test_sinkhorn_cuda():
    if not torch.cuda.is_available():
        raise ValueError("CUDA not available, skipping CUDA test.")

    device = 'cuda'
    # Create two random distributions
    torch.manual_seed(42)
    source = torch.rand(2, 2, device=device)
    target = torch.rand(2, 2, device=device)

    print("\n--- Input source ---")
    print(source)
    print("\n--- Input target ---")
    print(target)

    solver = tr.SinkhornSolver(reg=0.02, itrstep=100, device=device)
    source, target, cost_matrix, _ = solver.data_preprocess(source, target)

    torch_result = solver.sinkhorn(
        source=source,
        target=target,
        cost_matrix=cost_matrix,
        returngrad=True,
    )

    cuda_result = solver.sinkhorn_cuda(
        source=source,
        target=target,
        cost_matrix=cost_matrix,
    )

    torch_plan = torch_result["plan"].squeeze(0).squeeze(0)
    relative_error = torch.linalg.norm(cuda_result["plan"] - torch_plan) / torch.clamp(
        torch.linalg.norm(torch_plan), min=1e-12
    )

    print(f"CUDA relative error: {relative_error.item():.4e}")
    assert relative_error < 1e-4


def test_sinkhorn_log():
    if not torch.cuda.is_available():
        raise ValueError("CUDA not available, skipping CUDA test.")

    device = 'cuda'
    # Create two random distributions
    torch.manual_seed(42)
    source = torch.rand(2, 2, device=device).abs()
    target = torch.rand(2, 2, device=device).abs()

    solver = tr.SinkhornSolver(reg=0.02, itrstep=100, device=device)
    source, target, cost_matrix, cost_matrix_T = solver.data_preprocess(source, target)

    print("\n--- Input source ---")
    print(source)
    print("\n--- Input target ---")
    print(target)

    u, v = solver.sinkhorn_log_cuda(
        source=source,
        target=target,
        cost_matrix=cost_matrix,
        cost_matrix_T=cost_matrix_T,
    )

    print("\n--- Output u  ---")
    print(u)
    print("\n--- Output v  ---")
    print(v)


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
    test_sinkhorn_balanced_print()
    test_build_cost_matrix()
    test_sinkhorn_log()
    test_sinkhorn_cuda()
    loss_test()
