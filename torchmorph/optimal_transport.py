from typing import Optional

import torch
from torch import Tensor


def build_cost_matrix(shape, p=2, device=None) -> Tensor:
    """Pairwise L^p distance matrix between the points of a spatial grid.

    Returns a (d, d) cost matrix for the d flattened grid points of `shape`.
    Flatten grid-shaped distributions to rows of an (n, d) tensor and pass the
    result as `cost_matrix` to :class:`SinkhornSolver`.
    """
    coords = torch.stack(
        torch.meshgrid([torch.arange(s, device=device) for s in shape], indexing="ij"), dim=-1
    )
    coords = coords.reshape(-1, len(shape)).float()
    return torch.cdist(coords, coords, p=p)


class SinkhornSolver:
    """Entropy-regularized balanced optimal transport between batched histograms.

    Every method takes source and target as (n, d) tensors: n distributions of
    d bins each, sharing one (d, d) cost matrix. The standard convention
    K = exp(-C / epsilon) is used: smaller epsilon means weaker blurring and a
    sharper transport plan. All backends return the log scaling vectors
    (log_u, log_v); the dual potentials are f = epsilon * log_u and
    g = epsilon * log_v.
    """

    def __init__(self, epsilon=1.0, max_iter=100, threshold=0.0, p=2, device=None):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if threshold < 0:
            raise ValueError("threshold must be non-negative.")
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.threshold = threshold
        self.p = p
        self.device = device

    def data_preprocess(
        self,
        source: Tensor,
        target: Tensor,
        cost_matrix: Optional[Tensor] = None,
        dtype=torch.float32,
    ):
        """Validate (n, d) inputs, clamp negatives, and normalize each row to unit mass.

        When `cost_matrix` is None the d bins are treated as points on a line.
        """
        if source.ndim != 2 or source.shape != target.shape:
            raise ValueError("source and target must be (n, d) tensors of the same shape.")
        device = torch.device(self.device) if self.device is not None else source.device
        d = source.shape[1]
        if cost_matrix is None:
            cost_matrix = build_cost_matrix((d,), self.p, device)
        elif cost_matrix.shape != (d, d):
            raise ValueError(
                f"cost_matrix must have shape ({d}, {d}), got {tuple(cost_matrix.shape)}."
            )

        source = source.to(device=device, dtype=dtype).clamp(min=0)
        target = target.to(device=device, dtype=dtype).clamp(min=0)
        cost_matrix = cost_matrix.to(device=device, dtype=dtype).contiguous()
        source = source / source.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return source, target, cost_matrix

    def solve(
        self,
        source: Tensor,
        target: Tensor,
        cost_matrix: Optional[Tensor] = None,
        *,
        backend: str = "torch",
        return_plan: bool = True,
        return_potentials: bool = False,
        return_grad: bool = False,
        return_distance: bool = False,
        dtype=torch.float32,
    ):
        """Run Sinkhorn end-to-end on (n, d) source and target distributions.

        backend is "torch" (batched matmuls, any device), "cuda" (scaling-form
        kernel), or "cuda_log" (log-domain kernel, stable for small epsilon).
        The (n, d, d) plan is always reconstructed in log space.
        """
        source, target, cost_matrix = self.data_preprocess(source, target, cost_matrix, dtype)
        backends = {
            "torch": self.sinkhorn_torch,
            "cuda": self.sinkhorn_cuda,
            "cuda_log": self.sinkhorn_log_cuda,
        }
        if backend not in backends:
            raise ValueError(f"Unknown backend {backend!r}; expected one of {sorted(backends)}.")
        log_u, log_v = backends[backend](source, target, cost_matrix)

        output = {
            "source": source,
            "target": target,
            "cost_matrix": cost_matrix,
            "backend": backend,
        }
        if return_potentials:
            output["log_u"], output["log_v"] = log_u, log_v
        if return_plan or return_distance:
            plan = torch.exp(log_u.unsqueeze(-1) - cost_matrix / self.epsilon + log_v.unsqueeze(-2))
            if return_plan:
                output["plan"] = plan
            if return_distance:
                output["distance"] = (plan * cost_matrix).sum(dim=(-2, -1))
        if return_grad:
            output["grad_source"], output["grad_target"] = self.gradient(log_u, log_v)
        return output

    def sinkhorn_torch(self, source: Tensor, target: Tensor, cost_matrix: Tensor):
        """Batched Sinkhorn in scaling form with dense matmuls.

        Expects preprocessed (n, d) inputs; returns (log_u, log_v).
        """
        K = torch.exp(-cost_matrix / self.epsilon)
        u = torch.ones_like(source)
        v = torch.ones_like(target)
        for i in range(self.max_iter):
            u_prev = u
            u = source / (v @ K.mT + 1e-12)
            v = target / (u @ K + 1e-12)
            if self.threshold > 0 and i % 10 == 0:
                if (u - u_prev).abs().sum(dim=-1).max() <= self.threshold:
                    break
        return u.log(), v.log()

    @torch.no_grad()
    def sinkhorn_cuda(self, source: Tensor, target: Tensor, cost_matrix: Tensor):
        """Scaling-form Sinkhorn on the fused CUDA kernel; returns (log_u, log_v)."""
        from torchmorph import _C

        K = torch.exp(-cost_matrix / self.epsilon)
        u = torch.ones_like(source)
        v = torch.ones_like(target)
        u, v = _C.sinkhorn_fastiter(
            source.contiguous(), target.contiguous(), K, K.mT.contiguous(), u, v, self.max_iter
        )
        return u.log(), v.log()

    @torch.no_grad()
    def sinkhorn_log_cuda(self, source: Tensor, target: Tensor, cost_matrix: Tensor):
        """Log-domain Sinkhorn on the fused CUDA kernel; stable for small epsilon.

        Returns (log_u, log_v) without ever leaving log space.
        """
        from torchmorph import _C

        log_u = torch.zeros_like(source)
        log_v = torch.zeros_like(target)
        return _C.sinkhorn_logiter(
            source.log().contiguous(),
            target.log().contiguous(),
            cost_matrix,
            cost_matrix.mT.contiguous(),
            log_u,
            log_v,
            self.max_iter,
            self.epsilon,
        )

    def gradient(self, log_u: Tensor, log_v: Tensor):
        """Gradients of the entropic OT cost w.r.t. the marginals.

        These are the dual potentials f = epsilon * log_u and g = epsilon * log_v,
        centered per distribution to fix the additive gauge freedom.
        """
        f = self.epsilon * log_u
        g = self.epsilon * log_v
        return f - f.mean(dim=-1, keepdim=True), g - g.mean(dim=-1, keepdim=True)
