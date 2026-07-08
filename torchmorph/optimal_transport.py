from typing import Optional

import torch
from torch import Tensor, nn


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


def _use_fused_kernels(t: Tensor) -> bool:
    return t.is_cuda and t.dtype == torch.float32


def _scaling_iterations(a, b, cost_matrix, epsilon, max_iter, threshold):
    """Plain scaling-form Sinkhorn on (n, d) marginals; returns (log_u, log_v)."""
    K = torch.exp(-cost_matrix / epsilon)
    if _use_fused_kernels(a):
        from torchmorph import _C

        u, v = _C.sinkhorn_fastiter(
            a.contiguous(),
            b.contiguous(),
            K,
            K.mT.contiguous(),
            torch.ones_like(a),
            torch.ones_like(b),
            max_iter,
        )
        return u.log(), v.log()

    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for i in range(max_iter):
        u_prev = u
        u = a / (v @ K.mT + 1e-12)
        v = b / (u @ K + 1e-12)
        if threshold > 0 and i % 10 == 0:
            if (u - u_prev).abs().sum(dim=-1).max() <= threshold:
                break
    return u.log(), v.log()


def _log_iterations(a, b, cost_matrix, epsilon, max_iter, threshold):
    """Log-domain Sinkhorn on (n, d) marginals; returns (log_u, log_v)."""
    log_a, log_b = a.log(), b.log()
    if _use_fused_kernels(a):
        from torchmorph import _C

        return _C.sinkhorn_logiter(
            log_a.contiguous(),
            log_b.contiguous(),
            cost_matrix,
            cost_matrix.mT.contiguous(),
            torch.zeros_like(a),
            torch.zeros_like(b),
            max_iter,
            epsilon,
        )

    log_K = -cost_matrix / epsilon
    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)
    for i in range(max_iter):
        log_u_prev = log_u
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(-2), dim=-1)
        log_v = log_b - torch.logsumexp(log_K.mT + log_u.unsqueeze(-2), dim=-1)
        if threshold > 0 and i % 10 == 0:
            if (log_u - log_u_prev).abs().sum(dim=-1).max() <= threshold:
                break
    return log_u, log_v


class _SinkhornDistance(torch.autograd.Function):
    """Transport distance <P, C> with envelope-theorem gradients.

    The backward pass returns the centered dual potentials, i.e. the exact
    gradients of the entropic OT cost w.r.t. the (normalized) marginals.
    """

    @staticmethod
    def forward(ctx, a, b, cost_matrix, epsilon, max_iter, threshold, log_space):
        iterate = _log_iterations if log_space else _scaling_iterations
        log_u, log_v = iterate(a, b, cost_matrix, epsilon, max_iter, threshold)
        plan = torch.exp(log_u.unsqueeze(-1) - cost_matrix / epsilon + log_v.unsqueeze(-2))
        distance = (plan * cost_matrix).sum(dim=(-2, -1))
        f = epsilon * log_u
        g = epsilon * log_v
        ctx.save_for_backward(f - f.mean(dim=-1, keepdim=True), g - g.mean(dim=-1, keepdim=True))
        return distance

    @staticmethod
    def backward(ctx, grad_output):
        f, g = ctx.saved_tensors
        grad = grad_output.unsqueeze(-1)
        return grad * f, grad * g, None, None, None, None, None


class SinkhornSolver(nn.Module):
    """Entropy-regularized balanced optimal transport as a differentiable module.

    forward(source, target, cost_matrix=None) takes two (n, d) batches of
    histograms sharing one (d, d) cost matrix and returns the (n,) transport
    distances <P, C>, with K = exp(-C / epsilon). Gradients w.r.t. source and
    target are the centered dual potentials (envelope theorem), so the module
    can be used directly as a loss.

    log_space=True runs the iterations entirely in the log domain, which is
    numerically stable for small epsilon. The implementation is picked from
    the input automatically: CUDA float32 tensors use the fused kernels,
    everything else uses pure torch ops (where `threshold` > 0 enables early
    stopping; the fused kernels always run `max_iter` iterations).
    """

    def __init__(self, epsilon=1.0, max_iter=100, threshold=0.0, p=2, log_space=False):
        super().__init__()
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
        self.log_space = log_space

    def extra_repr(self):
        return (
            f"epsilon={self.epsilon}, max_iter={self.max_iter}, "
            f"threshold={self.threshold}, log_space={self.log_space}"
        )

    def data_preprocess(self, source, target, cost_matrix: Optional[Tensor] = None):
        """Validate (n, d) inputs, clamp negatives, and normalize each row to unit mass.

        All ops are differentiable, so forward gradients flow through the
        normalization. When `cost_matrix` is None the d bins are treated as
        points on a line.
        """
        if source.ndim != 2 or source.shape != target.shape:
            raise ValueError("source and target must be (n, d) tensors of the same shape.")
        if not source.is_floating_point():
            raise ValueError("source and target must be floating-point tensors.")
        d = source.shape[1]
        if cost_matrix is None:
            cost_matrix = build_cost_matrix((d,), self.p, source.device)
        elif cost_matrix.shape != (d, d):
            raise ValueError(
                f"cost_matrix must have shape ({d}, {d}), got {tuple(cost_matrix.shape)}."
            )
        cost_matrix = cost_matrix.to(device=source.device, dtype=source.dtype).contiguous()

        source = source.clamp(min=0)
        target = target.clamp(min=0)
        source = source / source.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return source, target, cost_matrix

    def forward(self, source: Tensor, target: Tensor, cost_matrix: Optional[Tensor] = None):
        """(n,) transport distances between the rows of source and target."""
        source, target, cost_matrix = self.data_preprocess(source, target, cost_matrix)
        return _SinkhornDistance.apply(
            source,
            target,
            cost_matrix,
            self.epsilon,
            self.max_iter,
            self.threshold,
            self.log_space,
        )

    @torch.no_grad()
    def plan(self, source: Tensor, target: Tensor, cost_matrix: Optional[Tensor] = None):
        """(n, d, d) transport plans, reconstructed in log space."""
        source, target, cost_matrix = self.data_preprocess(source, target, cost_matrix)
        log_u, log_v = self._iterate(source, target, cost_matrix)
        return torch.exp(log_u.unsqueeze(-1) - cost_matrix / self.epsilon + log_v.unsqueeze(-2))

    @torch.no_grad()
    def potentials(self, source: Tensor, target: Tensor, cost_matrix: Optional[Tensor] = None):
        """Centered dual potentials (f, g), each (n, d).

        These are the gradients of the entropic OT cost w.r.t. the marginals.
        """
        source, target, cost_matrix = self.data_preprocess(source, target, cost_matrix)
        log_u, log_v = self._iterate(source, target, cost_matrix)
        f = self.epsilon * log_u
        g = self.epsilon * log_v
        return f - f.mean(dim=-1, keepdim=True), g - g.mean(dim=-1, keepdim=True)

    def _iterate(self, a, b, cost_matrix):
        iterate = _log_iterations if self.log_space else _scaling_iterations
        return iterate(a, b, cost_matrix, self.epsilon, self.max_iter, self.threshold)
