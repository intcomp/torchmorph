from typing import Optional

import torch
from torch import Tensor, nn


def build_cost_matrix(shape, p=2, device=None) -> Tensor:
    """Build pairwise distances between points on a spatial grid

    The grid contains ``d = prod(shape)`` points in row-major order. Flatten
    grid-shaped distributions to ``(n, d)`` before passing this matrix to
    :class:`SinkhornSolver`.

    Args:
        shape (sequence[int]): Spatial grid shape.
        p (float): Norm degree passed to :func:`torch.cdist`.
        device (torch.device or str, optional): Device for the returned matrix.

    Returns:
        torch.Tensor: Symmetric ``float32`` cost matrix with shape ``(d, d)``.

    Example:
        ```pycon
        >>> import torchmorph as tm
        >>> cost = tm.build_cost_matrix((2, 3))
        >>> cost.shape
        torch.Size([6, 6])
        >>> cost[0, 1].item()
        1.0
        ```
    """
    coords = torch.stack(
        torch.meshgrid([torch.arange(s, device=device) for s in shape], indexing="ij"), dim=-1
    )
    coords = coords.reshape(-1, len(shape)).float()
    return torch.cdist(coords, coords, p=p)


def _use_fused_kernels(t: Tensor) -> bool:
    return t.is_cuda and t.dtype == torch.float32


def _transpose_or_self(matrix: Tensor) -> Tensor:
    """Return the contiguous transpose, reusing the input when it is symmetric.

    Cost matrices from build_cost_matrix are always symmetric, so this
    usually saves a (d, d) copy.
    """
    return matrix if torch.equal(matrix, matrix.mT) else matrix.mT.contiguous()


_GRAPH_MIN_ITER = 100  # below this, plain launches beat the graph-capture setup cost
_GRAPH_CHUNK = 25


def _run_fused(launch, max_iter, device):
    """Run `launch(k)`, which enqueues k in-place Sinkhorn iterations, max_iter times in total.

    Long runs are launch-latency-bound (two tiny kernels per iteration), so a
    chunk of iterations is captured into a CUDA graph once and replayed. The
    iterations are fixed-point steps on ping-pong buffers, so a fallback that
    runs extra iterations is always safe.
    """
    if max_iter < _GRAPH_MIN_ITER or torch.cuda.is_current_stream_capturing():
        launch(max_iter)
        return
    try:
        with torch.cuda.device(device):
            # Warm up on a side stream (required before capture), doing the
            # first chunk of real iterations in the process.
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                launch(_GRAPH_CHUNK)
            torch.cuda.current_stream().wait_stream(side_stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(_GRAPH_CHUNK)
            for _ in range(max_iter // _GRAPH_CHUNK - 1):
                graph.replay()
            remainder = max_iter % _GRAPH_CHUNK
            if remainder:
                launch(remainder)
    except RuntimeError:
        launch(max_iter)  # graphs unavailable; extra iterations are harmless


def _scaling_iterations(a, b, cost_matrix, epsilon, max_iter, threshold):
    """Plain scaling-form Sinkhorn on (n, d) marginals; returns (log_u, log_v)."""
    K = torch.exp(-cost_matrix / epsilon)
    if _use_fused_kernels(a):
        from torchmorph import _C

        a, b = a.contiguous(), b.contiguous()
        K_T = _transpose_or_self(K)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        _run_fused(lambda k: _C.sinkhorn_fastiter(a, b, K, K_T, u, v, k), max_iter, a.device)
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

        log_a, log_b = log_a.contiguous(), log_b.contiguous()
        cost_T = _transpose_or_self(cost_matrix)
        log_u = torch.zeros_like(a)
        log_v = torch.zeros_like(b)
        _run_fused(
            lambda k: _C.sinkhorn_logiter(
                log_a, log_b, cost_matrix, cost_T, log_u, log_v, k, epsilon
            ),
            max_iter,
            a.device,
        )
        return log_u, log_v

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
    """Solve entropy-regularized balanced optimal transport

    The module accepts batches of flattened histograms and returns transport
    costs that are differentiable with respect to both marginals. CUDA
    ``float32`` inputs use fused kernels; other device and dtype combinations
    use PyTorch operations. Set ``log_space=True`` for better stability at small
    regularization values.

    Args:
        epsilon (float): Positive entropy-regularization strength.
        max_iter (int): Positive maximum number of Sinkhorn iterations.
        threshold (float): Nonnegative early-stopping tolerance checked by the
            PyTorch implementation. Fused CUDA kernels always run ``max_iter``.
        p (float): Norm used by the default one-dimensional cost matrix.
        log_space (bool): Run Sinkhorn iterations in the log domain.

    Example:
        ```pycon
        >>> import torch
        >>> import torchmorph as tm
        >>> source = torch.tensor([[1.0, 0.0, 0.0]])
        >>> target = torch.tensor([[0.0, 0.0, 1.0]])
        >>> solver = tm.SinkhornSolver(epsilon=1.0, max_iter=200)
        >>> solver(source, target).shape
        torch.Size([1])
        ```
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

    def data_preprocess(
        self,
        source,
        target,
        cost_matrix: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Validate and normalize batched transport marginals

        Negative values are clamped to zero and every row is normalized to unit
        mass. These operations remain in the autograd graph. If no cost matrix
        is supplied, the ``d`` bins are treated as points on a line.

        Args:
            source (torch.Tensor): Floating-point source marginals with shape
                ``(n, d)``.
            target (torch.Tensor): Floating-point target marginals with the same
                shape as ``source``.
            cost_matrix (torch.Tensor, optional): Shared pairwise cost matrix
                with shape ``(d, d)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Normalized source,
            normalized target, and contiguous cost matrix on the source device
            and with the source dtype.

        Example:
            ```pycon
            >>> import torch
            >>> import torchmorph as tm
            >>> solver = tm.SinkhornSolver()
            >>> a, b, cost = solver.data_preprocess(
            ...     torch.tensor([[1.0, 1.0]]), torch.tensor([[3.0, 1.0]])
            ... )
            >>> a.sum().item(), b.sum().item(), cost.shape
            (1.0, 1.0, torch.Size([2, 2]))
            ```
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

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        cost_matrix: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute transport costs between corresponding marginal rows

        Args:
            source (torch.Tensor): Floating-point source marginals with shape
                ``(n, d)``.
            target (torch.Tensor): Floating-point target marginals with shape
                ``(n, d)``.
            cost_matrix (torch.Tensor, optional): Shared cost matrix with shape
                ``(d, d)``. If ``None``, uses a one-dimensional grid cost.

        Returns:
            torch.Tensor: Differentiable transport costs with shape ``(n,)``.

        Example:
            ```pycon
            >>> import torch
            >>> import torchmorph as tm
            >>> solver = tm.SinkhornSolver(max_iter=200)
            >>> solver(torch.ones(2, 4), torch.ones(2, 4)).shape
            torch.Size([2])
            ```
        """
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
    def plan(
        self,
        source: Tensor,
        target: Tensor,
        cost_matrix: Optional[Tensor] = None,
    ) -> Tensor:
        """Reconstruct transport plans in the log domain

        This method runs without gradient tracking.

        Args:
            source (torch.Tensor): Floating-point source marginals with shape
                ``(n, d)``.
            target (torch.Tensor): Floating-point target marginals with shape
                ``(n, d)``.
            cost_matrix (torch.Tensor, optional): Shared cost matrix with shape
                ``(d, d)``.

        Returns:
            torch.Tensor: Transport plans with shape ``(n, d, d)``.

        Example:
            ```pycon
            >>> import torch
            >>> import torchmorph as tm
            >>> solver = tm.SinkhornSolver(max_iter=200)
            >>> solver.plan(torch.ones(1, 3), torch.ones(1, 3)).shape
            torch.Size([1, 3, 3])
            ```
        """
        source, target, cost_matrix = self.data_preprocess(source, target, cost_matrix)
        log_u, log_v = self._iterate(source, target, cost_matrix)
        return torch.exp(log_u.unsqueeze(-1) - cost_matrix / self.epsilon + log_v.unsqueeze(-2))

    @torch.no_grad()
    def potentials(
        self,
        source: Tensor,
        target: Tensor,
        cost_matrix: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute centered dual potentials for both marginals

        The potentials are the envelope-theorem gradients of the entropic
        transport cost with respect to normalized marginals. This method runs
        without gradient tracking.

        Args:
            source (torch.Tensor): Floating-point source marginals with shape
                ``(n, d)``.
            target (torch.Tensor): Floating-point target marginals with shape
                ``(n, d)``.
            cost_matrix (torch.Tensor, optional): Shared cost matrix with shape
                ``(d, d)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Source and target potentials,
            each with shape ``(n, d)`` and zero mean along the last axis.

        Example:
            ```pycon
            >>> import torch
            >>> import torchmorph as tm
            >>> solver = tm.SinkhornSolver(max_iter=200)
            >>> f, g = solver.potentials(torch.ones(1, 3), torch.ones(1, 3))
            >>> f.shape, g.shape
            (torch.Size([1, 3]), torch.Size([1, 3]))
            ```
        """
        source, target, cost_matrix = self.data_preprocess(source, target, cost_matrix)
        log_u, log_v = self._iterate(source, target, cost_matrix)
        f = self.epsilon * log_u
        g = self.epsilon * log_v
        return f - f.mean(dim=-1, keepdim=True), g - g.mean(dim=-1, keepdim=True)

    def _iterate(self, a, b, cost_matrix):
        iterate = _log_iterations if self.log_space else _scaling_iterations
        return iterate(a, b, cost_matrix, self.epsilon, self.max_iter, self.threshold)
