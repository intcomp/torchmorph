from typing import Optional

import torch
from torch import Tensor

# from torchmorph import _C


# generate cost matrix for multi-dimensional tensor like 2D or 3D tensor
def build_cost_matrix(
    shape, device: str = 'cuda', p=2  # Euclidean distance by default
) -> torch.Tensor:

    coords = torch.stack(torch.meshgrid([torch.arange(s) for s in shape], indexing="ij"), dim=-1)
    coords_flat = coords.view(-1, coords.size(-1)).float()

    cost_matrix = torch.cdist(coords_flat, coords_flat, p=p).to(device)
    return cost_matrix


def data_preprocess(
    source: Tensor,
    target: Tensor,
    cost_matrix: Optional[Tensor] = None,
    p: int = 2,
    force_batched: bool = False,
    device: str = 'cuda',
    dtype=torch.float32,
):
    """Validate, move, reshape, and normalize Sinkhorn input distributions.

    If batch_flag is True, 1D/2D inputs are reshaped to (1, 1, N).
    Higher-dimensional inputs are reshaped from (B, C, *Spatial) to (B, C, N).
    """
    # check dimensions,reshape and choose strategy
    if not source.ndim == target.ndim or not source.shape == target.shape:
        raise ValueError(
            "Source and target must have the same number of dimensions and the same shape."
        )
    elif source.ndim <= 2:
        if cost_matrix is None:
            cost_matrix = build_cost_matrix(source.shape, device=device, p=p)
        if force_batched:
            source = source.view(1, 1, -1)
            target = target.view(1, 1, -1)
        else:
            source = source.view(-1)
            target = target.view(-1)
    elif source.ndim > 2:
        # cut and reshape source and target to coordinate tensors, then calculate cost matrix
        B, C = source.shape[:2]  # get batch and channel dimensions
        if cost_matrix is None:
            spatial_shape = source.shape[2:]
            cost_matrix = build_cost_matrix(spatial_shape, device=device, p=p)
        source = source.view(B, C, -1)
        target = target.view(B, C, -1)

    # move to specified device(cuda) and ensure dtype float32
    source = source.to(device=device, dtype=dtype)
    target = target.to(device=device, dtype=dtype)
    cost_matrix = cost_matrix.to(device=device, dtype=dtype)

    # ensure non-negativity
    source = torch.clamp(source, min=0)
    target = torch.clamp(target, min=0)

    # normalize source and target to make them valid probability distributions
    source /= torch.clamp(source.sum(dim=-1, keepdim=True).to(device), min=1e-12)
    target /= torch.clamp(target.sum(dim=-1, keepdim=True).to(device), min=1e-12)

    return source, target, cost_matrix


def sinkhorn_balanced(
    source: Tensor,
    target: Tensor,
    cost_matrix: Optional[Tensor] = None,
    p: int = 2,
    reg: float = 1.0,
    itrstep: int = 0,
    threshold: float = 0,
    returngrad: bool = False,
    device: str = 'cuda',
    verbose: bool = False,
):
    source, target, cost_matrix = data_preprocess(
        source,
        target,
        cost_matrix=cost_matrix,
        p=p,
        device=device,
        force_batched=True,
    )
    result = sinkhorn_balanced_batch(
        source,
        target,
        cost_matrix=cost_matrix,
        reg=reg,
        itrstep=itrstep,
        threshold=threshold,
        returnuv=returngrad,
        device=device,
        verbose=verbose,
    )

    if returngrad:
        return result
    return result["plan"]


def sinkhorn_balanced_batch(
    source: Tensor,  # (B,C,*Spatial)
    target: Tensor,  # (B,C,*Spatial)
    cost_matrix: Tensor,
    reg: float = 1.0,
    itrstep: int = 0,
    threshold: float = 0,
    returnuv: bool = False,  # whether to return the gradient of the dual potential
    device: str = 'cuda',
    verbose: bool = False,
):
    """Compute a balanced entropic optimal transport plan with Sinkhorn iterations.

    Builds K = exp(-reg * C), iteratively updates scaling vectors u and v,
    and returns the transport plan P.

    Args:
        source (Tensor): Preprocessed source distribution with shape (B, C, N).
        target (Tensor): Preprocessed target distribution with shape (B, C, N).
        cost_matrix (Tensor): Pairwise cost matrix with shape (N, N).
        reg (float): Inverse regularization parameter.
        itrstep (int): Number of Sinkhorn iterations.
        threshold (float): Optional convergence threshold.
        returnuv (bool): If True, also return u and v.

    Returns:
        dict: Transport plan, and optionally scaling vectors u and v.
    """
    # Device check
    if not torch.cuda.is_available():
        raise ValueError("CUDA device not available.")

    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("source and target must be preprocessed to shape (B, C, N).")

    # initialize u and v, and expand dimensions for batch and channel
    u = torch.ones_like(source, device=device)
    v = torch.ones_like(target, device=device)
    k = torch.exp(-cost_matrix * reg)

    B, C = source.shape[:2]
    k = k.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)

    if reg <= 0:
        raise ValueError("Regularization parameter must be positive.")

    # generate sinkhorn iteration judged by threshold or iteration times
    if itrstep < 0 or threshold < 0:
        raise ValueError("iterstep and threshold must be non-negative.")
    elif itrstep == 0 and threshold == 0:
        raise ValueError("Invalid iteration or step settings. ")

    if threshold > 0 and itrstep == 0:
        itrstep = 500

    # iteration
    convergence_flag = False
    convergence_check = 10
    for i in range(itrstep):
        u_old = u
        u = source / (torch.matmul(k, v.unsqueeze(-1)).squeeze(-1) + 1e-12)
        v = target / (torch.matmul(k.transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1) + 1e-12)
        if i % convergence_check == 0 and (u_old - u).abs().sum().mean() < threshold:
            convergence_flag = True
            break

    # use broadcasting to calculate the optimal transport plan P from u, v and K
    P = u.unsqueeze(-1) * k * v.unsqueeze(-2)

    if verbose:
        if convergence_flag:
            print(f"Sinkhorn iteration completed in {i+1} steps.Convergence: {convergence_flag}")
        else:
            print(f"Sinkhorn iteration completed in {i+1} steps,Covergence didn't complete.")

    # calculate gradient after iteration covergence ,return results
    if returnuv:
        return {"plan": P, "u": u, "v": v}
    else:
        return {"plan": P}


def sinkhorn_balanced_nobatch(
    source: Tensor,  # (N,)
    target: Tensor,  # (N,)
    cost_matrix: Tensor,
    reg: float = 1.0,
    itrstep: int = 0,
    threshold: float = 0,
    returnuv: bool = False,  # whether to return the gradient of the dual potential
    device: str = 'cuda',
):
    device = device if torch.cuda.is_available() else 'cpu'

    # initialize u and v, and expand dimensions for batch and channel
    u = torch.ones_like(source, device=device)
    v = torch.ones_like(target, device=device)
    k = torch.exp(-cost_matrix * reg)

    if reg <= 0:
        raise ValueError("Regularization parameter must be positive.")

    # generate sinkhorn iteration judged by threshold or iteration times
    if itrstep < 0 or threshold < 0:
        raise ValueError("iterstep and threshold must be non-negative.")
    elif itrstep == 0 and threshold == 0:
        raise ValueError("Invalid iteration or step settings. ")

    if threshold > 0 and itrstep == 0:
        itrstep = 500

    # iteration
    convergence_check = 10
    for i in range(itrstep):
        u_old = u
        u = source / (torch.matmul(k, v.unsqueeze(-1)).squeeze(-1) + 1e-12)
        v = target / (torch.matmul(k.transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1) + 1e-12)
        if i % convergence_check == 0 and (u_old - u).abs().sum().mean() < threshold:
            break

    # use broadcasting to calculate the optimal transport plan P from u, v and K
    P = u.unsqueeze(-1) * k * v.unsqueeze(-2)

    # calculate gradient after iteration covergence ,return results
    if returnuv:
        return {"plan": P, "u": u, "v": v}
    else:
        return {"plan": P}


def sinkhorn_gradient(u: Tensor, v: Tensor, reg: float):
    u = torch.log(torch.clamp(u, min=1e-12))
    v = torch.log(torch.clamp(v, min=1e-12))

    u = u - u.mean(dim=-1, keepdim=True)  # remove constants
    v = v - v.mean(dim=-1, keepdim=True)

    grad_source = u / reg
    grad_target = v / reg

    return grad_source, grad_target


def sinkhorn_balanced_cuda(
    source: Tensor,
    target: Tensor,
    cost_matrix: Tensor,
    reg: float = 1.0,
    itrstep: int = 100,
    returnuv: bool = False,
):
    from torchmorph import _C

    N = source.shape[-1]
    K = torch.exp(-cost_matrix * reg)
    u = torch.ones_like(source)
    v = torch.ones_like(target)

    u, v = _C.sinkhorn_fastiter(source, target, K, u, v, int(itrstep), int(N))

    P = u.unsqueeze(-1) * K * v.unsqueeze(-2)

    if returnuv:
        return {"plan": P, "u": u, "v": v}
    else:
        return {"plan": P}


def sinkhorn_balanced_log(
    source: Tensor,
    target: Tensor,
    cost_matrix: Tensor,
    reg: float = 1.0,
    itrstep: int = 100,
    dtype=torch.float32,
):
    from torchmorph import _C

    source = source.to(dtype=dtype).log()
    target = target.to(dtype=dtype).log()
    N = source.shape[-1]
    u = torch.zeros_like(source)
    v = torch.zeros_like(target)

    u, v = _C.sinkhorn_logiter(source, target, cost_matrix, u, v, int(itrstep), int(N), float(reg))

    u = u.exp()
    v = v.exp()

    return u, v
