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
    device: str = 'cuda',
):
    # check dimensions,reshape and choose strategy
    if not source.ndim == target.ndim or not source.shape == target.shape:
        raise ValueError(
            "Source and target must have the same number of dimensions and the same shape."
        )
    elif source.ndim == 2:
        if cost_matrix is None:
            cost_matrix = build_cost_matrix(source.shape, device=device, p=p)
        source = source.view(1, 1, -1)
        target = target.view(1, 1, -1)
    elif source.ndim > 2:
        # cut and reshape source and target to coordinate tensors, then calculate cost matrix
        B, C = source.shape[:2]  # get batch and channel dimensions
        if cost_matrix is None:
            spatial_shape = source.shape[2:]
            cost_matrix = build_cost_matrix(spatial_shape, device=device, p=p)
        source = source.view(B, C, -1)
        target = target.view(B, C, -1)

    # move to specified device(cuda) and ensure dtype float32
    source = source.to(device=device, dtype=torch.float32)
    target = target.to(device=device, dtype=torch.float32)
    cost_matrix = cost_matrix.to(device=device, dtype=torch.float32)

    # ensure non-negativity
    source = torch.clamp(source, min=0)
    target = torch.clamp(target, min=0)

    # normalize source and target to make them valid probability distributions
    source /= torch.clamp(source.sum(dim=-1, keepdim=True).to(device), min=1e-12)
    target /= torch.clamp(target.sum(dim=-1, keepdim=True).to(device), min=1e-12)

    return source, target, cost_matrix


def sinkhorn_balanced_full(
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
    )
    result = sinkhorn_balanced(
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


# sinkhorn iteration
def sinkhorn_balanced(
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

    Uses the Gibbs kernel K = exp(-reg * C), where `reg` is the inverse
    regularization scale. Source and target are clamped to be non-negative,
    normalized over the flattened spatial dimension, and matched by iterative
    Sinkhorn scaling.

    Args:
        source (Tensor): Source distribution, shape (B, C, *Spatial) or (*Spatial).
        target (Tensor): Target distribution with the same shape as source.
        cost_matrix (Tensor, optional): Pairwise cost matrix of shape (N, N).
        p (int): Lp norm used to build the cost matrix when not provided.
        reg (float): Positive inverse regularization parameter.
        itrstep (int): Number of Sinkhorn iterations.
        threshold (float): Stopping threshold on the change of the source scaling.
        returngrad (bool): If True, also return scaling vectors u and v.
        device (str): Computation device, typically 'cuda'.

    Returns:
        dict: Contains "plan" and, when requested, "u" and "v".

    Raises:
        ValueError: If dimensions, regularization, device, or stopping settings
            are invalid.
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
    if itrstep == 0 and threshold > 0:
        step = 1000
    elif itrstep == 0 and threshold == 0:
        raise ValueError("Invalid iteration or step settings. ")
    else:
        step = itrstep

    # iteration
    convergence_flag = False
    for i in range(step):
        u_old = u.clone()
        u = source / (torch.matmul(k, v.unsqueeze(-1)).squeeze(-1) + 1e-12)
        v = target / (torch.matmul(k.transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1) + 1e-12)
        if (u_old - u).abs().sum().mean() < threshold:
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

    u, v = _C.sinkhorn_uv_cuda(source, target, K, u, v, int(itrstep), int(N))

    P = u.unsqueeze(-1) * K * v.unsqueeze(-2)

    if returnuv:
        return {"plan": P, "u": u, "v": v}
    else:
        return {"plan": P}
