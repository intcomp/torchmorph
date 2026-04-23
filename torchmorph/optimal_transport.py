import torch
from typing import Optional
from torch import Tensor

#generate cost matrix for multi-dimensional tensor like 2D or 3D tensor
def build_cost_matrix(
    shape,
    device:     str = 'cuda',
    p           =2 #Euclidean distance by default
    )-> torch.Tensor:
 
    coords = torch.stack(torch.meshgrid([torch.arange(s) for s in shape], indexing="ij"), dim=-1)
    coords_flat = coords.view(-1, coords.size(-1)).float()
    

    cost_matrix = torch.cdist(coords_flat, coords_flat, p=p).to(device)
    return cost_matrix

#sinkhorn iteration
def sinkhorn_balanced(
        source:         Tensor,          #(B,C,*Spatial)
        target:         Tensor,          #(B,C,*Spatial)
        cost_matrix:    Optional[Tensor] = None,
        reg:            float=1.0,
        itrstep:        int=0,
        threshold:      float=0,
        returngrad:     bool=False,      #whether to calculate gradient
        device=         'cuda'
        ):
    
    ''' Calculate the optimal transport plan using Sinkhorn iterations in the balanced case. 
    
    Args:
        source is the input tensor, 
        target is the output tensor,
        cost_matrix is the cost of transporting mass from source to target,
        reg is the regularization parameter, and itrstep is the number of iterations to perform.
    Returns:
        The optimal transport plan P, and optionally the gradient if returngrad is True.
    '''
    #Device check
    if not torch.cuda.is_available():
        raise ValueError("CUDA device not available.")

    #get batch and channel dimensions
    B,C = source.shape[:2]
    #check dimensions,reshape and choose strategy
    if not source.ndim == target.ndim:
        raise ValueError("Source and target must have the same number of dimensions.")
    elif source.ndim == 2:
        if cost_matrix is None:
            cost_matrix=build_cost_matrix(source, device=device, p=2)
        source=source.view(-1)
        target=target.view(-1)
    elif source.ndim > 2:
        #cut and reshape source and target to coordinate tensors, then calculate cost matrix
        if cost_matrix is None:
            spatial_shape = source.shape[2:]
            cost_matrix=build_cost_matrix(spatial_shape, device=device, p=2)
        source=source.view(B,C,-1)
        target=target.view(B,C,-1)
        

    source/=source.sum(dim = -1,keepdim = True).to(device)
    target/=target.sum(dim = -1,keepdim = True).to(device)
    
    #initialize u and v, and expand dimensions for batch and channel
    u = torch.ones_like(source, device = device)
    v = torch.ones_like(target, device = device)
    k = torch.exp(- cost_matrix / reg)

    k=k.unsqueeze(0).unsqueeze(0).expand(B,C,-1,-1)

    #generate sinkhorn iteration judged by threshold or iteration times
    if itrstep == 0 and threshold > 0:
        step = 1000
    elif itrstep == 0 and threshold == 0:
        raise ValueError("Invalid iteration or step settings. ")
    else:
        step = itrstep
    
    #iteration
    convergence_flag = False
    for i in range(step):
        u_old = u.clone()
        u = source / (torch.matmul(k, v.unsqueeze(-1)).squeeze(-1) + 1e-9)
        v = target / (torch.matmul(k.transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1) + 1e-9)
        if (u_old - u).abs().sum().mean() < threshold:
            convergence_flag = True
            break


    #diagonalize u and v, then calculate optimal transport plan P
    u = torch.diag_embed(u)
    v = torch.diag_embed(v)      
    P = u @ k @ v

    if convergence_flag == True:
        print(f"Sinkhorn iteration completed in {i+1} steps.Convergence: {convergence_flag}")
    else:
        print(f"Sinkhorn iteration completed in {i+1} steps,Covergence didn't complete." )

    #calculate gradient after iteration covergence ,return results    
    if returngrad == True:
        log_u = torch.log(u)
        mean_log_u = sum(log_u) / k
        grad = (log_u - mean_log_u) / reg #gradient
        return P, grad
    else:
        return P
