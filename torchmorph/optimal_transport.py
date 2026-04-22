import torch
from torch import Tensor

def build_cost_matrix_1d(
    len_a:      int,
    len_b:      int,
    device:     str = 'cuda',
    dtype:      torch.dtype = torch.float32,
    p:          int = 2                         #wasserstein distance order, default is 2 
) -> torch.Tensor:

    pos_a = torch.arange(len_a, device=device, dtype=dtype)   
    pos_b = torch.arange(len_b, device=device, dtype=dtype)   

    diff = pos_a[:, None] - pos_b[None, :]

    if p == 1:
        M = torch.abs(diff)
    elif p == 2:
        M = diff.pow(2)
    else:
        M = torch.abs(diff).pow(p)

    return M


#sinkhorn iteration
def sinkhorn_binary(
        source:         Tensor,          #(B,C,*Spatial)
        target:         Tensor,          #(B,C,*Spatial)
        cost_matrix:    Tensor,
        reg:            float=1.0,
        itrstep:        int=0,
        threshold:      float=0,
        device=         'cuda'
        ):
    
    ''' Calculate the optimal transport plan using Sinkhorn iterations in the binary case. 
    
    Args:
        source is the input tensor, 
        target is the output tensor,
        cost_matrix is the cost of transporting mass from source to target,
        reg is the regularization parameter, and itrstep is the number of iterations to perform.
    '''
    #Device check
    if not torch.cuda.is_available():
        raise ValueError("CUDA device not available.")

    #check dimensions,and reshape to (B,C,N)
    if not source.ndim == target.ndim:
        raise ValueError("Source and target must have the same number of dimensions.")
    else:
        source=source.view(source.size(0),source.size(1),-1)
        target=target.view(target.size(0),target.size(1),-1)
        
    #prepare for iteration
    cost_matrix=build_cost_matrix_1d(source.size(-1),target.size(-1),device=device)

    source/=source.sum(dim = -1,keepdim = True).to(device)
    target/=target.sum(dim = -1,keepdim = True).to(device)
    
    u = torch.ones_like(source, device = device)
    v = torch.ones_like(target, device = device)
    k = torch.exp(- cost_matrix / reg)

    #generate sinkhorn iteration judged by threshold or iteration times
    if itrstep == 0:
        step = 1000
    for i in range(step):
        if ((source / (k @ v)) - u).abs().sum().mean() < threshold:
            convergence_flag = True
            break
        else:
            u = source / (k @ v)
            v = target / (k.t() @ u)
            P = torch.diag(u) @ k @ torch.diag(v)
            convergence_flag = False

    if threshold == 0:
        print(f"Sinkhorn iteration completed in {i+1} steps.")
    else:
        print(f"Sinkhorn iteration completed in {i+1} steps. Convergence: {convergence_flag}")
    return P
