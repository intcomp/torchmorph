from typing import Optional

import torch
from torch import Tensor


def build_cost_matrix(shape, p=2, device='cuda') -> torch.Tensor:  # Euclidean distance by default
    """Build the pairwise ground-cost matrix for a spatial grid.

    The shape defines grid coordinates, and p selects the distance norm.
    Returns an (N, N) matrix for N flattened grid points.
    """
    coords = torch.stack(torch.meshgrid([torch.arange(s) for s in shape], indexing="ij"), dim=-1)
    coords_flat = coords.view(-1, coords.size(-1)).float()

    cost_matrix = torch.cdist(coords_flat, coords_flat, p=p).to(device)
    return cost_matrix


class SinkhornSolver:
    def __init__(self, reg=0.02, itrstep=100, threshold=0, p=2, device='cuda'):
        self.reg = reg
        self.itrstep = itrstep
        self.threshold = threshold
        self.device = device
        self.p = p

    def data_preprocess(
        self,
        source: Tensor,
        target: Tensor,
        cost_matrix: Optional[Tensor] = None,
        force_batched: bool = False,
        dtype=torch.float32,
    ):
        """Preprocess source and target distributions for Sinkhorn solvers.

        This checks matching shapes, flattens spatial axes, clamps negatives,
        normalizes mass, and prepares the cost matrix transpose.
        """
        # check dimensions,reshape and choose strategy
        if not source.ndim == target.ndim or not source.shape == target.shape:
            raise ValueError(
                "Source and target must have the same number of dimensions and the same shape."
            )
        elif source.ndim <= 2:
            if cost_matrix is None:
                cost_matrix = build_cost_matrix(source.shape, self.p, self.device)
            if force_batched:
                source = source.reshape(1, 1, -1)
                target = target.reshape(1, 1, -1)
            else:
                source = source.reshape(-1)
                target = target.reshape(-1)
        elif source.ndim > 2:
            # cut and reshape source and target to coordinate tensors, then calculate cost matrix
            B, C = source.shape[:2]  # get batch and channel dimensions
            if cost_matrix is None:
                spatial_shape = source.shape[2:]
                cost_matrix = build_cost_matrix(spatial_shape, self.p, self.device)
            source = source.reshape(B, C, -1)
            target = target.reshape(B, C, -1)

        # move to specified device(cuda) and ensure dtype float32
        source = source.to(device=self.device, dtype=dtype)
        target = target.to(device=self.device, dtype=dtype)
        cost_matrix = cost_matrix.to(device=self.device, dtype=dtype)

        # ensure non-negativity
        source = torch.clamp(source, min=0)
        target = torch.clamp(target, min=0)

        # normalize source and target to make them valid probability distributions
        source /= torch.clamp(source.sum(dim=-1, keepdim=True).to(self.device), min=1e-12)
        target /= torch.clamp(target.sum(dim=-1, keepdim=True).to(self.device), min=1e-12)

        if not torch.allclose(cost_matrix, cost_matrix.transpose(-2, -1)):
            cost_matrix_T = cost_matrix.transpose(-2, -1).contiguous()
        else:
            cost_matrix_T = cost_matrix

        return source, target, cost_matrix, cost_matrix_T

    def sinkhorn(
        self,
        source: Tensor,
        target: Tensor,
        cost_matrix: Optional[Tensor] = None,
        returngrad: bool = False,
        verbose: bool = False,
    ):
        """Compute a balanced Sinkhorn transport plan from raw inputs.

        The function preprocesses inputs, runs the batched solver, and returns
        either the plan or auxiliary variables when gradients are requested.
        """
        source, target, cost_matrix, _ = self.data_preprocess(
            source,
            target,
            cost_matrix=cost_matrix,
            force_batched=True,
        )
        result = self.sinkhorn_batch(
            source, target, cost_matrix=cost_matrix, returnuv=returngrad, verbose=verbose
        )

        if returngrad:
            return result
        return result["plan"]

    def run_once(
        self,
        source: Tensor,
        target: Tensor,
        cost_matrix: Optional[Tensor] = None,
        *,
        use_cuda: bool = False,
        log_domain: bool = False,
        return_plan: bool = True,
        return_uv: bool = False,
        return_grad: bool = False,
        return_distance: bool = False,
        verbose: bool = False,
        dtype=torch.float32,
    ):
        """Run the full Sinkhorn workflow from raw inputs.

        This wrapper preprocesses raw source and target tensors, selects one
        Sinkhorn backend, optionally reconstructs the transport plan, and
        optionally returns the OT distance, dual-scaling vectors, or gradients.
        """
        source, target, cost_matrix, cost_matrix_T = self.data_preprocess(
            source,
            target,
            cost_matrix=cost_matrix,
            force_batched=True,
            dtype=dtype,
        )

        need_uv = return_uv or return_grad or log_domain

        if log_domain:
            u, v = self.sinkhorn_log_cuda(
                source=source,
                target=target,
                cost_matrix=cost_matrix,
                cost_matrix_T=cost_matrix_T,
                dtype=dtype,
            )
            backend = "cuda_log"
            result = {"u": u, "v": v}
        elif use_cuda:
            result = self.sinkhorn_cuda(
                source=source,
                target=target,
                cost_matrix=cost_matrix,
                returnuv=need_uv,
            )
            backend = "cuda"
        else:
            result = self.sinkhorn_batch(
                source=source,
                target=target,
                cost_matrix=cost_matrix,
                returnuv=need_uv,
                verbose=verbose,
            )
            backend = "torch"

        output = {
            "source": source,
            "target": target,
            "cost_matrix": cost_matrix,
            "cost_matrix_T": cost_matrix_T,
            "backend": backend,
        }

        plan = None
        if return_plan or return_distance:
            if "plan" in result:
                plan = result["plan"]
            else:
                k = torch.exp(-cost_matrix * self.reg)
                plan = result["u"].unsqueeze(-1) * k * result["v"].unsqueeze(-2)

        if return_plan:
            output["plan"] = plan

        if return_distance:
            output["distance"] = (plan * cost_matrix).sum(dim=(-2, -1))

        if return_uv or return_grad:
            output["u"] = result["u"]
            output["v"] = result["v"]

        if return_grad:
            grad_source, grad_target = self.gradient(result["u"], result["v"])
            output["grad_source"] = grad_source
            output["grad_target"] = grad_target

        return output

    def sinkhorn_batch(
        self,
        source: Tensor,  # (B,C,*Spatial)
        target: Tensor,  # (B,C,*Spatial)
        cost_matrix: Tensor,
        returnuv: bool = False,  # whether to return the gradient of the dual potential
        verbose: bool = False,
    ):
        """Compute batched balanced Sinkhorn iterations in PyTorch.

        Inputs should be preprocessed to shape (B, C, N). The solver builds
        K = exp(-reg * C), updates u and v, and returns the transport plan.
        """
        # Device check
        if not torch.cuda.is_available():
            raise ValueError("CUDA device not available.")

        if source.ndim != 3 or target.ndim != 3:
            raise ValueError("source and target must be preprocessed to shape (B, C, N).")

        # initialize u and v, and expand dimensions for batch and channel
        u = torch.ones_like(source, device=self.device)
        v = torch.ones_like(target, device=self.device)
        k = torch.exp(-cost_matrix * self.reg)

        B, C = source.shape[:2]
        k = k.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)

        if self.reg <= 0:
            raise ValueError("Regularization parameter must be positive.")

        # generate sinkhorn iteration judged by threshold or iteration times
        if self.itrstep < 0 or self.threshold < 0:
            raise ValueError("iterstep and threshold must be non-negative.")
        elif self.itrstep == 0 and self.threshold == 0:
            raise ValueError("Invalid iteration or step settings. ")

        itrstep = self.itrstep
        if self.threshold > 0 and self.itrstep == 0:
            itrstep = 500

        # iteration
        convergence_flag = False
        convergence_check = 10
        for i in range(itrstep):
            u_old = u
            u = source / (torch.matmul(k, v.unsqueeze(-1)).squeeze(-1) + 1e-12)
            v = target / (torch.matmul(k.transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1) + 1e-12)
            if i % convergence_check == 0 and (u_old - u).abs().sum().mean() < self.threshold:
                convergence_flag = True
                break

        # use broadcasting to calculate the optimal transport plan P from u, v and K
        P = u.unsqueeze(-1) * k * v.unsqueeze(-2)

        if verbose:
            if convergence_flag:
                print(
                    f"Sinkhorn iteration completed in {i+1} steps.Convergence: {convergence_flag}"
                )
            else:
                print(f"Sinkhorn iteration completed in {i+1} steps,Covergence didn't complete.")

        # calculate gradient after iteration covergence ,return results
        if returnuv:
            return {"plan": P, "u": u, "v": v}
        else:
            return {"plan": P}

    def sinkhorn__nobatch(
        self,
        source: Tensor,  # (N,)
        target: Tensor,  # (N,)
        cost_matrix: Tensor,
        returnuv: bool = False,  # whether to return the gradient of the dual potential
    ):
        """Compute Sinkhorn iterations for one flat distribution pair.

        Source and target should be one-dimensional probability vectors.
        This path is a simple PyTorch reference for the non-batched case.
        """

        # initialize u and v, and expand dimensions for batch and channel
        u = torch.ones_like(source, device=self.device)
        v = torch.ones_like(target, device=self.device)
        k = torch.exp(-cost_matrix * self.reg)

        if self.reg <= 0:
            raise ValueError("Regularization parameter must be positive.")

        # generate sinkhorn iteration judged by threshold or iteration times
        if self.itrstep < 0 or self.threshold < 0:
            raise ValueError("iterstep and threshold must be non-negative.")
        elif self.itrstep == 0 and self.threshold == 0:
            raise ValueError("Invalid iteration or step settings. ")

        itrstep = self.itrstep
        if self.threshold > 0 and self.itrstep == 0:
            itrstep = 500

        # iteration
        convergence_check = 10
        for i in range(itrstep):
            u_old = u
            u = source / (torch.matmul(k, v.unsqueeze(-1)).squeeze(-1) + 1e-12)
            v = target / (torch.matmul(k.transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1) + 1e-12)
            if i % convergence_check == 0 and (u_old - u).abs().sum().mean() < self.threshold:
                break

        # use broadcasting to calculate the optimal transport plan P from u, v and K
        P = u.unsqueeze(-1) * k * v.unsqueeze(-2)

        # calculate gradient after iteration covergence ,return results
        if returnuv:
            return {"plan": P, "u": u, "v": v}
        else:
            return {"plan": P}

    def gradient(self, u: Tensor, v: Tensor):
        """Convert Sinkhorn scaling vectors to source and target gradients.

        The scaling vectors are logged, centered to remove additive constants,
        and rescaled according to this module's regularization convention.
        """
        u = torch.log(torch.clamp(u, min=1e-12))
        v = torch.log(torch.clamp(v, min=1e-12))

        u = u - u.mean(dim=-1, keepdim=True)  # remove constants
        v = v - v.mean(dim=-1, keepdim=True)

        grad_source = u / self.reg
        grad_target = v / self.reg

        return grad_source, grad_target

    def _validate_single_cuda_distribution(self, source: Tensor, target: Tensor):
        if source.ndim != 3 or target.ndim != 3:
            raise ValueError("CUDA Sinkhorn inputs must be preprocessed to shape (B, C, N).")
        if source.shape[:2] != (1, 1) or target.shape[:2] != (1, 1):
            raise ValueError("CUDA Sinkhorn kernels currently support only B=C=1 inputs.")

    def sinkhorn_cuda(
        self,
        source: Tensor,
        target: Tensor,
        cost_matrix: Tensor,
        returnuv: bool = False,
    ):
        """Run the CUDA Sinkhorn solver in standard scaling form.

        The CUDA kernel updates u and v, after which the Python wrapper
        reconstructs the transport plan from u, K, and v.
        """
        self._validate_single_cuda_distribution(source, target)
        from torchmorph import _C

        N = source.shape[-1]
        K = torch.exp(-cost_matrix * self.reg)
        u = torch.ones_like(source)
        v = torch.ones_like(target)

        u, v = _C.sinkhorn_fastiter(source, target, K, u, v, int(self.itrstep), int(N))

        P = u.unsqueeze(-1) * K * v.unsqueeze(-2)

        if returnuv:
            return {"plan": P, "u": u, "v": v}
        else:
            return {"plan": P}

    def sinkhorn_log_cuda(
        self,
        source: Tensor,
        target: Tensor,
        cost_matrix: Tensor,
        cost_matrix_T: Tensor,
        dtype=torch.float32,
    ):
        """Run the CUDA Sinkhorn solver in log-domain form.

        Inputs are converted to log probabilities for numerical stability.
        The returned u and v are exponentiated scaling vectors.
        """
        self._validate_single_cuda_distribution(source, target)
        from torchmorph import _C

        with torch.no_grad():
            source = source.to(dtype=dtype).log()
            target = target.to(dtype=dtype).log()
            N = source.shape[-1]
            u = torch.zeros_like(source)
            v = torch.zeros_like(target)

            u, v = _C.sinkhorn_logiter(
                source,
                target,
                cost_matrix,
                cost_matrix_T,
                u,
                v,
                int(self.itrstep),
                int(N),
                float(self.reg),
            )

            u = u.exp()
            v = v.exp()

        return u, v
