# Optimal Transport

TorchMorph provides a differentiable Sinkhorn solver that runs on CPU or CUDA.
CUDA `float32` inputs automatically use the fused implementation.

::: torchmorph.build_cost_matrix

::: torchmorph.SinkhornSolver
    options:
      members:
        - data_preprocess
        - forward
        - plan
        - potentials
