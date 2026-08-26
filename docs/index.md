# torchmorph

[![arXiv](https://img.shields.io/badge/arXiv-2608.24738-b31b1b.svg)](https://arxiv.org/abs/2608.24738)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-FFD21E.svg)](https://huggingface.co/papers/2608.24738)

GPU-accelerated morphological image processing operations for PyTorch.

## Installation

```bash
pip install torchmorph
```

## Quick Start

```python
import torch
import torchmorph as tm

x = torch.zeros(1, 1, 64, 64, device='cuda')
x[0, 0, 10:20, 10:20] = 1

dist = tm.euclidean_distance_transform(x)
dilated = tm.binary_dilation(x)
```

Spatial morphology and distance-transform functions accept CUDA tensors shaped
as `(B, C, Spatial...)`, with one to eight spatial dimensions. Binary operators
treat nonzero values as foreground.

Optimal transport works on CPU and CUDA with batches of flattened histograms:

```python
source = torch.rand(8, 64, device="cuda")
target = torch.rand(8, 64, device="cuda")

solver = tm.SinkhornSolver(epsilon=1.0, max_iter=200)
cost = tm.build_cost_matrix((8, 8), device=source.device)
distances = solver(source, target, cost)
```

## API Reference

- [Distance Transforms](distance_transforms.md) — Euclidean, Chamfer, and brute-force distance transforms
- [Structuring Elements](structuring_elements.md) — Multidimensional connectivity structures
- [Binary Morphology](binary_morphology.md) — Binary dilation, erosion, filling, and related operators
- [Grayscale Morphology](grayscale_morphology.md) — Grayscale dilation, erosion, gradients, and top-hats
- [Optimal Transport](optimal_transport.md) — Cost matrices and differentiable Sinkhorn transport
