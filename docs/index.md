# torchmorph

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

## API Reference

- [Distance Transforms](api/distance_transforms.md) — Euclidean, Chamfer, and brute-force distance transforms
- [Morphological Operations](api/morphological_ops.md) — Binary dilation and erosion
