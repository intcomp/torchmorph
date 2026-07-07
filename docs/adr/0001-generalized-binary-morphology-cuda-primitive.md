# 0001 Generalized Binary Morphology CUDA Primitive

## Status

Accepted

## Context

TorchMorph already uses a generalized CUDA primitive for grey erosion and grey dilation. Binary erosion and binary dilation were implemented in Python with PyTorch padding and convolution helpers. That implementation kept SciPy-aligned behavior, but it made binary morphology less consistent with the project goal of CUDA-accelerated batch-parallel morphology.

Binary erosion and binary dilation are also the base operations for binary opening, binary closing, binary propagation, and binary hole filling. Optimizing only the public erosion and dilation wrappers would leave later binary operators without a clear reusable CUDA foundation.

## Decision

Implement a generalized binary morphology CUDA primitive, modeled after the grey morphology CUDA primitive, and dispatch CUDA binary erosion and binary dilation through it. The primitive supports batch-channel tensors with arbitrary spatial rank up to the same practical maximum used by the grey implementation. Python remains responsible for iteration loops, mask application, output copying, and composite operators.

Do not start with specialized 2D or 3D kernels. Keep those as later optimizations once the generalized primitive is correct and benchmarked.

## Consequences

The first CUDA implementation favors broad SciPy-aligned behavior and shared maintenance shape over peak 2D/3D performance. This gives binary propagation and binary hole filling a reusable base and keeps the public Python API stable.

Specialized 2D and 3D kernels may still be added later as fast paths when benchmark data shows that fixed-rank kernels would materially improve common workloads.
