# TorchMorph

> ⚡ CUDA-Accelerated, Batch-Parallel Morphological Transformations for PyTorch

TorchMorph is a lightweight, extensible library that brings **GPU-accelerated morphological operations** into the PyTorch ecosystem.  
It provides a **clean Python API** backed by **custom CUDA kernels**, enabling highly efficient, **batch-parallel** transformations for real-time and large-scale vision tasks.

---

## 🚀 Key Features

- ⚡ **CUDA Acceleration** – All operators are implemented with native CUDA kernels for maximum throughput.  
- 🧩 **Seamless PyTorch Integration** – Accepts and returns `torch.Tensor` objects, fully compatible with autograd and CUDA streams.  
- 🧠 **Highly Batch-Parallel** – Optimized to process large batches and multi-dimensional inputs concurrently.  
- 🧱 **Modular Design** – Each operation is isolated in its own kernel, making it easy to add or extend transformations.  
- ✅ **Lightweight & Self-Contained** – No third-party dependencies beyond PyTorch and a working CUDA toolkit.  

---

## 📦 Installation

Clone and build locally:

```bash
git clone https://github.com/torchmorph/torchmorph.git
cd torchmorph
pip install -e .
```
