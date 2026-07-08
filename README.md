# TorchMorph

> ⚡ CUDA-Accelerated, Batch-Parallel Morphological Transformations for PyTorch

TorchMorph is a lightweight, extensible library that brings **GPU-accelerated morphological operations** into the PyTorch ecosystem.
It provides a **clean Python API** backed by **custom CUDA kernels**, enabling highly efficient, **batch-parallel** transformations for real-time and large-scale vision tasks.

---

## 🚀 Key Features

- ⚡ **CUDA Acceleration** – All operators are implemented with native CUDA kernels for maximum throughput.
- 🧩 **PyTorch Tensor Integration** – Accepts and returns CUDA `torch.Tensor` objects for direct use in tensor-processing pipelines.
- 🧠 **Highly Batch-Parallel** – Optimized to process large batches and multi-dimensional inputs concurrently.
- 🧱 **Modular Design** – Public operators reuse shared CUDA morphology primitives and compose them into higher-level transformations.
- ✅ **Lightweight & Self-Contained** – No third-party dependencies beyond PyTorch and a working CUDA toolkit.


> **NaN values:** NaN propagation is not guaranteed to match SciPy. Replace or
> otherwise handle NaN values before calling a morphological operator when
> deterministic missing-value behavior is required.

---

## 📦 Local build and development install

The instructions in this section are for **building TorchMorph from this repository** and using it locally in a development environment. TorchMorph builds a custom CUDA extension during install, which means your PyTorch runtime and CUDA compiler (`nvcc`) must be compatible.

### Recommended workflow (Stable & Decoupled)

1. Create and activate a fresh conda environment:
   ```bash
   conda create -n torchmorph python=3.11 -y
   conda activate torchmorph
   ```

2. Install the stable PyTorch release (e.g., CUDA 12.4):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```

3. Ensure you have a compatible `nvcc` compiler.
   ```bash
   conda install -c nvidia cuda-nvcc=12.4 -y
   ```

4. Install TorchMorph's dependencies and build the extension:
   ```bash
   pip install -r "requirements-dev.txt"
   pip install --no-build-isolation -e .
   ```

### Verify the environment before building

Check your PyTorch and CUDA compiler versions:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__} | PyTorch CUDA: {torch.version.cuda}')"
nvcc --version
```
As long as your `nvcc` version matchs your PyTorch CUDA version (e.g., `nvcc 12.4` with PyTorch `cu124`), the extension will compile successfully.

### Minimal validation

After installation succeeds, verify import and a simple CUDA kernel call:

```python
import torch
import torchmorph as tm

print(torch.__version__, torch.version.cuda, torch.cuda.is_available())

if torch.cuda.is_available():
    x = torch.rand(1, 1, 8, 8, device="cuda")
    y = tm.grey_dilation(x, size=3)
    print(y.shape)
else:
    print("CUDA not available; install verified for import only.")
```
