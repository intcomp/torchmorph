import os
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_extensions():
    # Find all CUDA and C++ source files automatically
    src_dir = os.path.join("torchmorph", "csrc")
    sources = glob.glob(os.path.join(src_dir, "*.cu")) + glob.glob(os.path.join(src_dir, "*.cpp"))

    extension = CUDAExtension(
        name="torchmorph._C",
        sources=sources,
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "--use_fast_math", "-std=c++17"],
        },
    )

    return [extension]


setup(
    name="torchmorph",
    version="0.1.0",
    author="Kai Zhao",
    description="CUDA-accelerated morphological transformations for PyTorch",
    url="https://github.com/kaizhao-shu/torchmorph",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["torch"],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
    zip_safe=False,
)
