"""Build MuxServe's head-wise KV ops ahead of time, in place:

    python -m sglang.kernels.ops.attention.headwise

compiles csrc/ into the extension _C next to __init__.py. It is the build MuxServe's SGLang
baseline used before the ops moved here (tools/build_sglang_ext.py: setuptools with torch's
CUDAExtension, -O2 -std=c++20 for both the C++ compiler and nvcc, TORCH_CUDA_ARCH_LIST or
else 8.0 for the A100). nvcc must be on PATH. The build runs in a temporary directory, so
setuptools reads no pyproject.toml of the checkout, and only the extension is left behind.
"""
import importlib
import os
import tempfile
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")  # A100

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PACKAGE = "sglang.kernels.ops.attention.headwise"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[len(PACKAGE.split(".")) - 1]  # the directory that holds the sglang package
CSRC = HERE / "csrc"
SOURCES = ("bindings.cpp", "cache_headwise.cu", "attention_headwise_kernels.cu")
FLAGS = ["-O2", "-std=c++20"]

# This process imported an earlier build already; write the new one as a new file, not over it.
for earlier in HERE.glob("_C.*.so"):
    earlier.unlink()
start = os.getcwd()
with tempfile.TemporaryDirectory(prefix="sglang-headwise-build-") as build:
    os.chdir(build)
    try:
        setup(
            name="sglang-muxserve-headwise",
            script_args=["build_ext", "--inplace", "--build-temp", os.path.join(build, "temp")],
            package_dir={"": str(ROOT)},
            ext_modules=[
                CUDAExtension(
                    f"{PACKAGE}._C",
                    [str(CSRC / name) for name in SOURCES],
                    include_dirs=[str(CSRC)],
                    extra_compile_args={"cxx": FLAGS, "nvcc": FLAGS},
                )
            ],
            cmdclass={"build_ext": BuildExtension},
        )
    finally:
        os.chdir(start)

print(f"built {importlib.import_module(PACKAGE + '._C').__file__}")
