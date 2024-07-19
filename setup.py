# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import glob
import torch
import subprocess
from typing import List
from setuptools import find_packages, setup
from packaging.version import Version, parse
from torch.utils.cpp_extension import (
    CUDA_HOME,
    CUDAExtension,
    BuildExtension,
)

ROOT_DIR = os.path.dirname(__file__)
MAIN_CUDA_VERSION = "12.1"

DISTRIBUTION_NAME = "flute-kernel"
LIBRARY_NAME = "flute"
CUTLASS_PATH = "/workspace/cutlass/"


# References:
# https://github.com/pytorch/extension-cpp/blob/master/setup.py
# https://github.com/vllm-project/vllm/blob/main/setup.py
# https://github.com/flashinfer-ai/flashinfer/blob/main/python/setup.py
# https://github.com/microsoft/BitBLAS/blob/main/setup.py
def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    assert CUDA_HOME is not None, "CUDA_HOME is not set"
    nvcc_output = subprocess.check_output(
        [os.path.join(CUDA_HOME, "bin/nvcc"), "-V"],
        universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def find_version(filepath: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]",
            fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def get_version() -> str:
    version = find_version(get_path(LIBRARY_NAME, "__init__.py"))
    cuda_version = str(get_nvcc_cuda_version())
    if cuda_version != MAIN_CUDA_VERSION:
        cuda_version_str = cuda_version.replace(".", "")[:3]
        version += f"+cu{cuda_version_str}"

    return version


def get_extensions() -> List:

    extra_link_args = []
    extra_compile_args = {
        "cxx": ["-std=c++17"],
        "nvcc": ["-std=c++17"],
    }
    include_dirs = [
        os.path.join(CUTLASS_PATH, "include"),
        os.path.join(CUTLASS_PATH, "tools/util/include"),
    ]

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, LIBRARY_NAME, "csrc")
    sources = (
        list(glob.glob(os.path.join(extensions_dir, "*.cpp"))) +
        list(glob.glob(os.path.join(extensions_dir, "*.cu"))))

    ext_modules = [
        CUDAExtension(
            f"{LIBRARY_NAME}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=include_dirs,
        )
    ]

    return ext_modules


setup(
    name=DISTRIBUTION_NAME,
    version=get_version(),
    packages=find_packages(),
    include_package_data=True,
    ext_modules=get_extensions(),
    install_requires=["torch"],
    cmdclass={"build_ext": BuildExtension},
)
