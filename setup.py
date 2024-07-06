# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CUDAExtension,
    BuildExtension,
)

LIBRARY_NAME = "flute"
CUTLASS_PATH = "/workspace/cutlass/"


def get_extensions():

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
    name=LIBRARY_NAME,
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=get_extensions(),
    install_requires=["torch"],
    cmdclass={"build_ext": BuildExtension},
)
