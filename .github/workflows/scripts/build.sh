#!/bin/bash

python_executable=python$1
cuda_home=/usr/local/cuda-$2

# Update paths
PATH=${cuda_home}/bin:$PATH
LD_LIBRARY_PATH=${cuda_home}/lib64:$LD_LIBRARY_PATH

# Install requirements
$python_executable -m pip install --upgrade pip
$python_executable -m pip install --upgrade wheel
$python_executable -m pip install --upgrade build
$python_executable -m pip install --upgrade twine
$python_executable -m pip install --upgrade patchelf
$python_executable -m pip install --upgrade packaging
$python_executable -m pip install --upgrade auditwheel
$python_executable -m pip install -r requirements.txt

# Limit the number of parallel jobs to avoid OOM
# export MAX_JOBS=1
# Make sure release wheels are built for the following architectures
export TORCH_CUDA_ARCH_LIST="8.0 8.6"
# Build
$python_executable -m build --no-isolation

# https://pypi.org/project/cuda-ext-example/
auditwheel repair \
    dist/flute-*.whl \
    --plat manylinux_2_34_x86_64 \
    --exclude libc10.so \
    --exclude libc10_cuda.so \
    --exclude libtorch_cpu.so
