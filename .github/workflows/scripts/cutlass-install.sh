#!/bin/bash

mkdir -p /workspace
cd /workspace

git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout v3.4.1
cd ..
