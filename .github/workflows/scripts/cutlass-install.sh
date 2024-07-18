#!/bin/bash

sudo mkdir -p /workspace
sudo chmod -R 777 /workspace/

cd /workspace

git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout v3.4.1
cd ..
