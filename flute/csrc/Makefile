# Set the path to the cutlass library
CUTLASS_PATH = /workspace/cutlass

# Set the path to the cutlass tools
CUTLASS_TOOLS_PATH = /workspace/cutlass/tools/util

output: qgemm_kernel_2.cu
	nvcc qgemm_kernel_2.cu \
		-std=c++17 \
		-arch=sm_86 \
		--expt-relaxed-constexpr \
		-I$(CUTLASS_PATH)/include/ \
		-I$(CUTLASS_TOOLS_PATH)/include/ \
		-lineinfo \
		-o example

#  CUDA_VISIBLE_DEVICES=0 ncu --set full --import-source yes -f -o out --page details ./example

debug: qgemm_kernel_2.cu
	nvcc qgemm_kernel_2.cu \
		-std=c++17 \
		-arch=sm_86 \
		--expt-relaxed-constexpr \
		-I$(CUTLASS_PATH)/include/ \
		-I$(CUTLASS_TOOLS_PATH)/include/ \
		-g -G \
		-o example
