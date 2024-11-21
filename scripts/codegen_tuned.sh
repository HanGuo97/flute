set -e

cp flute/csrc/qgemm_kernel_generated.template.cu flute/csrc/qgemm_kernel_generated.cu
python -m flute.codegen_utils --tuned-no-M
