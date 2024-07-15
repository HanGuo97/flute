MODEL_DIRECTORY=$1

TENSOR_PARALLEL_SIZE=$(nvidia-smi --list-gpus | wc -l)
echo "TENSOR_PARALLEL_SIZE: ${TENSOR_PARALLEL_SIZE}"


for model in ${MODEL_DIRECTORY}/Meta-Llama-3-8B-Instruct-*16
do

python -m tests.vllm \
    --model ${model} \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --output-dir ./tests/

python -m tests.vllm \
    --model ${model} \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --output-dir ./tests/ \
    --generate

done


for model in ${MODEL_DIRECTORY}/Meta-Llama-3-8B-Instruct-*16-Fake
do

python -m tests.vllm \
    --model ${model} \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --fake \
    --output-dir ./tests/ \
    --generate

done


for model in ${MODEL_DIRECTORY}/Meta-Llama-3-70B-Instruct-*16
do

python -m tests.vllm \
    --model ${model} \
    --tokenizer meta-llama/Meta-Llama-3-70B-Instruct \
    --output-dir ./tests/ \
    --generate

python -m tests.vllm \
    --model ${model} \
    --tokenizer meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --output-dir ./tests/ \
    --generate

done


for model in ${MODEL_DIRECTORY}/Meta-Llama-3-70B-Instruct-*16-Fake
do

python -m tests.vllm \
    --model ${model} \
    --tokenizer meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --fake \
    --output-dir ./tests/ \
    --generate

done
