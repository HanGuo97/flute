# Installation

- **PyPI**
```bash
pip install -i https://test.pypi.org/simple/ flute
```

- **From source**
```bash
git clone https://github.com/HanGuo97/flute
cd flute
pip install -e .
```

# vLLM Integrations

1. Install the forked version of vLLM
```bash
git clone https://github.com/HanGuo97/vllm
cd vllm
pip install -e .  # This may take 5-10 minutes.
```

Alternatively, simply copy this [file](https://github.com/HanGuo97/vllm/blob/flute-integration/vllm/model_executor/layers/quantization/flute.py) into existing (editable) vLLM installation, and [add](https://github.com/HanGuo97/vllm/blob/flute-integration/vllm/model_executor/layers/quantization/__init__.py#L37) it to the list of supported quantization methods.

2. Specify the quantization method
```bash
python -m vllm.entrypoints.openai.api_server \
    --model [MODEL] \
    --tokenizer [TOKENIZER] \
    --tensor-parallel-size [TP_SIZE] \
    --quantization flute \
    --dtype half
```
