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

# Usages

### Command Line API

```bash
python -m flute.integrations.base \
    --pretrained_model_name_or_path meta-llama/Meta-Llama-3-70B-Instruct \
    --save_directory Meta-Llama-3-70B-Instruct-NF4 \
    --num_bits 4 \
    --group_size 128
```

### Python API
```python
from transformers import (
    LlamaForCausalLM,
    AutoModelForCausalLM)
import flute.integrations.base

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path,
    device_map="cpu",
    torch_dtype="auto")

if isinstance(model, LlamaForCausalLM):
    flute.integrations.base.prepare_model_flute(
        module=model.model.layers,
        num_bits=num_bits,
        group_size=group_size,
        fake=False)
else:
    # more models to come
    raise NotImplementedError
```

### vLLM Integration

1. Add `flute` to the [list](https://github.com/HanGuo97/vllm/blob/flute-integration/vllm/model_executor/layers/quantization/__init__.py#L37) of supported quantization methods.

Alternatively, install the forked version of vLLM
```bash
git clone https://github.com/HanGuo97/vllm
cd vllm
pip install -e .  # This may take 5-10 minutes.
```


2. Specify the quantization method
```bash
python -m vllm.entrypoints.openai.api_server \
    --model [MODEL] \
    --tokenizer [TOKENIZER] \
    --tensor-parallel-size [TP_SIZE] \
    --quantization flute \
    --dtype half
```
