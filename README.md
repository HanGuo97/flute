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

## HuggingFace

### Command Line API

```bash
python -m flute.integrations \
    --pretrained_model_name_or_path meta-llama/Meta-Llama-3-70B-Instruct \
    --save_directory Meta-Llama-3-70B-Instruct-NF4 \
    --num_bits 4 \
    --group_size 128
```

### Python API
```python

import os
import json
from transformers import (
    LlamaForCausalLM,
    AutoModelForCausalLM)

import flute
import flute.utils
import flute.integrations

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path,
    device_map="cpu",
    torch_dtype="auto")

if isinstance(model, LlamaForCausalLM):
    flute.integrations.prepare_model_flute(
        module=model.model.layers,
        num_bits=num_bits,
        group_size=group_size,
        fake=False)
else:
    # more models to come
    raise NotImplementedError

# save the model
model.save_pretrained(save_directory)

# save the config
config = {
    "num_sms": flute.NUM_SMS,
    "num_bits": num_bits,
    "group_size": group_size,
}
config_path = os.path.join(
    save_directory,
    "flute_config.json")
with open(config_path, "w") as f:
    json.dump(config, f)

```

### vLLM

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
