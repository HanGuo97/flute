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

# Compatibility

### Kernel
| Description      | Supported (via pip) | Supported (build from source) | Unsupported |
| ----------- | ----------- | ----------- | ----------- |
| Input dtypes   | `torch.float16` `torch.bfloat16` |  | `torch.float32` |
| Bits | `NF4` `NF3` | `NF2` `INT4` `INT3` `INT2` | |
| Group Sizes | `32` `64` `128` `256` | | |
| GPUs | `A100` `A6000` | `RTX 4090` `H100 (unoptimized)` | `V100` |

### Models

> [!WARNING]
> As of the current release, the kernel is shape-specialized due to legacy reasons (i.e., we tune tile sizes etc for each matrix shape). Please see the below chart for the supported use cases, as different platform and tensor parallel size changes the matrix shapes. We plan to add supports for a broad range of shapes in the near future. In the meantime, please let us know if you have any specific models in mind and we are happy to add support for them.

| Model      | 1 GPU (+ Pipeline Parallel) | 2 GPUs (Tensor Parallel) | 4 GPUs (Tensor Parallel) | Link |
| ----------- | ----------- | ----------- | ----------- | ----------- | 
| LLaMA-3 (8B) | ✅ | ❓ | ❓ | TBD |
| LLaMA-3 (70B) | ✅ | ✅ | ✅ | TBD |
| Gemma-2 (9B) | ✅ | ❓ | ❓ | TBD |
| Gemma-2 (27B) | ✅ | ✅ | ✅ | TBD |



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

**Method 1 (Recommended):** Using the monkey-patched entry-point

```diff
- python -m vllm.entrypoints.openai.api_server \
+ python -m flute.integrations.vllm vllm.entrypoints.openai.api_server \
    --model [MODEL] \
    --tokenizer [TOKENIZER] \
    --tensor-parallel-size [TP_SIZE] \
+    --quantization flute
```

**Method 2:** Alternatively, install the forked version of vLLM. 
```bash
git clone https://github.com/HanGuo97/vllm
cd vllm
pip install -e .  # This may take 5-10 minutes.
```
