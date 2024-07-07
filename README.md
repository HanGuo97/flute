
# Getting Started

Install FLUTE with pip or [from source](#build-from-source):
```bash
pip install -i https://test.pypi.org/simple/ flute
```

[FLUTE-quantized models](#models) can be directly served using exisiting frameworks such as vLLM.

```diff
- python -m vllm.entrypoints.openai.api_server \
+ python -m flute.integrations.vllm vllm.entrypoints.openai.api_server \
    --model [MODEL] \
    --tokenizer [TOKENIZER] \
    --tensor-parallel-size [TP_SIZE] \
+   --quantization flute
```

# Kernel Compatibility

| Description      | Supported (via pip) | Supported (build from source) |
| ----------- | ----------- | ----------- |
| Input dtypes   | `torch.float16` `torch.bfloat16` |  |
| Bits | `4bit` `3bit` | `2bit` |
| Group Sizes | `32` `64` `128` `256` | ❓ |
| GPUs | `A100` `A6000` | `RTX 4090` `H100` (unoptimized) |

# Models

> [!WARNING]
> As of the current release, the kernel is shape-specialized due to legacy reasons (i.e., we tune tile sizes etc for each matrix shape). Please see the below chart for the supported use cases, as different platform and tensor parallel size changes the matrix shapes. We plan to add supports for a broad range of shapes in the near future. In the meantime, please let us know if you have any specific models in mind and we are happy to add support for them.

| Model      | Single GPU / Pipeline Parallel | Tensor Parallel | Link |
| ----------- | ----------- | ----------- | ----------- | 
| LLaMA-3 (8B) | ✅ | | TBD |
| LLaMA-3 (70B) | ✅ | 2 or 4 GPUs  | TBD |
| Gemma-2 (9B) | ✅ |  | TBD |
| Gemma-2 (27B) | ✅ | 2 or 4 GPUs  | TBD |


### Quantizing Your Own Models

We provide two APIs to quantize a custom models. The easist way is to use the command line interface,
```bash
python -m flute.integrations.base \
    --pretrained_model_name_or_path meta-llama/Meta-Llama-3-70B-Instruct \
    --save_directory Meta-Llama-3-70B-Instruct-NF4 \
    --num_bits 4 \
    --group_size 128
```
The CLI essentially wraps around the following Python API,

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

# Build From Source

```bash
git clone https://github.com/HanGuo97/flute
cd flute
pip install -e .
```

**Note:** the build process requires having the local CUDA version (`nvcc --version`) match PyTorch's CUDA. In the unlikely situation in which the build process throws an error related to CUDA version mismatch, even if they should match, try adding `--no-build-isolation`.

# Integrations
### Alternative vLLM Integration

For users who prefer a non-monkey-patch solution, we also provide a forked version of vLLM. 
```bash
git clone https://github.com/HanGuo97/vllm
cd vllm
pip install -e .  # This may take 5-10 minutes.
```
