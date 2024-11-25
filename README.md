<p align="center">
    <img src="assets/flute-logo.png" alt="" width="40%" align="top" style="border-radius: 10px; padding-left: 120px; padding-right: 120px; background-color: white;">
</p>

<p align="center">
  <em><strong>FLUTE</strong>: Flexible Lookup Table Engine for LUT-quantized LLMs <br></em>
</p>

<div align="center">

  ![GitHub License](https://img.shields.io/github/license/HanGuo97/flute)
  <a href="https://pypi.org/project/flute-kernel/">![Version](https://img.shields.io/pypi/v/flute-kernel)</a>
  <a href="https://arxiv.org/abs/2407.10960">![arXiv](https://img.shields.io/badge/arXiv-2407.10960-b31b1b.svg)</a>
</div>

<div align="center">

[[Background](#background)] [[Benchmarks](#benchmarks)] [[Getting Started](#getting-started)] [[Compatibility](#support-and-compatibility)] [[Model Zoo](#model-zoo)]

</div>

# Update
- **October 5, 2024.** FLUTE will appear in EMNLP 2024 (Findings).
- **September 15, 2024.** Added [experimental support](#flute--huggingface) for loading pre-quantized FLUTE models in HuggingFace.
- **September 6, 2024.** Added (unlearned) NF-quantized LLaMA-3.1 (405B) models: [base](https://huggingface.co/radi-cho/Meta-Llama-3.1-405B-FLUTE/tree/nf_w4g64) and [instruction tuned](https://huggingface.co/radi-cho/Meta-Llama-3.1-405B-Instruct-FLUTE/tree/nf_w4g64).
- **August 31, 2024.** Added [support](#learned-normal-float-quantization-nfl) and [example](https://github.com/HanGuo97/flute/blob/main/examples/learnable_scales_eval.ipynb) for the Learned Normal Float (NFL) quantization.
- **August 26, 2024.** Added [support](#converting-bitsandbytes-model-into-flute-model) for converting `bitsandbytes` model into FLUTE model.
- **August 5, 2024.** Added quantized LLaMA-3.1 (8B/70B) models.
- **August 2, 2024.** Added support for RTX4090.
- **July 27, 2024.** Added support for LLaMA-3.1 (405B) and tuned BF16 performance. FP16 is still the recommended data type, especially for 3-bit settings.

# Installation

Install FLUTE with pip or [from source](#build-from-source):
```bash
# For CUDA 12.1
pip install flute-kernel
# For CUDA 11.8
pip install flute-kernel -i https://flute-ai.github.io/whl/cu118
```
Head over to [Getting Started](#getting-started) and try it out!

# Background
**Uniform quantization** converts full precision weights to lower-precision intervals of equal size. **Lookup table (LUT) quantization** is a flexible variant of non-uniform quantization which can map intervals to arbitrary values via a lookup table. 

<table align="center">
<tr>
<th>Uniform (Integer) Quantization</th>
<th>Lookup Table Quantization</th>
</tr>
<tr>
<td align="center">

$$\widehat{\mathbf{W}} = \mathtt{float}(\mathbf{Q}) \cdot \mathbf{s}$$

</td>
<td align="center">

$$\widehat{\mathbf{W}} = \mathtt{tableLookup}(\mathbf{Q}, \mathtt{table}) \cdot \mathbf{s}$$

</td>
</tr>
</table>

where $\mathbf{Q}$ denote the quantized weight, $\mathbf{s}$ the (group-wise) scales, and $\widehat{\mathbf{W}}$ the de-quantized weight. Here are some examples of the lookup table suppored in FLUTE.

<table align="center">
<tr>
<th>Examples</th>
<th>Notes</th>
</tr>
<tr>
<td align="left">

`int4`, `int3`, `int2`

</td>
<td align="left">

recovers uniform/integer quantization

</td>
</tr>
<tr>
<td align="left">

`fp4`, `fp3`, `fp2`

</td>
<td align="left">
</td>
</tr>
<tr>
<td align="left">

`nf4`, `nf3`, `nf2`

</td>
<td align="left">

generalizes the `nf4` data-format introduced in QLoRA

</td>
</tr>
</td>
</tr>
<tr>
<td align="left">

any arbitrary table

</td>
<td align="left">

you could even learn it!

</td>
</tr>
</table>

### New Models Powered by FLUTE
The flexibility of the kernel could lead to new quantization algorithms. As a proof of concept, we are releasing a few [models](#models) quantized using **Learned Normal Float (NFL)** --- a simple extension to the `nf4` data format introduced in QLoRA. NFL initialized the lookup table and the scales with those from NF quantization. Then, it uses calibration data to learn the scales via straight through estimation for for the gradient with respect to the scales.


# Benchmarks

For additional benchmarks, detailed breakdowns, and corresponding instruction-tuned models, please refer to the paper and the [model zoo](#model-zoo).

<p align="center">
  <img src="assets/intro-figure.jpg" />
</p>


### LLaMA-3.1
|               | Wiki PPL | C4 PPL    | LLM Eval Avg.  |               | Wiki PPL | C4 PPL   | LLM Eval Avg.  |
| -----------   | ---- | ----- | -----          | -----------   | ---- | ---- | -----          |
| LLaMA-3.1 (8B)  | 6.31 | 9.60  | 69.75          | LLaMA-3.1 (70B) | 2.82 | 7.18 | 75.45          |
| + NFL W4G64       | 6.24 | 10.06 | 69.13          | + NFL W4G64       | 3.09 | 7.53 | 74.84          |
| + NFL W3G64       | 7.23 | 11.83 | 65.66          | + NFL W3G64       | 4.29 | 8.91 | 72.65          |


### Gemma-2
|               | Wiki PPL | C4 PPL    | LLM Eval Avg.  |               | Wiki PPL | C4 PPL   | LLM Eval Avg.  |
| -----------   | ---- | ----- | -----          | -----------   | ---- | ---- | -----          |
| Gemma-2 (9B)  | 6.88 | 10.12 | 73.12          | Gemma-2 (27B) | 5.70 | 8.98 | 75.71          |
| + NFL W4G64       | 6.49 | 10.35 | 72.50          | + NFL W4G64       | 5.69 | 9.31 | 74.11          |


# Getting Started

## FLUTE + vLLM
FLUTE-quantized models ([Model Zoo](#models)) can be directly served using exisiting frameworks such as vLLM.

```diff
- python -m vllm.entrypoints.openai.api_server \
+ python -m flute.integrations.vllm vllm.entrypoints.openai.api_server \
    --model [MODEL] \
    --revision [REVISION] \
    --tensor-parallel-size [TP_SIZE] \
+   --quantization flute
```

For example, the following commmand runs the FLUTE-quantized LLaMA-3.1 (8B) on a single GPU.

```bash
python -m flute.integrations.vllm vllm.entrypoints.openai.api_server \
    --model radi-cho/Meta-Llama-3.1-8B-FLUTE \
    --quantization flute
```

We can then query the vLLM server as usual.
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "radi-cho/Meta-Llama-3.1-8B-FLUTE",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

## FLUTE + HuggingFace
FLUTE also runs out of the box with HuggingFace and its `accelerate` extension. This integration is mostly experimental and not optimized. Users sensitive to performance considerations should use the `vLLM` integration instead.

1. Loading a pre-quantized FLUTE model.

```diff
import flute.integrations.huggingface

- model = AutoModelForCausalLM.from_pretrained(
+ model = flute.integrations.huggingface.from_pretrained(
    "radi-cho/Meta-Llama-3.1-8B-FLUTE",
    # all of your favoriate HF flags will be forwarded
    device_map="auto")
```

2. Loading and quantizing a dense model.

```python
import flute.integrations.base
flute.integrations.base.prepare_model_flute(
    name="model.model.layers",
    module=model.model.layers,  # for LLaMA-3 and Gemma-2
    num_bits=num_bits,
    group_size=group_size,
    fake=False,
    handle_hooks=True)  # for `accelerate` hooks
```

After this, the model can be used as normal. Please checkout the quantization [guide](#quantizing-your-own-models) for more information.



# Support and Compatibility

## Kernel

| Description      | Supported (via pip) | Supported (build from source) |
| ----------- | ----------- | ----------- |
| Input dtypes   | `torch.float16` `torch.bfloat16` |  |
| Bits | `4bit` `3bit` | `2bit` |
| Group Sizes | `32` `64` `128` `256` | ❓ |
| GPUs | `A100` `A6000` `RTX 4090` | `H100` (unoptimized) |

> [!WARNING]
> In the current release, we noticed `torch.bfloat16` is slower than `torch.float16`. This likely because of lack of tuning, and that Ampere GPUs lack a hardware acceleration for `bfloat16` [vectorized atomic-add](https://github.com/HanGuo97/flute/blob/main/flute/csrc/cutlass_extensions_bf16.h#L27).

> [!WARNING]
> We noticed several numerically unstable situations using `bits=4, group-size=256, GPU=A100`, though this is relatively rare (8 of 9360 test cases failed). We also noticed correctness issues in some situations with `bits=4, group-size=256, dtype=bfloat16, GPU=RTX4090` (1 of 52 test cases failed). We will be looking into this, but we suggest avoiding these particular use cases (`W4G256`) for now. 

## Models

> [!NOTE]
> As of the current release, the kernel is shape-specialized due to legacy reasons (i.e., we tune tile sizes etc for each matrix shape). Please see the below chart for the supported use cases, as different platform and tensor parallel size changes the matrix shapes. We plan to add supports for a broad range of shapes in the near future. In the meantime, please let us know if you have any specific models in mind and we are happy to add support for them.

| Model      | Single GPU / Pipeline Parallel | Tensor Parallel |
| ----------- | ----------- | ----------- |
| LLaMA-3/3.1 (8B) | ✅ | |
| LLaMA-3/3.1 (70B) | ✅ | 2 or 4 GPUs  |
| LLaMA-3.1 (405B) | ✅ | 4 or 8 GPUs  |
| Gemma-2 (9B) | ✅ |  |
| Gemma-2 (27B) | ✅ | 2 or 4 GPUs  |


# Model Zoo

> [!NOTE]
> The models we release here are trained on more data and hence different from those in the paper.

> [!TIP]
> The HuggingFace Hub links are for `NFL W4G64` quantization by default. To use the `NFL W3G64` quantization, add `--revision nfl_w3g64`.


### [LLaMA-3.1 (8B)](https://huggingface.co/radi-cho/Meta-Llama-3.1-8B-FLUTE)

|             | Wiki | C4    | PIQA  | ARC-E | ARC-C | HellaSwag | Wino  | Avg.  |
| ----------- | ---- | ----- | ----- | ----- | ----- | --------- | ----- | ----- |
| Unquantized | 6.31 | 9.60  | 79.16 | 82.20 | 52.65 | 60.71     | 74.03 | 69.75 |
| NFL W4G64       | 6.24 | 10.06 | 79.38 | 81.61 | 51.54 | 59.57     | 73.56 | 69.13 |
| NFL W3G64       | 7.23 | 11.83 | 77.91 | 76.98 | 46.33 | 56.74     | 70.32 | 65.66 |


### [LLaMA-3.1 (70B)](https://huggingface.co/radi-cho/Meta-Llama-3.1-70B-FLUTE)

|             | Wiki | C4    | PIQA  | ARC-E | ARC-C | HellaSwag | Wino  | Avg.  |
| ----------- | ---- | ----- | ----- | ----- | ----- | --------- | ----- | ----- |
| Unquantized | 2.82 | 7.18  | 82.81 | 85.31 | 59.64 | 67.49     | 82.00 | 75.45 |
| NFL W4G64       | 3.09 | 7.53  | 83.03 | 85.52 | 58.19 | 67.04     | 80.43 | 74.84 |
| NFL W3G64       | 4.29 | 8.91  | 82.04 | 83.29 | 54.78 | 64.99     | 78.14 | 72.65 |

### [LLaMA-3.1 (405B)](https://huggingface.co/radi-cho/Meta-Llama-3.1-405B-FLUTE)
Note that the weights are in the branch `nf_w4g64` and thus `--revision nf_w4g64` is needed since these are not on the default branch.

### [LLaMA-3.1 Instruct (8B)](https://huggingface.co/radi-cho/Meta-Llama-3.1-8B-Instruct-FLUTE)

|             | Wiki | C4    |
| ----------- | ---- | ----- |
| NFL W4G64       | 6.78 | 11.11 |
| NFL W3G64       | 7.73 | 12.83 |


### [LLaMA-3.1 Instruct (70B)](https://huggingface.co/radi-cho/Meta-Llama-3.1-70B-Instruct-FLUTE)

|             | Wiki | C4    |
| ----------- | ---- | ----- |
| NFL W4G64       | 4.15 | 9.18  |
| NFL W3G64       | 4.74 | 9.48  |

### [LLaMA-3.1 Instruct (405B)](https://huggingface.co/radi-cho/Meta-Llama-3.1-405B-Instruct-FLUTE)
Note that the weights are in the branch `nf_w4g64` and thus `--revision nf_w4g64` is needed since these are not on the default branch.


### [LLaMA-3 (8B)](https://huggingface.co/radi-cho/Meta-Llama-3-8B-FLUTE)

|             | Wiki | C4    | PIQA  | ARC-E | ARC-C | HellaSwag | Wino  | Avg.  |
| ----------- | ---- | ----- | ----- | ----- | ----- | --------- | ----- | ----- |
| Unquantized | 6.1  | 9.2   | 79.9  | 80.1  | 50.4  | 60.2      | 72.8  | 68.6  |
| NFL W4G64       | 6.11 | 9.38  | 79.33 | 79.79 | 49.74 | 59.22     | 73.95 | 68.41 |
| NFL W3G64       | 7.13 | 11.06 | 78.78 | 76.22 | 44.37 | 56.69     | 70.32 | 65.28 |


### [LLaMA-3 (70B)](https://huggingface.co/radi-cho/Meta-Llama-3-70B-FLUTE)

|             | Wiki | C4   | PIQA  | ARC-E | ARC-C | HellaSwag | Wino  | Avg.  |
| ----------- | ---- | ---- | ----- | ----- | ----- | --------- | ----- | ----- |
| Unquantized | 2.9  | 6.9  | 82.4  | 86.9  | 60.3  | 66.4      | 80.6  | 75.3  |
| NFL W4G64       | 3.03 | 7.03 | 82.15 | 85.98 | 57.85 | 66.17     | 79.79 | 74.39 |
| NFL W3G64       | 4.15 | 8.10 | 80.74 | 83.71 | 55.29 | 64.05     | 78.45 | 72.45 |


### [LLaMA-3 Instruct (8B)](https://huggingface.co/radi-cho/Meta-Llama-3-8B-Instruct-FLUTE)

|             | Wiki | C4    |
| ----------- | ---- | ----- |
| NFL W4G64       | 6.78 | 10.61 |
| NFL W3G64       | 7.75 | 12.28 |


### [LLaMA-3 Instruct (70B)](https://huggingface.co/radi-cho/Meta-Llama-3-70B-Instruct-FLUTE)

|       | Wiki | C4    |
| ----- | ---- | ----- |
| NFL W4G64 | 3.67 | 7.95  |
| NFL W3G64 | 4.90 | 10.86 |


### [Gemma-2 (9B)](https://huggingface.co/radi-cho/gemma-2-9b-FLUTE)

|             | Wiki | C4    | PIQA  | ARC-E | ARC-C | HellaSwag | Wino  | Avg.  |
| ----------- | ---- | ----- | ----- | ----- | ----- | --------- | ----- | ----- |
| Unquantized | 6.88 | 10.12 | 81.39 | 87.37 | 61.35 | 61.23     | 74.27 | 73.12 |
| NFL W4G64       | 6.49 | 10.35 | 81.28 | 86.24 | 59.30 | 60.40     | 75.30 | 72.50 |
| NFL W3G64       | 7.06 | 11.14 | 80.52 | 83.16 | 55.46 | 58.28     | 72.69 | 70.02 |


### [Gemma-2 (27B)](https://huggingface.co/radi-cho/gemma-2-27b-FLUTE)

|             | Wiki | C4   | PIQA  | ARC-E | ARC-C | HellaSwag | Wino  | Avg.  |
| ----------- | ---- | ---- | ----- | ----- | ----- | --------- | ----- | ----- |
| Unquantized | 5.70 | 8.98 | 83.24 | 87.84 | 62.88 | 65.35     | 79.24 | 75.71 |
| NFL W4G64       | 5.69 | 9.31 | 82.53 | 86.45 | 59.22 | 64.13     | 78.21 | 74.11 |


### [Gemma-2 Instruct (9B)](https://huggingface.co/radi-cho/gemma-2-9b-it-FLUTE)

|             | Wiki | C4    |
| ----------- | ---- | ----- |
| NFL W4G64       | 6.88 | 11.02 |
| NFL W3G64       | 7.35 | 11.72 |

### [Gemma-2 Instruct (27B)](https://huggingface.co/radi-cho/gemma-2-27b-it-FLUTE)

|       | Wiki | C4    |
| ----- | ---- | ----- |
| NFL W4G64 | 5.91 | 9.71  |


## Quantizing Your Own Models

We provide two APIs to quantize a custom models. The easist way is to use the command line interface.

### Simple Normal Float Quantization

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
    Gemma2ForCausalLM,
    AutoModelForCausalLM)
import flute.integrations.base

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path,
    device_map="cpu",
    torch_dtype="auto")

if isinstance(model, (LlamaForCausalLM, Gemma2ForCausalLM)):
    flute.integrations.base.prepare_model_flute(
        name="model.model.layers",
        module=model.model.layers,
        num_bits=num_bits,
        group_size=group_size,
        fake=False)
else:
    # more models to come
    raise NotImplementedError
```

### Converting `bitsandbytes` Model into FLUTE Model

While FLUTE has its own Normal Float (NF) implementation, we could convert an existing HuggingFace model quantized via `bitsandbytes` into FLUTE format. To do so, just add two lines to the Python API,

```diff
flute.integrations.base.prepare_model_flute(
    name="model.model.layers",
    module=model.model.layers,
    num_bits=num_bits,
    group_size=group_size,
    fake=False,
+   prepare_bnb_layers=True,
+   default_bnb_dtype=torch.float16,
)
```

It's worth noting that we do not support double quantization, and the conversion will materialize the first-level scales.

### Learned Normal Float Quantization (NFL)

NFL initialized the lookup table and the scales with those from NF quantization. Then, it uses calibration data to learn the scales via straight through estimation for for the gradient with respect to the scales.

To use NFL quantization, call the following function before `prepare_model_flute`. We also provide an [example jupyter notebook](https://github.com/HanGuo97/flute/blob/main/examples/learnable_scales_eval.ipynb) to illustrate the entire process.

```python
import flute.integrations.learnable

flute.integrations.learnable.learn_scales(
    model=model,
    tokenizer=tokenizer,
    num_bits=num_bits,
    group_size=group_size,
    custom_corpora=list_of_corpora,
    samples=num_samples,
)
```

# Extending to New Models (Experimental)

At the moment, FLUTE kernel is specialized to the combination of GPU, matrix shapes, data types, bits, and group sizes. This means adding supporting new models requires tuning the kernel configurations for the corresponding use cases. We are hoping to add support for just-in-time tuning, but in the meantime, here are the ways to tune the kernel ahead-of-time.

### Step 1: Build the `raw` version of the library that exposes all templates.

1. Reset the previously tuned kernel,

```bash
cp flute/csrc/qgemm_kernel_generated.template.cu flute/csrc/qgemm_kernel_generated.cu
```

2. Un-comment the combination(s) to tune in `flute/csrc/qgemm_kernel_raw_generated.cu`,

```cpp
INSTANTIATE_TEMPLATE(NUM_SMs, DTYPE, cute::uint16_t, __half2, BITS, GROUP_SIZE);
```

<details>
<summary> Example for W4G64 on A100 </summary>

```diff
-// INSTANTIATE_TEMPLATE(108, cute::half_t    , cute::uint16_t, __half2       , 4, 64);
+INSTANTIATE_TEMPLATE(108, cute::half_t    , cute::uint16_t, __half2       , 4, 64);

-// INSTANTIATE_TEMPLATE(108, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 64);
+INSTANTIATE_TEMPLATE(108, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 64);
```

</details>

3. Remove settings _not tuned_ in `flute/csrc/qgemm.cpp`, `flute/__init__.py`, and `flute/ops.py`

> [!NOTE]
> Although including other settings could still build, it could break the linking process and require re-compiling the library.

<details>
<summary> Example for W4G64 on A100 </summary>


```diff
diff --git a/flute/csrc/qgemm.cpp b/flute/csrc/qgemm.cpp
index 84bae95..c4a0236 100644
--- a/flute/csrc/qgemm.cpp
+++ b/flute/csrc/qgemm.cpp
@@ -314,3 +313,0 @@ qgemm_raw_simple(const at::Tensor& input,
-        case 32:                                      \
-            RUN_QGEMM_RAW(T, NUM_BITS, 32);           \
-            break;                                    \
@@ -320,6 +316,0 @@ qgemm_raw_simple(const at::Tensor& input,
-        case 128:                                     \
-            RUN_QGEMM_RAW(T, NUM_BITS, 128);          \
-            break;                                    \
-        case 256:                                     \
-            RUN_QGEMM_RAW(T, NUM_BITS, 256);          \
-            break;                                    \
@@ -335,6 +325,0 @@ qgemm_raw_simple(const at::Tensor& input,
-        case 2:                                          \
-            RUN_QGEMM_RAW_SWITCH_GROUP_SIZE(T, 2);       \
-            break;                                       \
-        case 3:                                          \
-            RUN_QGEMM_RAW_SWITCH_GROUP_SIZE(T, 3);       \
-            break;                                       \
@@ -381 +366 @@ TORCH_LIBRARY(flute, m) {
-    // m.def("qgemm_raw_simple_80(Tensor input, Tensor weight, Tensor(a!) output, Tensor scales, Tensor table, Tensor table2, Tensor(b!) workspace, int
 num_bits, int group_size, int template_id) -> ()");
+    m.def("qgemm_raw_simple_80(Tensor input, Tensor weight, Tensor(a!) output, Tensor scales, Tensor table, Tensor table2, Tensor(b!) workspace, 
int num_bits, int group_size, int template_id) -> ()");
@@ -391 +376 @@ TORCH_LIBRARY_IMPL(flute, CUDA, m) {
-    // m.impl("qgemm_raw_simple_80", &qgemm_raw_simple<cute::Int<108>>);
+    m.impl("qgemm_raw_simple_80", &qgemm_raw_simple<cute::Int<108>>);
```

```diff
diff --git a/flute/__init__.py b/flute/__init__.py
index 34b1a26..f524841 100644
--- a/flute/__init__.py
+++ b/flute/__init__.py
@@ -69 +69 @@ QGEMM_SIMPLE_DICT = {
-# QGEMM_RAW_SIMPLE_DICT = {
+QGEMM_RAW_SIMPLE_DICT = {
@@ -71 +71 @@ QGEMM_SIMPLE_DICT = {
-#     108: cast(QGEMM_RAW_SIMPLE_TYPE, torch.ops.flute.qgemm_raw_simple_80),
+    108: cast(QGEMM_RAW_SIMPLE_TYPE, torch.ops.flute.qgemm_raw_simple_80),
@@ -73 +73 @@ QGEMM_SIMPLE_DICT = {
-# }
+}
@@ -76 +76 @@ qgemm_simple     = QGEMM_SIMPLE_DICT[NUM_SMS]
-qgemm_raw_simple = None  # QGEMM_RAW_SIMPLE_DICT[NUM_SMS]
+qgemm_raw_simple = QGEMM_RAW_SIMPLE_DICT[NUM_SMS]
```

```diff
diff --git a/flute/ops.py b/flute/ops.py
index 9fd91a2..80782ea 100644
--- a/flute/ops.py
+++ b/flute/ops.py
@@ -124 +124 @@ def _qgemm_simple_89_abstract(
-# @torch.library.impl_abstract("flute::qgemm_raw_simple_80")
+@torch.library.impl_abstract("flute::qgemm_raw_simple_80")
```

</details>

4. Build from source (see instructions below).

```bash
pip install -e . --no-build-isolation  # `--no-build-isolation` is optional
```

Depending on the number of configurations to tune, this could take time in the order of tens of minutes to hours.

### Step 2: Tune FLUTE on the new matrix shapes.

```python
import torch
from flute.tune import TuneTask, tune_tasks_legacy

tasks = [
    TuneTask(
        M=1,                  # batch size (x sequence length, usually 1 for token-by-token generation)
        N=1024,               # parameter dimension (note when using tensor-parallelism, this could change)
        K=4096,               # parameter dimension (note when using tensor-parallelism, this could change)
        num_bits=4,           # number of bits
        group_size=64,        # group size
        num_sms=108,          # number of streaming multiprocessors of the GPU
        dtype=torch.float16,  # data type
        device=torch.device("cuda:0")
    ),
]

tune_tasks_legacy(tasks)
```

After this step is complete, artifacts will be saved in `flute/data/`.

### Step 3: Build the newly-tuned kernel

```bash
# remove changes
git checkout -- flute/csrc/

# generating new dispatching logic based on tuning artifacts
bash scripts/codegen_tuned.sh

# remove changes
git checkout -- \
    flute/ops.py \
    flute/__init__.py

# Build
pip install -e . --no-build-isolation
```


Note that if only one data type is tuned, you will also need to edit `flute/utils.py`.

<details>
<summary> Example </summary>

```diff
diff --git a/flute/utils.py b/flute/utils.py
index 5add543..13f49c0 100644
--- a/flute/utils.py
+++ b/flute/utils.py
@@ -270,7 +270,7 @@ def pack(
 
         K, N = W.shape
         template_ids = []
-        for dtype in [torch.float16, torch.bfloat16]:
+        for dtype in [torch.float16]:
             template_id = TEMPLATE_TUNED_WITHOUT_M_CONFIGS[(
                 NUM_SMS,
                 num_bits,
```

</details>


Finally, please follow the examples in `tests/` to verify that the kernel is working correctly.

# Build From Source

1. Clone the CUTLASS library.

```bash
# Unfortunately, the path is hard-coded as of now. If you install CUTLASS
# in a different directory, please make sure the corresponding path in
# `setup.py` is updated.
cd /workspace

git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout v3.4.1
```

2. Build.

```bash
git clone https://github.com/HanGuo97/flute
cd flute
pip install -e .
```

**Note:** the build process requires having the local CUDA version (`nvcc --version`) match PyTorch's CUDA. In situations in which the build process throws an error related to CUDA version mismatch, try adding `--no-build-isolation`.


# Acknowledgement and Citation

Special thanks to Dmytro Ivchenko, Yijie Bei, and the Fireworks AI team for helpful discussion. If you find any of the models or code in this repo useful, please feel free to cite:

```bibtex
@inproceedings{flute2024,
  title={Fast Matrix Multiplications for Lookup Table-Quantized LLMs},
  author={Guo, Han and Brandon, William and Cholakov, Radostin and Ragan-Kelley, Jonathan and Xing, Eric and Kim, Yoon},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2024},
  pages={12419--12433},
  year={2024}
}
```
