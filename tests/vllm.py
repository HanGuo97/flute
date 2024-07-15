import os
import torch
import argparse
from typing import List
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.linear import LinearBase

import flute.integrations.vllm

# Error thresholds.
FP16_ERROR_THRESHOLD = 1.5e-3
BF16_ERROR_THRESHOLD = 1.0e-2


def test_vllm(
    fake_quantized_model: str,
    flute_quantized_model: str,
    tokenizer: str,
    verbose: bool,
) -> None:
    # vLLM integration
    flute.integrations.vllm.patch_vllm()

    llm_dense = LLM(  # type: ignore
        model=fake_quantized_model,
        tokenizer=tokenizer,
        gpu_memory_utilization=0.75)
    llm_flute = LLM(  # type: ignore
        model=flute_quantized_model,
        tokenizer=tokenizer,
        quantization="flute",
        gpu_memory_utilization=0.2)

    model_dense = llm_dense.llm_engine.model_executor.driver_worker.model_runner.model
    model_flute = llm_flute.llm_engine.model_executor.driver_worker.model_runner.model

    for (name_dense, module_dense), (name_flute, module_flute) in zip(model_dense.named_modules(), model_flute.named_modules()):
        if name_dense != name_flute:
            raise ValueError
        if not isinstance(module_dense, LinearBase):
            continue
        if not isinstance(module_flute, LinearBase):
            raise TypeError

        dtype = module_flute.scales.dtype
        if module_flute.tables.dtype != dtype:
            raise TypeError
        if module_dense.weight.dtype != dtype:
            raise TypeError
        if dtype == torch.float16:
            threshold = FP16_ERROR_THRESHOLD
        elif dtype == torch.bfloat16:
            threshold = BF16_ERROR_THRESHOLD
        else:
            raise NotImplementedError

        I = torch.eye(
            module_flute.input_size,
            dtype=dtype,
            device="cuda")

        W_, unused_bias = module_flute(I)
        if unused_bias is not None:
            raise NotImplementedError

        if not (W_.T == module_dense.weight).all():
            raise ValueError

        for batch_size in [1, 3, 7]:
            X = torch.randn(
                (batch_size, module_flute.input_size),
                dtype=dtype,
                device="cuda")
            Y_dense, unused_bias_dense = module_dense(X)
            Y_flute, unused_bias_flute = module_flute(X)
            if unused_bias_dense is not None or unused_bias_flute is not None:
                raise NotImplementedError

            error_dense = ((Y_dense - Y_flute).norm() / Y_dense.norm()).item()
            error_flute = ((Y_dense - Y_flute).norm() / Y_flute.norm()).item()
            if error_dense >= threshold or error_flute >= threshold:
                raise ValueError

            if verbose is True:
                print(f"{error_dense:.3e} {error_flute:.3e}", end=" ")

        if verbose is True:
            print()


# Sample prompts.
PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def test_vllm_generate(
    model: str,
    tokenizer: str,
    tensor_parallel_size: int,
    fake: bool,
    output_dir: str,
) -> List[str]:
    flute.integrations.vllm.patch_vllm()

    # Create a sampling params object.
    sampling_params = SamplingParams(  # type: ignore
        temperature=0.0,
        logprobs=1,
        prompt_logprobs=1,
        max_tokens=128)

    if fake is False:
        llm = LLM(  # type: ignore
            model=model,
            tokenizer=tokenizer,
            quantization="flute",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.95)
    else:
        llm = LLM(  # type: ignore
            model=model,
            tokenizer=tokenizer,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.95)

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(PROMPTS, sampling_params)

    sentences = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        sentences.append(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    if model.endswith("/"):
        raise ValueError
    output_path_prefix = os.path.basename(model)
    output_path = os.path.join(output_dir, f"{output_path_prefix}.tp-{tensor_parallel_size}.pth")
    torch.save(sentences, output_path)
    return sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    if args.generate is False:
        if args.model.endswith("/"):
            raise ValueError
        fake_quantized_model = f"{args.model}-Fake"
        flute_quantized_model = args.model
        if not os.path.isdir(fake_quantized_model):
            raise ValueError
        if not os.path.isdir(flute_quantized_model):
            raise ValueError
        test_vllm(
            fake_quantized_model=fake_quantized_model,
            flute_quantized_model=flute_quantized_model,
            tokenizer=args.tokenizer,
            verbose=True)
    else:
        test_vllm_generate(
            model=args.model,
            tokenizer=args.tokenizer,
            tensor_parallel_size=args.tensor_parallel_size,
            fake=args.fake,
            output_dir=args.output_dir)
