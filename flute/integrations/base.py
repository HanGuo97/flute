import os
import math
import json
import torch
import warnings
import argparse

from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    Gemma2ForCausalLM,
    AutoModelForCausalLM)
from accelerate.hooks import (
    ModelHook,
    add_hook_to_module)
from bitsandbytes.nn import (
    Linear4bit as BNBLinear4bit)
from typing import Optional, Dict

import flute
import flute.utils
import flute.nf_utils
import flute.integrations.bitsandbytes


def get_accelerate_hook(name: str, module: torch.nn.Module, allow: bool) -> Optional[ModelHook]:

    hook = getattr(module, "_hf_hook", None)
    if hook is not None and allow is not True:
        raise ValueError(f"`{name}` has accelerate `hook`")

    if hasattr(module, "_old_forward") and allow is not True:
        raise ValueError(f"`{name}` has accelerate `_old_forward`")

    for child_name, child in module.named_children():
        # we do not allow accelerate hooks in the children
        get_accelerate_hook(f"{name}.{child_name}", child, allow=False)

    return hook


class LearnableQuantizedLinear(torch.nn.Module):
    in_features : int
    out_features: int

    num_bits    : int
    group_size  : int
    symmetric   : bool
    weight      : torch.Tensor
    scales      : torch.Tensor
    values      : torch.Tensor
    pivots      : torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        num_bits: int = 4,
        group_size: int = 64,
        symmetric: bool = False
    ):
        super(LearnableQuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.symmetric = symmetric
        self.num_bits = num_bits

        self.values, self.pivots = flute.nf_utils.get_values_pivots(num_bits, False, dtype=torch.bfloat16)
        
        if weight is None:
            self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            assert weight.dtype == torch.bfloat16, "Training is currently only supported in bfloat16!"
            self.weight = torch.nn.Parameter(weight, requires_grad=False)

        if scales is None:
            self.scales = torch.nn.Parameter(torch.max(torch.abs(self.weight.view(-1, self.group_size)), dim=1, keepdim=True).values, requires_grad=True) # * nf4_constant
        else:
            self.scales = torch.nn.Parameter(scales, requires_grad=True)
        
        if bias is None:
            self.register_parameter('bias', None)
        else:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)

    
    def forward(self, inp: torch.Tensor):
        qweight = flute.nf_utils.manual_nf4(self.weight, absmax=self.scales, bits=self.num_bits, blocksize=self.group_size, values=self.values, pivots=self.pivots)

        return torch.nn.functional.linear(inp, qweight, self.bias)


def get_parent(module: torch.nn.Module, name_split: str):
    for n in name_split:
        module = getattr(module, n)
    return module


def learn_scales(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    num_bits: int = 4,
    group_size: int = 64,
    custom_corpora: Optional[str] = None,
    epochs: int = 1,
    lr: float = 0.0001,
    iters: int = 128,
    logging: bool = False,
) -> None:
    layer_types = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]

    for param in model.parameters():
        param.requires_grad = False

    print("Adding tunable scales to the linear layers...")
    for name, module in model.named_modules():
        name_split = name.split('.')
        if name_split[-1] in layer_types:
            q_layer = LearnableQuantizedLinear(
                module.in_features,
                module.out_features,
                weight=module.weight.data, 
                bias=module.bias.data if module.bias is not None else None,
                num_bits=num_bits,
                group_size=group_size
            )

            parent = get_parent(model, name_split[:-1])
            setattr(parent, name_split[-1], q_layer)

    print("Tokenizing corpora...")
    if custom_corpora == None:
        train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        corpora = [tokenizer("\n\n".join(train["text"]), return_tensors="pt")]
    else:
        corpora = [tokenizer(corpus, return_tensors="pt") for corpus in custom_corpora]

    print("Prepare model for training...")
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    max_length = 2048
    device = model.device
    bos_token_id = tokenizer.bos_token_id

    # Use BOS token in each sequence - especially important for Gemma
    stride = max_length - 1
    seq_len = iters * (max_length - 1)

    for epoch in range(epochs):
        print(f"Running epoch {epoch}...")

        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            for encodings in corpora:
                optimizer.zero_grad()
        
                end_loc = min(begin_loc + stride, seq_len)
                trg_len = end_loc - prev_end_loc
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                input_ids = torch.concat([torch.tensor([bos_token_id], dtype=torch.int64).unsqueeze(0).to(device), input_ids], dim=1)
        
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100
        
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
                neg_log_likelihood.backward()
                
                optimizer.step()
    
                if logging:
                    print(f"Step loss: {neg_log_likelihood.item()}.")
    
                prev_end_loc = end_loc
            
                if end_loc == seq_len:
                    break


# 2/4
@torch.no_grad()
def prepare_model_flute(
    name: str,
    module: torch.nn.Module,
    num_bits: int,
    group_size: int,
    fake: bool,
    handle_hooks: bool = False,
    prepare_bnb_layers: bool = True,
    default_bnb_dtype: Optional[torch.dtype] = None,
    custom_scales_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> None:

    warnings.warn(f"Quantization always happen on 1st GPU")

    # BNB layers always assume the unquantized weights are in FP32, regardless
    # of the actual dtype of the weights. Hence we cannot infer the dtype.
    if default_bnb_dtype is None:
        default_bnb_dtype = torch.float16

    def _replace_linear(_name: str, _module: torch.nn.Module) -> None:
        for child_name, child in _module.named_children():

            child_full_name = f"{_name}.{child_name}"

            if isinstance(child, torch.nn.Linear) or isinstance(child, LearnableQuantizedLinear):

                if isinstance(child, BNBLinear4bit):
                    if child.weight.dtype not in [torch.uint8]:
                        raise NotImplementedError
                    if prepare_bnb_layers is False:
                        raise ValueError
                    if num_bits != 4:
                        raise ValueError
                    if group_size != child.weight.quant_state.blocksize:
                        raise ValueError
                else:
                    if child.weight.dtype not in [torch.float16, torch.bfloat16]:
                        raise NotImplementedError

                if fake is True:
                    if isinstance(child, BNBLinear4bit):
                        raise NotImplementedError
                    # we primarily use the fake quantization to
                    # check the outputs of the quantized model
                    new_weight = flute.nf_utils.nf_quantize_2(
                        W=child.weight.to(device="cuda"),
                        num_bits=num_bits,
                        group_size=group_size,
                        dtype=child.weight.dtype)
                    # we use assignment instead of in-place copy to
                    # make sure there are no type casting operations.
                    child.weight = torch.nn.Parameter(
                        new_weight.to(device=child.weight.device),
                        requires_grad=False)
                    continue

                if handle_hooks is True:
                    # as of now, we do not support PyTorch hooks
                    # https://discuss.pytorch.org/t/how-to-check-where-the-hooks-are-in-the-model/120120
                    if len(child._backward_hooks) != 0:
                        raise NotImplementedError
                    if len(child._forward_hooks) != 0:
                        raise NotImplementedError
                    if len(child._forward_pre_hooks) != 0:
                        raise NotImplementedError

                    # the replacement will remove the accelerate hooks
                    maybe_hook = get_accelerate_hook(child_name, child, allow=True)

                if not isinstance(child, BNBLinear4bit):
                    flute_dtype = child.weight.dtype
                else:
                    flute_dtype = child.weight.quant_state.dtype
                    if flute_dtype == torch.float32:
                        flute_dtype = default_bnb_dtype
                        warnings.warn(f"BNB's `dtype` is `torch.float32`, changed to `{flute_dtype}`")

                setattr(
                    _module,
                    child_name,
                    FluteLinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        num_bits=num_bits,
                        group_size=group_size,
                        bias=(child.bias is not None),
                        device=child.weight.device,
                        dtype=flute_dtype))

                if custom_scales_dict is not None:
                    custom_scales = custom_scales_dict[child_full_name]
                else:
                    if isinstance(child, LearnableQuantizedLinear):
                        custom_scales = child.scales
                    else:
                        custom_scales = None

                if not isinstance(child, BNBLinear4bit):
                    _, _Q, scales, qmap = flute.nf_utils.nf_quantize(
                        W=child.weight.to(device="cuda"),
                        num_bits=num_bits,
                        group_size=group_size,
                        custom_scales=custom_scales)
                else:
                    _Q, scales, qmap = flute.integrations.bitsandbytes.convert_BNBLinear4bit(
                        bnb_module=child,
                        verify=True)

                Q  = flute.utils.pack(
                    _Q.T.contiguous(),
                    num_bits=num_bits,
                    group_size=group_size)

                new_child = getattr(_module, child_name)
                scales = scales.view(new_child.scales.shape)
                scales = scales.to(dtype=new_child.scales.dtype)
                qmap = qmap.to(dtype=new_child.tables.dtype)
                qmap2 = flute.utils.make_qmap2_from_qmap(qmap)

                new_child.weight.copy_(Q)
                new_child.scales.copy_(scales)
                new_child.tables.copy_(qmap)
                new_child.tables2.copy_(qmap2)
                if new_child.bias is not None:
                    new_child.bias.copy_(child.bias)

                # add the accelerate hook back
                if handle_hooks is True:
                    if maybe_hook is not None:
                        add_hook_to_module(
                            module=new_child,
                            hook=maybe_hook)

            else:
                _replace_linear(child_full_name, child)

    _replace_linear(name, module)


class FluteLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features", "num_bits", "group_size"]
    in_features : int
    out_features: int
    dtype       : torch.dtype

    num_bits    : int
    group_size  : int
    weight      : torch.Tensor
    scales      : torch.Tensor
    tables      : torch.Tensor
    tables2     : torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_bits: int,
        group_size: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:

        factory_kwargs = {"device": device, "dtype": dtype}
        if dtype not in [torch.float16, torch.bfloat16]:
            raise NotImplementedError
        if not isinstance(device, torch.device):
            raise NotImplementedError

        super().__init__()

        K = in_features
        N = out_features
        P = int(N / 16 * num_bits)
        G = int(K / group_size)
        tables = torch.arange(
            2 ** num_bits,
            dtype=dtype,
            device=device)

        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.group_size = group_size
        # scratch space used by the kernel
        self.workspace = flute.utils.get_workspace_streamk(device)

        self.register_buffer("weight", torch.empty((P, K), dtype=torch.int16, device=device))
        self.register_buffer("scales", torch.ones((N, G), dtype=dtype, device=device))
        self.register_buffer("tables", tables)
        self.register_buffer("tables2", flute.utils.make_qmap2_from_qmap(tables))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = flute.qgemm_simple(
            inputs,
            self.weight,
            self.scales,
            self.tables,
            self.tables2,
            self.workspace,
            self.num_bits,
            self.group_size,
        )

        if self.bias is not None:
            output.add_(self.bias)  # In-place add

        return output

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={False}, "
                f"num_bits={self.num_bits}, "
                f"group_size={self.group_size}")


def quantize_hf_model(
    pretrained_model_name_or_path: str,
    save_directory: str,
    num_bits: int,
    group_size: int,
    torch_dtype: str,
    fake: bool,
) -> None:
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        device_map="cpu",
        torch_dtype=torch_dtype)

    if isinstance(model, (LlamaForCausalLM, Gemma2ForCausalLM)):
        prepare_model_flute(
            name="model.model.layers",
            module=model.model.layers,
            num_bits=num_bits,
            group_size=group_size,
            fake=fake)
    else:
        raise NotImplementedError

    # save the model
    model.save_pretrained(save_directory)

    # save the config
    config = {
        "version": flute.__version__,
        "num_sms": flute.NUM_SMS,
        "num_bits": num_bits,
        "group_size": group_size,
    }
    config_path = os.path.join(
        save_directory,
        "flute_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--save_directory", type=str)
    parser.add_argument("--num_bits", type=int)
    parser.add_argument("--group_size", type=int)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()

    quantize_hf_model(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        save_directory=args.save_directory,
        num_bits=args.num_bits,
        group_size=args.group_size,
        torch_dtype=args.torch_dtype,
        fake=args.fake)
