import os
import json
import torch
import warnings
import argparse
from transformers import (
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
from flute.integrations.learnable import LearnableQuantizedLinear

FLUTE_CONFIG_FILE_NAME = "flute_config.json"


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
                    if isinstance(child, LearnableQuantizedLinear):
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
                        workspace_lazy_init=False,
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

                # use a smaller data type to save memory and
                # make sure this is a lossless conversion
                if not (_Q.to(dtype=torch.uint8) == _Q).all():
                    raise ValueError

                Q  = flute.utils.pack(
                    _Q.to(dtype=torch.uint8).T.contiguous(),
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
    __constants__ = ["in_features", "out_features", "num_bits", "group_size", "workspace_lazy_init"]
    in_features : int
    out_features: int
    dtype       : torch.dtype

    num_bits    : int
    group_size  : int
    weight      : torch.Tensor
    scales      : torch.Tensor
    tables      : torch.Tensor
    tables2     : torch.Tensor

    workspace_lazy_init: bool

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_bits: int,
        group_size: int,
        workspace_lazy_init: bool = False,
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
        self.workspace_lazy_init = workspace_lazy_init
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

        if self.workspace_lazy_init is True:
            workspace = flute.utils.get_workspace_streamk(inputs.device)
        else:
            workspace = self.workspace

        output = flute.qgemm_simple(
            inputs,
            self.weight,
            self.scales,
            self.tables,
            self.tables2,
            workspace,
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
        FLUTE_CONFIG_FILE_NAME)
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
