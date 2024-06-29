import os
import json
import torch
import argparse
from transformers import (
    LlamaForCausalLM,
    AutoModelForCausalLM)
from typing import Optional

import flute
import flute.utils
import flute.nf_utils

_WORKSPACE = flute.utils.make_workspace_streamk(torch.device("cuda"))


# 2/4
@torch.no_grad()
def prepare_model_flute(
    module: torch.nn.Module,
    num_bits: int,
    group_size: int,
    fake: bool,
) -> None:

    def _replace_linear(_module: torch.nn.Module) -> None:
        for name, child in _module.named_children():
            if isinstance(child, torch.nn.Linear):

                if fake is True:
                    # we primarily use the fake quantization to
                    # check the outputs of the quantized model
                    new_weight = flute.nf_utils.nf_quantize_2(
                        W=child.weight.to(device="cuda"),
                        num_bits=num_bits,
                        group_size=group_size,
                        dtype=torch.float16)
                    # we use assignment instead of in-place copy to
                    # make sure there are no type casting operations.
                    child.weight = torch.nn.Parameter(
                        new_weight,
                        requires_grad=False)
                    continue

                setattr(
                    _module,
                    name,
                    FluteLinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        num_bits=num_bits,
                        group_size=group_size,
                        bias=(child.bias is not None),
                        device=child.weight.device,
                        dtype=torch.float16))

                template_id = flute.TEMPLATE_TUNED_WITHOUT_M_CONFIGS[(
                    flute.NUM_SMS,
                    num_bits,
                    group_size,
                    child.out_features,  # N
                    child.in_features)]  # K
                new_child = getattr(_module, name)
                _, _Q, scales, qmap = flute.nf_utils.nf_quantize(
                    W=child.weight.to(device="cuda"),
                    num_bits=num_bits,
                    group_size=group_size)
                Q  = flute.utils.pack(
                    _Q.T.contiguous(),
                    num_bits=num_bits,
                    template_ids=[template_id])

                scales = scales.view(new_child.scales.shape)
                scales = scales.to(dtype=new_child.scales.dtype)
                qmap = qmap.to(dtype=new_child.tables.dtype)
                qmap2 = flute.utils.make_qmap2_from_qmap(qmap)

                new_child.weight.copy_(Q)
                new_child.scales.copy_(scales)
                new_child.tables.copy_(qmap)
                new_child.tables2.copy_(qmap2)

            else:
                _replace_linear(child)

    _replace_linear(module)


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
        if dtype not in [torch.float16]:
            raise NotImplementedError
        if bias:
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
        self.workspace = _WORKSPACE

        self.register_buffer("weight", torch.empty((P, K), dtype=torch.int16, device=device))
        self.register_buffer("scales", torch.ones((N, G), dtype=dtype, device=device))
        self.register_buffer("tables", tables)
        self.register_buffer("tables2", flute.utils.make_qmap2_from_qmap(tables))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return flute.qgemm_simple(
            inputs,
            self.weight,
            self.scales,
            self.tables,
            self.tables2,
            self.workspace,
            self.num_bits,
            self.group_size,
        )

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
    fake: bool,
) -> None:
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        device_map="auto",
        torch_dtype="auto")

    if isinstance(model, LlamaForCausalLM):
        prepare_model_flute(
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
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()

    quantize_hf_model(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        save_directory=args.save_directory,
        num_bits=args.num_bits,
        group_size=args.group_size,
        fake=args.fake)
