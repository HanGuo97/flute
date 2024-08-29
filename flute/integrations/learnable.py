import math
import torch

from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from typing import Optional

import flute
import flute.nf_utils

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
    samples: int = 128,
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
    seq_len = samples * (max_length - 1)

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