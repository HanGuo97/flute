import torch
from typing import Tuple
from bitsandbytes.nn import (
    Linear4bit as BNBLinear4bit)
from bitsandbytes.functional import (
    dequantize_4bit,
    dequantize_blockwise)


def convert_BNBLinear4bit(
    bnb_module: BNBLinear4bit,
    verify: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if not isinstance(bnb_module, BNBLinear4bit):
        raise TypeError

    bnb_qweight     = bnb_module.weight
    bnb_quant_state = bnb_module.weight.quant_state
    bnb_quant_table = bnb_module.weight.quant_state.code
    bnb_quant_dtype = bnb_module.weight.quant_state.dtype

    if not all([
        bnb_qweight.ndim == 2,
        bnb_qweight.shape[1] == 1,
        bnb_qweight.bnb_quantized is True,
        bnb_qweight.requires_grad is False,
        bnb_qweight.dtype == torch.uint8,
        bnb_qweight.quant_storage == torch.uint8,
        bnb_qweight.blocksize == bnb_quant_state.blocksize,
        bnb_qweight.quant_type == bnb_quant_state.quant_type,
        bnb_qweight.compress_statistics == bnb_quant_state.nested,
        bnb_module.quant_state is bnb_quant_state]):
        raise NotImplementedError

    # unpacked quantized weights
    qweight = torch.cat([
        (bnb_qweight.data >> 4) & 0b1111,
        (bnb_qweight.data >> 0) & 0b1111], dim=1)
    qweight = qweight.view(
        bnb_quant_state.shape)

    # get the scales
    if bnb_quant_state.nested:
        scales = dequantize_blockwise(
            A=bnb_quant_state.absmax,
            quant_state=bnb_quant_state.state2)
        scales = scales + bnb_quant_state.offset
    else:
        scales = bnb_quant_state.absmax

    # convert to the correct dtype
    if scales.dtype != bnb_quant_dtype:
        scales_casted = scales.to(dtype=bnb_quant_dtype)
    else:
        scales_casted = scales

    if bnb_quant_table.dtype != bnb_quant_dtype:
        bnb_quant_table_casted = bnb_quant_table.to(dtype=bnb_quant_dtype)
    else:
        bnb_quant_table_casted = bnb_quant_table

    if not all([
        scales.ndim == 1,
        scales.dtype == torch.float32,
        scales_casted.dtype == bnb_quant_dtype,
        bnb_quant_table.dtype == torch.float32,
        bnb_quant_table_casted.dtype == bnb_quant_dtype]):
        raise ValueError

    # double check that the conversion is lossless
    if verify is True:
        broadcasted_scales = (
            scales
            .unsqueeze(dim=-1)
            .expand(scales.shape[0], bnb_quant_state.blocksize)
            .reshape(qweight.shape))
        weight = (
            # `dequantize_4bit` function always performs dequantization in FP16
            bnb_quant_table[qweight.to(dtype=torch.int)] *
            broadcasted_scales).to(dtype=bnb_quant_dtype)
        weight_ = dequantize_4bit(
            A=bnb_qweight,
            quant_state=bnb_quant_state,
            # unused
            blocksize=bnb_quant_state.blocksize,
            quant_type=bnb_quant_state.quant_type)
        if not (weight == weight_).all():
            raise ValueError

    return qweight, scales_casted, bnb_quant_table_casted
