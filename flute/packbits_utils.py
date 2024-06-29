import math
import torch
import jaxtyping
from typing import Tuple, Union, Optional, NamedTuple

PackedDType = torch.int16
PackedNumBits = torch.iinfo(PackedDType).bits
FloatTensorType = jaxtyping.Float[torch.Tensor, "..."]
UInt8TensorType = jaxtyping.UInt8[torch.Tensor, "..."]
Int16TensorType = jaxtyping.Int16[torch.Tensor, "..."]
Int32TensorType = jaxtyping.Int32[torch.Tensor, "..."]
BinaryTensorType = jaxtyping.Bool[torch.Tensor, "..."]
PackedBinaryTensorType = Union[UInt8TensorType, Int16TensorType, Int32TensorType]


# https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
def to_binary(tensor: UInt8TensorType, num_bits: int, legacy: bool = True) -> BinaryTensorType:
    if tensor.dtype != torch.uint8:
        raise TypeError
    if num_bits > 8:
        raise NotImplementedError

    # Explicit casting, and the following code will
    # raise an Error if casting leads to overflow
    bits_max = torch.tensor(
        2 ** num_bits - 1,
        dtype=torch.uint8,
        device=tensor.device)
    if tensor.max() > bits_max:
        raise OverflowError

    if legacy is True:
        # When using `torch.compile`, the `pow` ops
        # requires floating point numbers, but the
        # `bitwise_and` requires integers.
        mask = 2 ** torch.arange(
            num_bits - 1, -1, -1,
            dtype=torch.float32,
            device=tensor.device)
        mask = mask.to(dtype=torch.uint8)
    else:
        # 1. The above casting is not necessary for PyTorch>=2.1
        # 2. We no longer reverse the bits directions
        mask = 2 ** torch.arange(
            num_bits,
            dtype=torch.uint8,
            device=tensor.device)

    return (
        tensor
        .unsqueeze(dim=-1)
        .bitwise_and(mask)
        .ne(0)
        .bool())


def from_binary(tensor: BinaryTensorType, num_bits: int, legacy: bool = True) -> UInt8TensorType:
    if tensor.dtype != torch.bool:
        raise TypeError
    if tensor.shape[-1] != num_bits:
        raise ValueError
    if num_bits > 8:
        raise NotImplementedError

    if legacy is True:
        mask = 2 ** torch.arange(
            num_bits - 1, -1, -1,
            dtype=torch.float32,
            device=tensor.device)
        mask = mask.to(dtype=torch.uint8)
    else:
        mask = 2 ** torch.arange(
            num_bits,
            dtype=torch.uint8,
            device=tensor.device)

    # This casting is somewhat unnecessary.
    tensor = tensor.to(dtype=torch.uint8)
    output = torch.sum(mask * tensor, dim=-1)
    output = output.to(dtype=torch.uint8)
    return output


def pack_bools_into_integers(
    tensor: BinaryTensorType,
    packed_dtype: torch.dtype,
) -> Tuple[PackedBinaryTensorType, int]:
    if tensor.ndim != 1:
        raise ValueError
    if tensor.dtype != torch.bool:
        raise TypeError
    if packed_dtype not in [torch.uint8, torch.int16, torch.int32]:
        raise NotImplementedError

    # number of bits in the packed dtype
    packed_num_bits = torch.iinfo(packed_dtype).bits

    remainder = (
        tensor.shape[-1] %
        packed_num_bits)
    if remainder > 0:
        padding_length = (
            packed_num_bits -
            remainder)
        padding = tensor.new_zeros(padding_length)
        tensor = torch.cat([tensor, padding], dim=-1)
    else:
        padding_length = 0

    # [-1, packed_num_bits]
    tensor = tensor.view(
        int(tensor.shape[-1] / packed_num_bits),
        packed_num_bits)

    # [1, packed_num_bits]
    bits = torch.arange(
        packed_num_bits,
        dtype=packed_dtype,
        device=tensor.device)
    bits = torch.unsqueeze(bits, dim=0)
    packed_tensor = (tensor << bits)
    packed_tensor = torch.sum(packed_tensor, dim=-1)
    packed_tensor = packed_tensor.to(dtype=packed_dtype)
    return packed_tensor, padding_length


def unpack_integers_into_bools(
    packed_tensor: PackedBinaryTensorType,
    padding_length: int,
    packed_dtype: torch.dtype,
) -> BinaryTensorType:
    if packed_tensor.ndim != 1:
        raise ValueError
    if packed_tensor.dtype != packed_dtype:
        raise TypeError
    if packed_dtype not in [torch.uint8, torch.int16, torch.int32]:
        raise NotImplementedError

    # number of bits in the packed dtype
    packed_num_bits = torch.iinfo(packed_dtype).bits

    # [1, packed_num_bits]
    bits = packed_tensor.new_tensor(
        1,
        dtype=packed_dtype)
    bits = bits << torch.arange(
        packed_num_bits,
        dtype=packed_dtype,
        device=packed_tensor.device)
    bits = torch.unsqueeze(
        bits,
        dim=0)
    unpacked_tensor = torch.unsqueeze(
        packed_tensor,
        dim=-1)
    unpacked_tensor = unpacked_tensor & bits
    if packed_dtype == torch.uint8:
        unpacked_tensor = unpacked_tensor > 0
    elif packed_dtype == torch.int32:
        # For signed integers such as int32, the 31st element is the
        # sign bit, so 0b10000000000000000000000000000000 = -2^31
        # The following line of code can be applied to both settings.
        # However, for legacy reasons, we only apply it to int32.
        unpacked_tensor = unpacked_tensor != 0
    else:
        raise NotImplementedError

    unpacked_tensor = unpacked_tensor.to(dtype=torch.bool)
    unpacked_tensor = unpacked_tensor.view(-1)
    if padding_length > 0:
        unpacked_tensor = unpacked_tensor[:-padding_length]
    return unpacked_tensor


def pack_integer_tensors(
    tensor: UInt8TensorType,
    num_bits: int,
) -> PackedBinaryTensorType:
    # Two major differences for faster dequantization
    # 1. `reverse=False`
    # 2. `packed_dtype=torch.int32`
    # 3. special implementation for `num_bits=3`
    # 4. does not support padding

    # [*tensor.shape, num_bits]
    binary_tensor = to_binary(
        tensor=tensor,
        num_bits=num_bits,
        legacy=False)

    if num_bits == 3:
        raise NotImplementedError

    # [tensor.numel() x num_bits]
    binary_tensor = binary_tensor.view(
        tensor.numel() * num_bits)
    binary_tensor = binary_tensor.contiguous()
    # [tensor.numel() x num_bits / 32]
    packed_tensor, padding_length = pack_bools_into_integers(
        tensor=binary_tensor,
        packed_dtype=PackedDType)
    if padding_length != 0:
        raise ValueError
    return packed_tensor
