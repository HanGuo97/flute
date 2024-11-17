import torch
import flute.utils
from typing import Tuple


def prepare_data(
    weight_original: torch.Tensor,  # [dim0 / vector_size, dim1]
    scales_original: torch.Tensor,  # [dim0 / group_size , dim1]
    grid: torch.Tensor,             # [2 ** (num_bits * vector_size), vector_size]
    num_bits: int,
    group_size: int,
    vector_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converting the HIGGS data to FLUTE format."""
    dim0 = int(weight_original.shape[0] * vector_size)
    dim1 = int(weight_original.shape[1])
    if weight_original.ndim != 2:
        raise ValueError
    if scales_original.ndim != 2:
        raise ValueError
    if grid.ndim != 2:
        raise ValueError
    if scales_original.shape[0] != int(dim0 / group_size):
        raise ValueError
    if scales_original.shape[1] != dim1:
        raise ValueError
    if grid.shape[0] != int(2 ** (num_bits * vector_size)):
        raise ValueError
    if grid.shape[1] != vector_size:
        raise ValueError
    if weight_original.dtype != torch.uint8:
        raise TypeError
    if scales_original.dtype != dtype:
        raise TypeError
    if grid.dtype != dtype:
        raise TypeError
    if weight_original.is_contiguous() is False:
        raise ValueError
    if scales_original.is_contiguous() is False:
        raise ValueError
    if grid.is_contiguous() is False:
        raise ValueError

    if vector_size == 2:
        if num_bits == 4:
            bit_mask = 0xF
        elif num_bits == 3:
            bit_mask = 0x7
        elif num_bits == 2:
            bit_mask = 0x3
        else:
            raise NotImplementedError

        # [dim0 / vector_size, vector_size, dim1]
        W = torch.stack([
            (weight_original >> num_bits) & bit_mask,
            (weight_original >>        0) & bit_mask],
            dim=1)
        W = W.view(dim0, dim1)

        qmap_size = int(2 ** num_bits)
        qmap = torch.arange(qmap_size, dtype=dtype, device=device)  # unused
        qmap2 = grid.view(qmap_size, qmap_size, vector_size)
        qmap2 = qmap2.view(dtype=torch.float32)
        qmap2 = qmap2.contiguous()

    elif vector_size == 1:
        W = weight_original
        qmap = grid.squeeze(dim=-1)
        qmap2 = flute.utils.make_qmap2_from_qmap(qmap)

    else:
        raise NotImplementedError

    Q = flute.utils.pack(
        W.contiguous(),
        num_bits=num_bits,
        group_size=group_size)
    S = scales_original.T.contiguous()
    return Q, S, qmap, qmap2


def prepare_data_transposed(
    weight_original: torch.Tensor,  # [dim0, dim1 / vector_size]
    scales_original: torch.Tensor,  # [dim0, dim1 / group_size]
    grid: torch.Tensor,             # [2 ** (num_bits * vector_size), vector_size]
    num_bits: int,
    group_size: int,
    vector_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight_original.ndim != 2:
        raise ValueError
    if scales_original.ndim != 2:
        raise ValueError
    return prepare_data(
        weight_original=weight_original.T.contiguous(),
        scales_original=scales_original.T.contiguous(),
        grid=grid,
        num_bits=num_bits,
        group_size=group_size,
        vector_size=vector_size,
        dtype=dtype,
        device=device)
