import torch
from typing import Tuple, Optional


def linspace(start, stop, num, dtype=torch.float32):
    steps = torch.arange(num, dtype=dtype, device=start.device) / (num - 1)

    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    return start[None] + steps*(stop - start)[None]


def get_values_pivots(bits=4, symmetric=False, dtype=torch.float32):

    dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
    offset = 0.5*(1/32 + 1/30)

    if symmetric:
        v = dist.icdf(linspace(torch.tensor(offset), torch.tensor(1 - offset), 2**(bits), dtype=dtype))
    else:
        v1 = -1 * dist.icdf(linspace(torch.tensor(1-offset), torch.tensor(0.5), 2**(bits-1)))
        v2 = dist.icdf(linspace(torch.tensor(0.5), torch.tensor(1-offset), 2**(bits-1)+1)[1:])
        v = torch.cat((v1, v2))

    v = v / torch.max(torch.abs(v))

    if bits == 4 and not symmetric:
        v = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])

    p = (v[1:] + v[:-1]) / 2
    return v.to(dtype).cuda().clone(), p.to(dtype).cuda().clone()


def manual_nf4(inp, absmax=None, bits=4, blocksize=128, return_stats=False, values=None, pivots=None):
    qx = inp.view(-1, blocksize)
    if absmax == None:
        absmax = torch.max(torch.abs(qx), dim=1, keepdim=True).values

    qx = qx / absmax
    index = torch.searchsorted(pivots, qx)
    dqx = values[index] * absmax

    if return_stats:
        return dqx.view(inp.size()), index.view(inp.size()), absmax.squeeze()
    
    return dqx.view(inp.size())


def nf_quantize(
    W: torch.Tensor,
    num_bits: int,
    group_size: int,
    custom_scales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values, pivots = get_values_pivots(num_bits, False)
    W_dequantized, W_quantized, absmax = manual_nf4(
        W,
        absmax=custom_scales,
        bits=num_bits,
        blocksize=group_size,
        values=values,
        pivots=pivots,
        return_stats=True)

    return (
        W_dequantized,  # fake-quantized weight
        W_quantized,    # quantized weight
        absmax,         # scale
        values,         # lookup table
    )


def nf_quantize_2(
    W: torch.Tensor,
    num_bits: int,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # fake quantization that mimics the kernel behavior
    # primarily used for testing purposes
    values, pivots = get_values_pivots(num_bits, False)
    qx = W.view(-1, group_size)
    absmax = torch.max(torch.abs(qx), dim=1, keepdim=True).values
    qx = qx / absmax
    index = torch.searchsorted(pivots, qx)
    # the main differences are the types
    dqx = values.to(dtype=dtype)[index] * absmax.to(dtype=dtype)
    return dqx.view(W.size())
