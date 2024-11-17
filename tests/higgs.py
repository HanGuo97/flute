import torch
import flute
import flute.utils
import flute.integrations.higgs


def vector_dequantize_higgs(
    weight_higgs: torch.Tensor, # [out_size, in_size / higgs_d]
    scales_higgs: torch.Tensor, # [out_size, in_size / group_size]
    grid: torch.Tensor, # [higgs_n, higgs_d]
) -> torch.Tensor:
    group_size = weight_higgs.shape[1] * grid.shape[1] // scales_higgs.shape[1]
    weight = grid[weight_higgs]  # [out_size, in_size//higgs_d, higgs_d]
    weight = weight.reshape(weight.shape[0], -1, group_size)  # [out_size, in_size//group_size, group_size]
    weight = weight * scales_higgs[..., None]  # [out_size, in_size//group_size, group_size]
    weight = weight.reshape(weight.shape[0], -1)  # [out_size, in_size]
    return weight


def vector_dequantize(
    weight_higgs: torch.Tensor,
    scales_higgs: torch.Tensor,
    grid: torch.Tensor,
    num_bits: int,
    group_size: int,
    vector_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    Q, S, tables, tables2 = flute.integrations.higgs.prepare_data_transposed(
        weight_original=weight_higgs,
        scales_original=scales_higgs,
        grid=grid,
        num_bits=num_bits,
        group_size=group_size,
        vector_size=vector_size,
        dtype=dtype,
        device=device)

    I = torch.eye(
        scales_higgs.shape[0],
        dtype=dtype,
        device=device)
    workspace = flute.utils.make_workspace_streamk(device=device)
    return flute.qgemm_simple(
        I,
        Q,
        S,
        tables,
        tables2,
        workspace,
        num_bits,
        group_size)


def test_vector_dequantize() -> None:
    N = 4096
    K = 4096
    group_size = 64
    device = torch.device("cuda")
    for num_bits in [4, 3, 2]:
        for vector_size in [2, 1]:
            for dtype in [torch.float16, torch.bfloat16]:
                num_codes = 2 ** (num_bits * vector_size)
                num_groups = int(K / group_size)
                num_vectors = int(K / vector_size)

                weight_higgs = torch.randint(
                    0, num_codes,
                    (N, num_vectors),
                    dtype=torch.uint8,
                    device=device)
                scales_higgs = torch.randn(
                    (N, num_groups),
                    dtype=dtype,
                    device=device)
                grid = torch.randn(
                    (num_codes, vector_size),
                    dtype=dtype,
                    device=device)

                outputs = vector_dequantize(
                    weight_higgs=weight_higgs,
                    scales_higgs=scales_higgs,
                    grid=grid,
                    num_bits=num_bits,
                    group_size=group_size,
                    vector_size=vector_size,
                    dtype=dtype,
                    device=device)

                outputs_higgs = vector_dequantize_higgs(
                    weight_higgs=weight_higgs.int(),
                    scales_higgs=scales_higgs,
                    grid=grid)

                if not (outputs == outputs_higgs.T).all():
                    raise ValueError
