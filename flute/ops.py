import torch


def _qgemm_simple_abstract(
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    table: torch.Tensor,
    table2: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    group_size: int,
) -> torch.Tensor:
    M = input.shape[0]
    N = scales.shape[0]
    return torch.empty(
        (M, N),
        dtype=input.dtype,
        device=input.device)


@torch.library.impl_abstract("flute::qgemm_simple_80")
def _qgemm_simple_80_abstract(
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    table: torch.Tensor,
    table2: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    group_size: int,
) -> torch.Tensor:
    return _qgemm_simple_abstract(
        input=input,
        weight=weight,
        scales=scales,
        table=table,
        table2=table2,
        workspace=workspace,
        num_bits=num_bits,
        group_size=group_size,
    )


@torch.library.impl_abstract("flute::qgemm_simple_86")
def _qgemm_simple_86_abstract(
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    table: torch.Tensor,
    table2: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    group_size: int,
) -> torch.Tensor:
    return _qgemm_simple_abstract(
        input=input,
        weight=weight,
        scales=scales,
        table=table,
        table2=table2,
        workspace=workspace,
        num_bits=num_bits,
        group_size=group_size,
    )


@torch.library.impl_abstract("flute::qgemm_simple_89")
def _qgemm_simple_89_abstract(
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    table: torch.Tensor,
    table2: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    group_size: int,
) -> torch.Tensor:
    return _qgemm_simple_abstract(
        input=input,
        weight=weight,
        scales=scales,
        table=table,
        table2=table2,
        workspace=workspace,
        num_bits=num_bits,
        group_size=group_size,
    )


@torch.library.impl_abstract("flute::qgemm_raw_simple_80")
def _qgemm_raw_simple_80_abstract(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    scales: torch.Tensor,
    table: torch.Tensor,
    table2: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    group_size: int,
    template_id: int,
) -> None:
    pass


@torch.library.impl_abstract("flute::qgemm_raw_simple_86")
def _qgemm_raw_simple_86_abstract(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    scales: torch.Tensor,
    table: torch.Tensor,
    table2: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    group_size: int,
    template_id: int,
) -> None:
    pass


@torch.library.impl_abstract("flute::qgemm_raw_simple_89")
def _qgemm_raw_simple_89_abstract(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    scales: torch.Tensor,
    table: torch.Tensor,
    table2: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    group_size: int,
    template_id: int,
) -> None:
    pass
