import torch


def _qgemm_simple_abstract(
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    tables: torch.Tensor,
    tables2: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    group_size: int,
) -> torch.Tensor:
    if not all([
        input.ndim >= 2,
        weight.ndim == 2,
        scales.ndim == 2,
        tables.ndim == 1,
        tables2.ndim == 3,
        workspace.ndim == 1,
    ]):
        raise ValueError

    dtype = input.dtype
    if dtype not in [torch.float16, torch.bfloat16]:
        raise TypeError

    if not all([
        weight.dtype == torch.int16,
        scales.dtype == dtype,
        tables.dtype == dtype,
        tables2.dtype == torch.float32,
        workspace.dtype == torch.uint8,
    ]):
        raise TypeError

    if not all([
        weight.shape[1] == input.shape[-1],  # K
        weight.shape[1] == scales.shape[1] * group_size,  # K
        weight.shape[0] == int(num_bits * (scales.shape[0] / 16)),  # P
        tables.shape[0] == 2 ** num_bits,
        tables2.shape[0] == 2 ** num_bits,
        tables2.shape[1] == 2 ** num_bits,
        tables2.shape[2] == 1,
    ]):
        raise ValueError

    N = scales.shape[0]
    return torch.empty(
        input.shape[:-1] + (N,),
        dtype=input.dtype,
        device=input.device)


@torch.library.impl_abstract("flute::qgemm_simple_80")
def _qgemm_simple_80_abstract(
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    tables: torch.Tensor,
    tables2: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    group_size: int,
) -> torch.Tensor:
    return _qgemm_simple_abstract(
        input=input,
        weight=weight,
        scales=scales,
        tables=tables,
        tables2=tables2,
        workspace=workspace,
        num_bits=num_bits,
        group_size=group_size,
    )


@torch.library.impl_abstract("flute::qgemm_simple_86")
def _qgemm_simple_86_abstract(
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    tables: torch.Tensor,
    tables2: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    group_size: int,
) -> torch.Tensor:
    return _qgemm_simple_abstract(
        input=input,
        weight=weight,
        scales=scales,
        tables=tables,
        tables2=tables2,
        workspace=workspace,
        num_bits=num_bits,
        group_size=group_size,
    )


@torch.library.impl_abstract("flute::qgemm_simple_89")
def _qgemm_simple_89_abstract(
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    tables: torch.Tensor,
    tables2: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    group_size: int,
) -> torch.Tensor:
    return _qgemm_simple_abstract(
        input=input,
        weight=weight,
        scales=scales,
        tables=tables,
        tables2=tables2,
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
    tables: torch.Tensor,
    tables2: torch.Tensor,
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
    tables: torch.Tensor,
    tables2: torch.Tensor,
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
    tables: torch.Tensor,
    tables2: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    group_size: int,
    template_id: int,
) -> None:
    pass
