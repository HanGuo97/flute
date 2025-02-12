import os
import click
import torch
import warnings
import triton.testing as triton_benchmark
from collections import defaultdict
from typing import Tuple, Dict, List, Optional, NamedTuple

import flute
import flute.utils

_TEMPLATES = {}
FP16_ERROR_THRESHOLD = 2.0e-3
BF16_ERROR_THRESHOLD = 1.1e-2


def prepare_flute_data(
    m: int,
    n: int,
    k: int,
    num_bits: int,
    group_size: int,
    dtype: torch.dtype,
    device: torch.device,
    bank_conflict_free: bool,
    template_id: int,
    num_sms: int,
) -> Dict:
    if n % 16 != 0:
        raise ValueError
    if k % group_size != 0:
        raise ValueError
    p = int(n / 16 * num_bits)
    g = int(k / group_size)

    Q = torch.randint(
        low=-2 ** 15,
        high=2 ** 15,
        size=(k, p),
        dtype=torch.int16,
        device=device)

    if bank_conflict_free is True:
        Q = torch.zeros_like(Q)

    A = torch.randn(
        (m, k),
        dtype=dtype,
        device=device)

    S = torch.randn(
        (g, n),
        dtype=dtype,
        device=device)

    D = torch.empty(
        (m, n),
        dtype=dtype,
        device=device)

    qmap = torch.arange(
        2 ** (num_bits),
        dtype=dtype,
        device=device)

    torch.cuda.synchronize()
    return {
        "A": A,
        "Q": Q.T.contiguous(),  # column-major layout
        "D": D,
        "S": S.T.contiguous(),  # column-major layout
        "qmap": qmap,
        "qmap2": flute.utils.make_qmap2_from_qmap(qmap),
        "workspace": flute.utils.make_workspace_streamk(device),
        "num_bits": num_bits,
        "group_size": group_size,
        "template_id": template_id,
        "num_sms": num_sms,
    }


def benchmark_flute(data: Dict, legacy: bool, n: int = 100) -> Dict[str, float]:
    if legacy is True:
        fn = lambda: flute.qgemm_raw_simple(
            data["A"],
            data["Q"],
            data["D"],
            data["S"],
            data["qmap"],
            data["qmap2"],
            data["workspace"],
            data["num_bits"],
            data["group_size"],
            data["template_id"])
    else:
        fn = lambda: flute.qgemm(
            data["A"],
            data["Q"],
            data["S"],
            data["qmap"],
            data["qmap2"],
            data["workspace"],
            data["num_bits"],
            data["group_size"],
            data["template_id"],
            data["num_sms"])

    triton_time = triton_benchmark.do_bench(fn, rep=n)
    return {"triton_time": triton_time}


@torch.no_grad()
def run_benchmark(
    M: int,
    N: int,
    K: int,
    num_bits: int,
    group_size: int,
    num_sms: int,
    dtype: torch.dtype,
    device: torch.device,
    seeds: List[int],
    legacy: bool,
) -> List[Dict]:

    results = []
    for seed in seeds:
        torch.manual_seed(seed)

        for template_id in flute.utils.get_template_ids(num_bits):

            if flute.utils.is_template_supported(
                M=M,
                N=N,
                K=K,
                num_bits=num_bits,
                template_id=template_id,
                num_sms=num_sms) is False:
                # print(f"{m} {n} {k} does not support {template_id}")
                continue

            flute_data = prepare_flute_data(
                m=M,
                n=N,
                k=K,
                num_bits=num_bits,
                group_size=group_size,
                dtype=dtype,
                device=device,
                bank_conflict_free=False,
                template_id=template_id,
                num_sms=num_sms)

            try:
                results.append({
                    "seed": seed,
                    "template_id": template_id,
                    **benchmark_flute(flute_data, legacy=legacy),
                })
            except RuntimeError as e:
                if str(e).startswith("CUDA error: invalid argument"):
                    # print(f"skipping {m} {n} {k} {num_bits} {template_id}")
                    continue
                if str(e).startswith("Unsupported template_id value"):
                    continue
                else:
                    raise e
            del flute_data

    return results


def get_template_key(
    M: int,
    N: int,
    K: int,
    num_bits: int,
    group_size: int,
    num_sms: int,
    dtype: torch.dtype,
    legacy: bool,
) -> Tuple:
    if legacy is True:
        return (
            num_sms,
            num_bits,
            group_size,
            M,
            N,
            K,
            str(dtype))
    else:
        # M < 16 dispatches to the same template
        return (
            "v1",
            max(M, 16),
            N,
            K,
            num_bits,
            group_size,
            num_sms,
            dtype)


def _tune(
    M: int,
    N: int,
    K: int,
    num_bits: int,
    group_size: int,
    num_sms: int,
    dtype: torch.dtype,
    device: torch.device,
    num_seeds: int,
    legacy: bool,
) -> int:

    template_key = get_template_key(
        M=M,
        N=N,
        K=K,
        num_bits=num_bits,
        group_size=group_size,
        num_sms=num_sms,
        dtype=dtype,
        legacy=legacy)

    if template_key in _TEMPLATES.keys():
        return _TEMPLATES[template_key]

    results = run_benchmark(
        M=M,
        N=N,
        K=K,
        num_bits=num_bits,
        group_size=group_size,
        num_sms=num_sms,
        dtype=dtype,
        device=device,
        seeds=list(range(num_seeds)),
        legacy=legacy)

    average_times = {}
    average_times_list = defaultdict(list)
    for result in results:
        time = result["triton_time"]
        template_id = result["template_id"]
        average_times_list[template_id].append(time)

    for template_id, times in average_times_list.items():
        average_times[template_id] = sum(times) / len(times)

    best_template_id = min(
        average_times.keys(),
        key=lambda tid: average_times[tid])
    _TEMPLATES[template_key] = best_template_id
    return best_template_id


class TuneMetaData(NamedTuple):
    M: int
    N: int
    K: int
    num_bits: int
    group_size: int
    num_sms: int
    dtype: torch.dtype
    device: torch.device
    template_id: int

    def to_dict(self) -> Dict:
        data = self._asdict()
        data["dtype"] = str(data["dtype"])
        data["device"] = str(data["device"])
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "TuneMetaData":
        dtype_name = data.get("dtype")
        if dtype_name == "torch.float32":
            data["dtype"] = torch.float32
        elif dtype_name == "torch.float16":
            data["dtype"] = torch.float16
        elif dtype_name == "torch.bfloat16":
            data["dtype"] = torch.bfloat16
        else:
            raise ValueError(f"Invalid dtype {dtype_name}")

        data["device"] = torch.device(data["device"])
        return cls(**data)


# this 2/4
@torch.no_grad()
def check(
    weight: torch.Tensor,
    weight_packed: torch.Tensor,
    metadata: TuneMetaData,
    uniform: bool,
    identity: bool,
) -> None:
    if identity is True:
        inputs = torch.eye(
            metadata.K,
            dtype=metadata.dtype,
            device=metadata.device)
    else:
        inputs = torch.randn(
            (metadata.M, metadata.K),
            dtype=metadata.dtype,
            device=metadata.device) / 100.

    scales = torch.randn(
        (metadata.N, int(metadata.K / metadata.group_size)),
        dtype=metadata.dtype,
        device=metadata.device)

    if uniform is True:
        tables = torch.arange(
            2 ** metadata.num_bits,
            dtype=metadata.dtype,
            device=metadata.device)
    else:
        tables = torch.randn(
            2 ** metadata.num_bits,
            dtype=metadata.dtype,
            device=metadata.device)

    tables2 = flute.utils.make_qmap2_from_qmap(tables)
    workspace = flute.utils.make_workspace_streamk(device=metadata.device)

    # ground truth
    weight_ = tables[flute.utils.safe_cast(weight, dtype=torch.int64)]
    scales_ = torch.repeat_interleave(scales, metadata.group_size, dim=1).T
    output_ = torch.mm(inputs, weight_ * scales_)

    qgemm_args = (
        inputs,
        weight_packed,
        scales,
        tables,
        tables2,
        workspace,
        metadata.num_bits,
        metadata.group_size,
        metadata.template_id,
        metadata.num_sms)
    output = flute.qgemm(*qgemm_args)

    if identity is True:
        torch.library.opcheck(flute.qgemm, qgemm_args)
    else:
        # `qgemm` is actually non-deterministic due to
        # how we implemented Stream-K reduction, so the
        # `opcheck` that asserts output equality with and
        # without compile will not pass
        opcheck_utils = tuple(
            c for c in torch.library._OPCHECK_DEFAULT_UTILS
            if c not in ["test_aot_dispatch_dynamic"])
        torch.library.opcheck(flute.qgemm, qgemm_args, test_utils=opcheck_utils)

    equal = (output_ == output).all().item()
    error  = ((output_ - output).norm() / output .norm()).item()
    error_ = ((output_ - output).norm() / output_.norm()).item()
    message = (
        f"WARNING: "
        f"M={str(metadata.M):<5} "
        f"N={str(metadata.N):<5} "
        f"K={str(metadata.K):<5} "
        f"num_bits={str(metadata.num_bits):<5} "
        f"group_size={str(metadata.group_size):<5} "
        f"dtype={str(metadata.dtype):<15} "
        f"uniform={str(uniform):<5} "
        f"error={error:.3e} "
        f"error_={error_:.3e}")

    if identity is True:
        if equal is not True:
            click.secho(message, bg="red")
    else:
        if metadata.dtype == torch.float16:
            threshold = FP16_ERROR_THRESHOLD
        elif metadata.dtype == torch.bfloat16:
            threshold = BF16_ERROR_THRESHOLD
        else:
            raise NotImplementedError

        if error > threshold or error_ > threshold:
            click.secho(message, fg="red")
        if not (error < threshold and error_ < threshold):
            # corner cases when error is NaN
            click.secho(message, fg="red")


def tune_and_pack(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    num_bits: int,
    group_size: int,
    num_seeds: int = 3,
    check_correctness: bool = True,
    check_num_seeds: int = 3,
) -> Tuple[torch.Tensor, TuneMetaData]:
    if inputs.ndim != 2:
        raise ValueError
    if weight.ndim != 2:
        raise ValueError
    if inputs.shape[1] != weight.shape[0]:
        raise ValueError
    M = inputs.shape[0]
    K, N = weight.shape
    dtype = inputs.dtype
    device = inputs.device
    num_sms = flute.utils.get_device_num_sms(device)

    template_id = _tune(
        M=M,
        N=N,
        K=K,
        num_bits=num_bits,
        group_size=group_size,
        num_sms=num_sms,
        dtype=dtype,
        device=device,
        num_seeds=num_seeds,
        legacy=False)

    weight_packed = flute.utils.pack(
        W=weight,
        num_bits=num_bits,
        template_ids=[template_id],
        num_sms=num_sms)

    metadata = TuneMetaData(
        M=M,
        N=N,
        K=K,
        num_bits=num_bits,
        group_size=group_size,
        num_sms=num_sms,
        dtype=dtype,
        device=device,
        template_id=template_id)

    if check_correctness is True:
        # sometimes the `weight` passed in can be on CPU, this is fine
        # since we don't really use `weight` during tuning, but for
        # checking we need to make sure they are on the proper device
        weight = weight.to(device=device)
        weight_packed = weight_packed.to(device=device)

        for uniform in [True, False]:
            for identity in [True, False]:
                for seed in range(check_num_seeds):
                    torch.manual_seed(seed)
                    check(
                        weight=weight,
                        weight_packed=weight_packed,
                        metadata=metadata,
                        uniform=uniform,
                        identity=identity)

    return weight_packed, metadata


class TuneTask(NamedTuple):
    M: int
    N: int
    K: int
    num_bits: int
    group_size: int
    num_sms: int
    dtype: torch.dtype
    device: torch.device


def tune_tasks_legacy(tasks: List[TuneTask], num_seeds: int = 3) -> None:
    for task in tasks:
        _tune(
            M=task.M,
            N=task.N,
            K=task.K,
            num_bits=task.num_bits,
            group_size=task.group_size,
            num_sms=task.num_sms,
            dtype=task.dtype,
            device=task.device,
            num_seeds=num_seeds,
            legacy=True)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(file_dir, "../flute/data/qgemm_kernel_raw_tuned_configs.pth")
    torch.save(_TEMPLATES, save_path)
    click.secho(f"Saved to {save_path}", fg="green")


def qgemm_v2(
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    table: torch.Tensor,
    table2: torch.Tensor,
    workspace: torch.Tensor,
    metadata: TuneMetaData,
    hadamard_size: Optional[int] = None,
) -> torch.Tensor:
    if hadamard_size is None:
        return flute.qgemm(
            input=input,
            weight=weight,
            scales=scales,
            table=table,
            table2=table2,
            workspace=workspace,
            num_bits=metadata.num_bits,
            group_size=metadata.group_size,
            template_id=metadata.template_id,
            num_sms=metadata.num_sms)
    else:
        return flute.qgemm_hadamard(
            input=input,
            weight=weight,
            scales=scales,
            table=table,
            table2=table2,
            workspace=workspace,
            num_bits=metadata.num_bits,
            group_size=metadata.group_size,
            hadamard_size=hadamard_size,
            template_id=metadata.template_id,
            num_sms=metadata.num_sms)


def maybe_tune_and_repack(
    weight: torch.Tensor,
    scales: torch.Tensor,
    metadata: TuneMetaData,
    example_batch_size: Optional[int] = None,
) -> Tuple[torch.Tensor, TuneMetaData]:

    if weight.device.type != "cuda":
        device = torch.device("cuda")
        warnings.warn(f"[FLUTE]: Moving data from {weight.device} to {device}.")
    else:
        device = weight.device

    if example_batch_size is None:
        example_batch_size = 1
        # warnings.warn(f"[FLUTE]: `example_batch_size` is not set, using {example_batch_size}.")

    num_sms = flute.utils.get_device_num_sms(device)
    if (metadata.M == example_batch_size) and (metadata.num_sms == num_sms):
        return weight, metadata

    warnings.warn(f"[FLUTE]: Tuning and repacking with "
                  f"batch size ({example_batch_size}) and "
                  f"metadata ({metadata._asdict()}).")

    # reconstruct the unpacked tensor
    Q_unpacked = flute.utils.unpack(
        weight=weight.to(device=device),
        scales=scales.to(device=device),
        workspace=flute.utils.get_workspace_streamk(device),
        num_bits=metadata.num_bits,
        group_size=metadata.group_size,
        template_id_packed=metadata.template_id,
        num_sms_packed=metadata.num_sms)

    # re-pack the tensors
    example_inputs = torch.randn(
        example_batch_size,
        metadata.K,
        dtype=scales.dtype,
        device=device)
    weight_repacked, tune_metadata = tune_and_pack(
        inputs=example_inputs,
        weight=Q_unpacked.T.contiguous().to(device="cpu"),
        num_bits=metadata.num_bits,
        group_size=metadata.group_size)
    weight_repacked = weight_repacked.to(device=weight.device)

    if not all([
        not isinstance(weight, torch.nn.Parameter),
        weight.requires_grad is False,
        weight_repacked.requires_grad is False,
        weight_repacked.shape == weight.shape,
        weight_repacked.dtype == weight.dtype,
        weight_repacked.device == weight.device]):
        raise ValueError

    return weight_repacked, tune_metadata
