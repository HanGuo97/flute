import os
import click
import torch
import triton.testing as triton_benchmark
from collections import defaultdict
from typing import Tuple, Dict, List, Optional, NamedTuple

import flute
import flute.utils

_TEMPLATES = {}


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
                template_id=template_id) is False:
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
            dtype)
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


def tune(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    tables: torch.Tensor,
    tables2: torch.Tensor,
    num_bits: int,
    group_size: int,
    num_seeds: int = 3,
) -> int:
    M = inputs.shape[0]  # [M, K]
    N = scales.shape[0]  # [N, G]
    K = weight.shape[1]  # [P, K]
    dtype = inputs.dtype
    device = inputs.device
    num_sms = flute.utils.get_device_num_sms(device)
    return _tune(
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
