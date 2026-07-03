"""Correctness tests for the W3 dual-arm kernel (tests/mma_w3_lut.cu,
tests/w3_smem_lut.cu), mirroring tests/kernel.py.

For every LLaMA-3 8B/70B shape in tests/shapes.py and both lookup arms:
the dispatched config and the AUTO fallback are checked against an fp64
reference with the exactness criterion of tests/mma_w3_lut.py::error_stats
(every element within 1 fp16 ULP or < noise absolute, zero violations),
and the two arms are checked bit-identical against each other.

  python -m tests.w3.w3_kernel            # full sweep
  python -m tests.w3.w3_kernel --quick    # (4096, 4096) only
"""
import argparse
import math

import click
import torch

from tests.w3.mma_w3_lut import arms, best_config, pack_w3, lut_planes, \
    error_stats, ABS_NOISE_BIG, GROUP_SIZE, NUM_BITS, DTYPE, DEVICE
from tests.shapes import LLAMA3_8B_SHAPES, LLAMA3_70B_SHAPES

DECODE_MS = [1, 3, 16]
BIG_MS = [64, 256]                  # big-kernel path; run on the small shape


def test_w3(N: int, K: int, ms, seed: int = 0) -> bool:
    torch.manual_seed(seed)
    W = torch.randint(0, 2 ** NUM_BITS, (K, N), dtype=torch.int64,
                      device=DEVICE)
    S = torch.randn((N, K // GROUP_SIZE), dtype=DTYPE, device=DEVICE) / 10.0
    qmap = torch.randn(2 ** NUM_BITS, dtype=DTYPE, device=DEVICE)
    Wdq64 = (qmap[W] *
             torch.repeat_interleave(S, GROUP_SIZE, dim=1).T).double()
    Qw = pack_w3(W)
    lo, hi = lut_planes(qmap)
    del W
    ok = True
    for M in ms:
        A = torch.randn(M, K, dtype=DTYPE, device=DEVICE) / 100.0
        ref = A.double() @ Wdq64
        # the absolute-noise floor is HMMA accumulator noise on near-zero
        # elements; it grows ~sqrt(serial chain length), i.e. ~sqrt(K)
        # (measured: K=28672 violators sit at <= 1.9e-6 vs 2.65e-6 bound)
        noise = (ABS_NOISE_BIG if M > 64 else 1e-6) * math.sqrt(K / 4096)
        outs = {}
        configs = {"dispatched": best_config(M, N, K),
                   "auto": ("prmt", 0, 0, 0, 0)}
        for label, (arm, pm, sw, nt, yt) in configs.items():
            out = arms()[arm].mma_gemm(A, Qw, S, lo, hi, GROUP_SIZE,
                                       pm, sw, nt, yt)
            s = error_stats(out, ref, noise=noise)
            good = s["rel"] < 1e-3 and s["viol"] == 0
            ok &= good
            if not good:
                click.secho(f"FAIL N={N} K={K} M={M} {label} ({arm}): "
                            f"rel {s['rel']:.2e} viol {s['viol']}", fg="red")
        # the two lookup arms must agree bit-for-bit (AUTO knobs each)
        for arm, ext in arms().items():
            outs[arm] = ext.mma_gemm(A, Qw, S, lo, hi, GROUP_SIZE,
                                     0, 0, 0, 0)
        if not torch.equal(outs["prmt"], outs["smem"]):
            ok = False
            click.secho(f"FAIL N={N} K={K} M={M}: arms differ bitwise",
                        fg="red")
    status = "OK" if ok else "FAIL"
    print(f"N={N:>6} K={K:>6} Ms={ms}: {status}", flush=True)
    return ok


def run_tests(quick: bool = False) -> bool:
    ok = test_w3(4096, 4096, DECODE_MS + BIG_MS)
    if quick:
        return ok
    for (N, K) in LLAMA3_8B_SHAPES + LLAMA3_70B_SHAPES:
        if (N, K) == (4096, 4096):
            continue
        ok &= test_w3(N, K, DECODE_MS)
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if run_tests(quick=args.quick):
        click.secho("all W3 correctness tests passed", fg="green")
    else:
        raise SystemExit(1)
