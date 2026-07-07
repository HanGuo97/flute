"""Per-shape autotuner for optimized_w3.

Sweeps the kernel's valid (thread_k, thread_n) x sms space for each (M, N, K),
times under do_bench (rep=100), exactness-gates every candidate against an fp64
reference, and rewrites dispatch.py with the winning configs.

    python -m tests.optimized_w3.autotune            # default cells, writes dispatch.py
    python -m tests.optimized_w3.autotune --dry      # print only, don't write

The pack is tile-independent, so each shape is packed once and every config is
timed on the same tensors.
"""
import argparse
import os
import warnings

import torch
import triton.testing as tb

from . import pack_w3, pack_scales, lut_planes
from .qgemm import _ext

warnings.filterwarnings("ignore")
DEV = torch.device("cuda")
GROUP = 128

# valid tiles by M bucket (from the kernel's CALL_IF table): M<=16 has two
# tiles, larger M only the fat-N tile.
_TILES = {16: [(128, 128), (64, 256)], 32: [(64, 256)], 64: [(64, 256)]}
_SMS = [54, 81, 108]

# (N, K) weight shapes to tune; edit for your model.
_CELLS = [(4096, 4096), (14336, 4096), (8192, 8192), (28672, 8192)]
_MS = [16, 32, 64]


def _db(fn):
    return tb.do_bench(fn, rep=100) * 1000.0  # ms -> us


def _sweep_cell(N, K, ms):
    torch.manual_seed(0)
    W = torch.randint(0, 8, (K, N), dtype=torch.int64, device=DEV)
    S = torch.randn((N, K // GROUP), dtype=torch.half, device=DEV) / 10.0
    qmap = torch.randn(8, dtype=torch.half, device=DEV)
    Wdq = (qmap[W] * torch.repeat_interleave(S, GROUP, dim=1).T).double()
    Wp, Sp = pack_w3(W, N, K), pack_scales(S, N, K)
    lo, hi = lut_planes(qmap)
    pw = torch.zeros(N // 128 * 16, dtype=torch.int32, device=DEV)
    ns = torch.cuda.get_device_properties(DEV).multi_processor_count
    out = {}
    for M in ms:
        A = torch.randn((M, K), dtype=torch.half, device=DEV) / 100.0
        ref = A.double() @ Wdq
        mb = 16 if M <= 16 else (32 if M <= 32 else 64)
        best = None
        for (tk, tn) in _TILES[mb]:
            for sm in [s for s in _SMS if s <= ns] + [-1]:
                try:
                    def fn(tk=tk, tn=tn, sm=sm):
                        C = torch.empty((M, N), dtype=torch.half, device=DEV)
                        _ext().mul(A, Wp, C, Sp, lo, hi, pw, tk, tn, sm, 8)
                        return C
                    e = ((fn().double() - ref).norm() / ref.norm()).item()
                    if e > 2e-3:
                        continue
                    t = _db(fn)
                    if best is None or t < best[0]:
                        best = (t, tk, tn, sm)
                except Exception:
                    continue
        if best:
            out[(mb, N, K)] = (best[1], best[2], best[3])
            print(f"  M={M:>2} {N}x{K}: (tk{best[1]},tn{best[2]},sms{best[3]}) "
                  f"{best[0]:.1f}us", flush=True)
    del W, S, Wp, Sp, Wdq
    torch.cuda.empty_cache()
    return out


def _write_dispatch(table, ns):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dispatch.py")
    lines = ['"""Tuned launch configs for optimized_w3, keyed by device SM count.',
             "",
             "    TUNED[sm_count][(M_bucket, N, K)] -> (thread_k, thread_n, sms)",
             "",
             "Regenerate with `python -m tests.optimized_w3.autotune`.",
             '"""', "", "TUNED = {", f"    {ns}: {{"]
    for (mb, N, K), (tk, tn, sm) in sorted(table.items()):
        lines.append(f"        ({mb:>2}, {N:>5}, {K}): ({tk}, {tn}, {sm}),")
    lines += ["    },", "}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="print only, don't write")
    args = ap.parse_args()
    ns = torch.cuda.get_device_properties(DEV).multi_processor_count
    print(f"autotune optimized_w3 on {torch.cuda.get_device_name()} (ns={ns})",
          flush=True)
    table = {}
    for (N, K) in _CELLS:
        table.update(_sweep_cell(N, K, _MS))
    if args.dry:
        print("\nTUNED table (dry run):")
        for k, v in sorted(table.items()):
            print(f"  {k}: {v}")
    else:
        _write_dispatch(table, ns)


if __name__ == "__main__":
    main()
