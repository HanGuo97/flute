"""Marlin-protocol bench: warm-L2, wall-clock, no cache control.

`benchmark()` below is VERBATIM from IST-DASLab/marlin bench.py
(https://github.com/IST-DASLab/marlin, Apache-2.0): 1 warmup + 10 timed
back-to-back launches on the SAME tensors, wall clock with sync only at the
ends, 1 s cool-down between benchmarks.  This is the warm-cache extreme of
the published protocols (triton do_bench = dirty-flush extreme;
clean_overwrite_bench = the serving state between them): any weight tensor
that fits in L2 is served from cache after the warmup call.  Marlin's README additionally
locks GPU clocks to base (`sudo nvidia-smi --lock-gpu-clocks`) -- run that
first where you have permission; it is not done here.

Batch sizes are Marlin's list capped at 32.  Every (shape, M) cell is
exactness-gated against an fp64 reference before its timing is trusted.

  python -m tests.w3.w3_marlin_bench
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch

import flute
import flute.utils
from tests.w3.mma_w3_lut import arms, best_config, pack_w3, lut_planes, \
    GROUP_SIZE, NUM_BITS, DTYPE, DEVICE

FLUTE_TIDS = (4, 5, 6, 7, 8, 9, 10, 11,          # every W3 template whose
              16, 17, 18, 19, 20, 21, 22, 23,    # packing layout exists
              28, 29, 30, 31, 32, 33, 34, 35)    # (TileP=32); best per cell
BATCHES = [1, 2, 4, 8, 16, 32]   # marlin bench.py batchsizes, capped at 32
SHAPES = [                       # (N, K), all from FLUTE's tests/shapes.py
    (4096, 4096),                # LLaMA-8B q/o proj
    (6144, 4096),                # LLaMA-8B fused qkv
    (14336, 4096),               # LLaMA-8B gate/up proj
    (4096, 14336),               # LLaMA-8B down proj
    (8192, 8192),                # LLaMA-70B attention
    (28672, 8192),               # LLaMA-70B gate/up proj
]


def benchmark(f, warmup=1, iter=10):
    # verbatim from IST-DASLab/marlin bench.py
    for i in range(warmup + iter):
        f()
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    time.sleep(1.)
    return res


def main():
    num_sms = flute.utils.get_device_num_sms(DEVICE)
    ws = flute.utils.make_workspace_streamk(device=DEVICE)
    l2_mb = torch.cuda.get_device_properties(DEVICE).L2_cache_size / 1e6
    print(f"marlin protocol on {torch.cuda.get_device_name()} "
          f"(L2 {l2_mb:.0f}MB); warm-L2 caveat applies to streams under L2")
    for (N, K) in SHAPES:
        torch.manual_seed(0)
        W = torch.randint(0, 2 ** NUM_BITS, (K, N), dtype=torch.int64,
                          device=DEVICE)
        S = torch.randn((N, K // GROUP_SIZE), dtype=DTYPE,
                        device=DEVICE) / 10.0
        qmap = torch.randn(2 ** NUM_BITS, dtype=DTYPE, device=DEVICE)
        Wdq = qmap[W] * torch.repeat_interleave(S, GROUP_SIZE, dim=1).T
        Qw = pack_w3(W)
        lo, hi = lut_planes(qmap)
        qmap2 = flute.utils.make_qmap2_from_qmap(qmap)
        Wu = W.to(torch.uint8).int()
        fq = {tid: flute.utils.pack(W=Wu, num_bits=NUM_BITS,
                                    template_ids=[tid], num_sms=num_sms)
              for tid in FLUTE_TIDS}
        Wdq64 = Wdq.double()
        del W, Wu
        qmb = N * K * 3 / 8 / 1e6
        print(f"\nN={N} K={K}  (W3 {qmb:.0f}MB / dense {N*K*2/1e6:.0f}MB)")
        print(f"{'M':>4} {'ours':>9} {'cfg':>20} {'flute':>9} {'tid':>4} "
              f"{'dense':>9} {'vs flute':>9} {'vs dense':>9}")
        for M in BATCHES:
            A = torch.randn(M, K, dtype=DTYPE, device=DEVICE) / 100.0
            arm, pm, sw, nt, yt = best_config(M, N, K)
            ext = arms()[arm]
            out = ext.mma_gemm(A, Qw, S, lo, hi, GROUP_SIZE, pm, sw, nt, yt)
            ref = A.double() @ Wdq64
            rel = ((out.double() - ref).norm() / ref.norm()).item()
            if rel > 1e-3:
                raise RuntimeError(f"wrong result N={N} K={K} M={M}: "
                                   f"rel={rel:.2e}")
            t_ours = benchmark(lambda: ext.mma_gemm(
                A, Qw, S, lo, hi, GROUP_SIZE, pm, sw, nt, yt))
            best, best_tid = None, None
            for tid in FLUTE_TIDS:
                fout = flute.qgemm(A, fq[tid], S, qmap, qmap2, ws, NUM_BITS,
                                   GROUP_SIZE, tid, num_sms)
                ferr = ((fout.double() - ref).norm() / ref.norm()).item()
                if ferr > 2e-3:      # gate each baseline cell too
                    continue
                t = benchmark(lambda tid=tid: flute.qgemm(
                    A, fq[tid], S, qmap, qmap2, ws, NUM_BITS, GROUP_SIZE,
                    tid, num_sms))
                if best is None or t < best:
                    best, best_tid = t, tid
            t_de = benchmark(lambda: A @ Wdq)
            cfg = f"{arm}/pm{pm}/sw{sw}/nt{nt}/yt{yt}"
            print(f"{M:>4} {t_ours*1e6:>9.1f} {cfg:>20} {best*1e6:>9.1f} "
                  f"{best_tid:>4} {t_de*1e6:>9.1f} {best/t_ours:>8.2f}x "
                  f"{t_de/t_ours:>8.2f}x", flush=True)
        del Qw, S, Wdq, Wdq64, fq
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
