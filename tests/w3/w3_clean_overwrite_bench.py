"""Clean-overwrite (serving-state) decode bench.

24 clones of the layer weights (well past L2 capacity) cycled like
transformer layers, per-call CUDA events, no flush: when a layer is
revisited its weights are L2-cold, but evicted by other layers' READ-ONLY
(clean) lines, so eviction is free -- the cache state a real decode step
sees.  Contrast triton do_bench (256 MB dirty write-flush = a writeback tax
no deployment pays) and the Marlin warm loop (same tensors every call = an
L2 residency no deployment gets).  Every (shape, M) cell is exactness-gated
against an fp64 reference before its timing is trusted.

  python -m tests.w3.w3_clean_overwrite_bench
"""
import sys, os
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
BATCHES = [1, 2, 4, 8, 16]
SHAPES = [                       # (N, K), from FLUTE's tests/shapes.py
    (4096, 4096),                # LLaMA-8B q/o proj
    (8192, 8192),                # LLaMA-70B attention
    (28672, 8192),               # LLaMA-70B gate/up proj
]
NLAYERS, ROUNDS = 24, 30


def ring_time(call, layers):
    for l in layers:                       # warmup + JIT/alloc settle
        call(l)
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True),
           torch.cuda.Event(enable_timing=True))
          for _ in range(ROUNDS * len(layers))]
    i = 0
    for _ in range(ROUNDS):
        for l in layers:
            ev[i][0].record()
            call(l)
            ev[i][1].record()
            i += 1
    torch.cuda.synchronize()
    ts = [s.elapsed_time(e) for s, e in ev]
    return sum(ts) / len(ts) * 1000.0      # mean us


def main():
    num_sms = flute.utils.get_device_num_sms(DEVICE)
    ws = flute.utils.make_workspace_streamk(device=DEVICE)
    print(f"clean-overwrite protocol on {torch.cuda.get_device_name()}: "
          f"{NLAYERS} layers x {ROUNDS} rounds, per-call events")
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
        Wdq64 = Wdq.double()
        del W
        ours_layers = [(Qw.clone(), S.clone()) for _ in range(NLAYERS)]
        de_layers = [Wdq.clone() for _ in range(NLAYERS)]
        As = {M: torch.randn(M, K, dtype=DTYPE, device=DEVICE) / 100.0
              for M in BATCHES}
        refs = {M: As[M].double() @ Wdq64 for M in BATCHES}
        t_ours, t_de, cfgs = {}, {}, {}
        for M in BATCHES:
            A = As[M]
            arm, pm, sw, nt, yt = best_config(M, N, K)
            ext = arms()[arm]
            out = ext.mma_gemm(A, Qw, S, lo, hi, GROUP_SIZE, pm, sw, nt, yt)
            rel = ((out.double() - refs[M]).norm() / refs[M].norm()).item()
            if rel > 1e-3:
                raise RuntimeError(f"wrong result N={N} K={K} M={M}: "
                                   f"rel={rel:.2e}")
            t_ours[M] = ring_time(lambda l: ext.mma_gemm(
                A, l[0], l[1], lo, hi, GROUP_SIZE, pm, sw, nt, yt),
                ours_layers)
            t_de[M] = ring_time(lambda l: A @ l, de_layers)
            cfgs[M] = f"{arm}/pm{pm}/sw{sw}/nt{nt}/yt{yt}"
        # FLUTE oracle: one tid's layer ring resident at a time
        best = {M: (None, None) for M in BATCHES}
        for tid in FLUTE_TIDS:
            fqt = flute.utils.pack(W=Wu, num_bits=NUM_BITS,
                                   template_ids=[tid], num_sms=num_sms)
            layers = [(fqt.clone(), S.clone()) for _ in range(NLAYERS)]
            for M in BATCHES:
                A = As[M]
                fout = flute.qgemm(A, fqt, S, qmap, qmap2, ws, NUM_BITS,
                                   GROUP_SIZE, tid, num_sms)
                ferr = ((fout.double() - refs[M]).norm() /
                        refs[M].norm()).item()
                if ferr > 2e-3:      # gate each baseline cell too
                    continue
                t = ring_time(lambda l, tid=tid: flute.qgemm(
                    A, l[0], l[1], qmap, qmap2, ws, NUM_BITS, GROUP_SIZE,
                    tid, num_sms), layers)
                if best[M][0] is None or t < best[M][0]:
                    best[M] = (t, tid)
            del fqt, layers
            torch.cuda.empty_cache()
        print(f"\nN={N} K={K}  (W3 {N*K*3/8/1e6:.0f}MB / "
              f"dense {N*K*2/1e6:.0f}MB)")
        print(f"{'M':>4} {'ours':>9} {'cfg':>20} {'flute':>9} {'tid':>4} "
              f"{'dense':>9} {'vs flute':>9} {'vs dense':>9}")
        for M in BATCHES:
            bt, btid = best[M]
            print(f"{M:>4} {t_ours[M]:>9.1f} {cfgs[M]:>20} {bt:>9.1f} "
                  f"{btid:>4} {t_de[M]:>9.1f} {bt/t_ours[M]:>8.2f}x "
                  f"{t_de[M]/t_ours[M]:>8.2f}x", flush=True)
        del Qw, S, Wu, Wdq, Wdq64, ours_layers, de_layers, As, refs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
