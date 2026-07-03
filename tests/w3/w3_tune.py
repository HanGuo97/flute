"""Dispatch autotune: sweep (perm, sw, nt, yt) x {prmt, smem} for one shape
under the clean-overwrite (serving) protocol and print DISPATCH-ready rows.

Every cell is exactness-gated against an fp64 reference before its timing
is trusted (a config that silently fails must not win a sweep).  Output
lines are (arm, pm, sw, nt, yt) per M, formatted for the DISPATCH table in
tests/mma_w3_lut.py.  Identical configs drift ~1-2 us between runs on A100;
treat sub-2-us differences between the best few rows as ties.

  python -m tests.w3.w3_tune                      # N=K=4096, M=1..32
  python -m tests.w3.w3_tune 14336 4096           # one shape
  python -m tests.w3.w3_tune 14336 4096 1,4,8,16  # and specific Ms
"""
import sys, os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch

from tests.w3.mma_w3_lut import arms, pack_w3, lut_planes, GROUP_SIZE, \
    NUM_BITS, DTYPE, DEVICE

NLAYERS, ROUNDS = 24, 20
GRID = [(pm, sw, nt, yt)
        for pm in (1, 2, 5, 6)
        for sw in (8, 16)
        for nt in (1, 2, 4)
        for yt in (1, 2)]


def ring_time(call, layers):
    for l in layers:
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
    return sum(ts) / len(ts) * 1000.0


def tune(N, K, ms):
    torch.manual_seed(0)
    W = torch.randint(0, 2 ** NUM_BITS, (K, N), dtype=torch.int64,
                      device=DEVICE)
    S = torch.randn((N, K // GROUP_SIZE), dtype=DTYPE, device=DEVICE) / 10.0
    qmap = torch.randn(2 ** NUM_BITS, dtype=DTYPE, device=DEVICE)
    Wdq64 = (qmap[W] *
             torch.repeat_interleave(S, GROUP_SIZE, dim=1).T).double()
    Qw = pack_w3(W)
    lo, hi = lut_planes(qmap)
    del W
    layers = [(Qw.clone(), S.clone()) for _ in range(NLAYERS)]
    cap = torch.cuda.get_device_capability(DEVICE)
    print(f"# sweep on {torch.cuda.get_device_name()} "
          f"(sm_{cap[0]}{cap[1]}), N={N} K={K}, clean-overwrite protocol")
    print(f"# DISPATCH key: ({cap[0]}, {cap[1]}, {N}, {K})")
    for M in ms:
        A = torch.randn(M, K, dtype=DTYPE, device=DEVICE) / 100.0
        ref = A.double() @ Wdq64
        rn = ref.norm()
        results = []
        for arm, ext in arms().items():
            for (pm, sw, nt, yt) in GRID:
                try:
                    out = ext.mma_gemm(A, Qw, S, lo, hi, GROUP_SIZE,
                                       pm, sw, nt, yt)
                    rel = ((out.double() - ref).norm() / rn).item()
                    if rel > 1e-3:
                        raise RuntimeError(f"wrong result rel={rel:.2e}")
                    t = ring_time(lambda l: ext.mma_gemm(
                        A, l[0], l[1], lo, hi, GROUP_SIZE, pm, sw, nt, yt),
                        layers)
                    results.append((t, arm, pm, sw, nt, yt))
                except Exception:
                    pass          # invalid combo or wrong result: skip
        auto = ring_time(lambda l: arms()["prmt"].mma_gemm(
            A, l[0], l[1], lo, hi, GROUP_SIZE, 0, 0, 0, 0), layers)
        results.sort()
        t, arm, pm, sw, nt, yt = results[0]
        top3 = " | ".join(f"{a}/pm{p}/sw{s}/nt{n}/yt{y} {tt:.1f}"
                          for tt, a, p, s, n, y in results[:3])
        print(f'    ({M}, ("{arm}", {pm}, {sw}, {nt}, {yt})),'
              f'   # {t:.1f}us  auto {auto:.1f}us  [{top3}]', flush=True)


def main():
    N, K = ((int(sys.argv[1]), int(sys.argv[2]))
            if len(sys.argv) >= 3 else (4096, 4096))
    ms = ([int(m) for m in sys.argv[3].split(",")]
          if len(sys.argv) >= 4 else [1, 2, 4, 8, 16, 32])
    tune(N, K, ms)


if __name__ == "__main__":
    main()
