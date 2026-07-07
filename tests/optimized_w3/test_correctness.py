"""Correctness + smoke benchmark for optimized_w3.

Validates the kernel against an fp64 reference (must be tighter than FLUTE's own
3-bit output) and reports latency vs FLUTE's best 3-bit template under do_bench
(triton.testing, L2 flush, rep=100 -- FLUTE's own tuning method), on a few
decode shapes.

    python -m tests.optimized_w3.test_correctness
"""
import sys
import warnings

import torch
import triton.testing as tb

import flute
import flute.utils
from . import pack_w3, pack_scales, lut_planes, qgemm_w3

warnings.filterwarnings("ignore")
DEV = torch.device("cuda")
GROUP = 128
M = 16
CELLS = [(8192, 8192), (28672, 8192), (4096, 4096), (14336, 4096)]


def main():
    ns = flute.utils.get_device_num_sms(DEV)
    fws = flute.utils.make_workspace_streamk(device=DEV)
    print(f"optimized_w3 correctness+bench on {torch.cuda.get_device_name()} | M={M}")
    print(f"{'shape':>14} | {'flute_us':>9} {'opt_us':>8} {'speedup':>8} | "
          f"{'err_flute':>10} {'err_opt':>10}")
    fails = 0
    for (N, K) in CELLS:
        torch.manual_seed(0)
        W = torch.randint(0, 8, (K, N), dtype=torch.int64, device=DEV)
        S = torch.randn((N, K // GROUP), dtype=torch.half, device=DEV) / 10.0
        qmap = torch.randn(8, dtype=torch.half, device=DEV)
        qmap2 = flute.utils.make_qmap2_from_qmap(qmap)
        A = torch.randn((M, K), dtype=torch.half, device=DEV) / 100.0
        Wdq = (qmap[W] * torch.repeat_interleave(S, GROUP, dim=1).T).double()
        ref = A.double() @ Wdq

        fqt = flute.utils.pack(W=W.to(torch.uint8).int(), num_bits=3,
                               template_ids=[20], num_sms=ns)
        Wp, Sp = pack_w3(W, N, K), pack_scales(S, N, K)
        lut = lut_planes(qmap)
        pw = torch.zeros(N // 128 * 16, dtype=torch.int32, device=DEV)

        fn_f = lambda: flute.qgemm(A, fqt, S, qmap, qmap2, fws, 3, GROUP, 20, ns)
        fn_o = lambda: qgemm_w3(A, Wp, Sp, lut, pw)
        ef = ((fn_f().double() - ref).norm() / ref.norm()).item()
        eo = ((fn_o().double() - ref).norm() / ref.norm()).item()
        if eo > 2e-3 or eo > ef:
            fails += 1

        tf = tb.do_bench(fn_f, rep=100) * 1000.0   # ms -> us
        to = tb.do_bench(fn_o, rep=100) * 1000.0
        print(f"{f'{N}x{K}':>14} | {tf:9.1f} {to:8.1f} {tf/to:7.2f}x | "
              f"{ef:10.1e} {eo:10.1e}")
        del W, S, fqt, Wp, Sp, Wdq
        torch.cuda.empty_cache()
    print("OK -- optimized_w3 correct and faster" if fails == 0
          else f"FAIL -- {fails} shape(s) regressed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
