"""W3 (3-bit) arbitrary-table register-LUT GEMM experiment.

Builds tests/mma_w3_lut.cu (prmt-based dequant, zero smem/LSU lookups, tight
6MB packing) and checks it against an fp64 reference with the same exactness
criterion as the W4 kernels, then benchmarks against FLUTE's W3 production
templates (best correct tid per M) and dense torch GEMM.

  python -m tests.w3.mma_w3_lut
"""
import numpy as np
import torch
import triton.testing as tb

DEVICE = torch.device("cuda")
DTYPE = torch.float16
FP16_MIN_NORMAL = 2.0 ** -14
ABS_NOISE = 1e-6      # HMMA accumulator noise ceiling, short chains (K-split)
# the large-M kernel accumulates K in one serial HMMA chain (no K-split);
# accumulator noise scales ~sqrt(chain length) -> ~3x ceiling
ABS_NOISE_BIG = 4e-6


def _ulp_key(t: torch.Tensor) -> torch.Tensor:
    u = t.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    return torch.where(u < 0x8000, u + 0x8000, 0x10000 - u)


def error_stats(out: torch.Tensor, ref: torch.Tensor,
                noise: float = ABS_NOISE) -> dict:
    """Exactness criterion: every element within 1 fp16 ULP of the correctly-
    rounded fp64 result, OR within `noise` absolute (HMMA's fp32 accumulator
    rounds intermediate sums differently from serial IEEE FMA, leaving
    ~1e-7-grade noise visible on near-zero elements; a genuine indexing or
    packing bug shows up as ~1e-3+ absolute error and fails this)."""
    ref_h = ref.to(DTYPE)
    normal = ref.abs() >= FP16_MIN_NORMAL
    ulp = (_ulp_key(out) - _ulp_key(ref_h)).abs()
    absdiff = (out.double() - ref).abs()
    viol = (ulp > 1) & (absdiff >= noise)
    return dict(
        rel=((out.double() - ref).norm() / ref.norm()).item(),
        viol=int(viol.sum().item()),
        ulp_normal=ulp[normal].max().item() if normal.any() else 0,
        max_abs=absdiff.max().item(),
        exact=(out == ref_h).float().mean().item(),
    )


N = K = 4096
GROUP_SIZE = 128
NUM_BITS = 3
CHECK_MS = [1, 2, 3, 4, 8, 13, 16, 24, 32, 48, 64]
CHECK_BIG_MS = [96, 256, 1024, 2048]
BENCH_MS = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def _load_ext():
    import os
    from torch.utils.cpp_extension import load
    here = os.path.dirname(os.path.abspath(__file__))
    return load(
        name="mma_w3_lut_ext",
        sources=[os.path.join(here, "mma_w3_lut.cu")],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=True,
    )


_ARMS = {}


def arms():
    """Both lookup arms of the same kernel (bit-identical outputs):
    "prmt" = register byte-permute LUT, "smem" = FLUTE-style paired table."""
    if not _ARMS:
        import os
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(__file__))
        _ARMS["prmt"] = load(
            name="mma_w3_lut_ext",
            sources=[os.path.join(here, "mma_w3_lut.cu")],
            extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
        _ARMS["smem"] = load(
            name="w3_smem_lut_ext",
            sources=[os.path.join(here, "w3_smem_lut.cu")],
            extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
    return _ARMS


# Per-(compute capability, N, K) dispatch: M-threshold -> (arm, pm, sw, nt,
# yt).  The same packed tensor serves every entry (the pack is keyed to the
# mma fragment map, not to these knobs).  AUTO falls through to the kernel's
# built-in per-M table (tuned on L40S at N=K=4096); explicit entries come
# from full (pm, sw, nt, yt) x {prmt, smem} grid re-sweeps under the
# clean-overwrite (serving) protocol on the named GPU/shape, every cell
# exactness-gated (see docs/w3_prmt_dequant.md).  Unlisted shapes/GPUs
# use AUTO.
AUTO = ("prmt", 0, 0, 0, 0)
DISPATCH = {
    # A100 (sm_80), LLaMA-8B q/o projection shape
    (8, 0, 4096, 4096): [
        (16, AUTO),
        (24, ("prmt", 1, 8, 2, 1)),
        (32, ("prmt", 5, 16, 2, 1)),
        (48, ("prmt", 5, 8, 4, 1)),
        (64, ("prmt", 2, 8, 4, 2)),
    ],
    # A100 (sm_80), LLaMA-8B gate/up projection shape
    (8, 0, 14336, 4096): [
        (4, AUTO),
        (16, ("smem", 2, 8, 2, 1)),
    ],
    # A100 (sm_80), LLaMA-8B down projection shape
    (8, 0, 4096, 14336): [
        (8, AUTO),
        (16, ("smem", 2, 8, 1, 1)),
    ],
    # A100 (sm_80), LLaMA-70B attention shape
    (8, 0, 8192, 8192): [
        (2, ("smem", 1, 8, 1, 1)),
        (4, ("prmt", 1, 16, 2, 1)),
        (16, ("smem", 2, 8, 2, 1)),
        (32, ("prmt", 5, 16, 2, 1)),
    ],
    # A100 (sm_80), LLaMA-70B gate/up projection shape
    (8, 0, 28672, 8192): [
        (4, ("smem", 1, 8, 2, 1)),
        (16, ("smem", 2, 8, 2, 1)),
        (32, ("smem", 5, 8, 4, 1)),
    ],
}


def best_config(M: int, N: int, K: int):
    cap = torch.cuda.get_device_capability(DEVICE)
    for m_max, cfg in DISPATCH.get((cap[0], cap[1], N, K), []):
        if M <= m_max:
            return cfg
    return AUTO


def w3_qgemm(A, Qw, S, lo, hi, group_size=GROUP_SIZE):
    """Dispatching entry point: picks lookup arm + template knobs by
    (M, N, K, GPU) from the swept table, AUTO elsewhere."""
    arm, pm, sw, nt, yt = best_config(A.shape[0], S.shape[0], A.shape[1])
    return arms()[arm].mma_gemm(A, Qw, S, lo, hi, group_size, pm, sw, nt, yt)


def pack_w3(W: torch.Tensor) -> torch.Tensor:
    """(K, N) codes 0..7 -> tight word-plane stream (int32), 3 bits/code.

    Lane record per (octet, 128-k chunk): 32 codes = 96 bits; code for step j,
    slot t sits at bit 12j + 3t (k = ci*128 + j*16 + (lane%4)*4 + t,
    n = oct*8 + lane//4). Stored as 3 word-planes of 32 lanes each.
    """
    Kd, Nd = W.shape
    if Kd % 128 or Nd % 8:
        raise ValueError(f"pack_w3 needs K % 128 == 0 and N % 8 == 0, "
                         f"got K={Kd} N={Nd}")
    c = W.cpu().numpy().astype(np.uint8)
    # process octets in column blocks to bound host memory (the layout is
    # octet-major, so per-block streams concatenate verbatim)
    out = []
    for n0 in range(0, Nd, 8192):
        cb = c[:, n0:n0 + 8192]
        noct, nch = cb.shape[1] // 8, Kd // 128
        oct_, ci, lane, j, t = np.indices((noct, nch, 32, 8, 4),
                                          dtype=np.int32)
        n = oct_ * 8 + (lane >> 2)
        k = ci * 128 + j * 16 + (lane & 3) * 4 + t
        codes = cb[k, n]                              # (noct, nch, 32, 8, 4)
        rec = np.zeros((noct, nch, 32, 12), np.uint8)  # 96-bit lane records
        for jj in range(8):
            for tt in range(4):
                pos = 12 * jj + 3 * tt
                bidx, off = pos >> 3, pos & 7
                v = codes[..., jj, tt]
                rec[..., bidx] |= (v << off) & 0xFF
                if off > 5:
                    rec[..., bidx + 1] |= v >> (8 - off)
        words = rec.view('<u4')                       # (noct, nch, 32, 3)
        planes = np.ascontiguousarray(words.transpose(0, 1, 3, 2))
        out.append(torch.from_numpy(planes.reshape(-1).view(np.int32)))
    return torch.cat(out).to(W.device)


def lut_planes(qmap: torch.Tensor):
    """(8,) fp16 -> (lut_lo, lut_hi) int64: lo/hi byte planes of the entries."""
    qb = qmap.cpu().numpy().view(np.uint8).reshape(8, 2)
    lo = int.from_bytes(qb[:, 0].tobytes(), "little")
    hi = int.from_bytes(qb[:, 1].tobytes(), "little")
    s64 = lambda x: x - (1 << 64) if x >= (1 << 63) else x
    return s64(lo), s64(hi)


def _build(seed: int):
    torch.manual_seed(seed)
    W = torch.randint(0, 2 ** NUM_BITS, (K, N), dtype=torch.int64, device=DEVICE)
    S = torch.randn((N, K // GROUP_SIZE), dtype=DTYPE, device=DEVICE) / 10.0
    qmap = torch.randn(2 ** NUM_BITS, dtype=DTYPE, device=DEVICE)
    Wdq = qmap[W] * torch.repeat_interleave(S, GROUP_SIZE, dim=1).T
    return W, S, qmap, Wdq


def check_correctness(ext, seeds=(0, 1)) -> bool:
    ok = True
    for seed in seeds:
        W, S, qmap, Wdq = _build(seed)
        Qw = pack_w3(W)
        lo, hi = lut_planes(qmap)
        Wdq64 = Wdq.double()
        for M in CHECK_MS + CHECK_BIG_MS:
            A = torch.randn(M, K, dtype=DTYPE, device=DEVICE) / 100.0
            out = ext.mma_gemm(A, Qw, S, lo, hi, GROUP_SIZE, 0, 0, 0, 0)
            noise = ABS_NOISE_BIG if M > 64 else 1e-6
            s = error_stats(out, A.double() @ Wdq64, noise=noise)
            good = s["rel"] < 1e-3 and s["viol"] == 0
            ok &= good
            print(f"seed {seed} M={M:>4}: rel {s['rel']:.2e}  "
                  f"violations {s['viol']}  ULP(normal) {s['ulp_normal']}  "
                  f"bit-exact {100*s['exact']:6.2f}%  {'OK' if good else '!! FAIL'}")
    return ok


def flute_w3_baseline():
    """Find correct FLUTE W3 tids and return packs for the fastest few."""
    import flute
    import flute.utils
    num_sms = flute.utils.get_device_num_sms(DEVICE)
    ws = flute.utils.make_workspace_streamk(device=DEVICE)
    W, S, qmap, Wdq = _build(0)
    qmap2 = flute.utils.make_qmap2_from_qmap(qmap)
    A = torch.randn(16, K, dtype=DTYPE, device=DEVICE) / 100.0
    ref = A @ Wdq
    # every W3 template with an implemented packing layout (TileP=32);
    # each is correctness-gated below before it may enter the baseline
    tids = [4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23,
            28, 29, 30, 31, 32, 33, 34, 35]
    scored = []
    for tid in tids:
        try:
            Q = flute.utils.pack(W=W, num_bits=NUM_BITS, template_ids=[tid],
                                 num_sms=num_sms)
            out = flute.qgemm(A, Q, S, qmap, qmap2, ws, NUM_BITS, GROUP_SIZE,
                              tid, num_sms)
            if ((out - ref).norm() / ref.norm()).item() < 2e-3:
                t = tb.do_bench(lambda: flute.qgemm(A, Q, S, qmap, qmap2, ws,
                                                    NUM_BITS, GROUP_SIZE, tid,
                                                    num_sms), rep=50)
                scored.append((t, tid, Q))
            else:
                del Q
        except Exception:
            pass
    scored.sort(key=lambda x: x[0])
    keep = scored[:8]
    print(f"FLUTE W3: {len(scored)} correct tids; keeping "
          f"{[t for _, t, _ in keep]}")
    return [(tid, Q) for _, tid, Q in keep], (W, S, qmap, qmap2, ws, num_sms, Wdq)


def bench(ext):
    import flute
    prods, (W, S, qmap, qmap2, ws, num_sms, Wdq) = flute_w3_baseline()
    Wdq = Wdq.contiguous()
    Qw = pack_w3(W)
    lo, hi = lut_planes(qmap)
    print(f"\n{'M':>5} {'w3-prmt':>9} {'flute-w3':>9} {'(tid)':>6} "
          f"{'torch':>9} {'vs flute':>9} {'vs torch':>9}")
    for M in BENCH_MS:
        A = torch.randn(M, K, dtype=DTYPE, device=DEVICE) / 100.0
        t_m = tb.do_bench(lambda: ext.mma_gemm(A, Qw, S, lo, hi, GROUP_SIZE,
                                               0, 0, 0, 0), rep=100)
        tp = {tid: tb.do_bench(lambda tid=tid, Q=Q: flute.qgemm(
                  A, Q, S, qmap, qmap2, ws, NUM_BITS, GROUP_SIZE, tid,
                  num_sms), rep=100)
              for tid, Q in prods}
        bt = min(tp, key=tp.get)
        t_t = tb.do_bench(lambda: A @ Wdq, rep=100)
        print(f"{M:>5} {t_m:>9.4f} {tp[bt]:>9.4f} {bt:>6} {t_t:>9.4f} "
              f"{tp[bt] / t_m:>8.2f}x {t_t / t_m:>8.2f}x")


def main():
    ext = _load_ext()
    print("\n=== correctness vs fp64 reference ===")
    ok = check_correctness(ext)
    print(f"\ncorrectness: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    if ok:
        print(f"\n=== benchmark (N=K=4096, W3 g128, "
              f"{torch.cuda.get_device_name()}) ===")
        bench(ext)


if __name__ == "__main__":
    main()
