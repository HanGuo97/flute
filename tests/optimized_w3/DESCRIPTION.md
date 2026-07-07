# optimized_w3 — a faster 3-bit LUT decode GEMM for FLUTE

A drop-in **3-bit LUT decode GEMM** for FLUTE that keeps FLUTE's learned
non-uniform quant map and offline-pack model, but replaces the CuTe mainloop with
a memory-bound-optimal one. Across the **entire M = 1–16 decode band** it is
**1.1–1.4× faster than FLUTE's best-tuned 3-bit kernel** while being **~2×
tighter in accuracy** — measured with FLUTE's own `do_bench` on an A100. It lifts
DRAM utilisation from **~36% → ~68%**, i.e. onto the same bandwidth roofline the
W4 kernels hit.

Self-contained under `tests/optimized_w3/`: CUDA kernel + PyTorch binding, offline
packing, learned-qmap prmt-LUT, data-driven per-shape dispatch, an autotuner, a
correctness+bench test, and docs (README, results, KERNEL_ANATOMY).

---

## Why this exists

Decode-phase GEMM at batch M ≤ 16 is **memory-bound**: it streams far more weight
bytes than it does math, so the roofline is DRAM bandwidth and *any dequant work
not hidden under memory traffic is pure overhead on the critical path.* FLUTE's
3-bit kernel dequantises via a two-plane bit-slice unpack **plus a shared-memory
paired-LDS** through the learned map, on a thin-tiled loop that hides latency by
running two CTAs/SM. That unpack + smem round-trip sits *on* the critical path
and the thin tile feeds each weight to only a few `mma`, so the kernel stalls
issue-bound at **~36% DRAM** — the dequant is *added to* memory time, not hidden
under it.

optimized_w3 keeps the two things FLUTE gets right — the **learned non-uniform
qmap** (its accuracy edge) and the **offline-pack model** — and rebuilds the
mainloop so dequant is **hidden under memory** (`time ≈ max(mem, compute)`),
putting the kernel on the DRAM roofline at **~68%**.

---

## Core contributions (what's novel here)

The memory-bound mainloop *structure* is adapted from **Marlin** (Frantar et al.,
IST-DASLab — see *Citation*), which showed a low-M mixed-precision GEMM can be
held at near-peak DRAM bandwidth by running **one fat CTA per SM** with a deep
prefetch pipeline and a large register accumulator, instead of relying on
multi-CTA occupancy. Marlin's fast path, however, dequantises **uniform** W4
levels with a `lop3` magic-bias arithmetic trick that **does not apply to a
learned, non-uniform map**. The novel contributions here carry that learned-LUT
case onto the memory-bound mainloop:

1. **In-register `prmt`-LUT dequant for a learned map.** The 8-entry learned qmap
   is held in registers as two fp16 byte-planes; dequant is a `prmt` (byte
   permute) that uses the fetched code word *itself* as the selector — **8 values
   = 1 shift + 8 `prmt`, zero shared-memory traffic.** `prmt` *is* the table
   lookup, so FLUTE's non-uniform accuracy is preserved while FLUTE's smem
   paired-LDS is eliminated entirely. This is the piece Marlin's arithmetic path
   can't express.

2. **Nibble + mma-fragment offline pack.** Each 3-bit code is packed into its own
   nibble, so the word loaded from DRAM *is already* the `prmt` selector — **no
   unpack ALU on the critical path** (vs FLUTE's cross-plane shift/mask/recombine).
   The packer also lays weights in `mma.sync` fragment order (`_base_perm`), so
   the weight tile needs **no `ldmatrix` transpose**: after the `cp.async` stage
   into shared memory, a plain vectorized 16-byte `ld.shared` drops each lane's
   bytes **directly into its `mma` B-fragment registers** (only the A tile still
   pays an `ldmatrix`). The weights still transit the `cp.async` pipeline buffer —
   that staging *is* the latency-hiding mechanism; what's eliminated is the
   `ldmatrix` and the unpack ALU. The one smem round-trip actually removed is
   FLUTE's dequant-table LDS, which contribution 1 serves from registers instead.
   Trade-off, stated honestly: a nibble spends 4 bits on a 3-bit code, so the
   weight stream is ~25% larger than FLUTE's 3.2-bit bit-slice — paid back many
   times over by moving DRAM utilisation 36% → 68% (net ~1.5× on the widest shape).

3. **FLUTE-qmap integration on the fat-tile mainloop.** Wiring the learned map,
   group scales, and offline pack through Marlin's fat register tile + 4-stage
   `cp.async` pipeline so the dequant overlaps the weight loads. The **fat
   register accumulator** is the lever that moves the roofline (36% → 68% DRAM),
   but the mechanism depends on where you are in the band: as M grows, the M loop
   is innermost so each dequantised weight is **reused across all `thread_m_blocks`
   `mma`** (classical arithmetic intensity). At the bottom of the band (M ≤ 16,
   `thread_m_blocks = 1`) there is **no such reuse** — there the 36% → 68% comes
   from **memory-level parallelism** (the 4-stage `cp.async` keeps 4 weight tiles
   in flight, saturating DRAM by request depth instead of by FLUTE's occupancy
   switching) plus **freeing issue slots**: with the smem-LDS, unpack, and
   `ldmatrix` all gone (contributions 1–2), the SM spends its issue bandwidth on
   *loads* rather than dequant bookkeeping — which is exactly what an issue-capped
   memory-bound kernel needs. Either way DRAM bandwidth, not instruction issue,
   becomes the bound.

**Adopted from Marlin (attributed, Apache-2.0):** one CTA/SM at 256 threads, the
4-stage `cp.async` pipeline with fully-unrolled static addressing, striped
**Stream-K** with in-L2 serial reduction, and L2 evict-first streaming of the
single-use weights.

### It's a co-design, not one knob
The same pipeline-depth knob has **opposite sign** on the two kernels
(28672×8192, M=16): this kernel at `Stages=2` is *slower* than FLUTE (165 µs vs
122 µs), and FLUTE at `Stages=4` is slower than FLUTE at 2. Only the **fat-tile ×
deep-pipe product** crosses (→ 88 µs). There is no incremental config-space path
from FLUTE — the crossing is the whole mainloop rewrite, which is why it ships as
a separate kernel.

---

## Results — full M = 1–16 decode band (A100-SXM4-80GB)

`triton.testing.do_bench` (L2 flush, rep=100 — FLUTE's own tuning method), vs
**FLUTE's best 3-bit template per shape** (autotuned, not just tid20). Both
kernels use identical output allocation (apples-to-apples). The speedup is
**flat across M = 1–16** because both kernels are weight-bandwidth-bound and the
`M×K` activation is negligible — so the decode band is won *uniformly*, not just
at one point.

| shape (N×K) | FLUTE (µs) | optimized_w3 (µs) | **speedup, M=1–16** | rel-err (FLUTE → opt) |
|---|---|---|---|---|
| 28672×8192  | 125–126 | 88–90 | **1.39–1.42×** | 4.2e-4 → **2.5e-4** |
| 8192×8192   |  46–50  | 35–37 | **1.28–1.33×** | 6.3e-4 → **2.8e-4** |
| 14336×4096  |  41–42  | 32–33 | **1.27–1.29×** | 5.1e-4 → **2.9e-4** |
| 4096×4096   |  22–22  | 20–20 | **1.09–1.12×** | 6.2e-4 → **3.1e-4** |

Per-M detail (speedup at M ∈ {1, 2, 4, 8, 16}):

| shape | M=1 | M=2 | M=4 | M=8 | M=16 |
|---|---|---|---|---|---|
| 28672×8192 | 1.42× | 1.42× | 1.41× | 1.41× | 1.39× |
| 8192×8192  | 1.33× | 1.30× | 1.28× | 1.29× | 1.29× |
| 14336×4096 | 1.29× | 1.27× | 1.28× | 1.27× | 1.28× |
| 4096×4096  | 1.11× | 1.09× | 1.12× | 1.10× | 1.10× |

- **Widest-N wins most.** 28672×8192 is the most memory-bound (largest
  weight-bandwidth share), so it gets the biggest lift — ~88 µs at ~68% DRAM, the
  W4 roofline.
- **~2× tighter accuracy, everywhere.** Same learned qmap on both sides, but the
  dequant here rounds to fp16 before the accumulate → relative error is roughly
  **half** FLUTE's own 3-bit output on every shape. Every autotune/test candidate
  is exactness-gated (`err ≤ 2e-3` and must not regress vs FLUTE) *before* it is
  allowed to be timed.

Reproduce: `python -m tests.optimized_w3.test_correctness` (A100). Numbers above
are a fresh `do_bench` sweep; run-to-run variance is ±~0.03× on the ratio.

---

## Citation

The mainloop structure — one CTA/SM, deep `cp.async` pipeline, fat register
tiles, striped Stream-K, static unrolling — derives from **Marlin**:

> E. Frantar, R. L. Castro, J. Chen, T. Hoefler, D. Alistarh.
> *MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language
> Models.* IST-DASLab. https://github.com/IST-DASLab/marlin (Apache-2.0).

The Apache-2.0 license header is preserved in `csrc/optimized_w3_kernel.cu`. The
**contributions in this package** are the W3-specific pieces Marlin's uniform-W4
path does not cover: the in-register `prmt`-LUT dequant for a *learned,
non-uniform* map, the nibble + `mma`-fragment offline pack, and the FLUTE-qmap
integration on the fat-tile mainloop.
