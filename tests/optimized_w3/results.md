# optimized_w3 — Optimizations & Results

A 3-bit LUT GEMM for the FLUTE decode path (M ≤ 64) that keeps FLUTE's learned
qmap and offline-pack model but runs on a Marlin-derived mainloop. Measured on
**NVIDIA A100-SXM4-80GB (108 SMs)**, M=16, group=128.

---

## Optimizations

Ordered by contribution. See `KERNEL_ANATOMY.md` for the mechanism of each.

1. **Fat register tile (arithmetic intensity)** — the output accumulator is a
   large per-thread register tile (`thread_n_blocks × thread_m_blocks`), so each
   weight fetched from DRAM feeds *many* `mma`. This raises the compute-to-memory
   ratio until **DRAM bandwidth**, not instruction issue, is the bottleneck. It is
   the single change that moves DRAM utilisation **36% → 68%**.

2. **In-register prmt-LUT dequant** — the learned 8-entry qmap is held in
   registers as two fp16 byte-planes; `prmt` (byte-permute) selects `qmap[code]`
   with the fetched nibble word *as the selector*. 8 values = 1 SHR + 8 prmt,
   **zero shared-memory** traffic. Replaces FLUTE's smem paired-LDS.

3. **Nibble pack + `_base_perm` fragment order** — each 3-bit code occupies its
   own nibble (fetched word *is* the prmt selector → no unpack ALU), and the
   offline `_base_perm` lays weights in `mma.sync` fragment order so a lane's
   coalesced 16B load lands directly in its B-fragment registers (**no shared-
   memory round-trip for B**). Replaces FLUTE's two-plane bit-slice.

4. **Static full unroll + 256 thr / 1 CTA/SM** — compile-time addressing (no
   per-iteration address/predicate math) and instruction-level parallelism that
   *fills* the deep pipeline, plus the register/smem budget to hold the fat tile.
   This is what makes #1 runnable.

5. **From the Marlin mainloop (kept):** striped Stream-K with in-L2 serial
   reduction, L2 evict-first streaming of the single-use weights, scale
   double-buffering.

**Deliberately kept from FLUTE:** the learned non-uniform qmap (accuracy) — which
forces dequant to remain a *table lookup* (#2), not arithmetic. And the
offline-pack model.

**Key finding — it is a co-design, not one knob.** At Stages=2 (shallow pipe)
this kernel is *slower* than FLUTE; the fat tile only pays *together* with the
deep pipe. Neither lever crosses alone, which is why there is no incremental path
from FLUTE — only the whole mainloop rewrite.

---

## Final results

Speedup vs **FLUTE's best 3-bit template per shape** (autotuned, not just tid20),
under two benchmark methods. Both are cold-weight; the ring cycles 24 distinct
weight tensors (per-call events), do_bench repeats one tensor with an L2 flush
(rep=100) — FLUTE's own tuning method.

Absolute latency (µs) and speedup, FLUTE-best vs optimized_w3 (do_bench, rep=100):

| shape (N×K) | FLUTE | optimized_w3 | **speedup** |
|---|---|---|---|
| 8192×8192  |  50.4 | 37.6 | **1.34×** |
| 28672×8192 | 135.0 | 90.2 | **1.50×** |
| 14336×4096 |  42.5 | 32.3 | **1.32×** |
| 4096×4096  |  23.1 | 20.4 | **1.13×** |

Reproducible via `python -m tests.optimized_w3.test_correctness` (M=16, A100).
Timed with `triton.testing.do_bench` (L2 flush, rep=100) — FLUTE's own tuning
method. Both kernels use identical output allocation (apples-to-apples).

- **Uniform win across the decode band:** 1.13–1.50× on all four shapes.
- **Widest N wins most:** 28672×8192 is the most memory-bound (largest
  weight-bandwidth share), so it gets the biggest win — opt ≈90µs at ~68% DRAM,
  the Marlin-W4 roofline, **1.50×**.
- **Baseline is FLUTE's best per shape**, not just tid20 (its do_bench-best is
  tid21/tid8 on some shapes); the win holds against the stronger baseline.

### Accuracy (relative error vs fp64, M=16) — tighter than FLUTE everywhere

| shape | FLUTE 3-bit | optimized_w3 |
|---|---|---|
| 8192×8192  | 6.3e-4 | **2.8e-4** |
| 28672×8192 | 4.2e-4 | **2.5e-4** |
| 14336×4096 | 5.1e-4 | **2.9e-4** |
| 4096×4096  | 8.3e-4 | **3.1e-4** |

Same learned qmap, but the dequant rounds to fp16 before accumulate → strictly
tighter than FLUTE's own 3-bit output.

---

## Reproduce

```bash
# env (A100): a libstdc++ with GLIBCXX_3.4.29 + CUDA on PATH
export LD_LIBRARY_PATH=/orcd/software/core/001/spack/pkg/gcc/12.2.0/yt6vabm/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/orcd/software/core/001/pkg/cuda/12.9.1

python -m tests.optimized_w3.test_correctness   # exactness + bench, both harnesses
python -m tests.optimized_w3.autotune           # re-tune -> dispatch.py
```

## Scope / limits
- Tuned for **A100 (108 SMs)**; `dispatch.py` is device-keyed, re-run `autotune`
  for other GPUs.
- Headline numbers are **M=16**. Larger-M and the flat-in-M input-staging
  (A-permute) optimisation are future work.
- Isolated-GEMM latency, not end-to-end tokens/sec.
