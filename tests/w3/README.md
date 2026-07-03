# W3 dual-arm dequant GEMM (register-PRMT / smem-table lookup) + multi-protocol benchmarking

Scope: a standalone 3-bit (W3) fused dequant-GEMM with two interchangeable,
bit-identical lookup arms — a register `prmt` (byte-permute) LUT and a
FLUTE-style shared-memory paired table — selected per (M, shape, GPU) by a
swept dispatch table, plus benchmarking under all three published protocol
families: triton `do_bench` (dirty L2, FLUTE's own harness), a
Marlin-verbatim warm bench, and a clean-overwrite serving protocol
introduced here. All code is additive under `tests/`; FLUTE's production
kernels are untouched.

Hardware/setup for every number below: A100-SXM4-80GB (sm_80, 108 SMs,
40 MB L2), CUDA 12.9, group_size=128, fp16 activations/scales,
fp32-accumulate HMMA. Shapes are (N, K) entries from FLUTE's own
`tests/shapes.py` (LLaMA-3 8B and 70B layers). The kernel was developed on
L40S (sm_89), which is why the built-in AUTO dispatch fallback is tuned
there; A100 is reported since it is in FLUTE's support matrix.

---

## 1. The kernel

3-bit is the case where PRMT is a *native* fit: an 8-entry LUT's byte plane
is exactly PRMT's 8-byte source pool, and a 3-bit code is literally a valid
PRMT selector nibble (bits [2:0] select, bit 3 stays clear). Dequant of a
4-code window is:

```
selector spread -> 2 independent prmt (lo/hi byte planes)
               -> 2 prmt interleaves into half2 pairs -> __hmul2 scale
```

zero shared-memory and zero LSU work for the lookup, ANY 8-entry fp16 table
(no codebook constraint), and bit-identical math to the paired-table path
(`__hmul2` reproduces production's fp16 dequant rounding exactly).

Weights are packed offline into mma-fragment order (a lane's coalesced 16B
load IS its m16n8k16 B-fragment codes; 3-bit codes tight-packed as three
word-planes per 128-k chunk in a single tensor, funnel-shift windows — no
bit-slicing across two streams). Explicit cross-chunk B double-buffering
(dequant j+1 before mma j); split-K across warps with smem reduction;
per-M dispatch over (perm, sw, nt, yt) template knobs; a TileM=64
big-kernel variant covers M>=96. The packed layout is keyed to the mma
fragment map only — one packed tensor serves every dispatch config, both
lookup arms, and every M.

**Two lookup arms, one kernel.** `tests/w3/w3_smem_lut.cu` is the identical
kernel with only `w3_dequant_step` swapped for a FLUTE-style 64-entry half2
paired shared-memory table. Outputs are bit-for-bit equal. Which arm is
faster flips with the regime (section 4), so the arm is a dispatch knob:
`w3_qgemm()` in `tests/w3/mma_w3_lut.py` selects (arm, perm, sw, nt, yt) per
(M, N, K, compute capability) from the swept `DISPATCH` table — the same
Python-side selection model FLUTE itself uses for template_ids.

Files: `tests/w3/mma_w3_lut.cu` (prmt arm + built-in AUTO host dispatch),
`tests/w3/w3_smem_lut.cu` (smem arm / controlled A/B), `tests/w3/mma_w3_lut.py`
(packing, LUT planes, exactness machinery, `DISPATCH`/`w3_qgemm`, do_bench
flow), `tests/w3/w3_kernel.py` (correctness suite over the LLaMA-3 8B/70B
shapes of tests/shapes.py, mirroring tests/kernel.py),
`tests/w3/w3_marlin_bench.py` and `tests/w3/w3_clean_overwrite_bench.py` (the
other two protocol benches), and `tests/w3/w3_tune.py` (the dispatch autotune:
sweeps (perm, sw, nt, yt) x {prmt, smem} for a shape under the
clean-overwrite protocol and prints DISPATCH-ready rows -- every DISPATCH
entry was produced by it, and unswept shapes/GPUs extend the table with
`python -m tests.w3.w3_tune N K`).

## 2. Correctness

Exactness bar (stricter than "allclose"): every element within 1 ULP of an
fp64 reference OR < 1e-6 absolute (4e-6 for M >= 256, where 256-step serial
HMMA accumulator chains add physical fp16-ULP noise near zero), and zero
violations. Measured rel error is 2.07-2.09e-4 — identical to the
fp16-dequant reference itself, and 3-5x tighter than FLUTE's W3 templates
(kMixed fp16-inner accumulation: 5.3e-4 to 1.2e-3 norm-relative, growing
with the Stream-K oversubscription factor). The two lookup arms agree
bit-for-bit. Every dispatch-sweep cell and every bench cell is
additionally gated on an fp64 check before its timing is trusted (a config
that silently fails must not win a sweep). Exactness passes on all six
shapes benchmarked, including the non-square 8B geometries (fused qkv
(6144, 4096), gate/up (14336, 4096), down (4096, 14336)).

**FLUTE W3 template audit** (isolated-process sweep, FLUTE's own pack +
`flute.qgemm` + its own 2e-3 norm bar; A100, official wheel `_C`): of the
36 W3 templates, the 12 TileP=64 ones cannot be packed at all — FLUTE's
`_pack_3bit` is implemented only for TileP=32 (`flute/utils.py`) — and
the remaining **24 are all numerically valid** at every shape benchmarked
here, M <= 32 (W4 for comparison: 143 of 144 valid). An earlier
classification of ours (only tids {4, 5, 10} correct, others wrong or
hanging) came from an L40S with a locally-compiled sm_89 `_C` and did not
reproduce on A100 with the shipped wheel; every number below uses the
full best-of-24 oracle. One systemic caveat stands: `flute.tune` selects
by speed alone among templates that launch (correctness is a post-hoc
warning printed for the winner only), which is safe exactly when every
candidate is valid — true on A100 with the wheel, demonstrably not true
of every (arch, binary) pair.

FLUTE baseline everywhere below: exhaustive per-cell sweep over all 24
packable W3 templates, each cell exactness-gated in situ before its
timing counts, best taken per cell. This is an oracle: it upper-bounds
what `flute.tune.tune_and_pack` (which commits to one template ahead of
time) could select. Ours is NOT per-cell: every protocol runs the same
fixed clean-overwrite-tuned `DISPATCH`.

## 3. The three benchmark protocols

Published quantized-GEMM numbers do not share an L2 cache state:

- **`do_bench` (dirty L2).** FLUTE's own harness (`flute/tune.py`) times
  with `triton.testing.do_bench`, which zeroes a 256 MB buffer before every
  timed iteration. That write-flush leaves L2 full of *dirty* lines: during
  the kernel's weight stream every incoming line must first write back a
  victim to DRAM. The paper's kernel figures (including its Marlin/BitBLAS/
  bitsandbytes baselines) run through this harness. Kept as-is here via
  `tests/w3/mma_w3_lut.py::bench`.
- **Marlin protocol (warm L2).** `tests/w3/w3_marlin_bench.py` carries
  IST-DASLab/marlin's `benchmark()` verbatim: 1 warmup + 10 back-to-back
  wall-clock iterations on the same tensors, no cache control, 1 s
  cool-down (their README also locks base clocks; requires sudo, not done
  here). Any weight stream that fits in L2 (6-25 MB W3 at 8B/70B-attention
  shapes vs 40 MB A100 L2) is served warm — a state real decode never sees,
  since the model's other layers evict it between visits.
- **Clean-overwrite protocol (serving state) — introduced here.**
  `tests/w3/w3_clean_overwrite_bench.py`: 24 clones of the layer
  weights (well past L2) cycled like transformer layers, per-call CUDA
  events, no flush. A revisited layer's weights are cold — but evicted by
  *clean* read-only lines, so eviction is free. That is the L2 state a real
  decode step sees: weights cold-but-clean, activations hot, the only dirty
  lines KB-scale outputs. Real serving sits between the two published
  extremes; this protocol is what the shipped dispatch targets.

Why the protocols disagree at small shapes and agree at large ones: the
dirty-flush tax is a roughly fixed few-microsecond cost — 25-40% of a
~20 us kernel on a 6 MB stream, noise on a 150 us kernel streaming 88 MB.
The warm protocol conversely flatters any sub-L2 stream. Dispatch tuned
under one protocol can mis-rank under another (measured repeatedly during
this work), so every dispatch table entry states its protocol: the shipped
tables are clean-overwrite-tuned.

## 4. Results (A100, dispatched, M <= 16, best-of-24 FLUTE oracle)

All times in microseconds, single runs of the shipped benches; "flute" is
the best valid W3 template per cell (tid in parentheses), "dense" is fp16
`torch` (cuBLAS) on the dequantized weight. Note which templates actually
win the oracle: overwhelmingly tids 8/9 and 20/21 (the Threads=128/
TileM=16 family at Stages 2-3 and SMs_Multiple 1-2) — templates the
earlier misclassification excluded, and the ones upstream's own shipped
tuned table favors. tid 10 (the best of the old 3-tid oracle) is never
the true per-cell best below M=32.

### Clean-overwrite protocol (`tests/w3/w3_clean_overwrite_bench.py`)

| shape (N, K) | M | ours | flute | dense | vs flute | vs dense |
|---|---|---|---|---|---|---|
| (4096, 4096)  | 1  | 19.4 | 24.6 (8) | 27.6 | **1.26x** | 1.42x |
|               | 2  | 20.0 | 24.7 (8) | 28.5 | **1.23x** | 1.42x |
|               | 4  | 19.6 | 24.3 (8) | 28.6 | **1.24x** | 1.46x |
|               | 8  | 20.5 | 24.7 (8) | 28.6 | **1.21x** | 1.39x |
|               | 16 | 22.6 | 24.9 (8) | 28.8 | **1.10x** | 1.27x |
| (8192, 8192)  | 1  | 43.4 | 43.5 (20) | 91.3 | 1.00x | 2.11x |
|               | 2  | 45.0 | 43.3 (20) | 91.4 | 0.96x | 2.03x |
|               | 4  | 47.2 | 43.4 (20) | 91.6 | 0.92x | 1.94x |
|               | 8  | 48.3 | 43.4 (20) | 92.0 | 0.90x | 1.91x |
|               | 16 | 48.2 | 43.8 (20) | 92.2 | 0.91x | 1.91x |
| (28672, 8192) | 1  | 126.1 | 121.1 (20) | 293.1 | 0.96x | 2.32x |
|               | 2  | 117.0 | 120.0 (20) | 287.5 | 1.03x | 2.46x |
|               | 4  | 125.2 | 120.5 (20) | 288.4 | 0.96x | 2.30x |
|               | 8  | 135.2 | 120.3 (20) | 293.8 | 0.89x | 2.17x |
|               | 16 | 136.1 | 120.7 (20) | 296.2 | 0.89x | 2.18x |

### Marlin protocol (`tests/w3/w3_marlin_bench.py`)

| shape (N, K) | M | ours | flute | dense | vs flute | vs dense |
|---|---|---|---|---|---|---|
| (4096, 4096)   | 1  | 17.1 | 20.9 (9) | 29.4 | **1.23x** | 1.72x |
|                | 2  | 17.3 | 21.6 (8) | 24.7 | **1.25x** | 1.43x |
|                | 4  | 17.6 | 21.3 (9) | 26.8 | **1.21x** | 1.52x |
|                | 8  | 18.3 | 23.4 (10) | 26.9 | **1.28x** | 1.47x |
|                | 16 | 24.6 | 21.6 (9) | 27.6 | 0.88x | 1.12x |
| (6144, 4096)   | 1  | 23.5 | 25.3 (20) | 42.4 | **1.08x** | 1.81x |
|                | 2  | 24.2 | 25.6 (21) | 43.4 | **1.06x** | 1.79x |
|                | 4  | 24.3 | 26.1 (22) | 45.0 | **1.07x** | 1.86x |
|                | 8  | 26.5 | 25.8 (21) | 43.7 | 0.97x | 1.65x |
|                | 16 | 25.6 | 27.3 (21) | 45.8 | **1.07x** | 1.79x |
| (14336, 4096)  | 1  | 41.5 | 40.3 (21) | 85.2 | 0.97x | 2.05x |
|                | 2  | 45.5 | 41.8 (21) | 81.2 | 0.92x | 1.78x |
|                | 4  | 46.5 | 41.9 (21) | 79.1 | 0.90x | 1.70x |
|                | 8  | 44.6 | 40.4 (21) | 80.9 | 0.91x | 1.82x |
|                | 16 | 46.8 | 42.3 (21) | 82.5 | 0.90x | 1.76x |
| (4096, 14336)  | 1  | 43.2 | 42.8 (20) | 83.4 | 0.99x | 1.93x |
|                | 2  | 47.7 | 43.0 (21) | 85.0 | 0.90x | 1.78x |
|                | 4  | 48.6 | 42.4 (20) | 84.6 | 0.87x | 1.74x |
|                | 8  | 53.0 | 42.5 (20) | 85.3 | 0.80x | 1.61x |
|                | 16 | 55.5 | 44.0 (20) | 85.3 | 0.79x | 1.54x |
| (8192, 8192)   | 1  | 41.7 | 46.4 (20) | 91.4 | **1.11x** | 2.19x |
|                | 2  | 47.8 | 45.9 (21) | 91.7 | 0.96x | 1.92x |
|                | 4  | 50.2 | 46.9 (20) | 92.7 | 0.93x | 1.85x |
|                | 8  | 51.4 | 45.4 (20) | 92.2 | 0.88x | 1.79x |
|                | 16 | 52.1 | 45.9 (20) | 91.7 | 0.88x | 1.76x |
| (28672, 8192)  | 1  | 112.7 | 129.5 (20) | 294.9 | **1.15x** | 2.62x |
|                | 2  | 125.8 | 129.7 (20) | 290.8 | 1.03x | 2.31x |
|                | 4  | 134.2 | 130.9 (20) | 291.4 | 0.98x | 2.17x |
|                | 8  | 147.7 | 130.1 (20) | 298.4 | 0.88x | 2.02x |
|                | 16 | 148.5 | 131.4 (20) | 299.0 | 0.88x | 2.01x |

Under FLUTE's own do_bench harness at 4096^2 the same dispatch measures
1.17x / 1.14x / 1.11x / 0.97x vs the 24-tid oracle at M=1/4/8/16
(1.9-2.3x vs dense); the big-kernel band trails at M=32-512 and wins
again at 1024-2048 (1.04-1.31x).

So with the corrected oracle the honest headline is narrower than the
earlier 3-tid-baseline draft of this report claimed: **the robust win is
the (4096, 4096)-class decode band — 1.21-1.28x (Marlin), 1.21-1.26x
(clean-overwrite), 1.11-1.17x (do_bench) at M <= 8 — plus M=1 at the
large shapes under the Marlin protocol (1.11-1.15x) and most of the
fused-qkv (6144, 4096) band (1.06-1.08x).** Everywhere else FLUTE's
Threads=128/TileM=16 templates at SMs_Multiple=2 (tids 20/21) match or
beat the shipped dispatch by 4-21%, mostly at deep K and M >= 4. For
context vs the paper's own headline: FLUTE's A100 figure shows ~2.1x vs
torch.mm at 8192^2 (reproduced here at 1.89-1.94x with their harness and
autotuner; the 3-4x end of their range is the A6000's weak dense
baseline).

**Honest losses (vs the 24-tid oracle) the current dispatch does not
recover:** M=16-32 at (4096, 4096) under warm/dirty protocols (0.83-0.97x
— reuse regime begins; clean-overwrite still wins 1.10x at 16), the deep-K
shapes at M >= 4 (0.79-0.98x — tids 20/21 pair TileM=16 with a staged smem
pipeline and 2x grid oversubscription; our 2-stage register buffer has
less latency-hiding depth at K >= 8192), the (4096, 14336) down-proj
generally (0.79-0.99x), and large M (>= 32 under do_bench, 0.53-0.83x
until the big kernel wins again at >= 1024). Several losing cells run the
AUTO (L40S-tuned) fallback — (14336, 4096) M 1-4, (4096, 14336) all M,
(6144, 4096) all M — so part of the gap is unswept dispatch, not kernel
ceiling; `python -m tests.w3.w3_tune N K` extends the table. The
remainder is the design boundary stated below.

### Existing lookup (smem paired table) vs prmt — the controlled A/B

Identical kernel, only `w3_dequant_step` differs. The winner flips with
the regime:

- **prmt wins the mid-band, M=4-16** (up to +14-15% at 4096^2 M=8, both
  protocols) — per-SM memory pressure exposes the smem lookup's LDS
  latency in the dequant->mma chain.
- **smem wins the issue-bound extremes**: M=1 at deep K (e.g. +11% at
  8192^2 — long inner loop, ~9 resident CTAs/SM hide all latency, and the
  currency becomes instruction count: 2 values/LDS vs ~1/ALU-op) and
  M>=32.
- Dispatch interacts with the arm: the two arms prefer *different* perm
  modes (prmt: pm1 direct-A; smem: pm2 permuted-A), so an arm benched
  under the other's dispatch can look ~25% worse than it is. Per-arm
  sweeps are mandatory; the `DISPATCH` table was swept per arm.

So the honest attribution: the vs-FLUTE decode win comes from the kernel
*design* — fragment-order packing (no smem staging/`ldmatrix` reorder on a
no-reuse operand), no phantom TileM rows at M=1 (split-K instead), single
launch, register B double-buffering — while the lookup arm is a
several-percent regime-dependent knob on top. The design lever is a
fixed-overhead deletion, so it is largest where kernels are shortest:
~1.2-1.3x at 4096-class shapes and at M=1 generally, shrinking to parity
or worse at 70B-class shapes for M >= 4 (where FLUTE's staged smem
pipeline plus Stream-K oversubscription earns its keep hiding DRAM
latency over K=8192), and negative at M >= 16-32 where weight reuse
returns.

### Design boundary (why W3, and not W4, and not inside FLUTE's kernel)

- W4 (16-entry LUT) needs a 2-level prmt tree (6 lookup + 2 interleave
  prmts/step): measured tie vs FLUTE's vectorized qmap2 at M=8-256 and
  5-6% loss at 1024-2048 (+36% executed instructions). FLUTE's smem qmap2
  choice is right at 4 bits; register-prmt wins at <= 8 entries.
- Grafting prmt INTO FLUTE's own W3 mainloop loses 12-22% everywhere and
  collapses tid 10 by 20x: FLUTE's fully-unrolled ~30-lookup dequant makes
  byte-perm temporaries scale with the unroll (+25 regs) in a
  register-saturated loop; cute arrays demote to local memory. The smem
  lookup is the only dequant that needs ZERO auxiliary registers, which is
  precisely why it fits FLUTE's mainloop. Register-prmt requires the
  register headroom this kernel's design keeps (40-130 regs).

## 5. What is NOT in this PR

The pipelined-dequant variant (separate branch, measured slower than
fused), the microbench harness, the warp-shuffle and in-FLUTE byteperm
graft arms (measured dead ends, removed), dispatch sweeps for the cells
still on AUTO fallback and for M>32 and GPUs other than A100/L40S, a
deeper (3-4 stage) prefetch pipeline + grid oversubscription for the
long-K regime where FLUTE's tids 20/21 currently win, and any
model-level integration (an nn.Linear wrapper + end-to-end decode tok/s
validation of the clean-overwrite numbers is the natural follow-up).
