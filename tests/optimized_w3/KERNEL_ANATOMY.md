# optimized_w3 — Kernel Anatomy

How the kernel works, what it changes vs FLUTE's 3-bit production kernel, and why
those changes pay off. For the measured numbers see `results.md`.

**Problem.** Decode GEMM `C[M,N] = A[M,K] · dequant(Wq[K,N])`, weights 3-bit,
group-quantised scales, **M ≤ 64** (memory-bound). FLUTE stores weights `(K, N)`.

**One-line idea.** Keep FLUTE's two levers of value — the **learned non-uniform
8-entry qmap** (its accuracy) and the **offline-pack model** — but replace the
CuTe mainloop with a **Marlin-derived mainloop** so the dequant is *hidden under*
memory instead of *added to* it. DRAM utilisation goes ~36% → ~68%.

---

## 1. FLUTE's 3-bit production kernel (the baseline, config tid20)

`SMs_Multiple=2` (216 blocks, **2 CTA/SM**), **128 threads/CTA**, **2 pipeline
stages**, thin tiles (TileM16/TileK64/TileP32). Dequant = **two-plane bit-slice
unpack + a shared-memory paired-LDS through `qmap2`**. Hides latency by
**occupancy** (the 2nd CTA issues while the 1st stalls).

```
kernel flute_qgemm(A, Bq, S, qmap2, ...):          # 128 threads, 2 CTA/SM
    for k_tile in streamk_range():
        cp.async(smemA, smemB <- next tile)         # 2-deep
        for packed 32b word w in smemB:             # 10 codes / 32b, TWO PLANES
            code   = unpack_two_plane(w)            # shift+mask+recombine across planes
            b_frag = LDS qmap2[code]                # shared-mem paired LDS
            b_frag = hmul2(b_frag, scale)
        accum += TiledMMA(smemA, b_frag)            # thin N-tile: few mma / weight
    streamk_reduce(accum)
```

Result: 36% DRAM. The per-code unpack + smem LDS sits **on the critical path**,
and the thin N-tile feeds each dequanted weight to only a few `mma` — so the
kernel is *work/issue-bound*, dequant time is **added** to memory time.

> FLUTE is **not** a fixed 2-stage kernel: it ships 36 3-bit configs spanning
> Stages {2,3,4,5}, Threads {128,256}, SMs_Multiple {1,2}, and autotunes per
> shape. tid20 is the config its tuner selects for the decode band (measured:
> deeper stages are monotonically worse there — a deeper pipe costs smem and
> adds no MLP in this memory-bound regime). We benchmark against FLUTE's *best*
> config per shape, not just tid20.

## 2. optimized_w3 (this kernel)

`sms` blocks (**1 CTA/SM**), **256 threads/CTA**, **4 pipeline stages**, fat
register tiles. Dequant = **in-register prmt-LUT**. Hides latency by **ILP
inside one CTA** (deep pipe + register tiles + fully-unrolled static addressing).

```
kernel optimized_w3(A, Bq, S, lut, ...):            # 256 threads, 1 CTA/SM
    FragC accum[thread_m_blocks][4][2] = 0          # FAT register tile
    fill 4-stage cp.async pipe
    for k_tile in striped_streamk_range():          # fully unrolled body
        cp.async(smemA, smemB <- k+4)               # 4-deep prefetch
        ldmatrix frag_a <- smemA[k]
        for w in smemB[k]:                          # 8 codes / 32b, ONE plane (nibbles)
            frag_b = dequant_w3(w, lut)             # 2 prmt from registers, no smem
            frag_b = scale(frag_b, S_group)
        for mb, n: accum[mb][n] += mma(frag_a[mb], frag_b[n])   # many mma / weight
    serial_in_L2_reduce(accum)                       # lock-stepped across column slices
```

```
dequant_w3(sel /*fetched nibble word == prmt selector*/, L /*qmap as byte planes*/):
    lo = prmt(L.lo0, L.lo1, sel)     # low bytes of qmap[code] for 4 codes
    hi = prmt(L.hi0, L.hi1, sel)     # high bytes
    return interleave(lo, hi)        # 8 values = 1 SHR + 8 prmt, zero smem
```

Result: 68% DRAM. dequant + mma live in one CTA's ILP, overlapped with the
4-deep cp.async; the fat tile amortises each weight load over many mma. Dequant
is **hidden under** memory (`time ≈ max(mem, compute)`).

---

## 3. What changed, and why each is necessary

| # | Change | vs FLUTE | Role |
|---|--------|----------|------|
| 1 | **Fat register tile** (many mma / weight load) | thin TileP=32 | **the lever** — raises arithmetic intensity so DRAM (not issue) bounds → 36%→68% |
| 2 | **In-register prmt-LUT dequant** | smem paired-LDS | removes the smem round-trip for the dequant table |
| 3 | **Nibble pack + `_base_perm` fragment order** | two-plane, flat | removes per-code unpack ALU and the smem round-trip for B |
| 4 | **Static unroll + 256 thr / 1 CTA/SM** | dynamic, 128 thr / 2 CTA | ILP + register/smem budget that makes #1 runnable |

**Kept, deliberately:** FLUTE's learned qmap. Because it is *non-uniform*, dequant
must be a **table lookup** — Marlin-W4's lop3 magic-bias arithmetic only works for
uniform levels. So we keep the table but serve it via **prmt in registers** (#2)
instead of smem. `prmt` *is* the lookup; qmap is what it reads from.

**Came free with the mainloop** (already present, no work): striped Stream-K +
in-L2 serial reduction, L2 evict-first streaming of the single-use weights, and
scale double-buffering.

### Why it's a co-design, not one knob (measured, single variable)
The same pipeline knob has **opposite sign** on the two kernels (28672×8192, M16):

| Stages | FLUTE | optimized_w3 |
|---|---|---|
| 2 | **122µs (best)** | 165µs (*slower than FLUTE*) |
| 4 | 134µs (worse) | **88µs (best)** |

optimized_w3 at Stages=2 **loses to FLUTE** — the fat-tile structure without the
deep pipe loses, and the deep pipe without the structure (on FLUTE) also loses.
Only the **fat-tile × deep-pipe product** crosses. That is why no incremental
config-space path exists between the two kernels (FLUTE's Threads, MMA-layout and
pack are static-asserted as one welded unit): the crossing is the mainloop rewrite.

---

## 4. Tuning granularities

| Granularity | Knob | FLUTE | optimized_w3 | Tunable here |
|---|---|---|---|---|
| Grid / SM | block count / `sms` | 216 (2/SM) | `sms` (1/SM) | ✅ `sms` in dispatch (≤ device SMs) |
| CTA | threads/CTA | 128 | 256 | fixed (kernel) |
| Tile | (thread_k, thread_n) | TileP-locked | (128,128) or (64,256) | ✅ autotuned per shape |
| Pipeline | stages | 2–5 (tuner) | 4 | fixed (4 is best, measured) |
| Pack | layout | two-plane 16B | nibble 16B + perm | fixed |
| Dequant | mechanism | smem LDS | register prmt | fixed |

The autotuned surface for this kernel is **(thread_k, thread_n) × sms per
(M, N, K)** — see `dispatch.py` / `autotune.py`. Wide-N/deep-K shapes want the fat
`(64,256)` tile; the rest want `(128,128)`; `sms=108` (or `-1` auto) generally
best on A100. `sms` must never exceed the device SM count — an oversubscribed grid
breaks the striped lock-order reduction.

---

## Provenance
The mainloop derives from Marlin (Frantar et al., IST-DASLab, Apache-2.0; see the
license header in `csrc/`). The 3-bit prmt-LUT dequant, the nibble+`_base_perm`
pack, and the FLUTE-qmap integration are the additions here.
