// W3 (3-bit) LUT GEMM with an ARBITRARY 8-entry table held entirely in
// registers: dequant is prmt-based (byte-permute), zero shared-memory and
// zero LSU traffic for the lookup. Derived from tests/mma_w4_lut.cu (which is
// left untouched); same fragment-order packing idea, same A path, same mma.
//
//   - Weights packed tightly: 3 bits/code -> 6MB at 4096^2 (75% of W4).
//     A lane's 128-k chunk record is 32 codes = 96 bits = 3 words, stored as
//     three coalesced word-planes.
//   - Per step: one compile-time funnel-shift extracts the 12-bit window,
//     one selector spread feeds TWO INDEPENDENT prmt lookups (lo/hi byte
//     planes -- they pipeline, the register-file analogue of the paired-qmap2
//     trick), two more prmts interleave into half2, then __hmul2 (identical
//     fp16 dequant rounding to the W4 kernels).
//   - The 8-entry fp16 table travels as two int64 kernel arguments.
//
// Why (from the component microbench): at M<=16 the production kernel spends
// ~58-71% in pipeline machinery (cp.async stages, s2r copies, barriers,
// Stream-K bookkeeping) and the tensor core is ~idle; a stock-template sweep
// shows ALL 138 correct configs bottom out at ~25.3us for M<=16. This kernel
// keeps the tensor core (so it scales M=1..64, unlike a CUDA-core GEMV which
// hits the FFMA wall at M>=8) but deletes the machinery:
//
//   - Weights are packed OFFLINE in mma-fragment order: each lane's coalesced
//     16B load contains exactly the code bytes whose paired-LUT lookups are
//     that lane's mma.sync B-fragment registers. gmem -> reg -> dequant ->
//     mma, no smem staging, no ldmatrix, no shuffle.
//   - One byte -> qmap2[byte] -> half2 == one B-fragment register pair;
//     __hmul2 with the scale reproduces production's fp16 dequant rounding
//     exactly (Wdq = fp16(qmap[code] * scale)), then HMMA accumulates in fp32.
//   - Split-K inside the CTA: 4 warps share one 8-column tile, each owns K/4.
//     Grid = N/8 CTAs (512 at N=4096) -- enough blocks at M=1, one smem
//     fragment reduction + one barrier at the end.
//   - A is read per-fragment via __ldg (M<=64 rows; L1-resident; rows >= M
//     read as zero, so any M in 1..16*MTILES is exact).
//
// Packing (built by tests/mma_lut.py): for CTA c, warp w, lane l, uint4 u,
// step j = 8u + (byte/2), ab = byte%2:
//   n = 8c + l/4
//   k = w*(K/4) + j*16 + (l%4)*2 + ab*8 + nibble   (low nibble first)
// flat uint4 index = ((c*4 + w) * (K/512) + u) * 32 + l   (coalesced per warp)

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

#define FULL_MASK 0xffffffffu

__device__ __forceinline__ void
mma_m16n8k16(float c[4], const unsigned a[4], const unsigned b[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

// fp16-accumulate HMMA: 2x the tensor-core rate of the fp32-accumulate form
// (362 vs 181 TFLOP/s on L40S) at the cost of accumulation precision
__device__ __forceinline__ void
mma_m16n8k16_f16(unsigned c[2], const unsigned a[4], const unsigned b[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

// ---- W3 register-LUT machinery ----
struct W3Lut { unsigned lo0, lo1, hi0, hi1; };  // byte planes of 8 fp16 entries

__device__ __forceinline__ uint3
w3_load(const unsigned* __restrict__ p)   // one lane's 96-bit chunk record
{
    uint3 r;
    r.x = __ldcs(p);
    r.y = __ldcs(p + 32);
    r.z = __ldcs(p + 64);
    return r;
}

// 12-bit window (4 x 3-bit codes) at step j; j is compile-time under full
// unrolling so the word picks and shift amounts fold to constants
__device__ __forceinline__ unsigned
w3_window(const uint3 q, const int j)
{
    const unsigned w[3] = {q.x, q.y, q.z};
    const int bit = 12 * j;
    const int a = bit >> 5;
    const unsigned hi = (a < 2) ? w[a + 1] : 0u;
    return __funnelshift_r(w[a], hi, bit & 31) & 0xFFFu;
}

// dequant 4 codes: one selector spread feeds two INDEPENDENT prmt lookups
// (lo/hi byte planes dual-issue/pipeline), two prmts interleave into half2
// pairs, then the scale multiply -- same fp16 rounding as the W4 path
// SMEM-LUT VARIANT (A/B vs the prmt register scheme): FLUTE-style paired
// table -- 64 entries of half2, entry i = (T[i&7], T[i>>3]) -- staged in
// shared memory once per tile, one LDS per code pair. Same __hmul2 path,
// bit-identical results.
__device__ __forceinline__ void
w3_stage_lut(unsigned* lutw, const W3Lut L)
{
    for (int i = threadIdx.x; i < 64; i += blockDim.x) {
        const unsigned a = i & 7, c = i >> 3;
        const unsigned la = __byte_perm(L.lo0, L.lo1, a) & 0xFFu;
        const unsigned ha = __byte_perm(L.hi0, L.hi1, a) & 0xFFu;
        const unsigned lc = __byte_perm(L.lo0, L.lo1, c) & 0xFFu;
        const unsigned hc = __byte_perm(L.hi0, L.hi1, c) & 0xFFu;
        lutw[i] = (la | (ha << 8)) | ((lc | (hc << 8)) << 16);
    }
}

__device__ __forceinline__ void
w3_dequant_step(unsigned b[2], const unsigned win,
                const unsigned* __restrict__ lutw, const half2 sc2)
{
    unsigned v0 = lutw[win & 0x3Fu];          // halfs (c0, c1)
    unsigned v1 = lutw[(win >> 6) & 0x3Fu];   // halfs (c2, c3)
    const half2 h0 = __hmul2(*reinterpret_cast<const half2*>(&v0), sc2);
    const half2 h1 = __hmul2(*reinterpret_cast<const half2*>(&v1), sc2);
    b[0] = *reinterpret_cast<const unsigned*>(&h0);
    b[1] = *reinterpret_cast<const unsigned*>(&h1);
}

// guarded A loads: rows >= M contribute zeros (exactness for any M).
// Fragment slots are relabeled (packing side) so a lane's 4 k's are
// contiguous: one 8-byte load per row covers both A registers of that row.
__device__ __forceinline__ uint2
load_a4(const half* __restrict__ A, int row, int k, int M, int K)
{
    if (row >= M) {
        return make_uint2(0u, 0u);
    }
    return *reinterpret_cast<const uint2*>(A + (size_t)row * K + k);
}

// pre-pass: permute A (M x K, rows >= M as zero) into fragment order, so the
// main loop fetches a lane's whole 8-half mma A-fragment with ONE coalesced
// LDG.128 instead of 4 scattered LDG.32 (the LSU-throughput limiter at M>4).
// Pure relocation of values -> main-kernel results are bit-identical.
__device__ __forceinline__ void
ap_write_item(const half* __restrict__ A, uint4* __restrict__ AP,
              int idx, int M, int K)
{
    const int lane = idx & 31;
    const int s    = (idx >> 5) % (K / 16);
    const int mt   = idx / ((K / 16) * 32);
    const int g  = lane >> 2;
    const int ka = s * 16 + (lane & 3) * 4;
    const int m0 = mt * 16 + g;
    const uint2 ra = load_a4(A, m0,     ka, M, K);
    const uint2 rb = load_a4(A, m0 + 8, ka, M, K);
    AP[idx] = make_uint4(ra.x, rb.x, ra.y, rb.y);   // a0, a1, a2, a3
}

__global__ void
a_perm_kernel(const half* __restrict__ A, uint4* __restrict__ AP,
              const int nitems, const int M, const int K)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nitems) {
        ap_write_item(A, AP, idx, M, K);
    }
}

// PERM=true: one LDG.128 from the fragment-order A built by a_perm_kernel
// (extra kernel launch, but minimal LSU work in the mainloop);
// PERM=false: two guarded LDG.64 straight from A (no pre-pass launch).
// STRONG: AP is being produced DURING this kernel (flag-pipelined fusion),
// so the __ldg non-coherent path is illegal (its contract is kernel-lifetime
// immutability; under fine-grained flags it can be filled with
// pre-production lines -- measured wrong results). Strong relaxed loads go
// to L2; nothing is lost, each 128B AP line is consumed by exactly one
// warp-load.
template <int MTILES, bool PERM, bool STRONG = false>
__device__ __forceinline__ void
load_a_frags(unsigned a[MTILES][4], const half* __restrict__ A,
             const uint4* __restrict__ AP,
             int s, int lane, int M, int K, int mt_base)
{
    #pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
        if (PERM && STRONG) {
            const uint4* p =
                &AP[((size_t)(mt_base + mt) * (K / 16) + s) * 32 + lane];
            uint4 v;
            asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                         : "l"(p));
            a[mt][0] = v.x;
            a[mt][1] = v.y;
            a[mt][2] = v.z;
            a[mt][3] = v.w;
        } else if (PERM) {
            const uint4 v = __ldg(
                &AP[((size_t)(mt_base + mt) * (K / 16) + s) * 32 + lane]);
            a[mt][0] = v.x;
            a[mt][1] = v.y;
            a[mt][2] = v.z;
            a[mt][3] = v.w;
        } else {
            const int g  = lane >> 2;
            const int ka = s * 16 + (lane & 3) * 4;
            const uint2 ra = load_a4(A, (mt_base + mt) * 16 + g,     ka, M, K);
            const uint2 rb = load_a4(A, (mt_base + mt) * 16 + g + 8, ka, M, K);
            a[mt][0] = ra.x;
            a[mt][1] = rb.x;
            a[mt][2] = ra.y;
            a[mt][3] = rb.y;
        }
    }
}

// NTILES: column-octets per CTA. Each A fragment is loaded ONCE and reused
// across NTILES mmas, dividing A/AP traffic by NTILES -- the limiter at
// larger M, where the 8-column tile makes every CTA re-read all of A.
// The packed stream is octet-major, so a CTA just reads NTILES adjacent
// octet streams; the packing itself is unchanged.
// FLAGS: AP is produced in-kernel piece-by-piece (piece = one row-tile x one
// 128-k chunk, 256 uint4 items); before touching a chunk's AP slice the warp
// acquire-spins on that piece's flag (== gen) instead of the whole grid
// crossing a barrier. The spin issues after the chunk's B prefetch, so its
// L2 latency hides under the B DRAM latency.
template <int MTILES, int SW, bool PERM, int NTILES, bool FLAGS = false>
__device__ __forceinline__ void
mma_tile_body(const half*     __restrict__ A,     // (M, K)
              const uint4*    __restrict__ AP,    // fragment-order A (PERM)
              const unsigned* __restrict__ Q,     // W3 word-plane stream
              const half*     __restrict__ S,     // (N, K/group)
              const W3Lut                  L,     // register LUT (8 entries)
                    half*     __restrict__ D,     // (M, N)
              const int M, const int K, const int N, const int group,
              const int colblock, const int mt_base,
              const int* flags = nullptr, const int gen = 0)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int col0 = colblock * 8 * NTILES;
    const int g    = lane >> 2;        // B-fragment column / C row base
    const int t2   = (lane & 3) * 2;   // fragment k / C col offset

    __shared__ float red[SW - 1][32][4 * MTILES * NTILES];
    __shared__ unsigned lutw[64];
    __syncthreads();   // separates `red`/`lutw` reuse across persistent tiles
    w3_stage_lut(lutw, L);
    __syncthreads();

    // packing is warp-count independent: an octet's K is a flat sequence of
    // K/128 chunk-uint4s; warp w owns chunks [w*nui, (w+1)*nui)
    const int nui   = K / (128 * SW);  // uint4s per lane (8 mma steps each)
    const int kbase = warp * nui * 128;
    const unsigned* q4[NTILES];
    const half*     Srow[NTILES];
    #pragma unroll
    for (int nt = 0; nt < NTILES; ++nt) {
        q4[nt] = Q + ((size_t)(colblock * NTILES + nt) * (K / 128)
                      + (size_t)warp * nui) * 96 + lane;
        Srow[nt] = S + (size_t)(col0 + nt * 8 + g) * (K / group);
    }

    float acc[NTILES][MTILES][4];
    #pragma unroll
    for (int nt = 0; nt < NTILES; ++nt) {
        #pragma unroll
        for (int mt = 0; mt < MTILES; ++mt) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc[nt][mt][r] = 0.0f;
            }
        }
    }

    // software pipeline: weight uint4 prefetched one iteration ahead in
    // registers (a warp-local cp.async smem ring was tried and LOST -- with
    // only 2-4 chunks per lane there is no depth to exploit and the smem
    // round-trip + wait_group serialization cost ~15-75%; a 2-deep register
    // ring was tried and ALSO lost -- ptxas scatters the distance-2 loads
    // through the body via a shared address register, long-scoreboard
    // stalls 1.21 -> 2.09/issue), scale hoisted to
    // one load per uint4 (its 8 steps span exactly one 128-code group), A
    // fragments ping-pong prefetched one mma step ahead. B fragments are
    // EXPLICITLY double-buffered (FLUTE-style), all shapes: step j+1 is
    // dequantized into the other half of bb before mma j issues, including
    // across the chunk boundary (next chunk's step 0 from qnext with a
    // prefetched scale).
    uint3 qv[NTILES];
    half2 sc2[NTILES];
    #pragma unroll
    for (int nt = 0; nt < NTILES; ++nt) {
        qv[nt]  = w3_load(q4[nt]);
        sc2[nt] = __half2half2(__ldg(&Srow[nt][kbase / group]));
    }

    unsigned bb[2][NTILES][2];                 // ping-pong B fragments
    #pragma unroll
    for (int nt = 0; nt < NTILES; ++nt) {      // prologue: dequant step 0
        w3_dequant_step(bb[0][nt], w3_window(qv[nt], 0), lutw, sc2[nt]);
    }

    for (int u = 0; u < nui; ++u) {
        uint3 qnext[NTILES];
        half2 scnext[NTILES];
        #pragma unroll
        for (int nt = 0; nt < NTILES; ++nt) {
            qnext[nt]  = (u + 1 < nui)
                       ? w3_load(q4[nt] + (size_t)(u + 1) * 96)
                       : make_uint3(0u, 0u, 0u);
            scnext[nt] = (u + 1 < nui)
                       ? __half2half2(__ldg(&Srow[nt][(kbase + (u + 1) * 128)
                                                      / group]))
                       : sc2[nt];
        }
        const int ku = kbase + u * 128;
        const int s0 = ku / 16;                 // global mma step index

        if (FLAGS) {
            // chunk's AP pieces ready? Unconditional do-while ld.acquire:
            // EVERY execution performs at least one acquire whose result
            // feeds the loop exit, so the AP loads below are data/control-
            // dependent on a real acquire. (A "fast path" that compares a
            // relaxed-prefetched value and branches around the load was
            // tried and is WRONG: it leaves a path with no memory op at
            // all, and ptxas hoists the AP loads above it -- wrong results
            // whenever AP's reused allocation isn't already identical.)
            // The acquire's ~200ns L2 round trip overlaps the qnext B
            // prefetch issued above.
            #pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                const int* fp = flags + (mt_base + mt) * (K / 128)
                              + warp * nui + u;
                int f;
                asm volatile("ld.acquire.gpu.global.s32 %0, [%1];"
                             : "=r"(f) : "l"(fp) : "memory");
                while (f != gen) {
                    __nanosleep(64);
                    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];"
                                 : "=r"(f) : "l"(fp) : "memory");
                }
            }
        }

        unsigned areg[2][MTILES][4];
        load_a_frags<MTILES, PERM, FLAGS>(areg[0], A, AP, s0, lane, M, K, mt_base);

        #pragma unroll
        for (int j = 0; j < 8; ++j) {          // 8 mma k-steps per uint4
            if (j < 7) {
                load_a_frags<MTILES, PERM, FLAGS>(areg[(j + 1) & 1], A, AP,
                                           s0 + j + 1, lane, M, K, mt_base);
                #pragma unroll
                for (int nt = 0; nt < NTILES; ++nt) {
                    w3_dequant_step(bb[(j + 1) & 1][nt],
                                    w3_window(qv[nt], j + 1), lutw, sc2[nt]);
                }
            } else if (u + 1 < nui) {          // cross-chunk: next step 0
                #pragma unroll
                for (int nt = 0; nt < NTILES; ++nt) {
                    w3_dequant_step(bb[0][nt], w3_window(qnext[nt], 0), lutw,
                                    scnext[nt]);
                }
            }

            #pragma unroll
            for (int nt = 0; nt < NTILES; ++nt) {
                #pragma unroll
                for (int mt = 0; mt < MTILES; ++mt) {
                    mma_m16n8k16(acc[nt][mt], areg[j & 1][mt], bb[j & 1][nt]);
                }
            }
        }
        #pragma unroll
        for (int nt = 0; nt < NTILES; ++nt) {
            qv[nt]  = qnext[nt];
            sc2[nt] = scnext[nt];
        }
    }

    // cross-warp split-K reduction of identically-shaped C fragments
    if (warp > 0) {
        #pragma unroll
        for (int nt = 0; nt < NTILES; ++nt) {
            #pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    red[warp - 1][lane][(nt * MTILES + mt) * 4 + r] = acc[nt][mt][r];
                }
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int nt = 0; nt < NTILES; ++nt) {
            #pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    float v = acc[nt][mt][r];
                    #pragma unroll
                    for (int wp = 0; wp < SW - 1; ++wp) {
                        v += red[wp][lane][(nt * MTILES + mt) * 4 + r];
                    }
                    acc[nt][mt][r] = v;
                }
                // C fragment: c0/c1 -> row g,   cols t2, t2+1
                //             c2/c3 -> row g+8, cols t2, t2+1
                const int n = col0 + nt * 8 + t2;
                const int r0 = (mt_base + mt) * 16 + g;
                if (r0 < M) {
                    D[(size_t)r0 * N + n]     = __float2half(acc[nt][mt][0]);
                    D[(size_t)r0 * N + n + 1] = __float2half(acc[nt][mt][1]);
                }
                if (r0 + 8 < M) {
                    D[(size_t)(r0 + 8) * N + n]     = __float2half(acc[nt][mt][2]);
                    D[(size_t)(r0 + 8) * N + n + 1] = __float2half(acc[nt][mt][3]);
                }
            }
        }
    }
}

template <int MTILES, int SW, bool PERM, int NTILES>
__global__ void __launch_bounds__(SW * 32)
mma_w3_lut_kernel(const half*     __restrict__ A,
                  const uint4*    __restrict__ AP,
                  const unsigned* __restrict__ Q,
                  const half*     __restrict__ S,
                  const W3Lut                  L,
                        half*     __restrict__ D,
                  const int M, const int K, const int N, const int group)
{
    mma_tile_body<MTILES, SW, PERM, NTILES>(A, AP, Q, S, L, D, M, K, N,
                                            group, blockIdx.x,
                                            blockIdx.y * MTILES);
}

// Fused single launch: phase 1 grid-strides the A-permute, a software grid
// barrier separates phases (safe: the grid is capped at the queried
// co-resident block capacity; remaining tiles are covered by the persistent
// loop), phase 2 walks the mma tiles. Removes the pre-pass launch (~1.8us).
template <int MTILES, int SW, int NTILES>
__global__ void __launch_bounds__(SW * 32)
mma_w3_lut_fused_kernel(const half*     __restrict__ A,
                        uint4*          __restrict__ APw,
                        const unsigned* __restrict__ Q,
                        const half*     __restrict__ S,
                        const W3Lut                  L,
                              half*     __restrict__ D,
                        const int M, const int K, const int N, const int group,
                        const int nitems, const int ntiles, const int ncolb,
                        int* bws)   // persistent barrier state {count, gen}
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nitems;
         idx += gridDim.x * blockDim.x) {
        ap_write_item(A, APw, idx, M, K);
    }
    __threadfence();   // publish AP writes before arriving at the barrier
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile int* cnt = bws;
        volatile int* gen = bws + 1;
        const int g0 = *gen;
        if (atomicAdd(bws, 1) == (int)gridDim.x - 1) {
            *cnt = 0;          // self-reset for the next launch
            __threadfence();
            atomicAdd(bws + 1, 1);
        } else {
            while (*gen == g0) {
            }
        }
    }
    __syncthreads();
    __threadfence();

    for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
        mma_tile_body<MTILES, SW, true, NTILES>(
            A, APw, Q, S, L, D, M, K, N, group,
            t % ncolb, (t / ncolb) * MTILES);
    }
}


// ---------------------------------------------------------------------------
// Flag-pipelined fusion (perm=6): the fused kernel's grid barrier costs as
// much as the separate permute launch it replaces (~1.9us -- every CTA waits
// for the LAST permute item even though each warp only ever reads the AP
// pieces of its own k-chunks). Here piece p (one row-tile x one 128-k chunk
// = 256 uint4 items) is produced whole by CTA (p % grid), published with a
// release-store of the launch generation, and consumers acquire-spin on the
// specific piece right before reading it. No global synchronization: the
// only wait is first-chunk pipeline fill, and B prefetches issue before
// every spin. Grid <= co-resident capacity (host-enforced), so all
// producers run without waiting on any flag -> deadlock-free.
// ---------------------------------------------------------------------------
template <int MTILES, int SW, int NTILES>
__global__ void __launch_bounds__(SW * 32)
mma_w3_lut_flag_kernel(const half*     __restrict__ A,
                       uint4*          __restrict__ APw,
                       const unsigned* __restrict__ Q,
                       const half*     __restrict__ S,
                       const W3Lut                  L,
                             half*     __restrict__ D,
                       const int M, const int K, const int N, const int group,
                       const int npieces, const int ntiles, const int ncolb,
                       int* flags, const int gen)
{
    for (int p = blockIdx.x; p < npieces; p += gridDim.x) {
        const int base = p * 256;
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            ap_write_item(A, APw, base + i, M, K);
        }
        __threadfence();   // publish this thread's items
        __syncthreads();   // all threads' items published
        if (threadIdx.x == 0) {
            asm volatile("st.release.gpu.global.s32 [%0], %1;"
                         :: "l"(flags + p), "r"(gen));
        }
    }

#ifdef W3_FLAG_DIAG_BARRIER
    // DIAGNOSTIC: grid barrier after produce (flags+1020 as barrier state).
    // Correct results here => producers fine, consumer waits broken.
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile int* gen2 = flags + 1021;
        const int g0 = *gen2;
        if (atomicAdd(flags + 1020, 1) == (int)gridDim.x - 1) {
            *(flags + 1020) = 0;
            __threadfence();
            atomicAdd(flags + 1021, 1);
        } else {
            while (*gen2 == g0) {
            }
        }
    }
    __syncthreads();
    __threadfence();
#endif

    for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
        mma_tile_body<MTILES, SW, true, NTILES, true>(
            A, APw, Q, S, L, D, M, K, N, group,
            t % ncolb, (t / ncolb) * MTILES, flags, gen);
    }
}


// ---------------------------------------------------------------------------
// PFA (prefetch-all) decode variant. A streaming probe showed the decode
// floor in the benchmark harness is the B-stream itself (~17us for 6MB: the
// do_bench write-flush leaves L2 full of dirty lines whose writebacks
// contend with our reads; load width/eviction hints are irrelevant), and
// that firing ALL of a warp's chunk loads up front reaches that floor
// exactly. Here a lane's whole B slice (nui <= 4 chunk records = 48B + 4
// scales per octet) is loaded into registers BEFORE any A-side work -- in
// the fused kernel even before the A-permute phase and grid barrier, since
// B depends on nothing -- so the mma loop runs with zero B-load stalls and
// the A machinery prices at its issue cost only, overlapped with the
// stream.
// ---------------------------------------------------------------------------
template <int PFN, int NTILES>
struct W3Pf {
    uint3 q[PFN][NTILES];
    half2 s[PFN][NTILES];
};

// issue-pinned streaming load: asm volatile so ptxas can NOT sink it toward
// its consumer (it re-scatters intrinsic loads next to first use, which is
// exactly what fire-all prefetch must prevent; measured lsb/issue 1.2->2.5
// with __ldcs here)
__device__ __forceinline__ uint3
w3_load_pinned(const unsigned* __restrict__ p)
{
    uint3 r;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(r.x) : "l"(p));
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(r.y) : "l"(p + 32));
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(r.z) : "l"(p + 64));
    return r;
}

template <int SW, int NTILES>
__device__ __forceinline__ void
w3_prefetch_all(W3Pf<32 / SW, NTILES>& pf,
                const unsigned* __restrict__ Q,
                const half*     __restrict__ S,
                const int K, const int group,
                const int colblock, const int warp, const int lane)
{
    constexpr int PFN = 32 / SW;
    const int nui   = K / (128 * SW);
    const int kbase = warp * nui * 128;
    #pragma unroll
    for (int nt = 0; nt < NTILES; ++nt) {      // the 6MB stream first
        const unsigned* q4 = Q + ((size_t)(colblock * NTILES + nt) * (K / 128)
                                  + (size_t)warp * nui) * 96 + lane;
        #pragma unroll
        for (int u = 0; u < PFN; ++u) {
            pf.q[u][nt] = (u < nui) ? w3_load_pinned(q4 + (size_t)u * 96)
                                    : make_uint3(0u, 0u, 0u);
        }
    }
    const int col0 = colblock * 8 * NTILES;
    const int g    = lane >> 2;
    #pragma unroll
    for (int nt = 0; nt < NTILES; ++nt) {
        const half* Srow = S + (size_t)(col0 + nt * 8 + g) * (K / group);
        #pragma unroll
        for (int u = 0; u < PFN; ++u) {
            pf.s[u][nt] = (u < nui)
                ? __half2half2(__ldg(&Srow[(kbase + u * 128) / group]))
                : __half2half2(__ushort_as_half(0));
        }
    }
}

template <int MTILES, int SW, bool PERM, int NTILES>
__device__ void
mma_tile_body_pfa(const half*  __restrict__ A,
                  const uint4* __restrict__ AP,
                  const W3Lut               L,
                        half*  __restrict__ D,
                  const int M, const int K, const int N,
                  const int colblock, const int mt_base,
                  const W3Pf<32 / SW, NTILES>& pf)
{
    constexpr int PFN = 32 / SW;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int col0 = colblock * 8 * NTILES;
    const int g    = lane >> 2;
    const int t2   = (lane & 3) * 2;

    __shared__ float red[SW - 1][32][4 * MTILES * NTILES];
    __shared__ unsigned lutw[64];
    __syncthreads();   // separates `red`/`lutw` reuse across persistent tiles
    w3_stage_lut(lutw, L);
    __syncthreads();

    const int nui = K / (128 * SW);

    float acc[NTILES][MTILES][4];
    #pragma unroll
    for (int nt = 0; nt < NTILES; ++nt) {
        #pragma unroll
        for (int mt = 0; mt < MTILES; ++mt) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc[nt][mt][r] = 0.0f;
            }
        }
    }

    unsigned bb[2][NTILES][2];                 // ping-pong B fragments
    #pragma unroll
    for (int nt = 0; nt < NTILES; ++nt) {      // prologue: dequant step 0
        w3_dequant_step(bb[0][nt], w3_window(pf.q[0][nt], 0), lutw, pf.s[0][nt]);
    }

    #pragma unroll
    for (int u = 0; u < PFN; ++u) {            // unrolled: pf stays in regs
        if (u >= nui) {
            break;
        }
        const int un = (u + 1 < PFN) ? u + 1 : 0;   // cross-chunk pf index
        const int s0 = (warp * nui + u) * 8;        // global mma step index

        unsigned areg[2][MTILES][4];
        load_a_frags<MTILES, PERM>(areg[0], A, AP, s0, lane, M, K, mt_base);

        #pragma unroll
        for (int j = 0; j < 8; ++j) {          // 8 mma k-steps per chunk
            if (j < 7) {
                load_a_frags<MTILES, PERM>(areg[(j + 1) & 1], A, AP,
                                           s0 + j + 1, lane, M, K, mt_base);
                #pragma unroll
                for (int nt = 0; nt < NTILES; ++nt) {
                    w3_dequant_step(bb[(j + 1) & 1][nt],
                                    w3_window(pf.q[u][nt], j + 1), lutw,
                                    pf.s[u][nt]);
                }
            } else if (u + 1 < nui) {          // cross-chunk: next step 0
                #pragma unroll
                for (int nt = 0; nt < NTILES; ++nt) {
                    w3_dequant_step(bb[0][nt], w3_window(pf.q[un][nt], 0), lutw,
                                    pf.s[un][nt]);
                }
            }

            #pragma unroll
            for (int nt = 0; nt < NTILES; ++nt) {
                #pragma unroll
                for (int mt = 0; mt < MTILES; ++mt) {
                    mma_m16n8k16(acc[nt][mt], areg[j & 1][mt], bb[j & 1][nt]);
                }
            }
        }
    }

    // cross-warp split-K reduction of identically-shaped C fragments
    if (warp > 0) {
        #pragma unroll
        for (int nt = 0; nt < NTILES; ++nt) {
            #pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    red[warp - 1][lane][(nt * MTILES + mt) * 4 + r] = acc[nt][mt][r];
                }
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int nt = 0; nt < NTILES; ++nt) {
            #pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    float v = acc[nt][mt][r];
                    #pragma unroll
                    for (int wp = 0; wp < SW - 1; ++wp) {
                        v += red[wp][lane][(nt * MTILES + mt) * 4 + r];
                    }
                    acc[nt][mt][r] = v;
                }
                const int n = col0 + nt * 8 + t2;
                const int r0 = (mt_base + mt) * 16 + g;
                if (r0 < M) {
                    D[(size_t)r0 * N + n]     = __float2half(acc[nt][mt][0]);
                    D[(size_t)r0 * N + n + 1] = __float2half(acc[nt][mt][1]);
                }
                if (r0 + 8 < M) {
                    D[(size_t)(r0 + 8) * N + n]     = __float2half(acc[nt][mt][2]);
                    D[(size_t)(r0 + 8) * N + n + 1] = __float2half(acc[nt][mt][3]);
                }
            }
        }
    }
}

template <int MTILES, int SW, bool PERM, int NTILES>
__global__ void __launch_bounds__(SW * 32)
mma_w3_lut_kernel_pfa(const half*     __restrict__ A,
                      const uint4*    __restrict__ AP,
                      const unsigned* __restrict__ Q,
                      const half*     __restrict__ S,
                      const W3Lut                  L,
                            half*     __restrict__ D,
                      const int M, const int K, const int N, const int group)
{
    W3Pf<32 / SW, NTILES> pf;
    w3_prefetch_all<SW, NTILES>(pf, Q, S, K, group, blockIdx.x,
                                threadIdx.x >> 5, threadIdx.x & 31);
    mma_tile_body_pfa<MTILES, SW, PERM, NTILES>(A, AP, L, D, M, K, N,
                                                blockIdx.x,
                                                blockIdx.y * MTILES, pf);
}

template <int MTILES, int SW, int NTILES>
__global__ void __launch_bounds__(SW * 32)
mma_w3_lut_fused_kernel_pfa(const half*     __restrict__ A,
                            uint4*          __restrict__ APw,
                            const unsigned* __restrict__ Q,
                            const half*     __restrict__ S,
                            const W3Lut                  L,
                                  half*     __restrict__ D,
                            const int M, const int K, const int N,
                            const int group, const int nitems,
                            const int ntiles, const int ncolb,
                            int* bws)
{
    // B-stream first: the host caps the grid at ntiles, so this CTA's first
    // tile is blockIdx.x -- its weight loads can issue BEFORE the A-permute
    // phase and grid barrier and retire during them.
    W3Pf<32 / SW, NTILES> pf;
    w3_prefetch_all<SW, NTILES>(pf, Q, S, K, group, blockIdx.x % ncolb,
                                threadIdx.x >> 5, threadIdx.x & 31);

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nitems;
         idx += gridDim.x * blockDim.x) {
        ap_write_item(A, APw, idx, M, K);
    }
    __threadfence();   // publish AP writes before arriving at the barrier
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile int* cnt = bws;
        volatile int* gen = bws + 1;
        const int g0 = *gen;
        if (atomicAdd(bws, 1) == (int)gridDim.x - 1) {
            *cnt = 0;          // self-reset for the next launch
            __threadfence();
            atomicAdd(bws + 1, 1);
        } else {
            while (*gen == g0) {
            }
        }
    }
    __syncthreads();
    __threadfence();

    for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
        if (t != (int)blockIdx.x) {            // persistent tail tiles
            w3_prefetch_all<SW, NTILES>(pf, Q, S, K, group, t % ncolb,
                                        threadIdx.x >> 5, threadIdx.x & 31);
        }
        mma_tile_body_pfa<MTILES, SW, true, NTILES>(
            A, APw, L, D, M, K, N, t % ncolb, (t / ncolb) * MTILES, pf);
    }
}


// ---------------------------------------------------------------------------
// Large-M variant (M > 64). The decode kernel collapses at large M because
// (a) every CTA re-reads A per 8/16-column tile and (b) weights are
// re-dequantized once per 16-row tile. Production tops out at TileM=32, so at
// M=2048 it dequantizes every weight 64x -- the reason it loses 2x to dense
// cuBLAS there. This kernel: TileM=64 (4 mma row-tiles, halves prod's dequant
// passes), TileN = 8 warps * 8*NT cols, A staged in smem via a cp.async
// double buffer ONCE per CTA k-chunk and reused by all warps/octets, B
// streamed gmem->reg through the same fragment-order packing, fp32 HMMA, no
// K-split (M*N tiles supply the grid), direct fragment stores.
// ---------------------------------------------------------------------------
#define BIG_TM   64                    // rows per CTA (4 mma tiles)
#define BIG_PAD  8                     // smem A row pad (halves) vs bank walk
#define BIG_PITCH (128 + BIG_PAD)

__device__ __forceinline__ void
cp_async_16(void* smem_dst, const void* gmem_src)
{
    const unsigned saddr = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(saddr), "l"(gmem_src));
}

__device__ __forceinline__ void cp_commit()
{
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_wait_1()
{
    asm volatile("cp.async.wait_group 1;\n");
}

// cooperative load of one 64-row x 128-k A chunk into the staging buffer;
// rows >= M are zero-filled (exactness for any M)
__device__ __forceinline__ void
big_stage_a(half* __restrict__ buf, const half* __restrict__ A,
            int m0, int kb, int M, int K, int tid)
{
    #pragma unroll
    for (int i = 0; i < 4; ++i) {                  // 1024 16B items / 256 thr
        const int item = tid + i * 256;
        const int row  = item >> 4;
        const int kseg = (item & 15) * 8;
        half* dst = buf + row * BIG_PITCH + kseg;
        if (m0 + row < M) {
            cp_async_16(dst, A + (size_t)(m0 + row) * K + kb + kseg);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}

// FACC: mixed accumulation -- fp16 HMMA (2x rate) over 64-k spans, spilled
// into fp32 master accumulators (production's kMixed scheme; not exact).
// ablation: runtime bitmask for component microbenchmarking in ONE binary
// (no DCE / re-optimization across settings): 1 = skip B gmem load,
// 2 = skip A cp.async staging, 4 = skip dequant, 8 = skip mma.
// BPF: B-prefetch distance. true = next chunk's 96-bit records + scale load
// one full chunk ahead and the dequant double buffer carries across the
// chunk boundary (hides the ~25us B-load latency exposed at M=64-128 / NT1);
// false = B words load at chunk start, dequant prologue per chunk (the held
// prefetch registers cost occupancy on the wider NT2/NT4 shapes). The
// explicit dequant-(j+1)-before-mma-(j) double buffer is active EITHER way;
// BPF only moves the load issue point. Swept per M like nt/sw/yt.
template <int NTILES, bool FACC, bool BPF>   // column-octets per warp; CTA = 8 warps
__global__ void __launch_bounds__(256)
mma_w3_lut_big_kernel(const half*     __restrict__ A,     // (M, K)
                      const unsigned* __restrict__ Q,     // W3 word-plane stream
                      const half*     __restrict__ S,     // (N, K/group)
                      const W3Lut                  L,     // register LUT
                            half*     __restrict__ D,     // (M, N)
                      const int M, const int K, const int N, const int group,
                      const int ablation)
{
    const bool skip_bload   = ablation & 1;
    const bool skip_astage  = ablation & 2;
    const bool skip_dequant = ablation & 4;
    const bool skip_mma     = ablation & 8;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int m0   = blockIdx.y * BIG_TM;
    const int g    = lane >> 2;
    const int t2   = (lane & 3) * 2;

    __shared__ half As[2][BIG_TM * BIG_PITCH];
    __shared__ unsigned lutw[64];
    w3_stage_lut(lutw, L);
    __syncthreads();

    const int oct0 = (blockIdx.x * 8 + warp) * NTILES;  // this warp's octets
    const unsigned* q4[NTILES];
    const half*     Srow[NTILES];
    #pragma unroll
    for (int nt = 0; nt < NTILES; ++nt) {
        q4[nt]   = Q + (size_t)(oct0 + nt) * (K / 128) * 96 + lane;
        Srow[nt] = S + (size_t)((oct0 + nt) * 8 + g) * (K / group);
    }

    float acc[NTILES][4][4];   // [octet][row tile][fragment]
    #pragma unroll
    for (int nt = 0; nt < NTILES; ++nt) {
        #pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc[nt][mt][r] = 0.0f;
            }
        }
    }

    unsigned acch[NTILES][4][2];   // fp16 inner accumulators (FACC only)
    if (FACC) {
        #pragma unroll
        for (int nt = 0; nt < NTILES; ++nt) {
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                acch[nt][mt][0] = 0u;
                acch[nt][mt][1] = 0u;
            }
        }
    }

    const int nchunks = K / 128;
    if (!skip_astage) {
        big_stage_a(As[0], A, m0, 0, M, K, tid);
    }
    cp_commit();
    if (nchunks > 1 && !skip_astage) {
        big_stage_a(As[1], A, m0, 128, M, K, tid);
    }
    cp_commit();

    // B stream pipelining (see BPF above): the ablation showed the per-chunk
    // qv load right after cp_wait+sync exposed ~25us of latency at M=64-128
    // (36% of runtime, the largest single component).
    uint3 qv[NTILES];
    half2 sc2[NTILES];
    unsigned bb[2][NTILES][2];
    if constexpr (BPF) {                       // prologue for chunk 0
        #pragma unroll
        for (int nt = 0; nt < NTILES; ++nt) {
            qv[nt]  = skip_bload ? make_uint3(0u, 0u, 0u) : w3_load(q4[nt]);
            sc2[nt] = __half2half2(__ldg(&Srow[nt][0]));
            if (skip_dequant) {
                bb[0][nt][0] = 0u;
                bb[0][nt][1] = 0u;
                bb[1][nt][0] = 0u;
                bb[1][nt][1] = 0u;
            } else {
                w3_dequant_step(bb[0][nt], w3_window(qv[nt], 0), lutw, sc2[nt]);
            }
        }
    }

    for (int ci = 0; ci < nchunks; ++ci) {
        uint3 qnext[NTILES];
        half2 scnext[NTILES];
        if constexpr (BPF) {                   // next chunk, one ahead
            #pragma unroll
            for (int nt = 0; nt < NTILES; ++nt) {
                qnext[nt]  = (skip_bload || ci + 1 >= nchunks)
                           ? make_uint3(0u, 0u, 0u)
                           : w3_load(q4[nt] + (size_t)(ci + 1) * 96);
                scnext[nt] = (ci + 1 < nchunks)
                           ? __half2half2(__ldg(&Srow[nt][((ci + 1) * 128)
                                                          / group]))
                           : sc2[nt];
            }
        }
        cp_wait_1();             // chunk ci staged
        __syncthreads();
        const half* a_s = As[ci & 1];

        if constexpr (!BPF) {                  // this chunk, at chunk start
            #pragma unroll
            for (int nt = 0; nt < NTILES; ++nt) {
                qv[nt]  = skip_bload ? make_uint3(0u, 0u, 0u)
                                     : w3_load(q4[nt] + (size_t)ci * 96);
                sc2[nt] = __half2half2(__ldg(&Srow[nt][(ci * 128) / group]));
                if (skip_dequant) {
                    bb[0][nt][0] = 0u;
                    bb[0][nt][1] = 0u;
                    bb[1][nt][0] = 0u;
                    bb[1][nt][1] = 0u;
                } else {
                    w3_dequant_step(bb[0][nt], w3_window(qv[nt], 0), lutw,
                                    sc2[nt]);
                }
            }
        }

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            // A fragments for all 4 row tiles from smem (slot-relabeled:
            // a lane's 4 k's are contiguous -> one LDS.64 per row)
            unsigned afr[4][4];
            const int kl = j * 16 + (lane & 3) * 4;
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const uint2 ra = *reinterpret_cast<const uint2*>(
                    a_s + (mt * 16 + g) * BIG_PITCH + kl);
                const uint2 rb = *reinterpret_cast<const uint2*>(
                    a_s + (mt * 16 + g + 8) * BIG_PITCH + kl);
                afr[mt][0] = ra.x;
                afr[mt][1] = rb.x;
                afr[mt][2] = ra.y;
                afr[mt][3] = rb.y;
            }
            if (!skip_dequant) {
                if (j < 7) {
                    #pragma unroll
                    for (int nt = 0; nt < NTILES; ++nt) {
                        w3_dequant_step(bb[(j + 1) & 1][nt],
                                        w3_window(qv[nt], j + 1), lutw, sc2[nt]);
                    }
                } else if (BPF && ci + 1 < nchunks) {   // cross-chunk: step 0
                    #pragma unroll
                    for (int nt = 0; nt < NTILES; ++nt) {
                        w3_dequant_step(bb[0][nt], w3_window(qnext[nt], 0), lutw,
                                        scnext[nt]);
                    }
                }
            }
            if (!skip_mma) {
                #pragma unroll
                for (int nt = 0; nt < NTILES; ++nt) {
                    #pragma unroll
                    for (int mt = 0; mt < 4; ++mt) {
                        if (FACC) {
                            mma_m16n8k16_f16(acch[nt][mt], afr[mt],
                                             bb[j & 1][nt]);
                        } else {
                            mma_m16n8k16(acc[nt][mt], afr[mt], bb[j & 1][nt]);
                        }
                    }
                }
            }
            // mixed accumulation: spill the fp16 inner accumulators into the
            // fp32 masters every 64 k (4 mma steps), like production's kMixed
            if (FACC && (j & 3) == 3) {
                #pragma unroll
                for (int nt = 0; nt < NTILES; ++nt) {
                    #pragma unroll
                    for (int mt = 0; mt < 4; ++mt) {
                        const float2 f0 = __half22float2(
                            *reinterpret_cast<const half2*>(&acch[nt][mt][0]));
                        const float2 f1 = __half22float2(
                            *reinterpret_cast<const half2*>(&acch[nt][mt][1]));
                        acc[nt][mt][0] += f0.x;
                        acc[nt][mt][1] += f0.y;
                        acc[nt][mt][2] += f1.x;
                        acc[nt][mt][3] += f1.y;
                        acch[nt][mt][0] = 0u;
                        acch[nt][mt][1] = 0u;
                    }
                }
            }
        }
        __syncthreads();         // all warps done with buffer ci&1
        if (ci + 2 < nchunks && !skip_astage) {
            big_stage_a(As[ci & 1], A, m0, (ci + 2) * 128, M, K, tid);
        }
        cp_commit();             // keep group count in lockstep
        if constexpr (BPF) {
            #pragma unroll
            for (int nt = 0; nt < NTILES; ++nt) {
                qv[nt]  = qnext[nt];
                sc2[nt] = scnext[nt];
            }
        }
    }

    #pragma unroll
    for (int nt = 0; nt < NTILES; ++nt) {
        #pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const int n  = (oct0 + nt) * 8 + t2;
            const int r0 = m0 + mt * 16 + g;
            if (r0 < M) {
                D[(size_t)r0 * N + n]     = __float2half(acc[nt][mt][0]);
                D[(size_t)r0 * N + n + 1] = __float2half(acc[nt][mt][1]);
            }
            if (r0 + 8 < M) {
                D[(size_t)(r0 + 8) * N + n]     = __float2half(acc[nt][mt][2]);
                D[(size_t)(r0 + 8) * N + n + 1] = __float2half(acc[nt][mt][3]);
            }
        }
    }
}


static W3Lut make_lut(int64_t lut_lo, int64_t lut_hi)
{
    W3Lut L;
    L.lo0 = (unsigned)((uint64_t)lut_lo & 0xFFFFFFFFu);
    L.lo1 = (unsigned)((uint64_t)lut_lo >> 32);
    L.hi0 = (unsigned)((uint64_t)lut_hi & 0xFFFFFFFFu);
    L.hi1 = (unsigned)((uint64_t)lut_hi >> 32);
    return L;
}

torch::Tensor
mma_w3_lut(torch::Tensor A,      // (M, K) fp16
           torch::Tensor Q,      // (N*K*3/32,) int32 = W3 word-plane stream
           torch::Tensor S,      // (N, K/group) fp16
           int64_t lut_lo,       // bytes 0-7: lo-byte plane of the 8 entries
           int64_t lut_hi,       // bytes 0-7: hi-byte plane
           int64_t group,
           int64_t perm,         // A path: 0 = auto, 1 = direct, 2 = pre-permuted
           int64_t sw,           // warps per CTA: 0 = auto, 8, 16
           int64_t nt,           // column-octets per CTA: 0 = auto, 1, 2
           int64_t yt,           // M-split blocks (grid.y): 0 = auto, 1, 2
           int64_t pfa)          // 1 = fire-all B prefetch (decode, nui<=4)
{
    TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kHalf && A.is_contiguous());
    TORCH_CHECK(Q.is_cuda() && Q.scalar_type() == torch::kInt && Q.is_contiguous());
    TORCH_CHECK(S.is_cuda() && S.scalar_type() == torch::kHalf && S.is_contiguous());

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = S.size(0);
    TORCH_CHECK((int64_t)Q.numel() * 32 == (int64_t)N * K * 3, "packed W3 Q size mismatch");
    const W3Lut L = make_lut(lut_lo, lut_hi);
    TORCH_CHECK(S.size(1) * group == K);
    TORCH_CHECK(K % 2048 == 0, "warp K-split needs K % (128*SW) == 0");
    TORCH_CHECK(N % 8 == 0);
    TORCH_CHECK(group % 128 == 0, "scale hoist assumes one group per uint4 (128 codes)");
    TORCH_CHECK(M >= 1, "M must be positive");

    auto D = torch::empty({M, N}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    if (M > 64) {   // large-M kernel: TileM=64, smem-staged A, no K-split
        if (nt == 0) {  // measured winners: wider N tile pays off as M grows
            nt = (M <= 128) ? 1 : (M <= 256) ? 2 : 4;
        }
        // chunk-ahead B prefetch (BPF), swept winners: ON for NT1 (hides the
        // exposed B-load latency, -25% at M=64-128) and for NT4 at M<=512
        // (grid < 1 wave: 128 CTAs on 142 SMs, register cost is free); OFF
        // for NT2 and for NT4 at M>=1024 where the held prefetch registers
        // (156 REG = 1 CTA/SM) cost a second resident CTA
        const bool bpf = (nt == 1) || (nt == 4 && M <= 512);
        TORCH_CHECK(nt == 1 || nt == 2 || nt == 4, "nt must be 1, 2 or 4");
        TORCH_CHECK(N % (64 * nt) == 0 && K % 256 == 0);
        const dim3 grid(N / (64 * (int)nt), (M + 63) / 64);
#define LAUNCH_BIG(NT, BPF)                                                   \
    mma_w3_lut_big_kernel<NT, false, BPF><<<grid, 256, 0, stream>>>(          \
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),                \
        reinterpret_cast<const unsigned*>(Q.data_ptr<int32_t>()),             \
        reinterpret_cast<const half*>(S.data_ptr<at::Half>()),                \
        L,                                                                    \
        reinterpret_cast<half*>(D.data_ptr<at::Half>()),                      \
        M, K, N, (int)group, 0)
        if (nt == 1) {
            if (bpf) { LAUNCH_BIG(1, true); } else { LAUNCH_BIG(1, false); }
        } else if (nt == 2) {
            if (bpf) { LAUNCH_BIG(2, true); } else { LAUNCH_BIG(2, false); }
        } else {
            if (bpf) { LAUNCH_BIG(4, true); } else { LAUNCH_BIG(4, false); }
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return D;
    }

    const int mtiles = (M <= 16) ? 1 : (M <= 32) ? 2 : (M <= 48) ? 3 : 4;
    // perm=1 direct / perm=2 pre-permuted (extra ~1.8us launch, minimal
    // mainloop LSU work) / perm=4 TIMING-ONLY (pre-pass skipped, AP stale,
    // invalid values) / perm=5 fused single launch (permute phase + grid
    // barrier + tiles).
    // perm==0 auto: measured winners per M band (L40S, N=K=4096):
    //   M=1: direct sw8; 2-32: fused sw8 nt2; 33-48: permuted sw8 nt1;
    //   49-64: permuted sw8 nt2 yt2
    // W3 sweep winners. nt=4 (32-column CTAs) sweeps M>=8: the band's old
    // limiter was A-side L2 traffic -- every 8/16-col CTA re-read all of A,
    // ~128MB through L2 at M=64; nt4 quarters it (nt8 over-widens: only 64
    // column-blocks left -> grid starvation). perm=6 (flag-pipelined
    // fusion) wins M=3..16: the fused grid barrier costs ~1.9us -- as much
    // as the separate permute launch it replaces -- and per-piece ready
    // flags remove it (0.0235 -> 0.0216 ms at M=16). At M>=32 the barrier
    // amortizes and perm5/perm2 stay ahead.
    // ring-protocol (serving) retune, kept in lockstep with mma_w3_lut.cu
    // for fair A/B -- see comments there
    if (perm == 0) {
        if (M <= 8)       { perm = 1; sw = sw ? sw : 8; nt = nt ? nt : 1; yt = yt ? yt : 1; }
        else if (M <= 16) { perm = 1; sw = sw ? sw : 8; nt = nt ? nt : 2; yt = yt ? yt : 1; }
        else if (M <= 24) { perm = 1; sw = sw ? sw : 8; nt = nt ? nt : 4; yt = yt ? yt : 2; }
        else if (M <= 32) { perm = 5; sw = sw ? sw : 8; nt = nt ? nt : 4; yt = yt ? yt : 1; }
        else if (M <= 48) { perm = 2; sw = sw ? sw : 8; nt = nt ? nt : 4; yt = yt ? yt : 1; }
        else              { perm = 5; sw = sw ? sw : 8; nt = nt ? nt : 4; yt = yt ? yt : 2; }
    }
    const bool use_perm = (perm == 2 || perm == 4 || perm == 5 || perm == 6);
    const bool skip_prepass = (perm == 4 || perm == 5 || perm == 6);
    auto AP = use_perm
        ? torch::empty({(int64_t)mtiles * (K / 16) * 32 * 4},
                       A.options().dtype(torch::kInt))
        : torch::empty({0}, A.options().dtype(torch::kInt));

    if (use_perm && !skip_prepass) {
        const int nperm = mtiles * (K / 16) * 32;
        const int pgrid = (nperm + 255) / 256;
        a_perm_kernel<<<pgrid, 256, 0, stream>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            reinterpret_cast<uint4*>(AP.data_ptr<int32_t>()), nperm, M, K);
    }

#define LAUNCH_MMA(MT, SW, PERM, NT, YT)                                      \
    mma_w3_lut_kernel<MT, SW, PERM, NT>                                       \
        <<<dim3(N / (8 * (NT)), YT), (SW) * 32, 0, stream>>>(                 \
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),                \
        reinterpret_cast<const uint4*>(AP.data_ptr<int32_t>()),               \
        reinterpret_cast<const unsigned*>(Q.data_ptr<int32_t>()),             \
        reinterpret_cast<const half*>(S.data_ptr<at::Half>()),                \
        L,                                                                    \
        reinterpret_cast<half*>(D.data_ptr<at::Half>()),                      \
        M, K, N, (int)group)

#define DISPATCH_PERM(MT, SW, NT, YT)                                         \
    do {                                                                      \
        if (use_perm) LAUNCH_MMA(MT, SW, true, NT, YT);                       \
        else          LAUNCH_MMA(MT, SW, false, NT, YT);                      \
    } while (0)

    // taller CTAs = more resident warps per SM; NT widens the CTA's N tile
    // (A reuse); YT splits the row tiles across blocks (occupancy at high M,
    // free: rows are independent, no extra reduction)
    if (sw == 0) {
        sw = (M <= 16) ? 16 : 8;
    }
    if (nt == 0) {
        nt = (M <= 16) ? 1 : 2;
    }
    if (yt == 0) {
        yt = (mtiles == 4) ? 2 : 1;
    }
    TORCH_CHECK(K % (128 * sw) == 0, "K not divisible by 128*sw");
    TORCH_CHECK(N % (8 * nt) == 0, "N not divisible by 8*nt");
    TORCH_CHECK(nt == 1 || nt == 2 || nt == 4 || nt == 8,
                "nt must be 1, 2, 4 or 8");
    TORCH_CHECK(yt >= 1 && yt <= 4, "yt must be 1..4");
    TORCH_CHECK(mtiles % yt == 0, "yt must divide the row-tile count");
    const int mt_cta = mtiles / (int)yt;   // row tiles per CTA
    // reduction buffer = (SW-1)*32*16*MT*NT bytes; 48KB static smem cap
    TORCH_CHECK(nt < 4 || (sw == 8 && mt_cta * nt <= 12)
                       || (sw == 16 && mt_cta * nt <= 4),
                "nt=4/8 smem: sw8 needs mt_cta*nt<=12, sw16 mt_cta*nt<=4");
    // PFA holds a lane's whole B slice in registers: 32/SW chunk records
    TORCH_CHECK(!pfa || (mt_cta <= 2 && nt <= 4 && perm != 4
                         && K / (128 * (int)sw) <= 32 / (int)sw),
                "pfa needs mt_cta<=2, nt<=4, nui<=32/sw");

    if (perm == 5 || perm == 6) {
        static torch::Tensor bws;  // persistent barrier state {count, gen}
        if (!bws.defined()) {
            bws = torch::zeros({2}, A.options().dtype(torch::kInt));
        }
        const int ncolb  = N / (8 * (int)nt);
        const int ntiles = ncolb * (int)yt;
        const int nitems = mtiles * (K / 16) * 32;
        const int nsms   = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

#define LAUNCH_FUSED(MT, SWv, NT)                                             \
    do {                                                                      \
        int nb = 0;                                                           \
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(                        \
            &nb, mma_w3_lut_fused_kernel<MT, SWv, NT>, (SWv) * 32, 0);        \
        TORCH_CHECK(nb > 0, "fused kernel does not fit");                     \
        const int grid = std::min(nb * nsms, ntiles);                         \
        mma_w3_lut_fused_kernel<MT, SWv, NT><<<grid, (SWv) * 32, 0, stream>>>(\
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),            \
            reinterpret_cast<uint4*>(AP.data_ptr<int32_t>()),                 \
            reinterpret_cast<const unsigned*>(Q.data_ptr<int32_t>()),         \
            reinterpret_cast<const half*>(S.data_ptr<at::Half>()),            \
            L,                                                                \
            reinterpret_cast<half*>(D.data_ptr<at::Half>()),                  \
            M, K, N, (int)group, nitems, ntiles, ncolb,                       \
            bws.data_ptr<int32_t>());                                         \
    } while (0)

#define LAUNCH_FUSED_PFA(MT, SWv, NT)                                         \
    do {                                                                      \
        int nb = 0;                                                           \
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(                        \
            &nb, mma_w3_lut_fused_kernel_pfa<MT, SWv, NT>, (SWv) * 32, 0);    \
        TORCH_CHECK(nb > 0, "fused pfa kernel does not fit");                 \
        const int grid = std::min(nb * nsms, ntiles);                         \
        mma_w3_lut_fused_kernel_pfa<MT, SWv, NT>                              \
            <<<grid, (SWv) * 32, 0, stream>>>(                                \
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),            \
            reinterpret_cast<uint4*>(AP.data_ptr<int32_t>()),                 \
            reinterpret_cast<const unsigned*>(Q.data_ptr<int32_t>()),         \
            reinterpret_cast<const half*>(S.data_ptr<at::Half>()),            \
            L,                                                                \
            reinterpret_cast<half*>(D.data_ptr<at::Half>()),                  \
            M, K, N, (int)group, nitems, ntiles, ncolb,                       \
            bws.data_ptr<int32_t>());                                         \
    } while (0)

        if (perm == 6) {
            TORCH_CHECK(!pfa, "pfa not supported with perm=6");
            static torch::Tensor flags;   // piece-ready flags (gen-stamped)
            static int gen = 0;
            if (!flags.defined()) {
                flags = torch::zeros({1024}, A.options().dtype(torch::kInt));
            }
            const int npieces = mtiles * (K / 128);
            TORCH_CHECK(npieces <= 1024, "flag buffer too small");
            ++gen;

#define LAUNCH_FLAG(MT, SWv, NT)                                              \
    do {                                                                      \
        int nb = 0;                                                           \
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(                        \
            &nb, mma_w3_lut_flag_kernel<MT, SWv, NT>, (SWv) * 32, 0);         \
        TORCH_CHECK(nb > 0, "flag kernel does not fit");                      \
        const int grid = std::min(nb * nsms, ntiles);                         \
        mma_w3_lut_flag_kernel<MT, SWv, NT><<<grid, (SWv) * 32, 0, stream>>>( \
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),            \
            reinterpret_cast<uint4*>(AP.data_ptr<int32_t>()),                 \
            reinterpret_cast<const unsigned*>(Q.data_ptr<int32_t>()),         \
            reinterpret_cast<const half*>(S.data_ptr<at::Half>()),            \
            L,                                                                \
            reinterpret_cast<half*>(D.data_ptr<at::Half>()),                  \
            M, K, N, (int)group, npieces, ntiles, ncolb,                      \
            flags.data_ptr<int32_t>(), gen);                                  \
    } while (0)

            if (mt_cta == 1) {
                if      (sw == 16 && nt == 1) LAUNCH_FLAG(1, 16, 1);
                else if (sw == 16 && nt == 2) LAUNCH_FLAG(1, 16, 2);
                else if (sw == 16)            LAUNCH_FLAG(1, 16, 4);
                else if (nt == 1)             LAUNCH_FLAG(1, 8, 1);
                else if (nt == 2)             LAUNCH_FLAG(1, 8, 2);
                else if (nt == 4)             LAUNCH_FLAG(1, 8, 4);
                else                          LAUNCH_FLAG(1, 8, 8);
            } else if (mt_cta == 2) {
                if (nt == 1)      LAUNCH_FLAG(2, 8, 1);
                else if (nt == 2) LAUNCH_FLAG(2, 8, 2);
                else              LAUNCH_FLAG(2, 8, 4);
            } else if (mt_cta == 3) {
                if (nt == 1)      LAUNCH_FLAG(3, 8, 1);
                else if (nt == 2) LAUNCH_FLAG(3, 8, 2);
                else              LAUNCH_FLAG(3, 8, 4);
            } else {
                if (nt == 1) LAUNCH_FLAG(4, 8, 1); else LAUNCH_FLAG(4, 8, 2);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return D;
        }
        if (pfa) {
            if (mt_cta == 1) {
                if      (sw == 16 && nt == 1) LAUNCH_FUSED_PFA(1, 16, 1);
                else if (sw == 16 && nt == 2) LAUNCH_FUSED_PFA(1, 16, 2);
                else if (sw == 16)            LAUNCH_FUSED_PFA(1, 16, 4);
                else if (nt == 1)             LAUNCH_FUSED_PFA(1, 8, 1);
                else if (nt == 2)             LAUNCH_FUSED_PFA(1, 8, 2);
                else                          LAUNCH_FUSED_PFA(1, 8, 4);
            } else {
                if      (nt == 1) LAUNCH_FUSED_PFA(2, 8, 1);
                else if (nt == 2) LAUNCH_FUSED_PFA(2, 8, 2);
                else              LAUNCH_FUSED_PFA(2, 8, 4);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return D;
        }
        if (mt_cta == 1) {
            if      (sw == 16 && nt == 1) LAUNCH_FUSED(1, 16, 1);
            else if (sw == 16 && nt == 2) LAUNCH_FUSED(1, 16, 2);
            else if (sw == 16)            LAUNCH_FUSED(1, 16, 4);
            else if (nt == 1)             LAUNCH_FUSED(1, 8, 1);
            else if (nt == 2)             LAUNCH_FUSED(1, 8, 2);
            else if (nt == 4)             LAUNCH_FUSED(1, 8, 4);
            else                          LAUNCH_FUSED(1, 8, 8);
        } else if (mt_cta == 2) {
            if (nt == 1)      LAUNCH_FUSED(2, 8, 1);
            else if (nt == 2) LAUNCH_FUSED(2, 8, 2);
            else              LAUNCH_FUSED(2, 8, 4);
        } else if (mt_cta == 3) {
            if (nt == 1)      LAUNCH_FUSED(3, 8, 1);
            else if (nt == 2) LAUNCH_FUSED(3, 8, 2);
            else              LAUNCH_FUSED(3, 8, 4);
        } else {
            if (nt == 1) LAUNCH_FUSED(4, 8, 1); else LAUNCH_FUSED(4, 8, 2);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return D;
    }
#define LAUNCH_MMA_PFA(MT, SW, PERM, NT, YT)                                  \
    mma_w3_lut_kernel_pfa<MT, SW, PERM, NT>                                   \
        <<<dim3(N / (8 * (NT)), YT), (SW) * 32, 0, stream>>>(                 \
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),                \
        reinterpret_cast<const uint4*>(AP.data_ptr<int32_t>()),               \
        reinterpret_cast<const unsigned*>(Q.data_ptr<int32_t>()),             \
        reinterpret_cast<const half*>(S.data_ptr<at::Half>()),                \
        L,                                                                    \
        reinterpret_cast<half*>(D.data_ptr<at::Half>()),                      \
        M, K, N, (int)group)

#define DISPATCH_PERM_PFA(MT, SW, NT, YT)                                     \
    do {                                                                      \
        if (use_perm) LAUNCH_MMA_PFA(MT, SW, true, NT, YT);                   \
        else          LAUNCH_MMA_PFA(MT, SW, false, NT, YT);                  \
    } while (0)

#define DISPATCH_MT_PFA(MT)                                                   \
    do {                                                                      \
        if      (sw == 16 && nt == 1) DISPATCH_PERM_PFA(MT, 16, 1, yt);       \
        else if (sw == 16 && nt == 2) DISPATCH_PERM_PFA(MT, 16, 2, yt);       \
        else if (nt == 1)             DISPATCH_PERM_PFA(MT, 8, 1, yt);        \
        else if (nt == 2)             DISPATCH_PERM_PFA(MT, 8, 2, yt);        \
        else                          DISPATCH_PERM_PFA(MT, 8, 4, yt);        \
    } while (0)

#define DISPATCH_MT(MT)                                                       \
    do {                                                                      \
        if      (sw == 16 && nt == 1) DISPATCH_PERM(MT, 16, 1, yt);           \
        else if (sw == 16 && nt == 2) DISPATCH_PERM(MT, 16, 2, yt);           \
        else if (nt == 1)             DISPATCH_PERM(MT, 8, 1, yt);            \
        else if (nt == 2)             DISPATCH_PERM(MT, 8, 2, yt);            \
        else                          DISPATCH_PERM(MT, 8, 4, yt);            \
    } while (0)

    if (pfa) {
        if (mt_cta == 1) {
            if (sw == 16 && nt == 4) DISPATCH_PERM_PFA(1, 16, 4, yt);
            else                     DISPATCH_MT_PFA(1);
        } else {
            DISPATCH_MT_PFA(2);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return D;
    }
    if (mt_cta == 1) {
        if      (sw == 16 && nt == 4) DISPATCH_PERM(1, 16, 4, yt);
        else if (nt == 8)             DISPATCH_PERM(1, 8, 8, yt);
        else                          DISPATCH_MT(1);
    }
    else if (mt_cta == 2) DISPATCH_MT(2);
    else if (mt_cta == 3) DISPATCH_MT(3);
    else {  // MT4 + sw16/nt4 exceeds the 48KB static smem limit (reduction)
        TORCH_CHECK(sw == 8 && nt <= 2, "MT4 supports sw=8, nt<=2 only");
        if (nt == 1) DISPATCH_PERM(4, 8, 1, yt); else DISPATCH_PERM(4, 8, 2, yt);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return D;
}


// big kernel with the runtime ablation bitmask exposed (component
// attribution; M >= 64 allowed so the decode band can be cross-checked).
// bit 1 = skip B load, 2 = skip A staging, 4 = skip dequant, 8 = skip mma
torch::Tensor
mma_w3_big_abl(torch::Tensor A, torch::Tensor Q, torch::Tensor S,
               int64_t lut_lo, int64_t lut_hi, int64_t group,
               int64_t nt, int64_t bpf, int64_t facc, int64_t ablation)
{
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = S.size(0);
    const W3Lut L = make_lut(lut_lo, lut_hi);
    TORCH_CHECK(M >= 64 && (nt == 1 || nt == 2 || nt == 4));
    TORCH_CHECK(N % (64 * nt) == 0 && K % 256 == 0);
    auto D = torch::empty({M, N}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(N / (64 * (int)nt), (M + 63) / 64);
#define LAUNCH_BIG_ABL(NT, FACC, BPF)                                         \
    mma_w3_lut_big_kernel<NT, FACC, BPF><<<grid, 256, 0, stream>>>(           \
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),                \
        reinterpret_cast<const unsigned*>(Q.data_ptr<int32_t>()),             \
        reinterpret_cast<const half*>(S.data_ptr<at::Half>()),                \
        L,                                                                    \
        reinterpret_cast<half*>(D.data_ptr<at::Half>()),                      \
        M, K, N, (int)group, (int)ablation)
#define DISPATCH_ABL(NT)                                                      \
    do {                                                                      \
        if (facc) {                                                           \
            if (bpf) { LAUNCH_BIG_ABL(NT, true, true); }                      \
            else     { LAUNCH_BIG_ABL(NT, true, false); }                     \
        } else {                                                              \
            if (bpf) { LAUNCH_BIG_ABL(NT, false, true); }                     \
            else     { LAUNCH_BIG_ABL(NT, false, false); }                    \
        }                                                                     \
    } while (0)
    if (nt == 1) {
        DISPATCH_ABL(1);
    } else if (nt == 2) {
        DISPATCH_ABL(2);
    } else {
        DISPATCH_ABL(4);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mma_gemm", &mma_w3_lut,
          "W3 register-LUT GEMM (A, Q_w3packed, S, lut_lo, lut_hi, group_size, perm, sw, nt, yt, pfa) -> D",
          py::arg("A"), py::arg("Q"), py::arg("S"), py::arg("lut_lo"),
          py::arg("lut_hi"), py::arg("group"), py::arg("perm"),
          py::arg("sw"), py::arg("nt"), py::arg("yt"), py::arg("pfa") = 0);
    m.def("mma_big_abl", &mma_w3_big_abl,
          "big kernel, ablation bitmask (1=Bload 2=Astage 4=dequant 8=mma)");
}
