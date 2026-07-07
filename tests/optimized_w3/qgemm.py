"""optimized_w3 -- a drop-in 3-bit LUT GEMM for the FLUTE decode path.

Keeps FLUTE's two levers of value -- the learned, non-uniform 8-entry quant map
(its accuracy contribution) and the offline-pack model -- but runs on a
Marlin-derived mainloop (deep cp.async pipeline, striped Stream-K, fat register
tiles, in-register prmt-LUT dequant) instead of FLUTE's CuTe mainloop. On the
memory-bound decode band (M<=64) this raises DRAM utilisation ~36% -> ~68% and
gives 1.1-1.5x over FLUTE's best 3-bit template, at strictly tighter accuracy.

See KERNEL_ANATOMY.md for the design and results.md for the measured numbers.

    from tests.optimized_w3 import pack_w3, pack_scales, lut_planes, qgemm_w3
    Wp  = pack_w3(codes, N, K)          # codes: (K, N) ints 0..7
    Sp  = pack_scales(scales, N, K)     # scales: (N, K/group) fp16
    lut = lut_planes(qmap)              # ONCE per layer (host sync)
    ws  = torch.zeros(N // 128 * 16, dtype=torch.int32, device=dev)
    out = qgemm_w3(A, Wp, Sp, lut, ws)  # A: (M, K) fp16 -> (M, N) fp16
"""
import os

import numpy as np
import torch
from torch.utils.cpp_extension import load as _load

from .dispatch import TUNED

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = None


def _ext():
    """Lazily JIT-compile and cache the CUDA extension."""
    global _EXT
    if _EXT is None:
        _EXT = _load(
            name="optimized_w3_cuda",
            sources=[os.path.join(_HERE, "csrc", "optimized_w3_binding.cpp"),
                     os.path.join(_HERE, "csrc", "optimized_w3_kernel.cu")],
            extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
    return _EXT


# --------------------------------------------------------------------------- #
# Offline pack: one 3-bit code per nibble (fetched word IS the prmt selector)
# + the mma-fragment interleave, so a lane's coalesced 16B load lands directly
# in its mma.sync B-fragment registers (no shared-memory round-trip for B).
# --------------------------------------------------------------------------- #
def _base_perm():
    """Inverse of the mma.sync.m16n8k16 lane->element map."""
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [2 * (i % 4), 2 * (i % 4) + 1,
                        2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])
    return np.array(perm)


_PERM = _base_perm()
_SCALE_PERM = [i + 8 * j for i in range(8) for j in range(8)]


def pack_w3(codes, N, K):
    """codes: (K, N) int tensor of 3-bit values (0..7) -> B int32 (K/16, N*2)."""
    tile = 16
    w = codes.to(torch.int64).cpu()
    w = w.reshape((K // tile, tile, N // tile, tile)).permute((0, 2, 1, 3))
    w = w.reshape((K // tile, N * tile))
    res = w.reshape((-1, _PERM.size))[:, _PERM].reshape(w.shape)
    units = res.reshape((-1, 32)).numpy().astype(np.uint64)
    q = np.zeros((units.shape[0], 4), dtype=np.uint64)
    for i in range(32):
        q[:, i // 8] |= units[:, i] << (4 * (i % 8))
    q = q.astype(np.uint32).view(np.int32)
    return torch.from_numpy(q).reshape((K // tile, N * tile // 8)).to(codes.device)


def pack_scales(S, N, K, groupsize=128):
    """S: (N, K/group) fp16 -> (K/group, N) with the scale permutation."""
    s = S.t().contiguous().cpu()
    s = s.reshape((-1, len(_SCALE_PERM)))[:, _SCALE_PERM]
    return s.reshape((-1, N)).contiguous().to(S.device)


def lut_planes(qmap):
    """(8,) fp16 quant map -> (lut_lo, lut_hi) int64 lo/hi fp16 byte planes.

    These are the prmt source operands; the learned qmap IS the table, prmt
    reads from it in registers. Does a host sync -- call ONCE per layer, never
    in the hot loop.
    """
    qb = qmap.cpu().numpy().view(np.uint8).reshape(8, 2)
    lo = int.from_bytes(qb[:, 0].tobytes(), "little")
    hi = int.from_bytes(qb[:, 1].tobytes(), "little")
    s64 = lambda x: x - (1 << 64) if x >= (1 << 63) else x
    return s64(lo), s64(hi)


# --------------------------------------------------------------------------- #
# Shape -> launch config dispatch (see dispatch.py / autotune.py)
# --------------------------------------------------------------------------- #
_NUM_SMS = None


def best_config(M, N, K, device):
    """(thread_k, thread_n, sms) for this shape: tuned table, else heuristic."""
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    mb = 16 if M <= 16 else (32 if M <= 32 else 64)
    cfg = TUNED.get(_NUM_SMS, {}).get((mb, N, K))
    if cfg is None:
        # heuristic: wide-N layers want the fat (64,256) tile even at M<=16
        cfg = (64, 256, -1) if (M > 16 or N >= 3 * K) else (128, 128, -1)
    return cfg


def qgemm_w3(A, weight, scales, lut, workspace,
             thread_k=-1, thread_n=-1, sms=-1, max_par=8):
    """3-bit LUT matmul on the optimized_w3 mainloop.

    A: (M, K) fp16; weight/scales from pack_w3/pack_scales; lut from lut_planes
    (precompute once per layer); workspace: zeros(N//128*16) int32. Leaving
    thread_k/thread_n at -1 uses the tuned dispatch for the shape.
    """
    lut_lo, lut_hi = lut
    if thread_k == -1 and thread_n == -1:
        M, K, N = A.shape[0], A.shape[1], scales.shape[1]
        thread_k, thread_n, sms = best_config(M, N, K, A.device)
    C = torch.empty((A.shape[0], scales.shape[1]), dtype=torch.half, device=A.device)
    _ext().mul(A, weight, C, scales, lut_lo, lut_hi, workspace,
               thread_k, thread_n, sms, max_par)
    return C
