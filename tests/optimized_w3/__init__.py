"""optimized_w3 -- 3-bit LUT GEMM on a Marlin-derived mainloop for FLUTE decode.

Public API:
    pack_w3(codes, N, K)          offline weight pack (nibble + fragment perm)
    pack_scales(scales, N, K)     offline scale pack
    lut_planes(qmap)              learned 8-entry qmap -> prmt byte planes
    qgemm_w3(A, Wp, Sp, lut, ws)  the matmul (shape-dispatched launch config)
"""
from .qgemm import pack_w3, pack_scales, lut_planes, qgemm_w3, best_config

__all__ = ["pack_w3", "pack_scales", "lut_planes", "qgemm_w3", "best_config"]
