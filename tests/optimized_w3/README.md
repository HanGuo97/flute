# optimized_w3

A drop-in **3-bit LUT GEMM** for the FLUTE decode path (M ≤ 64). Keeps FLUTE's
learned quant map and offline-pack model, but runs on a **Marlin-derived
mainloop** (deep cp.async pipeline, striped Stream-K, fat register tiles,
in-register prmt-LUT dequant). On the memory-bound decode band this raises DRAM
utilisation ~36% → ~68% and gives **1.1–1.6× over FLUTE's best 3-bit template**,
at strictly tighter accuracy.

- **[results.md](results.md)** — optimizations and measured numbers.
- **[KERNEL_ANATOMY.md](KERNEL_ANATOMY.md)** — how it works and why.

## Layout
```
optimized_w3/
├── __init__.py           public API
├── qgemm.py              pack_w3 / pack_scales / lut_planes / qgemm_w3 + dispatch
├── autotune.py           per-shape autotuner -> writes dispatch.py
├── dispatch.py           tuned (M,N,K) -> (thread_k, thread_n, sms) table (data)
├── test_correctness.py   exactness vs fp64 + smoke bench (both harnesses)
├── csrc/
│   ├── optimized_w3_kernel.cu    Marlin-derived mainloop + W3 prmt-LUT dequant
│   └── optimized_w3_binding.cpp
├── README.md · results.md · KERNEL_ANATOMY.md
```

## Usage
```python
import torch
from tests.optimized_w3 import pack_w3, pack_scales, lut_planes, qgemm_w3

N, K, M, group = 8192, 8192, 16, 128
Wp  = pack_w3(codes, N, K)                    # codes: (K, N) ints 0..7
Sp  = pack_scales(scales, N, K)               # scales: (N, K/group) fp16
lut = lut_planes(qmap)                        # 8-entry fp16 qmap; once per layer
ws  = torch.zeros(N // 128 * 16, dtype=torch.int32, device="cuda")
out = qgemm_w3(A, Wp, Sp, lut, ws)            # A: (M, K) fp16 -> (M, N) fp16
```
The launch config is dispatched from `dispatch.py` by shape; pass
`thread_k/thread_n/sms` explicitly to override.

## Build env (A100)
The extension JIT-compiles on first import and needs a libstdc++ providing
`GLIBCXX_3.4.29` plus CUDA:
```bash
export LD_LIBRARY_PATH=/orcd/software/core/001/spack/pkg/gcc/12.2.0/yt6vabm/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/orcd/software/core/001/pkg/cuda/12.9.1
python -m tests.optimized_w3.test_correctness
```

## Provenance
The mainloop derives from Marlin (Frantar et al., IST-DASLab, Apache-2.0 — see
the license header in `csrc/`). The 3-bit prmt-LUT dequant, the nibble +
fragment-order pack, and the FLUTE-qmap integration are the additions here.
