"""Linear interpolation of audio at fractional points (warp/resample) — a GPU/CPU backend.

Equivalent to `np.interp(src, np.arange(len(y)), y)` with edges clamped (no extrapolation).
The conform pipeline runs several of these warps at FULL length (@44.1 kHz × channels): the
output resample of audio onto the reference grid (align) plus the band warps
(_warp_by_off0/_warp_piecewise). On CPU (`np.interp`) that is ~16-22s; on GPU the same linear
lerps run ×10 faster (measured).

The GPU branch computes in float64 with the SAME formula as np.interp (y0 + f·(y1−y0)) → bit-exact
with the CPU path (IEEE addition/multiplication are exactly commutative; dividing by a grid step
of 1.0 is exact). The CPU fallback (`np.interp`) is bit-exact by construction, since it IS
np.interp. fp64 on the warp is memory-bound, so it costs almost no extra time.

DURATION LAW: the GPU branch is CHUNKED (by `block` points) — VRAM stays constant, it does not
grow with the file's length.
GPU-FIRST LAW: CUDA available → GPU, otherwise CPU (a consumer with no GPU is never left out).
"""

from __future__ import annotations

import numpy as np
import torch

# Block window (src points). ~90s@44.1k; the block's y-window plus src is tens of MB, so VRAM
# stays constant at any length.
WARP_BLOCK = 4_000_000


def cuda_ok() -> bool:
    return torch.cuda.is_available()


def warp_interp(y, src, *, block: int = WARP_BLOCK) -> np.ndarray:
    """Linear interpolation of `y` values (on the integer grid 0..len(y)-1) at fractional points `src`.
    Clamps the edges like np.interp (no extrapolation). GPU, chunked, when CUDA is available,
    otherwise CPU (bit-exact either way).
    Returns: float32 shaped like `src`."""
    y = np.ascontiguousarray(y, dtype=np.float32)
    src = np.asarray(src, dtype=np.float64)
    n = int(y.shape[0])
    if n < 2 or not cuda_ok():
        return np.interp(src, np.arange(n), y).astype(np.float32)   # CPU fallback — bit-exact

    out = np.empty(src.shape, dtype=np.float32)
    for s0 in range(0, len(src), block):
        sb = src[s0:s0 + block]
        lo = min(max(0, int(np.floor(sb.min()))), n - 2)    # the block's y-window; clamp lo<=n-2 keeps the slice non-empty
        hi = max(min(n, int(np.ceil(sb.max())) + 2), lo + 2)  # a block entirely past the audio end clamps to the edge (like np.interp), no crash
        # float64 with the SAME formula as np.interp -> bit-exact with CPU; the final cast to float32 happens at the end
        yt = torch.from_numpy(y[lo:hi]).to("cuda", torch.float64)
        st = torch.from_numpy(sb).to("cuda")                 # src is already float64
        m = int(yt.shape[0])
        idx = torch.clamp(torch.floor(st).long() - lo, 0, m - 2)
        f = (st - lo - idx.double()).clamp_(0.0, 1.0)        # fraction in float64; clamp means no extrapolation
        y0 = yt[idx]
        out[s0:s0 + block] = (y0 + f * (yt[idx + 1] - y0)).to(torch.float32).cpu().numpy()
    return out
