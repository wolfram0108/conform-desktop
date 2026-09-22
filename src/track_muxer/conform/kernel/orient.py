"""Frame orientation in SRM feature space: a horizontally mirrored dub is recovered without a
second decode, because the SRM vector of a mirrored frame follows from the original one.

SRM vector = [rk | rd], two GH×GW maps (features.build_srm). KB is symmetric, so rk of the
mirrored frame is rk with columns reversed. D1 is the first horizontal difference
rd[x] = g[x-1] - g[x] (ndimage.convolve reverses the kernel), so for g'[x] = g[W-1-x]:
rd'[x] = -rd[W-x]. Both maps are unit-normalised, which mirroring preserves.
"""

from __future__ import annotations

import numpy as np
import torch

from track_muxer.conform.kernel.coarse import CMIN, DEV, K, VHI, W

GW, GH = 128, 72                 # SRM grid (features.GW/GH); imported here to keep the kernel package standalone
PROBE_SAMPLE = 300               # dub frames sampled for the orientation probe
PROBE_BLOCK = 1 << 14            # frames per block when mirroring a full feature file


def mirror_srm(v: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Exact SRM of the horizontally mirrored frames, block-wise; `out` may be a memmap."""
    n = len(v)
    if out is None:
        out = np.empty_like(v)
    for s in range(0, n, PROBE_BLOCK):
        x = np.asarray(v[s:s + PROBE_BLOCK]).reshape(-1, 2, GH, GW)
        y = np.empty_like(x)
        y[:, 0] = x[:, 0, :, ::-1]
        y[:, 1, :, 1:] = -x[:, 1, :, :0:-1]
        y[:, 1, :, 0] = y[:, 1, :, 1]        # column 0 has no source column W; repeat the edge
        out[s:s + len(x)] = y.reshape(len(x), -1)
    return out


def _strong_matches(cs: torch.Tensor, cr: torch.Tensor) -> int:
    """Number of sampled dub frames with a reliable ref match (coarse._anchors criterion)."""
    G = cs @ cr.T
    bestv, best = G.max(1)
    offs = torch.arange(-W, W + 1, device=DEV)
    idx = (best.unsqueeze(1) + offs).clamp(0, cr.shape[0] - 1)
    G.scatter_(1, idx, -2.0)
    second = G.max(1).values
    return int(((bestv > VHI) & ((bestv - second) > CMIN)).sum().item())


def orientation_probe(srm_s: np.ndarray, srm_r: np.ndarray, n_sample: int = PROBE_SAMPLE,
                      ref_thinned: bool = False) -> dict:
    """Strong-match counts of a frame sample in each orientation against the whole (thinned) ref.
    Cheap: only the sample is mirrored, the ref matrix is shared. ref_thinned: srm_r is already
    srm_r[::K]. -> {"plain": n, "mirror": n, "sample": n}."""
    idx = np.unique(np.linspace(0, len(srm_s) - 1, min(n_sample, len(srm_s))).astype(np.int64))
    smp = np.asarray(srm_s[idx]).astype(np.float32)
    cr = torch.from_numpy(np.asarray(srm_r if ref_thinned else srm_r[::K]).astype(np.float32)).to(DEV)
    if len(idx) < 2 or cr.shape[0] < 2:
        return {"plain": 0, "mirror": 0, "sample": int(len(idx))}
    plain = _strong_matches(torch.from_numpy(smp).to(DEV), cr)
    mirror = _strong_matches(torch.from_numpy(mirror_srm(smp)).to(DEV), cr)
    return {"plain": plain, "mirror": mirror, "sample": int(len(idx))}
