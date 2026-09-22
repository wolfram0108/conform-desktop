"""The coarse pass: the support is the CONSENSUS of reliable frames (lis_nd, coarse_robust). The
device is chosen by itself: cuda when present, else cpu."""

from __future__ import annotations

import numpy as np
import torch

K = 8; W = 80; VHI = 0.60; CMIN = 0.12
CWIN = 8000; COV = 2000; CSR = 4000     # windowed coarse: window / overlap / ref search half-band (frames)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def lis_nd(a):
    """Indices of the longest NON-DECREASING subsequence of a."""
    n = len(a)
    if n == 0: return []
    tails = []; back = [-1] * n
    for i in range(n):
        x = a[i]; lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if a[tails[mid]] <= x: lo = mid + 1
            else: hi = mid
        if lo > 0: back[i] = tails[lo - 1]
        if lo == len(tails): tails.append(i)
        else: tails[lo] = i
    k = tails[-1]; out = []
    while k != -1: out.append(k); k = back[k]
    return out[::-1]


def _anchors(srm_s, srm_r):
    """G=cs@cr.T -> RELIABLE anchors (synth_frame, ref_frame), LOCAL. The coarse_robust core
    (same VHI/CMIN criterion), reused by both the full and the windowed pass."""
    cs = torch.from_numpy(srm_s[::K].astype(np.float32)).to(DEV)
    cr = torch.from_numpy(srm_r[::K].astype(np.float32)).to(DEV)
    Nr = cr.shape[0]
    if cs.shape[0] < 2 or Nr < 2:
        return np.array([], np.int64), np.array([], np.int64)
    G = cs @ cr.T                                  # Ns×Nr
    bestv, best = G.max(1)
    offs = torch.arange(-W, W + 1, device=DEV)
    idx = (best.unsqueeze(1) + offs).clamp(0, Nr - 1)
    G.scatter_(1, idx, -2.0)                        # zero out the neighbourhood of best
    second = G.max(1).values
    rel = ((bestv > VHI) & ((bestv - second) > CMIN)).cpu().numpy()
    bestc = best.cpu().numpy()
    return np.where(rel)[0] * K, bestc[rel] * K     # synth frames of the anchors, their ref


def coarse_robust(srm_s, srm_r):
    """The FULL coarse pass (the whole matrix G). Memory and time grow with long files -- use
    coarse_windowed then (band-limited, matching this pass on the cases it was checked against)."""
    aj, ar = _anchors(srm_s, srm_r)
    nrel = len(aj)
    if nrel < 5:
        return np.zeros(len(srm_s), np.int64), 0, 0, np.array([], np.int64)
    keep = lis_nd(ar.tolist())                      # chain monotonic in ref
    aj2 = aj[keep]; ar2 = ar[keep]
    off = np.interp(np.arange(len(srm_s)), aj2, ar2 - aj2).astype(np.int64)
    return off, nrel, len(keep), aj2                # aj2 = synth frames of the chain (for band's edge rescue)


def coarse_windowed(srm_s, srm_r, win=CWIN, ov=COV, sr=CSR, on_prog=None):
    """The WINDOWED coarse pass: synth windows with overlap, the seed offset CARRIES OVER from
    window to window, ref search limited to a BAND of +-sr around the seed. Memory and time are
    bounded by the window size (do not grow with file length). Same anchor criterion (_anchors).
    Returns the same shape as coarse_robust.
    Matched the full pass bit-exact on 26 real/synthetic cases.
    on_prog(frac) -- optional per-window progress hook."""
    N = len(srm_s); Rn = len(srm_r)
    gj_all, gr_all = [], []
    seed = 0; a = 0
    while a < N:
        b = min(a + win, N)
        r1 = max(0, a + seed - sr); r2 = min(Rn, b + seed + sr)
        aj, ar = _anchors(srm_s[a:b], srm_r[r1:r2])
        if on_prog is not None:
            on_prog(b / N)
        if len(aj):
            gj = aj + a; gr = ar + r1
            gj_all.append(gj); gr_all.append(gr)
            seed = int(np.median(gr - gj))          # seed of the next window comes from this
        if b >= N:
            break
        a += win - ov
    if not gj_all:
        return np.zeros(N, np.int64), 0, 0, np.array([], np.int64)
    gj = np.concatenate(gj_all); gr = np.concatenate(gr_all)
    o = np.argsort(gj, kind="stable"); gj, gr = gj[o], gr[o]
    uj, ui = np.unique(gj, return_index=True); ur = gr[ui]   # dedup by synth frame (overlaps)
    keep = lis_nd(ur.tolist())                                # chain monotonic in ref
    uj, ur = uj[keep], ur[keep]
    off = np.interp(np.arange(N), uj, ur - uj).astype(np.int64)
    return off, len(gj), len(keep), uj                        # uj = synth frames of the chain (for band's edge rescue)
