"""The tracking band over a whole video: a narrow reference window around the draft line for each
chunk, Drop-DTW inside it, the chunks glued by how deep a frame sits in its chunk."""

from __future__ import annotations

import numpy as np
import torch

from track_muxer.conform.kernel.dropdtw import (
    backtrack,
    backtrack_affine,
    drop_dtw,
    drop_dtw_affine,
    drop_dtw_affine_guard,
    drop_dtw_affine_guard_struct,
)

_BAND_CUDA = torch.cuda.is_available()       # GPU backend for the band cost; no CUDA -> CPU reference path (fallback)


def _chunk_cost(ref_w, syn_w):
    """Cost C = 1 - cos for one CHUNK. CUDA -> a GPU matmul (the chunk slice is loaded onto
    VRAM and freed right after, so memory is O(chunk), not the whole SRM -- keeping to the
    unbounded-duration invariant). No CUDA -> CPU numpy (the reference path, bit-exact).
    ref_w/syn_w are float32 numpy chunk slices. Returns np.float64, for the numba DTW. (A
    GPU float32 matmul can differ from the CPU one by about 1 LSB; that is the GPU baseline
    the edge-density guard was validated against, and the algorithm itself is identical.)"""
    if _BAND_CUDA:
        rt = torch.from_numpy(ref_w).cuda(); st = torch.from_numpy(syn_w).cuda()
        C = (1.0 - rt @ st.t()).double().cpu().numpy()
        del rt, st
        return C
    return (1.0 - ref_w @ syn_w.T).astype(np.float64)

K = 8            # thinning for the coarse pass of the band layer
MARG = 120       # band HALF-WIDTH (frames) around the draft line
CHUNK = 4000     # chunk length (synth frames)
OVERLAP = 700    # chunk overlap (> the ~500-frame max correction, so a correction fits fully inside some chunk's center)
DROP = 0.20

# Edge rescue by anchor DENSITY: the band drops a bright head/edge with a low absolute cos, even
# when that stretch is genuinely in sync. The sign of real content is a DENSE chain of monotonic
# coarse-pass anchors nearby, not cos itself. It only applies at the EDGES (first/last chunk),
# where this issue occurs; in the body, dropping stays a valve — protecting the body too caused
# false rejections on heavily stretched dubs.
OC_WIN = 60       # density window, frames (±2.5s)
OC_MIN_N = 3      # min chain anchors in the window for a frame to count as "on a dense chain" and be protected from being dropped
OC_OFF_TOL = 20   # |chain shift jump| above this means a REAL cut/insert -> don't protect the zone around it


def onchain_density(chain_aj, chain_off, N, win=OC_WIN, min_n=OC_MIN_N, off_tol=OC_OFF_TOL):
    """bool[N]: frame k is protected from being dropped by the band when it sits near (±win)
    a DENSE (>=min_n) run of anchors from the MONOTONIC chain (the coarse pass's LIS) AND
    the chain's shift there is CONSISTENT (no jump/cut nearby).
    chain_aj holds the anchors' synth frames (sorted ascending), chain_off their shift
    (ar-aj). Why consistency matters: a dub is continuous across its own frames through an
    EXCISION (the gap is only on the REFERENCE side), so density alone is blind to an
    excision and would paper over it. A real excision/insertion shows up as a JUMP in the
    chain's shift greater than off_tol -- the zone ±win around it is NOT protected. A head
    or edge with a level, unchanging shift is protected."""
    aj = np.asarray(chain_aj, np.int64); of = np.asarray(chain_off, np.float64)
    if len(aj) < min_n:
        return np.zeros(N, np.bool_)
    k = np.arange(N)
    lo = np.searchsorted(aj, k - win, side="left")
    hi = np.searchsorted(aj, k + win, side="right")
    dense = (hi - lo) >= min_n
    # Chain EVENTS (cut/insert) are shift JUMPS. Protect only the head (before the first event)
    # and the tail (after the last one) — that's where this issue occurs. Leave the middle
    # (between events) untouched: otherwise density would paper over a cut and the band would
    # not jump across it (an observed opening cut of -81 s).
    steps = np.where(np.abs(np.diff(of)) > off_tol)[0]
    if len(steps):
        first_ev = (int(aj[steps[0]]) + int(aj[steps[0] + 1])) // 2
        last_ev = (int(aj[steps[-1]]) + int(aj[steps[-1] + 1])) // 2
    else:
        first_ev, last_ev = N, 0                              # no events -> the whole track counts as head/tail
    edge = (k <= first_ev) | (k >= last_ev)
    return dense & edge


def band_align(syn, ref, off, affine=False, OPEN=0.20, EXT=0.02, DSYN=0.20, MATCH_THR=None,
               on_prog=None, chain_aj=None, chain_off=None):
    """The tracking band: for each chunk, a narrow ref window around the draft line,
    Drop-DTW, then gluing.
    affine=True -> an affine penalty for a ref drop (a sharp seam).
    MATCH_THR given -> guard: dropping a synth frame is forbidden for frames with a good
    match (base_k<=MATCH_THR).
    The FIRST window has a free left end over synth: an advert or a splash at the start falls out by
    itself, the first frame is not nailed to ref. The anchors of the coarse pass are the support.
    chain_aj given -> EDGE RESCUE BY ANCHOR DENSITY: in the first/last chunk, a frame on a
    dense anchor chain (onchain_density) is NOT dropped even at low cos (a bright, in-sync
    head or edge). Does not apply in the body. chain_aj=None keeps the behavior BIT-FOR-BIT
    identical to the plain guarded path.
    on_prog(frac) is an optional per-chunk progress callback (0..1) for the web progress bar."""
    N = len(syn); Rn = len(ref)
    onc = (onchain_density(chain_aj, chain_off, N)
           if (chain_aj is not None and chain_off is not None and MATCH_THR is not None) else None)
    pred_full = np.full(N, -2, np.int64)         # -2 = unassigned, -1 = dropped (insert/edge)
    quality = np.full(N, -1, np.int64)           # how deep a frame sits inside its chunk (for gluing)
    starts = list(range(0, N, CHUNK - OVERLAP))
    nst = max(1, len(starts))
    for ist, a in enumerate(starts):
        if on_prog is not None:
            on_prog(ist / nst)
        b = min(a + CHUNK - 1, N - 1)
        ks = np.arange(a, b + 1)
        refk = ks + off[ks]
        r1 = max(0, int(refk.min()) - MARG); r2 = min(Rn - 1, int(refk.max()) + MARG)
        if r2 - r1 < 50: continue
        syn_w = syn[a:b + 1].astype(np.float32); ref_w = ref[r1:r2 + 1].astype(np.float32)
        C = _chunk_cost(ref_w, syn_w)                # GPU if CUDA is available, else CPU (memory O(chunk))
        if affine:
            if MATCH_THR is not None:
                if onc is not None and (a == 0 or b == N - 1):     # edge chunk -> density protects the head/tail
                    st = np.ascontiguousarray(onc[a:b + 1])
                    M, Dr, BM, BDr = drop_dtw_affine_guard_struct(
                        C, OPEN, EXT, DSYN, MATCH_THR, st, a == 0)
                else:                                              # body (or no chain_aj) -> the plain path, bit-exact
                    M, Dr, BM, BDr = drop_dtw_affine_guard(C, OPEN, EXT, DSYN, MATCH_THR, a == 0)
            else:
                M, Dr, BM, BDr = drop_dtw_affine(C, OPEN, EXT, DSYN)
            pred, _, _ = backtrack_affine(M, Dr, BM, BDr, r1)
        else:
            D, B = drop_dtw(C, DROP); pred, _, _ = backtrack(D, B, r1)
        for idx, kk in enumerate(range(a, b + 1)):
            q = min(kk - a, b - kk)              # distance to the chunk edge (larger is more reliable)
            if q > quality[kk]:
                quality[kk] = q; pred_full[kk] = pred[idx]
        if b == N - 1: break
    if on_prog is not None:
        on_prog(1.0)
    return pred_full

