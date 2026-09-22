# -*- coding: utf-8 -*-
"""Production entry point of the anchor pipeline, operating on in-memory arrays.

The signature follows the convention of align.py: mutates `out` IN PLACE along the ref's timeline,
and returns the residual in ms.

  out      -- (n,2) float32 @ sr_audio: the dub, already laid out on the ref grid (vision's output);
  ref_buf  -- (n,2) float32 @ sr_audio: the ref's audio on the same grid.

method ∈ {band, muq}: the meter of the shift. Detection is always full: the drift and the discrete
cuts, applied by a piecewise warp with silence at the joints.

Freeze is not applied in the warp: freeze is fixed at the vision layer (files without it never reach here).
Sign: rightward=+; frame=41.708 ms."""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np, torch
from scipy.ndimage import gaussian_filter1d
from .params import T as _DEFT, FRAME, SMAX, STEP, make_T
from . import detect
from .layers import layered_layout
from . import coarse_dtw                       # coarse DTW event detector (inserts/excisions outside the band window)
from .maps import band, muq, multispec
from ..interp_backend import warp_interp        # GPU/CPU block-wise linear interp (audio warps)
from ..memlog import memlog as _memlog
from ..progress import part          # memory-usage checkpoints (CONFORM_MEMLOG=1)

MAP_SR = {"band": 16000, "muq": 24000}
COARSE_LAG_S = 2.5      # window width of the coarse pass of the band layer, seconds (catches an opening-credits shift; ±0.7 s does not reach it)
# Gate against a false excision confirmed by band: coarse_dtw finds an excision on the full mix
# (cosine over a flattened descriptor), where vocal asymmetry between ref and dub (ref keeps the
# original vocal, the dub does not) can produce a false drop in a music passage -- observed as a
# 100 s phantom excision in a [109, 210] window that drove off0_ev to -20 s and produced 40 cuts.
# band is robust per band (weighted median over 48 bands) and can see true sync on the pre-DTW dub
# where the DTW cosine goes blind. An excision is rejected only when band confidently (w > GATE_W *
# median) sees a small residual (|o| < GATE_O_FR) inside its body -- the dub is actually in sync
# there, so there is no excision. A blind band (w ~ 0) never cuts, so real excisions survive. The
# gate applies to excisions only, not inserts: band sync rules out an excision, but an insert lies
# outside the band window, and gating it would falsely remove small inserts. Measured: a false
# excision at |o|=0.4, w=1.95 gets rejected by the gate; inserts are left ungated and pass through
# intact.
GATE_O_FR, GATE_W = 5.0, 0.5


def _wmedian(x, wts):
    """Weighted median (a robust statistic, no thresholds)."""
    if len(x) == 0:
        return 0.0
    o = np.argsort(x); x = x[o]; wts = wts[o]; cw = np.cumsum(wts)
    if cw[-1] <= 0:
        return float(np.median(x))
    return float(x[int(np.searchsorted(cw, 0.5 * cw[-1]))])


def _smooth_w(tt, yy, ww, win):
    """Robust smoothing: a weighted median over a window of ±win nodes (suppresses outliers). A copy
    of vision_detect._smooth_w on the local _wmedian -- the anchor package is self-contained."""
    n = len(yy); out = np.empty(n); half = max(1, win // 2)
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        out[i] = _wmedian(yy[a:b], np.maximum(ww[a:b], 1e-6))
    return out


def _zero_w_in_spans(w, T, spans):
    """Zero out band/muq anchor weight in vision's SILENCE zones (fill_spans, seconds): the dub is
    silent there, so there is nothing to compare against the ref -- a similarity anchor there is
    noise. Vision has already marked the stream; hearing does not reopen it, only avoids latching
    onto the discarded zones. Returns a copy of w."""
    if not spans:
        return w
    w = np.asarray(w, float).copy(); T = np.asarray(T, float)
    for a, b in spans:
        w[(T >= a) & (T <= b)] = 0.0
    return w


def _band_confirms_sync(o_band, w_band, T, te, tn):
    """True iff band on the pre-DTW dub CONFIDENTLY sees sync inside the body [te, tn] -> the DTW
    event is FALSE. band is robust per band (not blinded by the vocal asymmetry that trips up the
    DTW cosine). Conservative: a blind band (median w <= GATE_W) returns False (the event is NOT
    cut, so real excisions/inserts survive).

    Judged only on nodes whose band windows (±COARSE_LAG_S) lie strictly inside the body: a node at
    the body's edge measures a window overlapped by synced content outside it, so "sync"
    self-confirms off the edges -- without this restriction, a case with all 2-3 nodes touching the
    edges (audio-only, 5 s audio excision, 5.5 s body) gives median |o|=3.8<5, w=1.05 and would kill
    a real event. A short body (< 2*COARSE_LAG_S) legitimately falls outside the gate's reach ->
    False -> the event survives. A phantom excision with a ~100 s body, the original reason for
    this gate, stays caught -- it has dozens of internal nodes."""
    body = (T >= te + COARSE_LAG_S) & (T <= tn - COARSE_LAG_S)
    if not body.any():
        return False
    ref = T >= 30
    med = float(np.median(o_band[ref])) if ref.any() else 0.0
    return bool(np.median(np.abs(o_band[body] - med)) < GATE_O_FR and np.median(w_band[body]) > GATE_W)


def _coarse_off0(o_c, w_c, *, med_win_s=5.0, stab_tol=4.0, spans=None, T=_DEFT):
    """COARSE continuous off0 curve (frames on grid T) -- the trail the tracking band pass follows.
    Pure statistics over the already-computed wide measurement (o_c, w_c) = band.build_arr,
    ±COARSE_LAG_S (48 bands + weighted median PER BAND = tolerance to vocals). The caller runs this
    measurement ONCE per dub; it also serves as the judge for the DTW-excision gate
    (_band_confirms_sync). Robustness by statistics alone: an edge guard (a peak at the boundary is
    no real peak) + a neighbour-agreement gate + a weighted median. spans (seconds) are vision
    silence zones: weight is zeroed there (off0 is interpolated across them rather than latching on).
    -> off0."""
    glf = np.asarray(o_c, float); w0 = np.maximum(np.asarray(w_c, float), 1e-4)
    edge_k = COARSE_LAG_S * 1000.0 / FRAME
    w0 = np.where(np.abs(glf) > edge_k - 2.0, 0.0, w0)        # edge guard: an argmax at the ±2.5 s window edge is no real peak
    w0 = _zero_w_in_spans(w0, T, spans)                      # vision silence: off0 does not latch onto it
    half = max(1, int(round(med_win_s / STEP)))

    def _sm(weights):
        out = np.full(len(T), np.nan)
        for i in range(len(T)):
            lo, hi = max(0, i - half), min(len(T), i + half + 1)
            if weights[lo:hi].sum() > 0:
                out[i] = _wmedian(glf[lo:hi].copy(), weights[lo:hi].copy())
        return out

    m0 = _sm(w0)
    stable = (~np.isnan(m0)) & (np.abs(glf - m0) <= stab_tol)  # neighbour-agreement gate (stability)
    off0 = _sm(np.where(stable, w0, 0.0))
    ok = ~np.isnan(off0)
    off0 = np.interp(np.arange(len(T)), np.where(ok)[0], off0[ok]) if ok.any() else np.zeros(len(T))
    return off0.astype(np.float32)


def _warp_by_off0(dub_ch, sr, off0, T=_DEFT):
    """Warp a channel by the CONTINUOUS off0 curve (frames on grid T) -- a coarse correction."""
    n = len(dub_ch); t = np.arange(n) / sr
    dlt = np.interp(t, T, off0) * FRAME / 1000.0          # on grid T (cheap) -- CPU
    return warp_interp(dub_ch, (t + dlt) * sr)            # full-length warp -- GPU/CPU block-wise


def _events_outside_vision(inserts, cuts, spans, min_overlap=0.5):
    """Keep only DTW events the vision map has not already applied: a cut whose zone lies at least
    min_overlap inside vision silence, or an insert placed inside it, is vision's cut seen again by the
    audio layer (the DTW curve there is unsupported) — never an audio-only event."""
    if not spans:
        return list(inserts), list(cuts)
    def overlap(a, b):
        return sum(max(0.0, min(b, sb) - max(a, sa)) for sa, sb in spans)
    def inside(t):
        return any(sa <= t <= sb for sa, sb in spans)
    keep_c = [c for c in cuts if overlap(c[2], c[3]) < min_overlap * max(c[3] - c[2], 1e-9)]
    keep_i = [i for i in inserts if not inside(i[0])]
    return keep_i, keep_c


def _events_step_curve(curve, ts, cuts, inserts, T=_DEFT, w=None):
    """STEPWISE pre-correction of DTW EVENTS (removes inserts/shifts outside the band ±2.5 s
    window): off0_ev (frames on grid T). FLAT between events -- drift is left untouched, the coarse
    band pass below picks it up. Plateau levels are the median of the DTW curve in the segment
    between events; the reference level (0) is the segment with the most support. Sign matches off0
    (rightward=+): off0_ev = -(level of the DTW curve). Called ONLY for event tracks (where DTW
    found an insert or an excision); when there are no events, the caller does not build off0_ev
    (base=src).

    w (on grid ts) is the support of the DTW curve; w=0 where there is no sound (vision silence): the
    curve carries no data there, and a level taken from such a segment would shift the whole track."""
    cv = np.interp(np.asarray(T, float), np.asarray(ts, float), np.asarray(curve, float))
    wv = (np.interp(np.asarray(T, float), np.asarray(ts, float), np.asarray(w, float))
          if w is not None else np.ones(len(T)))
    ev = sorted([float(tc) for tc, _ in inserts] + [float(tc) for tc, _, _, _ in cuts])
    if not ev:
        return np.zeros(len(T), np.float32)
    bnds = [-1e9] + ev + [1e9]; off = np.zeros(len(T)); last = None; ref = None; ref_sup = -1.0
    for k in range(len(bnds) - 1):
        m = (T > bnds[k]) & (T <= bnds[k + 1]); sup = m & (wv > 0)
        if sup.any():
            last = float(np.median(cv[sup]))
            if float(wv[sup].sum()) > ref_sup:  # reference = the segment with the most support (the bulk)
                ref, ref_sup = last, float(wv[sup].sum())
        if m.any() and last is not None:
            off[m] = last                       # unsupported segment keeps the neighbour's level: no step
    if ref is None:
        return np.zeros(len(T), np.float32)
    return (-(off - ref)).astype(np.float32)


def _robust_drift_curve(o_total, w, cut_times, max_pct_s=1.25, T=_DEFT):
    """ROBUST fit of the drift curve (R2, replacing Nadaraya-Watson): resistant to the embedder's
    outliers on noisy data. BY SEGMENT between cuts (cut_times = production `det_cuts`):
      * confidence GATE: the level is computed only from anchors with w >= 0.5 * median(w>0); zones
        with w~=0 (band is blind: dark or repeated scenes) are not followed but bridged by
        interpolating the confident ones;
      * weighted MEDIAN (weight w^3, window of 48 nodes = 24 s) -- robust to outliers (vs. a plain
        mean);
      * GAUSSIAN sigma=3 -- rounds the median's steps into smooth ramps (tolerable to the ear);
      * speed CAP of max_pct_s %/s (the physical limit of drift), forward and backward.
    Breaks occur only at cuts (the warp STEPS there; the caller fills the joint with silence).
    Detection is left untouched. Validated against the dub cache library (1062 dubs, wobble
    25.6 -> 5.8) and confirmed visually on the plots."""
    o = np.asarray(o_total, float); w = np.asarray(w, float)
    sig = max(1.0, 3.0 / STEP)
    lim = (max_pct_s / 100.0) * STEP * 1000.0 / FRAME    # max |shift delta| between nodes (frames) at max_pct_s %/s
    gate = 0.5 * (np.median(w[w > 0]) if np.any(w > 0) else 1.0)   # confidence threshold (matches production coverage)
    cur = np.zeros(len(T)); last = 0.0
    bnds = [-1e9] + sorted(float(t) for t in cut_times) + [1e9]
    for k in range(len(bnds) - 1):                       # over segments between production cuts
        m = (T <= bnds[1]) if k == 0 else (T > bnds[k]) & (T <= bnds[k + 1])
        idx = np.where(m)[0]
        if len(idx) < 1:
            continue
        ci = idx[w[idx] >= gate]                         # confident anchors only (not w ~ 0)
        if len(ci) >= 2:
            lvl = _smooth_w(T[ci], o[ci], w[ci] ** 3, 48)   # robust level over confident anchors
            raw = np.interp(T[idx], T[ci], lvl)          # w ~ 0 zones are bridged, not followed
        elif len(ci) == 1:
            raw = np.full(len(idx), o[ci[0]])
        else:
            raw = np.full(len(idx), last)                # no support: hold the neighbour's level
        sm = gaussian_filter1d(raw, sig, mode="nearest")
        for i in range(1, len(sm)):                      # speed cap, forward pass
            sm[i] = min(max(sm[i], sm[i - 1] - lim), sm[i - 1] + lim)
        for i in range(len(sm) - 2, -1, -1):             # and backward pass
            sm[i] = min(max(sm[i], sm[i + 1] - lim), sm[i + 1] + lim)
        cur[idx] = sm; last = float(sm[-1])
    return cur


def _warp_piecewise(dub, wcurve, cut_times, sr, T=_DEFT, dst=None, on_prog=None):
    """Piecewise warp between cuts: a smooth wcurve within a piece (ramps), a sharp seam at a cut.
    cut_times=[] means a single piece (continuous, steps become ramps). The caller places silence at
    the cuts.

    dst, when given, is where the result is written; otherwise a new array is allocated. Passing an
    output buffer avoids a SECOND full copy of the track: on a 1.5 h film with 5.1 layout that would
    be 6 GB (see `_source_copy`).

    Memory does not grow with duration: indices are built per piece (`arange(s0, s1)`) rather than
    for the whole track, which would cost 2 GB of indices (`arange(n)`); a channel is fed to the
    interpolator as a WINDOW covering the requested points, instead of a copy of the whole channel.
    """
    n = len(dub)
    bnds = [0.0] + sorted(float(t) for t in cut_times) + [n / sr]
    out = np.empty_like(dub) if dst is None else dst
    # A piece spans up to the whole track; blocks keep index arrays bounded and let progress move.
    blocks = [(k, s) for k in range(len(bnds) - 1)
              for s in range(int(round(bnds[k] * sr)),
                             n if k == len(bnds) - 2 else int(round(bnds[k + 1] * sr)), _WARP_BLK)]
    for ib, (k, s0) in enumerate(blocks):
        c0, c1 = bnds[k], bnds[k + 1]
        s1 = min(s0 + _WARP_BLK, n if k == len(bnds) - 2 else int(round(c1 * sr)))
        if s1 <= s0:
            continue
        m = (T >= c0 - 0.6) & (T <= c1 + 0.6)
        tt = np.arange(s0, s1) / sr
        if m.sum() >= 2:
            dlt = np.interp(tt, T[m], wcurve[m]) * FRAME / 1000.0
        else:
            dlt = np.full(s1 - s0, float(np.interp(c0, T, wcurve)) * FRAME / 1000.0)
        src = (tt + dlt) * sr
        # Channel window covering the requested points. Bounds padded and clamped to [0, n], as in
        # conform's block-wise resample: the interpolator clamps edges the same way as over the whole
        # channel, so the values match bit for bit.
        a1 = min(max(0, int(np.floor(src.min())) - 1), n - 1)
        a2 = max(min(n, int(np.ceil(src.max())) + 2), a1 + 2)
        src_w = src - a1
        for ch in range(dub.shape[1]):
            out[s0:s1, ch] = warp_interp(dub[a1:a2, ch], src_w)
        if on_prog is not None:
            on_prog((ib + 1) / len(blocks))
    return out


_WARP_BLK = 60 * 44100   # one minute of audio per warp block

PCM_FS = 32768.0          # int16 full scale: fixed PCM -> [-1,1] conversion (matches ffmpeg s16 -> f32le)


def _mono_sr(stereo, sr_in, sr_out):
    """(n,2|n,) int16-range @ sr_in -> mono float32 [-1,1] @ sr_out (downmix + /32768 + resample).

    CONTRACT: the input is PCM at int16 range (±32768), like the `out`/`ref_buf` conform produces
    (wavfile.read s16 -> float). The conversion to [-1,1] uses a fixed scale, /PCM_FS (the same
    scale ffmpeg's f32le uses), against which the maps are calibrated. This is required:
    band/multispec use log1p(energy), which is NOT scale-invariant; MuQ expects [-1,1]."""
    import torchaudio.functional as AF
    mono = np.ascontiguousarray(stereo.mean(axis=1) if stereo.ndim == 2 else stereo, dtype=np.float32)
    t = torch.from_numpy(mono / PCM_FS)
    if sr_in != sr_out:
        t = AF.resample(t, sr_in, sr_out)
    return t.numpy().astype(np.float32)


def _eval_seglines(seglines, T=_DEFT):
    """Denoised shift (with steps) at each point of grid T, evaluated from the segment lines."""
    bnds = np.array([s[0] for s in seglines[1:]]) if len(seglines) > 1 else np.array([])
    idx = np.searchsorted(bnds, T, side="right")
    a = np.array([s[2] for s in seglines]); b = np.array([s[3] for s in seglines])
    return a[idx]*T + b[idx]


# ─────────────────────────────────────────────────────────────────────────────
# The 48-band envelope basis of the audio structure probe: the same bands and rate as the cached
# envelopes of the reference, so a cached reference is used as is.
# ─────────────────────────────────────────────────────────────────────────────
_PRE_DEV = "cuda" if torch.cuda.is_available() else "cpu"     # GPU-first, CPU-fallback
_PRE_SR, _PRE_HOP, _PRE_NFFT = 16000, 256, 2048
_PRE_FPS = _PRE_SR / _PRE_HOP                                 # 62.5 env-fps: fixed to match the basis of the CK4 reference-envelope cache
_PRE_TOL_S = 1.5                                              # window lags within this many seconds belong to one plateau
_STRUCT_HW_S, _STRUCT_STEP_S = 20.0, 15.0                     # structure probe: window ±hw / step on the laid stream
_STRUCT_FINE_HW_S = 1.0                                       # smoothing (s) of the frame-wise agreement at a jump
_STRUCT_MIN_SPAN_S = 2 * _STRUCT_HW_S                          # a plateau must span a whole window: only then do two of its windows share no sound and agree independently
_STRUCT_BATCH = 6                                             # windows per GPU batch (cc is [B, 48, block])
_STRUCT_REF_BLK = 1 << 14                                     # reference positions per correlation block
_STRUCT_ZONE_BLK = 1 << 16                                    # env frames per block of the jump refinement


def _pre_bands():
    from .maps.band import NB48 as _NB
    fb = torch.linspace(0, _PRE_SR / 2, _PRE_NFFT // 2 + 1); hi = min(_NB["fmax"], _PRE_SR / 2 - 1)
    edg = torch.logspace(np.log10(_NB["fmin"]), np.log10(hi), 49)
    BM = torch.zeros(48, _PRE_NFFT // 2 + 1)
    for b in range(48):
        BM[b, (fb >= edg[b]) & (fb < edg[b + 1])] = 1.0
    return BM.to(_PRE_DEV)


def _pre_benv(x16):
    """mono @16k float32 -> [48,Nf] z-normalised envelope IN HOST MEMORY (band.track_envelope, the
    basis of the CK4 reference-envelope cache)."""
    from .maps.band import track_envelope as _tenv
    BM = _pre_bands(); win = torch.hann_window(_PRE_NFFT).to(_PRE_DEV)
    return _tenv(x16, BM, _PRE_NFFT, _PRE_HOP, win)


@torch.no_grad()
def _structure_window_lags(er, ed, centers, hw):
    """Windows ±hw (env frames) of the laid stream at `centers`, each searched over the WHOLE ref:
    per-band argmax of the NORMALISED correlation, peak-weighted median across bands.
    -> (lag_s, conf); lag > 0: the window's sound sits later in the ref than where it lies now.

    SHAPE, NOT LOUDNESS: a match scores by normalised cross-correlation — the dot product divided
    by the norms of the window and of the reference segment under it. A plain dot product grows
    with the segment's loudness, because the envelope is normalised over the whole track and not
    per position; the loudest episode of a film then outscores the true place, every window drifts
    to a different loud spot, and those drifts agree often enough to found a plateau. Measured on a
    67-minute pair: 18 of 269 windows found their true lag by the dot product against 197 by this
    one. Forty seconds of sound are unique within a film, so the shape alone identifies them.

    DURATION LAW: the reference is correlated BLOCK BY BLOCK (overlap-save), never through one FFT
    the length of the track — that FFT was 3.1 GB of VRAM on a 90-minute film against 0.4 GB on a
    ten-minute episode. Segment norms are a running sum inside the block, so they cost the block
    and not the track.
    """
    Nf = min(er.shape[1], ed.shape[1]); er = er[:, :Nf]; ed = ed[:, :Nf]
    W = 2 * hw; n_pos = Nf - W + 1
    nfb = 1 << int(np.ceil(np.log2(min(_STRUCT_REF_BLK, n_pos) + W)))
    L = nfb - W + 1                                    # positions a block yields without wrap-around
    NB = er.shape[0]
    lags = np.full(len(centers), np.nan); conf = np.zeros(len(centers))

    for b0 in range(0, len(centers), _STRUCT_BATCH):
        ci = centers[b0:b0 + _STRUCT_BATCH]
        B = len(ci)
        D = torch.from_numpy(np.stack([ed[:, c - hw:c + hw] for c in ci])).to(_PRE_DEV)
        FD = torch.fft.rfft(D, nfb, dim=2)
        wn = D.norm(dim=2).clamp(min=1e-6)                     # [B, NB] window norms
        best = torch.full((B, NB), -float("inf"), device=_PRE_DEV)
        arg = torch.zeros((B, NB), dtype=torch.long, device=_PRE_DEV)
        for s0 in range(0, n_pos, L):
            k_end = min(L, n_pos - s0)
            seg = torch.from_numpy(er[:, s0:s0 + k_end + W - 1]).to(_PRE_DEV)
            FRb = torch.fft.rfft(seg, nfb, dim=1)
            cc = torch.fft.irfft(FRb.unsqueeze(0) * torch.conj(FD), nfb, dim=2)[:, :, :k_end]
            sq = torch.cumsum(seg.double() ** 2, dim=1)        # float64: 250k frames lose float32
            sn = (sq[:, W - 1:W - 1 + k_end]
                  - torch.nn.functional.pad(sq, (1, 0))[:, :k_end]).clamp(min=1e-12).sqrt().float()
            m, a = (cc / (wn[..., None] * sn.unsqueeze(0))).max(2)
            up = m > best
            arg = torch.where(up, a + s0, arg); best = torch.where(up, m, best)
            del seg, FRb, cc, sq, sn
        starts = torch.tensor([float(c - hw) for c in ci], device=_PRE_DEV).unsqueeze(1)
        band_lag = arg.float() - starts
        pk = best.clamp(min=0)                                 # weight: how alike the shapes are, 0..1
        order = torch.argsort(band_lag, 1)
        bl = torch.gather(band_lag, 1, order); pw = torch.gather(pk, 1, order)
        cw = torch.cumsum(pw, 1); mi_b = (cw < cw[:, -1:] * 0.5).sum(1).clamp(0, NB - 1)
        medl = bl.gather(1, mi_b[:, None]).squeeze(1)
        lags[b0:b0 + B] = (medl / _PRE_FPS).cpu().numpy()
        conf[b0:b0 + B] = pk.median(1).values.cpu().numpy()
        del D, FD, wn, best, arg
    return lags, conf


def _structure_plateaus(t_c, lags, conf, hw_s):
    """Consecutive agreeing windows -> plateaus [(t0, t1, lag_s, n_win, conf_sum)] in laid time.
    A run shorter than a whole window founds no plateau: its windows share sound and cannot confirm
    one another. Plateaus whose lags differ by no more than band's reach are one plateau (the fine
    layers absorb it)."""
    ok = np.isfinite(lags) & (conf > 0)
    runs = []; cur = None
    for i in np.where(ok)[0]:
        if cur is not None and abs(lags[i] - cur["lag"]) <= _PRE_TOL_S:
            cur["idx"].append(int(i)); cur["lag"] = _wmedian(lags[cur["idx"]], conf[cur["idx"]])
        else:
            if cur is not None:
                runs.append(cur)
            cur = {"idx": [int(i)], "lag": float(lags[i])}
    if cur is not None:
        runs.append(cur)
    merged = []
    for r in runs:
        # A plateau is founded by INDEPENDENT agreement. Windows are 2*hw long and step by less, so
        # neighbours share most of their sound and cannot confirm each other: three of them agree on
        # 10 s of common sound and a single false match carries all three. Only windows a whole
        # window apart share nothing, so the run must span at least that.
        if t_c[r["idx"][-1]] - t_c[r["idx"][0]] < _STRUCT_MIN_SPAN_S:
            continue
        if merged and abs(r["lag"] - merged[-1]["lag"]) <= COARSE_LAG_S:
            merged[-1]["idx"] += r["idx"]
            merged[-1]["lag"] = _wmedian(lags[merged[-1]["idx"]], conf[merged[-1]["idx"]])
        else:
            merged.append(r)
    return [(float(t_c[r["idx"][0]] - hw_s), float(t_c[r["idx"][-1]] + hw_s), float(r["lag"]),
             len(r["idx"]), float(conf[r["idx"]].sum())) for r in merged]


@torch.no_grad()
def _refine_jump(er, ed, lo, hi, la_f, lb_f, smooth):
    """Change point between two known lags inside env frames [lo, hi): before it the laid stream
    agrees with the ref at lag A, after it at lag B. Frame-wise band agreement (z-normalised
    envelopes) is box-smoothed over `smooth` frames; the split maximising (A before) + (B after)
    is returned as an env-frame index, or None when the zone is empty."""
    Nf = min(er.shape[1], ed.shape[1])
    lo = max(lo, 0, -la_f, -lb_f); hi = min(hi, Nf, Nf - la_f, Nf - lb_f)
    if hi - lo < 2 * smooth:
        return None
    # The zone between two plateaus can be as long as the film, so the agreement is computed in
    # blocks and only its per-frame value — one float per envelope frame — is kept (on the host).
    d = np.empty(hi - lo, np.float32)
    for z0 in range(lo, hi, _STRUCT_ZONE_BLK):
        z1 = min(hi, z0 + _STRUCT_ZONE_BLK)
        eb = torch.from_numpy(ed[:, z0:z1]).to(_PRE_DEV)
        ra = torch.from_numpy(er[:, z0 + la_f:z1 + la_f]).to(_PRE_DEV)
        rb = torch.from_numpy(er[:, z0 + lb_f:z1 + lb_f]).to(_PRE_DEV)
        d[z0 - lo:z1 - lo] = ((eb * ra).mean(0) - (eb * rb).mean(0)).cpu().numpy()
        del eb, ra, rb
    k = np.ones(smooth, np.float32) / smooth
    pad = smooth // 2
    d = np.convolve(d, k, "full")[pad:pad + (hi - lo)]  # same alignment as a padded conv1d
    c = np.cumsum(d, dtype=np.float64)
    score = 2 * c - c[-1]                              # sum(d before j) - sum(d after j)
    return int(lo + int(np.argmax(score)))


def _audio_structure(ref16, out16, T, vision_spans=None, ref_dsp=None):
    """Piecewise map of the laid stream's sound onto the ref: plateaus of constant lag found by a
    whole-track search, jumps refined to a few seconds. A single plateau within band's reach is
    nothing to do. -> None | dict(off0 [frames on T], gaps [(ref_a, ref_b)], plateaus, jumps)."""
    er = None
    if ref_dsp:
        from pathlib import Path as _P
        if _P(ref_dsp).exists():
            try:
                er = np.load(str(ref_dsp))
            except Exception:  # noqa: BLE001 — a broken cache is recomputed
                er = None
    if er is None:
        er = _pre_benv(ref16)
    ed = _pre_benv(out16)
    fps = _PRE_FPS; Nf = min(er.shape[1], ed.shape[1])
    hw = int(_STRUCT_HW_S * fps); step = int(_STRUCT_STEP_S * fps)
    centers = np.arange(hw, Nf - hw, step)
    if len(centers) < 2 or (len(centers) - 1) * _STRUCT_STEP_S < _STRUCT_MIN_SPAN_S:
        return None
    lags, conf = _structure_window_lags(er, ed, centers, hw)
    t_c = centers / fps
    for a, b in (vision_spans or []):                    # no dub sound there: the window measures nothing
        conf[(t_c >= a) & (t_c <= b)] = 0.0
    plateaus = _structure_plateaus(t_c, lags, conf, _STRUCT_HW_S)
    if not plateaus:
        return None
    if len(plateaus) == 1 and abs(plateaus[0][2]) <= COARSE_LAG_S:
        return None
    dur = Nf / fps
    # Jumps: the coarse windows straddle each boundary; the exact point is the change point between
    # the two known lags, searched from the last A centre minus a window to the first B centre plus one.
    smooth = int(_STRUCT_FINE_HW_S * fps)
    bounds = [0.0]; jumps = []
    for k in range(1, len(plateaus)):
        a_end = plateaus[k - 1][1] - _STRUCT_HW_S; b_beg = plateaus[k][0] + _STRUCT_HW_S
        la, lb = plateaus[k - 1][2], plateaus[k][2]
        # The search starts no earlier than the previous boundary: boundaries divide one timeline,
        # so they can only grow. Search windows of neighbouring jumps overlap, and without this the
        # points came back out of order -- a plateau ending before it began, dropped below in silence.
        lo = max(int((a_end - _STRUCT_HW_S) * fps), int(bounds[-1] * fps))
        j = _refine_jump(er, ed, lo, int((b_beg + _STRUCT_HW_S) * fps),
                         int(round(la * fps)), int(round(lb * fps)), smooth)
        t_j = max(j / fps if j is not None else 0.5 * (a_end + b_beg), bounds[-1])
        bounds.append(float(t_j)); jumps.append((float(t_j), float(lb - la)))
    bounds.append(dur)
    # Plateau k owns laid [bounds[k], bounds[k+1]) -> ref [.. + lag]; later plateaus override overlaps.
    ranges = []; dropped = []; kept = []
    for k, p in enumerate(plateaus):
        lag, n_win = p[2], p[3]
        ra, rb = max(0.0, bounds[k] + lag), min(dur, bounds[k + 1] + lag)
        if rb > ra:
            ranges.append((ra, rb, lag))
            kept.append((round(bounds[k], 2), round(bounds[k + 1], 2), round(lag, 3), n_win))
        else:                                  # squeezed out by its neighbours: it goes on the record
            dropped.append((round(bounds[k], 2), round(bounds[k + 1], 2), round(lag, 3), n_win))
    if not kept:
        return None
    off0 = np.full(len(T), np.nan)
    for ra, rb, lag in ranges:
        off0[(T >= ra) & (T < rb)] = -lag * 1000.0 / FRAME
    covered = ~np.isnan(off0)
    if not covered.any():
        return None
    off0 = np.interp(np.arange(len(T)), np.where(covered)[0], off0[covered])
    gaps = []; edge = 0.0
    for ra, rb, _ in sorted(ranges):
        if ra - edge >= STEP:
            gaps.append((float(edge), float(ra)))
        edge = max(edge, rb)
    if dur - edge >= STEP:
        gaps.append((float(edge), float(dur)))
    # Where one plateau's ref range hands over to the next: the lag steps there.
    handovers = [float(ra) for ra, _, _ in sorted(ranges)[1:]]
    return {"off0": off0.astype(np.float32), "gaps": gaps, "jumps": jumps, "handovers": handovers,
            "dropped": dropped,
            "plateaus": kept,
            "windows": int(len(centers)), "windows_used": int(sum(p[3] for p in kept))}


_COPY_BLK = 1 << 22          # 4M frames per chunk: the copy proceeds in blocks, so peak memory does not grow with duration


def _source_copy(out):
    """A snapshot of the sound taken BEFORE the warp. Lives next to `out`, not in RAM.

    Building it with `np.stack([out[:, c].copy() ...])` would hold a full copy of the whole track
    in memory, twice over: first a list of per-channel copies, then the `stack`. On a 1.5 h film
    with 5.1 layout that is 6 GB, peaking at 12 GB (observed process RSS spiking to 38 GB). Memory
    must not grow with duration, so the snapshot is written to disk next to `out` (itself a file)
    and copied in blocks.

    When `out` is a plain in-memory array (no disk backing), the whole array is kept in RAM
    directly. Either path produces identical values: same order, same dtype, a byte-for-byte copy.
    """
    fn = getattr(out, "filename", None)
    if fn is None:                            # not a file: keep the whole array in RAM (short tracks, tests)
        return np.stack([out[:, c].copy() for c in range(out.shape[1])], axis=1)
    d = Path(tempfile.mkdtemp(prefix="asrc_", dir=str(Path(fn).parent)))
    dst = np.memmap(d / "src.f32", dtype=np.float32, mode="w+", shape=out.shape)
    for s in range(0, out.shape[0], _COPY_BLK):
        dst[s:s + _COPY_BLK] = out[s:s + _COPY_BLK]
    dst._conform_tmpdir = str(d)              # the caller (audio_anchor) removes the directory
    return dst


def _map_channels(src_arr, fn):
    """Apply a per-channel transform `fn(channel) -> channel` to the whole track.

    `np.stack([fn(a[:, c]) for c ...], axis=1)` would hold both the list of per-channel results and
    the final array in memory at once -- two full copies of the track. Here the result is written
    per-channel, next to the source when the source lives on disk.
    """
    fn_path = getattr(src_arr, "filename", None)
    if fn_path is None:
        return np.stack([fn(src_arr[:, c]) for c in range(src_arr.shape[1])], axis=1)
    d = Path(tempfile.mkdtemp(prefix="amap_", dir=str(Path(fn_path).parent)))
    dst = np.memmap(d / "map.f32", dtype=np.float32, mode="w+", shape=src_arr.shape)
    for c in range(src_arr.shape[1]):
        dst[:, c] = fn(src_arr[:, c])
    dst._conform_tmpdir = str(d)
    return dst


def _drop_tmp(*arrs):
    """Remove the temporary files created by `_source_copy`/`_map_channels`."""
    for a in arrs:
        d = getattr(a, "_conform_tmpdir", None)
        if d:
            shutil.rmtree(d, ignore_errors=True)


def audio_anchor(out, ref_buf, fps_ref=None, *, method="band",
                 sr_audio=44100, drift_speed_pct=1.25, info=None, progress=None,
                 plot_dir=None, plot_stem=None, render_own=True, vision_spans=None, dsp_cache=None,
                 layout_map=False):
    """Map -> detect -> warp `out` in place. -> the residual, |median|, in ms.

    layout_map=True also puts the whole audio layout into info["time_map"]: the caller has no other
    layout of the pair (a dub without a video) and what follows the sound is laid by this one.

    info, when given, is filled with layout metrics (audio_cuts/max_step/coverage/span/drift...) and
    layers for the combined plot (info["band_layers"]: o/w/off0/wcurve/det_cuts/gcc). plot_dir+
    plot_stem, when given, produce raw output (npz+json). render_own=True also renders band's OWN
    PNG/HTML plot; render_own=False leaves plotting to conform, which draws the SINGLE combined plot
    (vision+audio), and band draws none of its own."""
    if progress is not None:
        progress.mark(0.0, "снимок дубля")
    n_out = out.shape[0]
    T = make_T(n_out / sr_audio)              # grid sized to the pair's actual length (any duration)
    _memlog('вход аудио-слоя')
    src = _source_copy(out)                   # dub before warping (the source), kept off RAM
    _memlog('снимок дубля')
    # ═══ STEP 1 — EVENTS (all that lies OUTSIDE the ±2.5 s band window): DTW events → gate → step ═══
    if progress is not None:
        progress.mark(0.08, "приведение к моно 16 кГц")
    ref16 = _mono_sr(ref_buf, sr_audio, 16000)       # ref@16k computed once for the whole band path (shared by
    dub16d = _mono_sr(src, sr_audio, 16000)          # DTW + the wide/fine passes + resid); the dub gets its own
    _memlog('моно 16к рефа и дубля')
    # Audio structure: the laid stream's sound may sit elsewhere in the ref than its video says
    # (a constant studio delay, or a broken encode whose sound jumps while the picture freezes).
    # A whole-track window search gives plateaus of constant lag; the stream is laid by them and
    # ref zones without dub sound become silence (filled from the ref later). One plateau within
    # band's reach is a no-op, so healthy pairs pass untouched.
    if progress is not None:
        progress.mark(0.16, "карта структуры звука")
    struct = _audio_structure(ref16, dub16d, T, vision_spans=vision_spans, ref_dsp=dsp_cache)
    if struct is not None:
        prev = src
        src = _map_channels(src, lambda ch: _warp_by_off0(ch, sr_audio, struct["off0"], T=T))
        for a, b in struct["gaps"]:
            src[int(a * sr_audio):int(b * sr_audio)] = 0.0
        _drop_tmp(prev)
        dub16d = _mono_sr(src, sr_audio, 16000)      # laid stream → coarse_dtw / wide pass measure it
        vision_spans = list(vision_spans or []) + [tuple(g) for g in struct["gaps"]]
        if info is not None:
            info["audio_structure"] = {k: struct[k] for k in ("plateaus", "jumps", "gaps", "dropped", "windows", "windows_used")}
            info["audio_global_offset_ms"] = round(-struct["plateaus"][0][2] * 1000.0, 1)
    # Coarse DTW event detector (inserts/excisions outside the ±2.5 s band window, post-vision). Sparse
    # and measured at 1 real event / 0 false positives out of 340 dubs. No event fires on most dubs, so
    # base=src and everything below runs bit-exact with the plain path. An event triggers a
    # pre-correction of the dub (remove the insert), then the same refinement follows.
    _memlog('перед детектором событий')
    if progress is not None:
        progress.mark(0.24, "поиск вставок и вырезов")
    dres = coarse_dtw.detect(ref16, dub16d, vspans=vision_spans, ref_cache=dsp_cache,
                             on_prog=part(progress, 0.24, 0.56))
    _memlog('после детектора событий')
    dtw_ins, dtw_cuts = _events_outside_vision(dres["events_inserts"], dres["events_cuts"], vision_spans)
    oc = wc = None                                   # wide band measurement, ±2.5 s (computed once per dub)
    if dtw_cuts:                                     # band confirmation gate: filter out false excisions (a drop mimicking an excision)
        oc, wc = band.build_arr(ref16, dub16d, T, maxlag=COARSE_LAG_S,
                                on_prog=part(progress, 0.56, 0.60))   # band measured on the pre-DTW dub (grid T)
        dtw_cuts = [c for c in dtw_cuts if not _band_confirms_sync(oc, wc, T, c[2], c[3])]
    if dtw_ins or dtw_cuts:
        off0_ev = _events_step_curve(dres["curve"], dres["ts"], dtw_cuts, dtw_ins, T=T, w=dres["w"])
        base = _map_channels(src, lambda ch: _warp_by_off0(ch, sr_audio, off0_ev, T=T))
        dub16b = _mono_sr(base, sr_audio, 16000)     # events shifted the dub: mono16 and the measurement are redone
        oc = wc = None
    else:
        off0_ev = None; base = src; dub16b = dub16d      # bit-exact plain path (no events)
    # ═══ STEP 2 — LINE (everything within ±2.5 s): wide pass → fine pass on the mono warp → detect → R2 ═══
    # The wide band measurement, ±2.5 s (tolerant of vocals), runs once per dub: the gate above
    # reuses the same result (a dub without events sees the same input, so the result is bit-exact).
    if progress is not None:
        progress.mark(0.56, "широкое измерение сдвига")
    if oc is None:
        oc, wc = band.build_arr(ref16, dub16b, T, maxlag=COARSE_LAG_S,
                                on_prog=part(progress, 0.56, 0.60))
    # off0 is the trail the tracking fine pass follows: pure statistics over (oc, wc).
    # vision_spans (vision silence) zero the anchor weights there: hearing does not latch onto discarded zones.
    _memlog('после широкого измерения')
    off0 = _coarse_off0(oc, wc, spans=vision_spans, T=T)
    if progress is not None:
        progress.mark(0.60, "измерение сдвига по частотным полосам" if method == "band"
                       else "измерение сдвига моделью MuQ")
    # The fine pass (±0.7 s) FOLLOWS off0: it RE-MEASURES the residual on the pre-warped dub. The cut
    # detector is co-adapted with this re-measurement (both o and w): statistics over the wide pass do
    # not replace it, as shown on the regression library. Only mono@MAP_SR is warped: the meter listens
    # to nothing else, so warping full 44.1 kHz stereo is wasted. The A/B equivalence holds on 180 dubs:
    # cuts are bit-exact in 160 of 163, with 3 borderline flips at the MIN_FR threshold.
    sr_map = MAP_SR[method]
    ref_map = ref16 if sr_map == 16000 else _mono_sr(ref_buf, sr_audio, sr_map)
    dub_map = dub16b if sr_map == 16000 else _mono_sr(base, sr_audio, sr_map)
    dub_map = _warp_by_off0(dub_map, sr_map, off0, T=T)
    _memlog('перед тонким проходом')
    want_diag = plot_dir is not None and bool(plot_stem)
    _bm = (band.build_arr(ref_map, dub_map, T, diag=want_diag, on_prog=part(progress, 0.60, 0.69))
           if method == "band" else muq.build_arr(ref_map, dub_map, T, diag=want_diag))
    if want_diag:
        o_res, w, diag = _bm
    else:
        o_res, w, diag = _bm[0], _bm[1], None
    _memlog('после тонкого прохода')
    w = _zero_w_in_spans(w, T, vision_spans)             # vision silence: fine anchors are not built there
    o = off0 + o_res                                      # full shift = coarse off0 + fine residual
    seglines, cuts = detect.detect(o, w, T=T)            # cuts = large steps (an opening-credits shift)
    det_cuts = list(cuts)
    if progress is not None:
        progress.mark(0.69, "построение кривой сдвига")
    # The drift curve and the warp: a detected cut is a break in the curve and silence at the joint.
    cut_times = [float(tc) for tc, _ in det_cuts]
    wcurve = _robust_drift_curve(o, w, cut_times, max_pct_s=drift_speed_pct, T=T)   # R2 robust fit, resistant to outliers unlike a Nadaraya-Watson kernel smoother
    # Written directly into `out` (the source lives in a separate buffer `base`): no intermediate
    # whole-track array is allocated, so peak memory excludes an extra full copy of the audio.
    if progress is not None:
        progress.mark(0.70, "перекладка звука по кривой")
    warped = _warp_piecewise(base, wcurve, cut_times, sr_audio, T=T, dst=out,
                             on_prog=part(progress, 0.70, 0.86))
    # Silence at the cuts: a continuous warp would replay sound across the joint.
    for tc, v in det_cuts:
        jms = v * FRAME
        if jms < 0:                                      # the dub lacks sound: silence of |jump|, centred
            h = abs(jms) / 1000.0 / 2.0
            a, b = int(max(0.0, tc - h) * sr_audio), int((tc + h) * sr_audio)
        else:                                            # the extra sound is cut out: a narrow seam
            a, b = int(max(0.0, tc - 0.15) * sr_audio), int((tc + 0.15) * sr_audio)
        warped[a:min(n_out, b)] = 0.0
    for _tc, _dv, te, tn in dtw_cuts:                    # a cut found by DTW: the dub has nothing for [te, tn]
        a2, b2 = int(float(te) * sr_audio), int(float(tn) * sr_audio)
        if b2 > a2:
            warped[max(0, a2):min(n_out, b2)] = 0.0
    _memlog('после варпа')
    _drop_tmp(src, base)                       # free the snapshot and the pre-correction buffer: nothing past this point needs them
    cuts = det_cuts
    # The plot line is band's drift fit (it tracks anchors o). DTW event steps (inserts/excisions) were
    # already removed by the audio pre-correction and are shown as separate markers (dtw_inserts/
    # dtw_cut_zones) rather than baked into the line -- otherwise the line would drift off the anchors
    # by the event size and stretch the scale.
    wcurve_disp = wcurve
    # residual measured by an independent multispectral pass (16k): ref@16k is reused, corr@16k is its own
    _memlog('перед замером остатка')
    if progress is not None:
        progress.mark(0.86, "замер остатка")
    corr16 = _mono_sr(out, sr_audio, 16000)
    resid = multispec.drift(ref16, corr16, T, on_prog=part(progress, 0.86, 0.94))
    m = (T >= 30); resid_fr = float(np.median(np.abs(resid[m])))   # no hardcoded upper bound: the whole length is used

    if progress is not None:
        progress.mark(0.94, "графики и сырьё")
    # --- alignment metrics (from the found cuts and the piecewise curve) ---
    sm = _eval_seglines(seglines, T=T)                   # denoised shift curve on grid T
    span_ms = float((sm.max() - sm.min()) * FRAME) if len(sm) else 0.0
    max_step_ms = float(max((abs(v) for _, v in det_cuts), default=0.0) * FRAME)
    sum_ms = float(sum(abs(v) for _, v in det_cuts) * FRAME)
    net_ms = float(sum(v for _, v in det_cuts) * FRAME)          # signed total: sum − |net| = movement that cancelled out
    drift_ms = float((sm[-1] - sm[0]) * FRAME - sum(v for _, v in det_cuts) * FRAME) if len(sm) else 0.0
    wmed = float(np.median(w[w > 0])) if np.any(w > 0) else 1.0
    coverage = float(np.mean((w / (wmed if wmed > 1e-9 else 1.0)) >= 0.5))

    # --- PLOTS: the track PNG (thumbnail) and the interactive HTML ---
    # There is no GCC witness: the fields kept for it in the raw data and in the plot layers stay empty.
    gl = gc = None
    plots = []
    if render_own and plot_dir is not None and plot_stem:
        try:
            from pathlib import Path as _P
            from . import plots as _plt
            pd = _P(plot_dir); pd.mkdir(parents=True, exist_ok=True)
            ttl = (f"{plot_stem} — {method} · резов {len(det_cuts)} · "
                   f"остаток {resid_fr * FRAME:+.0f}мс · дрейф {drift_ms:+.0f}мс · "
                   f"скорость ≤{drift_speed_pct:.2f}%/с")
            tp = pd / f"{plot_stem}__track.png"
            _plt.render_track(tp, o, w, off0, wcurve_disp, det_cuts, title=ttl, T=T)   # full warp curve (band + DTW events); T is the real grid (handles non-standard durations)
            plots.append({"kind": "track", "name": tp.name, "t": None, "v_ms": None})
            try:
                from . import plots_html as _ph
                hp = pd / f"{plot_stem}__track.html"
                _ph.render_track_html(hp, o, w, off0, gl, wcurve_disp, det_cuts, title=ttl, T=T)
                plots.append({"kind": "html", "name": hp.name, "t": None, "v_ms": None})
            except Exception:  # noqa: BLE001 -- HTML is optional (plotly); the PNG already exists
                pass
        except Exception:  # noqa: BLE001 -- plotting must never crash conform
            plots = []

    # --- Raw data next to the plot (npz+json), for investigation/filtering/isolation ---
    # Layers before collapsing: coarse off0, the warp curve, the time x lag surface, the per-band
    # breakdown (band), the independent GCC-PHAT witness (±2.5 s). Read-only, does not affect the audio.
    if want_diag:
        try:
            from pathlib import Path as _P2
            pd2 = _P2(plot_dir); pd2.mkdir(parents=True, exist_ok=True)
            npz = {"T": np.asarray(T, np.float32), "o": o.astype(np.float32),
                   "w": w.astype(np.float32), "off0": off0.astype(np.float32),
                   "wcurve": np.asarray(wcurve_disp, np.float32),
                   "dtw_inserts": (np.array(dtw_ins, np.float64) if dtw_ins else np.zeros((0, 2))),
                   "dtw_cut_zones": (np.array([(te, tn) for _t, _d, te, tn in dtw_cuts], np.float64)
                                     if dtw_cuts else np.zeros((0, 2))),
                   "o_coarse": oc.astype(np.float32), "w_coarse": wc.astype(np.float32),
                   "seglines": np.array(seglines, np.float64) if seglines else np.zeros((0, 4)),
                   "det_cuts": (np.array([(t, v) for t, v in det_cuts], np.float64)
                                if det_cuts else np.zeros((0, 2))),
                   "gcc_lag_fr": gl if gl is not None else np.zeros(0, np.float32),
                   "gcc_conf": gc if gc is not None else np.zeros(0, np.float32)}
            if diag:
                npz.update(diag)                               # surf/lags_fr/band_shift/band_prom/band_edges
            np.savez_compressed(pd2 / f"{plot_stem}__raw.npz", **npz)
            meta = {
                "method": method, "stem": plot_stem,
                "frame_ms": FRAME, "grid_step_s": STEP, "t0_s": float(T[0]),
                "n_grid": int(len(T)),
                "maxlag_band_s": float(band.MAXLAG if method == "band" else muq.MAXLAG),
                "gcc_maxlag_s": 2.5,
                "params": {"QPOW": detect.QPOW, "PEN": detect.PEN, "SMAX": SMAX,
                           "MIN_FR": detect.MIN_FR, "MSIZE_S": detect.MSIZE_S},
                "metrics": {"audio_cuts": len(det_cuts), "max_step_ms": max_step_ms,
                            "sum_ms": sum_ms, "net_ms": net_ms, "drift_ms": drift_ms, "span_ms": span_ms,
                            "coverage": coverage, "n_segments": len(seglines),
                            "resid_med_frames": resid_fr},
                "seglines": [[float(x) for x in s] for s in seglines],
                "det_cuts": [[round(float(t), 2), round(float(v), 3),
                              round(float(v * FRAME), 1)] for t, v in det_cuts],
                "arrays": {
                    "npz": f"{plot_stem}__raw.npz",
                    "T": "сетка времени, с",
                    "o": "сдвиг (argmax), кадры; правее=+",
                    "w": "уверенность band/muq, норм. по медиане",
                    "surf": "поверхность время×лаг (до argmax)",
                    "lags_fr": "ось лагов поверхности, кадры",
                    "band_shift": "[T,48] по-полосный сдвиг, кадры (только band)",
                    "band_prom": "[T,48] выраженность пика полосы (только band)",
                    "band_edges": "границы 48 полос, Гц (только band)",
                    "gcc_lag_fr": "лаг GCC-PHAT по окнам, кадры (свидетель, ±2.5с)",
                    "gcc_conf": "уверенность GCC (доля фазово-согласной полосы)",
                    "seglines": "[K,4] (t_lo,t_hi,a,b) ломаные варпа",
                    "det_cuts": "[M,2] (t_с, величина_кадры) найденные резы",
                },
            }
            (pd2 / f"{plot_stem}__raw.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            if info is not None:
                info["raw"] = f"{plot_stem}__raw.npz"
        except Exception:  # noqa: BLE001 -- raw output must never crash conform
            pass

    if info is not None:
        # layers for the combined plot (conform draws vision+audio together) -- on grid T, frames
        info["band_layers"] = {
            "T": np.asarray(T, np.float32), "o": o.astype(np.float32), "w": w.astype(np.float32),
            "off0": off0.astype(np.float32), "wcurve": np.asarray(wcurve_disp, np.float32),
            "det_cuts": [(float(t), float(v)) for t, v in det_cuts],
            # insert of dub audio (green vertical line + duration) / missing dub audio (orange zone) -- plot markers
            "dtw_inserts": [(float(tc), float(ln)) for tc, ln in dtw_ins],
            "dtw_cut_zones": [(float(te), float(tn)) for _tc, _dv, te, tn in dtw_cuts],
            "struct_gaps": [(float(a), float(b)) for a, b in (struct["gaps"] if struct else [])],
            "gcc": (gl if gl is not None else None)}
        info["anchor_method"] = method
        info["cuts"] = [(round(t, 1), round(v, 1)) for t, v in cuts]
        info["resid_med_frames"] = resid_fr
        info["audio_cuts"] = len(det_cuts)
        info["audio_max_step_ms"] = max_step_ms
        info["audio_sum_ms"] = sum_ms
        info["audio_net_ms"] = net_ms
        info["audio_drift_ms"] = drift_ms
        info["audio_span_ms"] = span_ms
        info["audio_coverage"] = coverage
        info["n_segments"] = len(seglines)
        info["plots"] = plots
    if info is not None and layout_map:
        # The drift line reads the event-corrected stream, which reads the structure-laid stream.
        layers = [(wcurve, [tc for tc, v in cuts if v >= 0],
                   [(tc - abs(v) * FRAME / 2000.0, tc + abs(v) * FRAME / 2000.0) for tc, v in cuts if v < 0]
                   + [(float(te), float(tn)) for _tc, _dv, te, tn in dtw_cuts])]
        if off0_ev is not None:
            layers.append((off0_ev, [float(tc) for tc, _ in dtw_ins] + [float(c[0]) for c in dtw_cuts], []))
        if struct is not None:
            layers.append((struct["off0"], struct["handovers"], struct["gaps"]))
        t_map, shift_s, map_cuts = layered_layout(T, layers, FRAME / 1000.0, STEP)
        info["time_map"] = {"T": t_map, "shift_s": shift_s, "cuts": map_cuts}
    if progress is not None:
        progress.mark(1.0, "готово")
    return resid_fr * FRAME


def detect_av_desync(out, ref_buf, sr_audio=44100, *, win_s=20.0, step_s=30.0,
                     maxlag_s=2.5, t0_s=60.0, big_frames=7.0, mad_max_frames=6.0):
    """Cross-modal detector of a STUDIO A/V desync (read-only, does not affect the wav).

    Idea: after the video layout, `out` is already on the ref grid. We measure the residual shift of
    the M&E audio against the ref audio with a WIDE window (PHAT, ±maxlag_s -- wider than band's
    0.7 s working window). On a healthy track it is ~0; when it is LARGE AND STABLE (little spread
    across windows), the source's audio is offset relative to its own video (the studio shifted the
    background/M&E in the mix). This is a source defect that cannot be reliably fixed (stem
    separation drags along the original speech), so it is flagged RED.

    The pattern was validated on a case with a known studio offset: median -40k frames / MAD 2.2k
    (flagged), against all 10 healthy tracks at ~0k / MAD ~0. PHAT is robust to periodicity, where
    band's NCC produces false peaks.

    -> dict(lag_frames, lag_ms, mad_frames, mad_ms, n_windows, danger) | None (too few windows)."""
    import numpy as np
    SR = 16000
    ref16 = _mono_sr(ref_buf, sr_audio, SR); out16 = _mono_sr(out, sr_audio, SR)
    n = min(len(ref16), len(out16)); dur = n / SR
    centers = np.arange(t0_s, dur - win_s, step_s)
    if len(centers) < 5:
        return None
    M = int(maxlag_s * SR); w = int(win_s * SR); nf = 1 << int(np.ceil(np.log2(2 * w)))
    lags = []
    for c in centers:
        s = int(c * SR - w // 2)
        a = ref16[s:s + w]; b = out16[s:s + w]
        if len(a) < w or len(b) < w:
            continue
        A = np.fft.rfft(a, nf); B = np.fft.rfft(b, nf)
        R = A * np.conj(B); R /= np.abs(R) + 1e-9
        cc = np.fft.irfft(R, nf)
        cc = np.concatenate([cc[-M:], cc[:M + 1]])      # lags [-M..M]
        k = int(np.argmax(cc))
        lags.append((k - M) / SR * 1000.0 / FRAME)      # shift in ref frames
    if len(lags) < 5:
        return None
    lags = np.array(lags, float)
    med = float(np.median(lags)); mad = float(np.median(np.abs(lags - med)))
    danger = (abs(med) > big_frames) and (mad < mad_max_frames)
    return {"lag_frames": med, "lag_ms": med * FRAME, "mad_frames": mad,
            "mad_ms": mad * FRAME, "n_windows": len(lags), "danger": danger}
