# -*- coding: utf-8 -*-
"""COARSE audio pass on DTW. Finds audio events (insertions/excisions) that vision does not see, on
a stream ALREADY laid out by vision (post-vision). Validated on 340 dubs (1 real event, 0 false
positives); license-free (DSP-48 band), no muq, no content-specific cues, memory-bounded.

Pipeline (_detect_one at scale HW): _benv -> windowed DSP descriptor (±HW, P points) -> _whiten
against the reference stats -> _banded_dtw (a band around the diagonal, plus colmin, plus a
SIGNAL-ADAPTIVE DIAGONAL PRIOR that pulls the path toward lag 0 ONLY where there is no match
nearby) -> (o, w) on the grid -> vision_detect.global_trend/build_curve -> _roundtrip_filter
(straighten a round trip larger than the fine band pass can reach) -> _classify. cross_scale: an
event counts as real only when HW6 and HW18 agree (short-window matcher and long-window arbiter,
|Δtc| <= 20 s).

Parameters are backed by measurements:
  AMERCE=0.04 (warp-step penalty), DIAG_PRIOR=2.0/SIG_* (signal gate, wide window diag in [2,8] x
  sig_hi in [0.40,0.50]), MINFR=80 fr (above the band layer's ±0.7 s reach), flat_fr=band's ±0.7 s
  (straightens a round trip). Sign convention: rightward = +; frame = 41.708 ms."""
import numpy as np, torch
from .maps import band as _B
from .params import FRAME, STEP
from ..progress import part
from ..kernel.dropdtw import drop_dtw_affine_guard_amerce, backtrack_affine
from .. import vision_detect as _VD
from .. import cache as _cache
from ..vision_detect import global_trend as _global_trend, build_curve as _build_curve, _wmedian

DEV = _B.DEV
# --- detector constants (distinct from band_align's 0.30/0.30) ---
HW_SHORT, HW_LONG, P = 6.0, 18.0, 32          # descriptor window (matcher/arbiter), points per window
AMERCE, DIAG_PRIOR = 0.04, 2.0                # warp-step penalty; diagonal-prior strength (range [2,8])
SIG_LO, SIG_HI, SIG_WIN = 0.20, 0.40, 40      # signal gate for the prior (sig_hi range [0.40,0.50]); smoothing ±10 s
MINFR = 80.0                                  # event size threshold, frames (> band's ±0.7 reach)
OPEN, EXT, DSYN, MATCH_THR = 0.20, 0.02, 0.20, 0.15   # affine kernel costs
CHUNK, OVERLAP, MARG = 800, 200, 160          # banded_dtw: chunk/overlap/band width (constant memory)
COARSE_FR = 2.5 * 1000.0 / FRAME              # ±2.5 s in frames (ret_tol round-trip)
BAND_FINE_FR = 0.7 * 1000.0 / FRAME           # reach of the band layer's fine pass, ±0.7 s (flat_fr round-trip)
CROSS_TOL_S = 20.0                            # cross-scale: |delta tc| <= this counts as confirmed
FPS = _B.NB48["sr"] / _B.NB48["hop"]          # env-fps 62.5 (used by classify for length)
_nfft, _hop, _sr = _B.NB48["nfft"], _B.NB48["hop"], _B.NB48["sr"]
_win = torch.hann_window(_nfft).to(DEV); _BM = _B._bands(48, 50.0, 14000.0, _sr, _nfft)
_fps_env = _sr / _hop


def _benv(x):
    """mono16 float32 → [48, Nf] band envelope in HOST memory (block by block on the device, so the
    peak is the block and not the track)."""
    return _B.track_envelope(x, _BM, _nfft, _hop, _win)


DESC_BATCH = 512          # centres per batch: the peak is set by the batch, never by the track


@torch.no_grad()
def _desc(E, hw):
    """[48,Nf] envelope in host memory → (descriptor [n, 48*P], ts[c]). The ±hw window is resampled
    to P points and flattened. GRID=STEP. Centres go in batches, and each batch lifts only the
    envelope span it reads — the device never holds the whole track."""
    Nf = E.shape[1]; dur = Nf / _fps_env; hwf = hw * _fps_env
    ts = np.arange(0, max(0.0, dur - 2 * hw), STEP) + hw
    if len(ts) == 0:
        return np.zeros((0, 48 * P), np.float32), ts
    pp = torch.linspace(0, 1, P, device=DEV)
    out = np.empty((len(ts), 48 * P), np.float32)
    for b0 in range(0, len(ts), DESC_BATCH):
        tb = ts[b0:b0 + DESC_BATCH]
        centers = torch.from_numpy(tb * _fps_env).to(DEV).float()
        wlo = (centers - hwf).floor(); whi = (centers + hwf).floor(); Wd = (whi - wlo).clamp(min=1)
        pos = (wlo[:, None] + pp[None, :] * (Wd[:, None] - 1)).clamp(0, Nf - 1)
        lo = pos.floor().long(); hi = (lo + 1).clamp(max=Nf - 1); fr = (pos - lo.float())
        e0 = int(lo.min().item()); e1 = int(hi.max().item()) + 1
        Eb = torch.from_numpy(E[:, e0:e1]).to(DEV)
        Elo = Eb[:, (lo - e0).reshape(-1)].reshape(48, len(tb), P)
        Ehi = Eb[:, (hi - e0).reshape(-1)].reshape(48, len(tb), P)
        seg = Elo * (1 - fr)[None] + Ehi * fr[None]
        out[b0:b0 + len(tb)] = seg.permute(1, 0, 2).reshape(len(tb), 48 * P).contiguous().cpu().numpy()
        del centers, pos, lo, hi, fr, Eb, Elo, Ehi, seg
    return out, ts


def _whiten(Rr, Dr):
    """Whitens descriptors against the REFERENCE'S STATISTICS, then L2-normalizes (audio cos ≈ 0.5, so normalization is needed)."""
    mu = Rr.mean(0, keepdims=True); sd = Rr.std(0, keepdims=True) + 1e-6
    wh = lambda o: ((o - mu) / sd) / (np.linalg.norm((o - mu) / sd, axis=1, keepdims=True) + 1e-8)
    return wh(Rr.astype(np.float64)), wh(Dr.astype(np.float64))


def _banded_dtw(R, Dd, off, *, amerce=AMERCE, diag_prior=DIAG_PRIOR, on_prog=None):
    """MEMORY-BOUNDED Drop-DTW (duration is not limited): a band of ±MARG around the TREND off,
    processed in CHUNK-sized pieces with OVERLAP. colmin per piece (audio cos ≈ 0.5).
    SIGNAL-ADAPTIVE DIAGONAL PRIOR applied after colmin: penalty diag_prior*|lag deviation from the
    trend|*wsig, where wsig is the weight of "no match NEARBY" (smoothed 1-colmin, gated by
    SIG_LO/SIG_HI) — it suppresses wandering in blind zones while leaving a real event free to
    move. -> pred[Nd] (ref index or -1)."""
    N = len(Dd); Rn = len(R); pred_full = np.full(N, -2, np.int64); quality = np.full(N, -1, np.int64)
    for a in range(0, N, CHUNK - OVERLAP):
        b = min(a + CHUNK - 1, N - 1); ks = np.arange(a, b + 1); refk = ks + off[ks]
        r1 = max(0, int(refk.min()) - MARG); r2 = min(Rn - 1, int(refk.max()) + MARG)
        if r2 - r1 < 50:
            if b == N - 1: break
            continue
        C = (1.0 - R[r1:r2 + 1] @ Dd[a:b + 1].T).astype(np.float64)
        colmin = C.min(0); C -= colmin[None, :]                        # relative to the frame's best match
        if diag_prior > 0:                                            # signal-adaptive prior (local)
            best = 1.0 - colmin
            sm = np.convolve(best, np.ones(SIG_WIN) / SIG_WIN, "same") if len(best) > SIG_WIN else best
            wsig = np.clip((SIG_HI - sm) / (SIG_HI - SIG_LO + 1e-9), 0.0, 1.0)   # 1 = no match nearby -> prior ON
            dev = np.arange(r1, r2 + 1)[:, None] - refk[None, :]
            C = C + diag_prior * np.abs(dev) * wsig[None, :]
        fs = (a == 0)
        M, Dr, BM, BDr = drop_dtw_affine_guard_amerce(C, OPEN, EXT, DSYN, MATCH_THR, amerce, fs)
        pred, _, _ = backtrack_affine(M, Dr, BM, BDr, r1)
        for idx, kk in enumerate(range(a, b + 1)):
            q = min(kk - a, b - kk)
            if q > quality[kk]: quality[kk] = q; pred_full[kk] = pred[idx]
        if on_prog is not None: on_prog((b + 1) / N)
        if b == N - 1: break
    pred_full[pred_full == -2] = -1
    return pred_full


def _aow(tr, sh, cs, T):
    """Anchors (reference time tr, shift sh, cos similarity cs) → (o,w) on grid T (window ±VIS_WIN, weighted median, agreement)."""
    order = np.argsort(tr); trs, shs, css = tr[order], sh[order], np.clip(cs[order], 0, None)
    o = np.full(len(T), np.nan); w = np.zeros(len(T))
    for i, t in enumerate(T):
        lo = int(np.searchsorted(trs, t - _VD.VIS_WIN)); hi = int(np.searchsorted(trs, t + _VD.VIS_WIN))
        if hi - lo < 1: continue
        ww = css[lo:hi]; ss = shs[lo:hi]; om = _wmedian(ss, ww)
        o[i] = om; w[i] = float(np.median(ww)) * float(np.mean(np.abs(ss - om) <= 2.0))
    g = ~np.isnan(o); o = np.interp(np.arange(len(T)), np.where(g)[0], o[g]) if g.any() else np.zeros(len(T))
    pos = w > 0
    if pos.any(): w = w / np.median(w[pos])
    return o, w


def _levels(cuts, curve, T, win=8.0):
    lv = []
    for tc, dv, te, tn in cuts:
        bef = curve[(T >= tc - win) & (T < tc)]; aft = curve[(T > tc) & (T <= tc + win)]
        lv.append((float(np.median(bef)) if len(bef) else np.nan, float(np.median(aft)) if len(aft) else np.nan))
    return lv


def _roundtrip_filter(cuts, curve, T, *, ret_tol_fr=COARSE_FR, flat_fr=BAND_FINE_FR, win=8.0):
    """Removes round-trip cuts (a lag drifted from the base and RETURNED to it) and straightens the
    curve. flat_fr=band's ±0.7 s: a round trip bigger than the fine band pass can reach is folded
    back into the BASE curve (band would not stretch to it; otherwise a spurious ~2 s match on the
    ending credits would remain and regress). A real correction is a permanent step that never
    returns, so it survives this filter."""
    curve = np.asarray(curve, float).copy(); n = len(cuts)
    if n < 2: return list(cuts), curve
    lv = _levels(cuts, curve, T, win); keep = [True] * n
    for k in range(n):
        if not keep[k]: continue
        base = lv[k][0]
        if not np.isfinite(base): continue
        for j in range(k + 1, n):
            la = lv[j][1]
            if np.isfinite(la) and abs(la - base) <= ret_tol_fr:
                for i in range(k, j + 1): keep[i] = False
                sp = (T > cuts[k][0]) & (T < cuts[j][0])
                mag = float(np.max(np.abs(curve[sp] - base))) if sp.any() else 0.0
                if mag > flat_fr: curve[(T >= cuts[k][0]) & (T <= cuts[j][0])] = base
                break
    return [c for i, c in enumerate(cuts) if keep[i]], curve


def _classify(cuts, tR, pred, drs, *, minfr=MINFR):
    """Turns build_curve's cuts into events: excision (drop≥.15 and |dv|≥minfr) or insertion (hwarp≥2 and length≥minfr)."""
    cutsL = []; insL = []
    for tc, dv, te, tn in cuts:
        s = int(np.searchsorted(tR, te)); e = max(int(np.searchsorted(tR, tn)), s + 1)
        drop = sum(1 for r in drs if s <= r < e) / (e - s)
        n_dub = int(((pred >= s) & (pred < e)).sum()); hw = n_dub / (e - s)
        if drop >= 0.15 and abs(dv) >= minfr:
            cutsL.append((float(tc), float(dv), float(te), float(tn)))
        elif hw >= 2.0:
            ln = max(STEP, (n_dub - (e - s)) * STEP)
            if ln >= minfr * FRAME / 1000.0: insL.append((float(tc), float(ln)))
    return cutsL, insL


def _detect_one(Rr, tR, Dr, tD, vspans, on_prog=None):
    """One scale: _whiten → _banded_dtw (diagonal prior) → (o,w) → global_trend/build_curve →
    _roundtrip_filter → _classify. post-vision: the trend off=0 (the stream is already laid out by
    vision). -> (events_cuts, events_inserts, curve[on tR], o, w)."""
    R, Dd = _whiten(Rr, Dr)
    off = np.clip(np.searchsorted(tR, tD), 0, len(tR) - 1) - np.arange(len(tD))   # ~0 (diagonal)
    pred = _banded_dtw(R, Dd, off.astype(np.int64), on_prog=on_prog)
    m = pred >= 0; di = np.where(m)[0]; ri = pred[m]; trf = tR[np.clip(ri, 0, len(tR) - 1)]
    drs = (set(range(int(ri.min()), int(ri.max()) + 1)) - set(int(x) for x in ri)) if m.any() else set()
    sh = (trf - tD[di]) * 1000.0 / FRAME
    cos = np.array([float(Dd[di[i]] @ R[np.clip(ri[i], 0, len(R) - 1)]) for i in range(len(di))])
    T = tR.copy(); o, w = _aow(trf, sh, cos, T)
    if vspans:
        for sa, sb in vspans: w[(T >= sa) & (T <= sb)] = 0.0       # vision silence: no anchors built there
    a, b = _global_trend(o, w, T, FPS)
    curve, cuts_all, _fill, _ores, _conf, _body = _build_curve(o, w, T, a, b)
    cuts_kept, curve = _roundtrip_filter(cuts_all, curve, T)
    cutsL, insL = _classify(cuts_kept, tR, pred, drs)
    return cutsL, insL, curve, o, w


def _ref_benv(ref_mono16, ref_cache):
    """The reference's band envelope with the CK4 on-disk cache (the expensive STFT is reused across
    the episode's dubs and across runs). Stored as f32 .npy, bit-for-bit exact (GPU→CPU→save→load→GPU).
    A broken cache does not fail the run — it is recomputed."""
    if ref_cache is not None:
        from pathlib import Path as _P
        p = _P(ref_cache)
        if p.exists():
            try:
                e = np.load(str(p))
                _cache._a("CK4 benv реф", True, p.name)
                return e
            except Exception:  # noqa: BLE001 -- a broken cache triggers a recompute
                pass
        _cache._a("CK4 benv реф", False, p.name)
        E = _benv(ref_mono16)
        try:
            p.parent.mkdir(parents=True, exist_ok=True); np.save(str(p), E.astype(np.float32))
        except Exception:  # noqa: BLE001
            pass
        return E
    return _benv(ref_mono16)


def detect(ref_mono16, dub_mono16, *, vspans=None, ref_cache=None, on_prog=None):
    """The FULL detector: cross-scale (HW6 matcher confirmed by the HW18 arbiter). Input is mono @16
    kHz (post-vision dub + reference). ref_cache (a .npy path) is CK4: the cache of the reference's
    band envelope (the reference is reused across dubs). -> dict: curve (frames, on the short
    window's ts grid), ts, o, w, events_cuts, events_inserts (confirmed by cross-scale)."""
    Er = _ref_benv(ref_mono16, ref_cache)
    if on_prog is not None: on_prog(0.15)
    Ed = _benv(dub_mono16)
    if on_prog is not None: on_prog(0.30)
    Rr6, tR6 = _desc(Er, HW_SHORT); Dr6, tD6 = _desc(Ed, HW_SHORT)
    cutsL, insL, curve, o, w = _detect_one(Rr6, tR6, Dr6, tD6, vspans, on_prog=part(on_prog, 0.30, 0.75))
    cand = [(t, dv, te, tn) for t, dv, te, tn in cutsL] + [(t, None, None, None) for t, ln in insL]
    conf_c, conf_i = cutsL, insL
    if cand:                                                       # cross-scale check runs lazily, only when candidates exist
        Rr18, tR18 = _desc(Er, HW_LONG); Dr18, tD18 = _desc(Ed, HW_LONG)
        lc, li, _cv, _o, _w = _detect_one(Rr18, tR18, Dr18, tD18, vspans, on_prog=part(on_prog, 0.75, 1.0))
        long_t = [t for t, dv, te, tn in lc] + [t for t, ln in li]
        ok = lambda tc: any(abs(t - tc) <= CROSS_TOL_S for t in long_t)
        conf_c = [(t, dv, te, tn) for t, dv, te, tn in cutsL if ok(t)]
        conf_i = [(t, ln) for t, ln in insL if ok(t)]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return dict(curve=curve, ts=tR6, o=o, w=w, events_cuts=conf_c, events_inserts=conf_i)
