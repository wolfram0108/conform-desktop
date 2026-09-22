# -*- coding: utf-8 -*-
"""Vision-layer analyzer: the sole video-map builder in conform (anchor-based layout plus cuts).
Vision reads the shift structure (plateaus/steps) more precisely than audio; the goal is to sit
on CONFIDENT anchors and place a cut only where the shift level actually changed.

Pipeline: video anchors (pred/cos) -> (o, w) on grid T (weight = cos . agree) -> removal of the
dub's real scale (`global_trend`, PAL/speed-up, no ceiling) -> `clean_cuts` (the DELTA decides:
cuts on a jump exceeding drift, transient islands bridged, edge filter) -> `build_curve` (robust
line between cuts) -> tg_s, monotonic by construction (a dub can never play backward). Cuts and
an unsupported head/tail are filled with SILENCE (spacing out content / no dub there) by the
caller; the final reference fill overlays it in sync.

Design principles:
  - CONFIDENT anchors (w>=W_CONF) are ground truth: the map must sit on them.
  - The DELTA (shift level) decides, not duration: a cut fires only on a SUSTAINED delta change;
    a delta that RETURNS (a down-up excursion) is a blind outlier -> bridged, not cut, regardless
    of how many anchors it spans.
  - An edge without support (head/tail) extrapolates the plateau and is filled with SILENCE
    (reference fill).
Sign convention: right = positive; frame = 41.708 ms.
"""
from __future__ import annotations

import numpy as np

from track_muxer.conform.anchor.params import T as GT, make_T

# Anchor grid/weight (insensitivity-plateau centers; tuned by a sweep over 36 tracks)
VIS_WIN = 0.3             # anchor aggregation window, s (plateau 0.2-0.5)
VIS_SMAX = 0.45          # slope ceiling for lines = drift SPEED limit, frames/s (playback physics)
VIS_MAX_SCALE_PCT = 15.0  # sanity guard for trusting the global slope, %/s (real PAL/NTSC <=+-5%;
                          # >15% means anchors are garbage -> scale is not removed)
VIS_SCALE_BIN_S = 25.0    # scale-estimation bin, s (plateau 15-40): suppresses anchor noise before the median

# Clean builder: 5 honest parameters
VIS_W_CONF = 0.5         # anchor confidence threshold (measurement reliability; w~=0 noise <-> w~=1 confident)
VIS_TOL = 6.0           # anchor jitter, frames (anchors jitter ~1.5 frames within a plateau)

# Adaptive curve fit BETWEEN cuts: production line (Theil-Sen) if it fits, otherwise an RDP polyline.
VIS_RDP_EPS = 3.0       # RDP: max deviation of the polyline from smoothed anchors, frames (fewest vertices)
VIS_SMOOTH_WIN = 8      # robust anchor smoothing window before RDP, nodes (suppresses outliers)
VIS_ADAPT_THR = 8.0     # adaptive: 90th-pct residual of the production line <= threshold -> line (bit-exact steps), else RDP

# Vision noise on ambiguous content (dark or repeated scenes, NTSC sources): the layout uses
# TV denoising (1D total variation / fused lasso). A short excursion, even a tall one, does not pay
# for a base jump and is absorbed; a persistent step pays and the base moves. The optimisation is
# global over the track, so neighbouring excursions cannot contaminate each other locally.
# Validated on 690 cached dubs: 0 regressions against the greedy scan, 20 fewer false cuts;
# the lambda sweep shows a wide plateau [1200,3000].
VIS_TV_LAM = 1800.0     # TV jump payoff threshold, frame*sample (plateau 1200-3000)
VIS_TV_STEP_MIN = 8.0   # |level delta| for a cut, frames (>8 removes fine TV staircasing on smooth wander)
VIS_OUT_THR = 12.0      # |o_res - B(TV)| outlier threshold -> excluded from the curve fit, frames


def _wmedian(x, wts):
    """Weighted median (a robust statistic, no thresholds)."""
    if len(x) == 0:
        return 0.0
    o = np.argsort(x); x = x[o]; wts = wts[o]; cw = np.cumsum(wts)
    if cw[-1] <= 0:
        return float(np.median(x))
    return float(x[int(np.searchsorted(cw, 0.5 * cw[-1]))])


# ── Baked 3:2 telecine: NTSC rips carry 23.976 film as 29.97 without inverse telecine, so every
#    ~5th frame is a temporal hybrid, SRM gets noisy and anchors jitter. Detection is the peak of
#    the autocorrelation of neighbour-frame dissimilarity at PERIOD 5, computed over the ready
#    128x72 SRM and therefore resolution-invariant. The reaction is a soft IVTC: the redundant
#    frame of each five is dropped, giving ~23.976 and clean matching. Non-NTSC and non-telecine
#    input is a bit-exact no-op. Proven on 132 synthetic clips and both seasons of an NTSC-sourced series.
VIS_TC_NTSC = (29.0, 30.5)      # NTSC fps range -> candidate for 3:2 telecine (PAL/film has no period-5)
VIS_TC_FLOOR = 0.05             # min period-5 prominence (guards against accidental argmax=5 on noise)
_TC_SAMPLE = 6000               # fixed detection sample size, frames (memory O(const), does not grow with length)


def _tele_signature(srm):
    """(period-5 prominence of the neighbour-frame dissimilarity autocorrelation over background,
    argmax over 2..12), on a fixed-size sample."""
    n = len(srm)
    off = min(_TC_SAMPLE, n // 4)
    a = np.asarray(srm[off:off + _TC_SAMPLE], np.float32)
    if len(a) < 600:
        a = np.asarray(srm[:_TC_SAMPLE], np.float32)
    if len(a) < 600:
        return 0.0, 0
    c = np.sum(a[:-1] * a[1:], axis=1)
    x = (1.0 - c).astype(np.float64); x = x - x.mean()
    if x.std() < 1e-9:
        return 0.0, 0
    ac = np.correlate(x, x, "full")[len(x) - 1:]; ac = ac / (ac[0] + 1e-12)
    g = lambda p: float(ac[p]) if p < len(ac) else 0.0       # noqa: E731
    return float(max(g(5), g(10)) - float(np.mean([g(3), g(4), g(6), g(7)]))), int(np.argmax(ac[2:13]) + 2)


def is_baked_telecine(srm, fps):
    """Cheap (fixed-size sample, NO thinning): returns (telecine, tele_score, argmax). The gate is
    NTSC fps AND argmax at period 5 AND prominence >= floor. Used for the CK3/passport gate
    before matching."""
    n = 0 if srm is None else len(srm)
    if n < 1000 or not (VIS_TC_NTSC[0] <= float(fps) <= VIS_TC_NTSC[1]):
        return False, 0.0, 0
    ts, am = _tele_signature(srm)
    return (am == 5 and ts >= VIS_TC_FLOOR), round(ts, 4), am


def _cadence_keep(srm, blk=4096):
    """Keep-frame mask: in each window of 5, drop the most redundant frame (max cos to the
    previous frame = a pulldown repeat). Neighbour similarity is computed block by block
    (memory O(block))."""
    n = len(srm)
    d = np.empty(n, np.float32); d[0] = -1.0
    for s in range(0, n, blk):
        lo = max(0, s - 1)
        a = np.asarray(srm[lo:s + blk], np.float32)
        if len(a) < 2:
            continue
        dd = np.sum(a[1:] * a[:-1], axis=1)
        d[lo + 1:lo + 1 + len(dd)] = dd
    keep = np.ones(n, bool)
    for w in range(0, n, 5):
        idx = [i for i in range(w, min(w + 5, n)) if i >= 1]
        if len(idx) >= 5:
            keep[max(idx, key=lambda i: d[i])] = False
    return keep


def detelecine(srm, fps, *, scratch=None):
    """Baked 3:2 telecine (NTSC): thin the cadence at the level of SRM features (a soft IVTC to
    ~23.976) for clean matching; otherwise a no-op returning the same srm and fps bit for bit.
    With `scratch` (the owner of the run's temporary files) the thinned SRM is a file-backed
    array, memory O(block); otherwise RAM. -> (srm, fps, info)."""
    is_tc, ts, am = is_baked_telecine(srm, fps)
    info = {"telecine": is_tc, "tele_score": ts, "argmax": am, "dropped": 0}
    if not is_tc:
        return srm, fps, info
    n = len(srm); keep = _cadence_keep(srm); n2 = int(keep.sum()); D = int(srm.shape[1])
    if scratch is not None:
        mp = scratch.sub("detc") / "f.f16"
        out = np.memmap(mp, dtype=np.float16, mode="w+", shape=(n2, D)); j = 0
        for s in range(0, n, 8192):
            sel = np.asarray(srm[s:s + 8192])[keep[s:s + 8192]]
            out[j:j + len(sel)] = sel; j += len(sel)
        out.flush(); del out
        srm2 = np.memmap(mp, dtype=np.float16, mode="r", shape=(n2, D))
    else:
        srm2 = np.asarray(srm)[keep]
    info["dropped"] = int(n - n2)
    return srm2, float(fps) * n2 / n, info


def vision_ow(pred, asg, cos, fps_ref, fps_dub, *, win_s=VIS_WIN, tol_fr=2.0, T=GT,
              ax_ref=None, ax_dub=None):
    """Video anchors -> (o, w) on the params.T grid. o = weighted median shift (weight=cos) within
    a window of +-win_s; w = median(cos) times the fraction of agreeing anchors
    (|shift-o|<=tol_fr, the band-layer's "agree" analogue). Empty nodes: o is interpolated, w=0.
    w is normalized by the median of its positive values.

    ax_ref/ax_dub is the frame time axis as actually measured (SrmFeatures.pts, VFR only): frame
    time is then ax[index] rather than index/fps -- on VFR content index/fps is off by up to
    hundreds of seconds (observed drift: 713 s). None falls back to index/fps, bit-exact with that
    formula. This is the only place where frame indices turn into time for the layout; fps_ref
    downstream is only the unit "reference frames" (it cancels out in build_map: multiplied by
    fps_ref here, divided by fps_ref in tg_s)."""
    t_ref = (ax_ref[pred[asg]] if ax_ref is not None
             else pred[asg].astype(np.float64) / fps_ref)
    t_dub = (ax_dub[asg] if ax_dub is not None
             else asg.astype(np.float64) / fps_dub)
    sh = (t_dub - t_ref) * fps_ref                       # shift in reference frames (right = +)
    cs = np.asarray(cos, np.float64)
    order = np.argsort(t_ref)
    trs, shs, css = t_ref[order], sh[order], cs[order]
    o = np.full(len(T), np.nan); w = np.zeros(len(T))
    for i, t in enumerate(T):
        lo = int(np.searchsorted(trs, t - win_s)); hi = int(np.searchsorted(trs, t + win_s))
        if hi - lo < 1:
            continue
        ww = np.maximum(css[lo:hi], 0.0); ss = shs[lo:hi]
        om = _wmedian(ss, ww)
        agree = float(np.mean(np.abs(ss - om) <= tol_fr))
        o[i] = om; w[i] = float(np.median(ww)) * agree
    g = ~np.isnan(o)
    if g.any():
        o = np.interp(np.arange(len(T)), np.where(g)[0], o[g])
    else:
        o[:] = 0.0
    pos = w > 0
    if pos.any():
        w = w / np.median(w[pos])
    return o.astype(np.float64), w.astype(np.float64)


def global_trend(o, w, T, fps_ref, *, bin_s=VIS_SCALE_BIN_S, max_pct=VIS_MAX_SCALE_PCT):
    """The GLOBAL layout slope = the dub's REAL scale (PAL/NTSC/any speed-up), read FROM the data,
    with no ceiling and no dependence on fps. Robust to both cuts and anchor noise:
      1) coarse BINS of ~bin_s, weighted MEDIAN of o per bin -> SUPPRESSES NOISE (on a noisy dub
         a per-step drift of ~0.3 frame would otherwise drown in the jitter of neighbouring grid
         nodes);
      2) MEDIAN of NEIGHBOURING bins' slopes -> a cut is a spike confined to one bin pair, and the
         median cuts it out.
    Rationale: a plain median of local node slopes is fragile to noise (measured: 0% recovered on
    a noisy source); least squares/`wfit` with wide pairs let cuts pull the fit (measured: -9.4%
    on a stepped dub). Binning plus neighbour-median is robust to both failure modes.
    Sanity guard: |slope| > max_pct %/s means the anchors are garbage -> (0, 0). Returns: (a, b)
    of the line a*T+b."""
    o = np.asarray(o, float); w = np.asarray(w, float); T = np.asarray(T, float)
    if len(T) < 3:
        return 0.0, 0.0
    dur = float(T[-1] - T[0])
    nb = max(2, int(dur / bin_s)) if dur > 0 else 0
    edges = np.linspace(T[0], T[-1], nb + 1) if nb >= 2 else None
    tc: list[float] = []; oc: list[float] = []; wc: list[float] = []
    if edges is not None:
        for i in range(nb):
            m = (T >= edges[i]) & (T < edges[i + 1]) & (w > 0)
            if int(m.sum()) >= 3:
                tc.append(0.5 * (edges[i] + edges[i + 1]))
                oc.append(_wmedian(o[m], w[m]))          # bin level (weighted median -- noise suppressed)
                wc.append(float(w[m].sum()))
    if len(tc) >= 3:                                      # slope of neighbouring bins -> median (cut filtered out)
        tca = np.asarray(tc); oca = np.asarray(oc); wca = np.asarray(wc)
        a = _wmedian(np.diff(oca) / np.diff(tca), np.minimum(wca[:-1], wca[1:]))
    else:                                                # too few bins -> fall back to local node slopes
        sl = np.diff(o) / np.maximum(np.diff(T), 1e-9); we = np.minimum(w[:-1], w[1:]); g = we > 0
        a = _wmedian(sl[g], we[g]) if g.any() else 0.0
    if (not np.isfinite(a)) or abs(a) / max(float(fps_ref), 1e-6) * 100.0 > max_pct:
        return 0.0, 0.0
    b = _wmedian(o - a * T, np.maximum(w, 1e-6))          # reference level (staircase center)
    return float(a), float(b)


def tv1d(y, lam):
    """1D total variation denoising (Condat 2013, direct O(n)). Minimizes 0.5*sum((y-x)^2) +
    lam*sum(|dx|). A piecewise-constant, ROBUST baseline: a short outlier (even a tall one) does
    not pay for two jumps of size lam and is absorbed; a sustained step does pay and the jump
    remains. Applied globally over the whole track."""
    y = np.asarray(y, float); N = len(y)
    x = np.empty(N)
    if N == 0:
        return x
    if N == 1:
        x[0] = y[0]; return x
    k = k0 = kminus = kplus = 0
    vmin = y[0] - lam; vmax = y[0] + lam
    umin = lam; umax = -lam
    while True:
        if k == N - 1:
            if umin < 0.0:
                x[k0:kminus + 1] = vmin
                k = k0 = kminus = kminus + 1
                if k >= N:
                    break
                kplus = k; vmin = y[k]; vmax = y[k] + 2 * lam; umin = lam; umax = -lam
            elif umax > 0.0:
                x[k0:kplus + 1] = vmax
                k = k0 = kplus = kplus + 1
                if k >= N:
                    break
                kminus = k; vmin = y[k] - 2 * lam; vmax = y[k]; umin = lam; umax = -lam
            else:
                x[k0:N] = vmin + umin / (k - k0 + 1)
                break
            continue
        umin += y[k + 1] - vmin
        umax += y[k + 1] - vmax
        if umin < -lam:                                          # negative jump
            x[k0:kminus + 1] = vmin
            k = k0 = kminus = kminus + 1
            kplus = k; vmin = y[k]; vmax = y[k] + 2 * lam; umin = lam; umax = -lam
        elif umax > lam:                                         # positive jump
            x[k0:kplus + 1] = vmax
            k = k0 = kplus = kplus + 1
            kminus = k; vmin = y[k] - 2 * lam; vmax = y[k]; umin = lam; umax = -lam
        else:
            k += 1
            if umin >= lam:
                vmin += (umin - lam) / (k - k0 + 1); umin = lam; kminus = k
            if umax <= -lam:
                vmax += (umax + lam) / (k - k0 + 1); umax = -lam; kplus = k
    return x


def clean_cuts(o, w, T, a_g, b_g, *, w_conf=VIS_W_CONF, smax=VIS_SMAX, tol=VIS_TOL,
               lam=VIS_TV_LAM, step_min=VIS_TV_STEP_MIN, exc_thr=VIS_OUT_THR):
    """Video-map cut boundaries via TV DENOISING (global, so a greedy scan cannot let one excursion
    contaminate its local neighbourhood). o_res -> TV baseline B (piecewise constant): a short
    excursion does not pay for a jump and is absorbed; a sustained step makes B step. CUTS are the
    boundaries of B's segments, but the level/delta at each cut comes from the REAL anchors
    (segment median with outliers removed -- TV understates delta by shrinking it ~lambda/m),
    plus a "sharp local jump" filter (removes fine TV staircasing on smooth wander). Outliers are
    |o_res - segment_level| > exc_thr, measured from the segment's TRUE level rather than the
    shrunken TV baseline B (otherwise a short segment ahead of a large step would be lost
    entirely) -> exc_mask marks them EXCLUDED from the curve fit.
    Returns: (cuts[(tc, delta, t_end, t_nxt)], o_res, conf, body, exc_mask)."""
    o = np.asarray(o, float); w = np.asarray(w, float); T = np.asarray(T, float)
    o_res = o - (a_g * T + b_g)                                   # detrend by scale: ripple+steps with the slope removed
    conf = w >= w_conf
    ti = T[conf]; ri = o_res[conf]
    n = len(ti)
    if n < 2:
        return [], o_res, conf, (float(T[0]), float(T[-1])), np.zeros(len(T), bool)
    B = tv1d(ri, lam)                                            # TV -- ONLY for SEGMENTATION (cut boundaries)
    conf_idx = np.where(conf)[0]
    body = (float(ti[0]), float(ti[-1]))
    bj = list(np.where(np.abs(np.diff(B)) > 0.5)[0])            # TV jump between anchors i and i+1
    seg_b = [0] + [i + 1 for i in bj] + [n]
    segs = [(seg_b[k], seg_b[k + 1]) for k in range(len(seg_b) - 1)]
    # The outlier is measured from the segment's TRUE level (median of real anchors), not from the
    # shrunken TV base B: TV understates a step by ~lambda/m, so a short segment ahead of a large
    # step (e.g. a synced lead-in before an insert) would otherwise fall entirely into the outlier
    # set, build_curve would drop it, and the alignment would sit on the global level instead (the
    # dub's opening seconds shifted by the insert's length). Anchor-derived levels are this
    # detector's stated design, so the outlier threshold follows the same rule.
    Blvl = np.empty(n)
    for a, b in segs:
        Blvl[a:b] = np.median(ri[a:b])
    exc = np.abs(ri - Blvl) > exc_thr                          # outliers: far from the level of their OWN segment

    def _lvl(a, b):                                             # segment level from REAL anchors (outliers excluded)
        sl = ri[a:b][~exc[a:b]]
        return float(np.median(sl)) if len(sl) else float(np.median(ri[a:b]))
    seg_lvl = [_lvl(a, b) for a, b in segs]
    cuts = []
    for k in range(len(segs) - 1):
        i = segs[k][1] - 1                                      # last anchor of segment k
        dv = seg_lvl[k + 1] - seg_lvl[k]                        # TRUE level delta (without TV shrinkage)
        sharp = any(abs(ri[j + 1] - ri[j]) > smax * (ti[j + 1] - ti[j]) + tol   # sharp local jump
                    for j in range(max(0, i - 2), min(n - 1, i + 3)))           # (not a smooth staircase from wander)
        if abs(dv) > step_min and sharp:
            cuts.append((0.5 * (ti[i] + ti[i + 1]), dv, float(ti[i]), float(ti[i + 1])))
    exc_mask = np.zeros(len(T), bool)
    exc_mask[conf_idx[exc]] = True
    return cuts, o_res, conf, body, exc_mask


def _theilsen(tt, yy, smax):
    """Production line: sparse Theil-Sen, slope clamped to +-smax."""
    n = len(tt)
    if n < 2:
        return 0.0, float(yy[0]) if n else 0.0
    step = max(1, n // 20)
    sl = [(yy[j] - yy[i]) / (tt[j] - tt[i])
          for i in range(0, n, step) for j in range(i + 1, n, step) if tt[j] > tt[i] + 1]
    s = float(np.clip(np.median(sl or [0.0]), -smax, smax))
    return s, float(np.median(yy - s * tt))


def _smooth_w(tt, yy, ww, win):
    """Robust smoothing: weighted median within a window of +-win nodes (suppresses outliers before RDP)."""
    n = len(yy); out = np.empty(n); half = max(1, win // 2)
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        out[i] = _wmedian(yy[a:b], np.maximum(ww[a:b], 1e-6))
    return out


def _rdp(t, y, eps):
    """Ramer-Douglas-Peucker: a polyline guaranteed to deviate by at most eps, with the fewest vertices."""
    keep = np.zeros(len(t), bool); keep[0] = keep[-1] = True
    stack = [(0, len(t) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        x0, y0, x1, y1 = t[a], y[a], t[b], y[b]
        sl = 0.0 if x1 == x0 else (y1 - y0) / (x1 - x0)
        d = np.abs(y[a:b + 1] - (y0 + sl * (t[a:b + 1] - x0)))
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[a + k] = True; stack.append((a, a + k)); stack.append((a + k, b))
    idx = np.where(keep)[0]
    return t[idx], y[idx]


def build_curve(o, w, T, a_g, b_g, *, smax=VIS_SMAX):
    """Clean map: on each span between cuts, ADAPTIVELY fit a production line (Theil-Sen,
    clamped to smax) when it sits on the anchors (90th-pct residual <= VIS_ADAPT_THR), reproducing
    stepped/linear dubs bit-exact; when it does not (the shift WANDERS in humps), fall back to an
    RDP polyline over the smoothed anchors. Cuts and exc_mask come from clean_cuts (TV
    segmentation, final). The fit uses only baseline anchors, not outliers. Outside body
    (head/tail), the plateau is extrapolated.
    Returns: (curve[frames on T], cuts, fill[(t_end, t_nxt) silence spans], o_res, conf, body)."""
    o = np.asarray(o, float); w = np.asarray(w, float); T = np.asarray(T, float)
    cuts, o_res, conf, body, exc_mask = clean_cuts(o, w, T, a_g, b_g, smax=smax)  # TV: final cuts
    base = a_g * T + b_g
    inbody = (T >= body[0]) & (T <= body[1])                      # outside the body (head/tail) -- don't extend the line past the edge
    fit_ok = conf & ~exc_mask                                     # fit ONLY on baseline anchors (not outliers)
    bnds = [float(T[0]) - 1] + [c[0] for c in cuts] + [float(T[-1]) + 1]
    curve = np.zeros(len(T))
    for k in range(len(bnds) - 1):
        m = (T > bnds[k]) & (T <= bnds[k + 1])
        idx = np.where(m)[0]
        if not len(idx):
            continue
        cm = fit_ok & m & inbody
        if int(cm.sum()) >= 2:
            tt = T[cm]; yy = o_res[cm]; ww = w[cm]
            sl, inter = _theilsen(tt, yy, smax)                  # production-line fit: robust Theil-Sen slope/intercept for this segment
            line_resid = float(np.percentile(np.abs(yy - (sl * tt + inter)), 90))
            if line_resid <= VIS_ADAPT_THR:                      # LINE fits -> step/linear dubs are reproduced BIT-EXACT
                curve[idx] = sl * T[idx] + inter + base[idx]
            else:                                                # WANDERS in humps -> RDP polyline over smoothed anchors
                ys = _smooth_w(tt, yy, ww, VIS_SMOOTH_WIN)
                nt, ny = _rdp(tt, ys, VIS_RDP_EPS)
                curve[idx] = np.interp(T[idx], nt, ny) + base[idx]
        else:
            curve[idx] = base[idx]
    fill = [(c[2], c[3]) for c in cuts]                           # silence spans (cuts)
    return curve, cuts, fill, o_res, conf, body


def build_map(pred, asg, cos, fps_ref, fps_dub, dur_ref, dt, ax_ref=None, ax_dub=None):
    """The video map from the clean builder -- conform's sole source of truth: real-scale removal
    plus clean_cuts (the DELTA decides) plus a robust layout. The `tg_s` map is the layout
    directly, with no monotonization: between cuts it is monotonic by construction (slope
    <= VIS_SMAX << fps, so the dub can never play backward). Excisions (spans with no dub content)
    are filled with SILENCE by the caller, driven by `cuts` (a negative delta means an excision of
    width |delta|/fps; tg_s dips locally at the cut, but that zone is zeroed out -- past it the dub
    is continuous).
    Returns: (grid, tg_s, cuts, o, w, curve).
      cuts [(tc, delta_frames, t_end, t_nxt)] are the layout's cuts, the sole cut detector; an
      excision is the gap BETWEEN anchors [t_end, t_nxt] (no dub content there); o/w/curve are for
      plotting.

    Real scale: the dub's global slope (PAL/speed-up/any fps) is removed with no ceiling
    (`global_trend`), and detection plus layout then run on the RESIDUAL, where the VIS_SMAX
    ceiling bounds the DEVIATION from the real scale rather than from zero. Otherwise a sped-up
    dub (e.g. PAL at -4.1%) would hit the ~1.9% ceiling and drift away. A healthy dub (slope~0) is
    unaffected; excisions (instantaneous steps) remain in the residual for cut detection."""
    T = make_T(dur_ref)                                   # grid from the REAL reference duration (any length)
    o, w = vision_ow(pred, asg, cos, fps_ref, fps_dub, T=T, ax_ref=ax_ref, ax_dub=ax_dub)
    a_g, b_g = global_trend(o, w, T, fps_ref)             # dub's real scale (ceiling removed)
    curve, cuts4, _fill, _o_res, _conf, _body = build_curve(o, w, T, a_g, b_g)
    # [(tc, delta_frames, t_end, t_nxt)] -- cut + excision boundaries = the gap BETWEEN plateau anchors
    # (t_end = last anchor of the plateau before, t_nxt = first anchor of the plateau after): no dub content there.
    cuts = [(float(tc), float(dv), float(te), float(tn)) for tc, dv, te, tn in cuts4]
    grid = np.arange(0, dur_ref, dt)
    shift = np.interp(grid, T, curve)                     # shift (reference frames) at the reference grid = alignment
    tg_s = grid + shift / fps_ref                         # dub time, s: frames -> seconds via the REAL fps_ref
    #   (NOT FRAME=23.976 -- gave a 1.25x scale on NTSC/PAL; shift is measured in reference frames by vision_ow)
    return grid, tg_s.astype(np.float64), cuts, o, w, curve

