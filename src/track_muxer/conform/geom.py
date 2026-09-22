"""Geometric registration of a dub to the reference -- for the vision layer, when the SRM
goes blind on a geometric mismatch (crop / zoom / anamorphic / letterbox bars / an
arbitrary fps).

Called from conform_features ONLY when the coarse pass is blind (coarse n_keep=~0): the
regular vision layer builds the SRM on a fixed 128x72 grid (anisotropic scale, the
descriptor is tied to pixel position) and goes blind when the dub's content is
geometrically shifted relative to the reference. This module recovers a GLOBAL transform
(one per file):

  STEP 1 -- synchronization WITHOUT a time model: frame fingerprints (geometry-invariant
    color/brightness/novelty aggregates, taken after the border crop, resampled onto a
    TIME axis so fps drops out) give local anchors (NCC of a window across the full
    length; prominence + mutual + fine checks reject false matches).
  STEP 2 -- geometry: LoFTR (a detector-free matcher, kornia) on the anchors gives dense
    correspondences -> a RANSAC affine dub<->ref in frame fractions -> a consensus MEDIAN
    bbox across anchors (not a single best one: a high-inlier anchor can still be
    misaligned in fps). This handles cross-source content (Blu-ray vs web) where a
    detector like ORB goes blind on flat, low-texture anime frames: zoom/letterbox/
    identity all fall out of one affine.

The consensus_G result gives a crop for the reference (ROI = the area visible in the dub)
and for the dub (the border crop): both feed build_srm(crop=...). The transform is an
axis-aligned affine (no rotation), which decomposes into crop + anisotropic scale, so no
warp is needed -- it maps onto ffmpeg's native vf chain.

GPU-first (LoFTR on CUDA), with a CPU fallback (slow). Memory is O(block): fingerprints
stream, LoFTR runs on individual anchor frames. Requires cv2 (Apache-2.0) + kornia
(Apache-2.0, LoFTR).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from track_muxer.conform.config import FFMPEG, FFPROBE
from track_muxer.conform import procreg
from track_muxer.conform.features import probe_video_duration
from track_muxer.conform.progress import part

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # geom unavailable without cv2: conform_features falls back to the regular path
    _HAS_CV2 = False

try:
    import torch
    import kornia.feature as _KF
    _HAS_LOFTR = True
except ImportError:  # without torch/kornia geom is unavailable: regular path
    _HAS_LOFTR = False

_LOFTR = None
_DEV = None


def _device():
    global _DEV
    if _DEV is None:
        _DEV = "cuda" if (_HAS_LOFTR and torch.cuda.is_available()) else "cpu"
    return _DEV


def _loftr():
    """Lazily loads the detector-free LoFTR matcher (kornia, Apache). Uses the GPU when available, otherwise the CPU."""
    global _LOFTR
    if _LOFTR is None:
        _LOFTR = _KF.LoFTR(pretrained="outdoor").eval().to(_device())
    return _LOFTR

# ── parameters ──
GRID_HZ = 10.0          # common time grid for the fingerprints (fps-invariant)
WIN_S = 12.0            # fingerprint window length, s (uniqueness of the time pattern)
K_POINTS = 12           # reference anchor points
EDGE_S = 40.0           # margin from the edges (opening/ending/credits)
DESC_W, DESC_H = 32, 18  # fingerprint decode size (H is a multiple of NZONES)
NZONES = 6              # horizontal zones (vertical profile) — composition, X-invariant
W_NOV = 3.0            # weight of the novelty (cut) channel in NCC
MUTUAL_TOL = 6.0       # tolerance for the backward match's return to t_ref, s
FINE_S, FINE_RNG, FINE_THR = 4.0, 6.0, 0.55  # narrow check of the anchor CENTER's sync
FRAME_H = 540          # common frame height for ORB (after cropping the borders)
LOFTR_W, LOFTR_H = 640, 384  # LoFTR input (multiple of 8)
LOFTR_CONF = 0.5       # LoFTR match confidence threshold
AFFINE_MIN_INL = 300   # minimum RANSAC affine inliers for an anchor to enter the consensus


def available() -> bool:
    return _HAS_CV2 and _HAS_LOFTR


def probe_fps(video: Path) -> float:
    out = procreg.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0",
                          str(video)], capture_output=True, text=True).stdout.strip()
    a, b = out.split("/") if "/" in out else (out, "1")
    return float(a) / float(b)


def probe_wh(video: Path) -> tuple[int, int]:
    out = procreg.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0",
                          str(video)], capture_output=True, text=True).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


def crop_detect(video: Path, *, thr: int = 20, samples=(120, 300, 500, 700, 900)) -> str:
    """Robust black-border detection: takes the max brightness of each row/column across a
    sample of frames (a border pixel must be black in ALL sampled frames; content that is
    bright even once is not mistaken for a border, unlike ffmpeg's cropdetect). Returns
    'W:H:X:Y' (the full frame if there are no borders)."""
    W, H = probe_wh(video)
    mrow = np.zeros(H, np.float32); mcol = np.zeros(W, np.float32); got = 0
    for t in samples:
        out = procreg.run([FFMPEG, "-ss", f"{t}", "-i", str(video), "-frames:v", "1",
                              "-vf", "format=gray", "-f", "rawvideo", "-", "-loglevel", "error"],
                             capture_output=True).stdout
        if len(out) < W * H:
            continue
        g = np.frombuffer(out[:W * H], np.uint8).reshape(H, W).astype(np.float32)
        mrow = np.maximum(mrow, g.max(1)); mcol = np.maximum(mcol, g.max(0)); got += 1
    if got == 0:
        return f"{W}:{H}:0:0"
    rr = np.where(mrow > thr)[0]; rc = np.where(mcol > thr)[0]
    if not len(rr) or not len(rc):
        return f"{W}:{H}:0:0"
    y0, y1 = int(rr[0]), int(rr[-1]); x0, x1 = int(rc[0]), int(rc[-1])
    cw = (x1 - x0 + 1) - ((x1 - x0 + 1) % 2); ch = (y1 - y0 + 1) - ((y1 - y0 + 1) % 2)
    if cw >= W - 2 and ch >= H - 2:
        return f"{W}:{H}:0:0"
    return f"{cw}:{ch}:{x0}:{y0}"


# ── STEP 1: fingerprints → anchors without a time model ──
def _decode_sig(video: Path, crop: str, on_prog=None, flip: bool = False):
    """Streaming decode (after the border crop) into a per-frame descriptor [N,22]:
    6 horizontal zones x RGB (18) + Y percentiles p10/p50/p90 (3) + novelty (1). Also
    returns fps. Memory is O(block)."""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(video), "-an",
           "-vf", f"crop={crop},{'hflip,' if flip else ''}scale={DESC_W}:{DESC_H},format=rgb24", "-vsync", "0",
           "-f", "rawvideo", "-"]
    fb = DESC_W * DESC_H * 3
    fps = probe_fps(video)
    n_expect = int(round((probe_video_duration(video) or 0.0) * fps)) if on_prog is not None else 0
    p = procreg.popen(cmd, stdout=subprocess.PIPE, bufsize=fb * 1024)
    rows: list[np.ndarray] = []
    n_done = 0
    prevY = None; buf = b""
    while True:
        chunk = p.stdout.read(fb * 1024)
        if not chunk:
            break
        buf += chunk
        k = len(buf) // fb
        if not k:
            continue
        blk = np.frombuffer(buf[:k * fb], np.uint8).reshape(k, DESC_H, DESC_W, 3).astype(np.float32)
        buf = buf[k * fb:]
        zones = blk.reshape(k, NZONES, DESC_H // NZONES, DESC_W, 3).mean(axis=(2, 3)).reshape(k, NZONES * 3)
        Y = blk.reshape(k, DESC_H * DESC_W, 3) @ np.array([0.299, 0.587, 0.114], np.float32)
        pcts = np.percentile(Y, [10, 50, 90], axis=1).T
        nov = np.empty(k, np.float32)
        for j in range(k):
            nov[j] = 0.0 if prevY is None else float(np.abs(Y[j] - prevY).mean())
            prevY = Y[j]
        rows.append(np.concatenate([zones, pcts, nov[:, None]], axis=1))
        n_done += k
        if on_prog is not None and n_expect:
            on_prog(min(0.999, n_done / n_expect))
    p.stdout.close()
    rc_geom = p.wait(); procreg.done(p)
    if rc_geom not in (0, None):   # a fingerprint decode failure must not pass silently
        raise RuntimeError(f"geom: декод слепков упал (ffmpeg rc={p.returncode}): {Path(video).name}")
    sig = np.concatenate(rows) if rows else np.zeros((0, NZONES * 3 + 4), np.float32)
    return sig, fps


def _to_grid(sig: np.ndarray, fps: float):
    t = np.arange(len(sig)) / fps
    g = np.arange(0, t[-1], 1 / GRID_HZ)
    return np.stack([np.interp(g, t, sig[:, c]) for c in range(sig.shape[1])], axis=1)


def _ncc_full(T: np.ndarray, D: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted normalized cross-correlation of window T against the full length of D (per-channel)."""
    Lw = len(T); acc = np.zeros(len(D) - Lw + 1)
    for c in range(T.shape[1]):
        Tz = T[:, c] - T[:, c].mean(); Tn = float(np.linalg.norm(Tz)) + 1e-9
        Dc = D[:, c]
        corr = fftconvolve(Dc, Tz[::-1], "valid")
        cs = np.cumsum(np.insert(Dc, 0, 0)); cs2 = np.cumsum(np.insert(Dc * Dc, 0, 0))
        ws = cs[Lw:] - cs[:-Lw]; ws2 = cs2[Lw:] - cs2[:-Lw]
        dstd = np.sqrt(np.maximum(ws2 - ws * ws / Lw, 1e-9))
        acc += w[c] * corr / (dstd * Tn + 1e-9)
    return acc / w.sum()


def _match_point(T, D, w):
    ncc = _ncc_full(T, D, w)
    j = int(np.argmax(ncc)); peak = float(ncc[j]); Lw = len(T)
    mask = np.ones(len(ncc), bool); mask[max(0, j - Lw):j + Lw] = False
    second = float(ncc[mask].max()) if mask.any() else -1.0
    return (j + Lw / 2) / GRID_HZ, peak, peak - second


def _fine_check(G_r, ci_s, G_d, td_coarse, w):
    Lf = int(FINE_S * GRID_HZ); half = Lf // 2; ci = int(ci_s * GRID_HZ)
    Tf = G_r[ci - half:ci - half + Lf]
    if len(Tf) < Lf:
        return -1.0, td_coarse
    jc = int(td_coarse * GRID_HZ) - half
    lo = max(0, jc - int(FINE_RNG * GRID_HZ)); hi = min(len(G_d) - Lf, jc + int(FINE_RNG * GRID_HZ))
    if hi <= lo:
        return -1.0, td_coarse
    seg = _ncc_full(Tf, G_d[lo:hi + Lf], w)
    if not len(seg):
        return -1.0, td_coarse
    jj = int(np.argmax(seg))
    return float(seg[jj]), (lo + jj + half) / GRID_HZ


def _find_anchors(ref: Path, dub: Path, on_prog=None, flip: bool = False):
    """Anchor candidates [(t_ref, t_dub, prom, fine)] after the prominence+mutual+fine checks.
    flip -- the dub is mirrored: its frames are flipped after the border crop."""
    cr_ref = crop_detect(ref); cr_dub = crop_detect(dub)
    sig_r, fps_r = _decode_sig(ref, cr_ref, part(on_prog, 0.05, 0.55)); G_r = _to_grid(sig_r, fps_r)
    sig_d, fps_d = _decode_sig(dub, cr_dub, part(on_prog, 0.55, 0.95), flip=flip); G_d = _to_grid(sig_d, fps_d)
    G_r = (G_r - G_r.mean(0)) / (G_r.std(0) + 1e-9)
    G_d = (G_d - G_d.mean(0)) / (G_d.std(0) + 1e-9)
    w = np.ones(G_r.shape[1]); w[-1] = W_NOV
    Lw = int(WIN_S * GRID_HZ); dur_r = len(G_r) / GRID_HZ
    centers = np.linspace(EDGE_S + WIN_S / 2, dur_r - EDGE_S - WIN_S / 2, K_POINTS)
    rows = []
    for tc in centers:
        i0 = int((tc - WIN_S / 2) * GRID_HZ); T = G_r[i0:i0 + Lw]
        if len(T) < Lw:
            continue
        td, peak, prom = _match_point(T, G_d, w)
        if peak > 0.45 and prom > 0.12:
            rows.append((float(tc), td, peak, prom))
    good = []
    for tc, td, peak, prom in rows:
        i0d = int((td - WIN_S / 2) * GRID_HZ); Td = G_d[i0d:i0d + Lw]
        tr_back = _match_point(Td, G_r, w)[0] if len(Td) >= Lw else -999
        if abs(tr_back - tc) > MUTUAL_TOL:
            continue
        fp, td_fine = _fine_check(G_r, tc, G_d, td, w)
        if fp >= FINE_THR:
            good.append((tc, td_fine, prom, fp))
    good.sort(key=lambda r: -r[3])
    return good, cr_ref, cr_dub


# ── STEP 2: geometry on anchors (ORB+RANSAC) ──
def _decode_frame(video: Path, t: float, crop: str, flip: bool = False):
    """Exact grayscale frame at time t (select+copyts, since ffmpeg -ss is inaccurate on
    long GOPs), after the border crop, height FRAME_H, width from the aspect ratio.
    Returns np.uint8 [h,tw] or None."""
    W, H = map(int, crop.split(":")[:2])
    tw = max(2, round(W / H * FRAME_H)); tw -= tw % 2
    coarse = max(0, int(t - 4))
    out = procreg.run([FFMPEG, "-ss", f"{coarse}", "-copyts", "-i", str(video),
                          "-vf", f"crop={crop},{'hflip,' if flip else ''}select=gte(t\\,{t}),scale={tw}:{FRAME_H},format=gray",
                          "-frames:v", "1", "-f", "rawvideo", "-", "-loglevel", "error"],
                         capture_output=True).stdout
    if len(out) < tw * FRAME_H:
        return None
    return np.frombuffer(out[:tw * FRAME_H], np.uint8).reshape(FRAME_H, tw)


def _even(v):
    v = int(round(v)); return v - (v % 2)


def _to_dev(img_gray, w, h):
    r = cv2.resize(img_gray, (w, h)).astype(np.float32) / 255.0
    return torch.from_numpy(r)[None, None].to(_device())


def _match(ref_gray, dub_gray):
    """LoFTR: dense correspondences dub<->ref on a LOFTR_W x LOFTR_H grid. Returns (k_dub, k_ref, conf)."""
    m = _loftr()
    with torch.no_grad():
        out = m({"image0": _to_dev(dub_gray, LOFTR_W, LOFTR_H),
                 "image1": _to_dev(ref_gray, LOFTR_W, LOFTR_H)})
    return (out["keypoints0"].cpu().numpy(), out["keypoints1"].cpu().numpy(),
            out["confidence"].cpu().numpy())


def _affine_frac(ref_gray, dub_gray):
    """RANSAC affine dub<->ref from the LoFTR matches (in frame FRACTIONS) -> fractional
    bboxes for the consensus. Returns dict(sx, sy, n_inl, dub_in_ref, ref_in_dub) or None.
    Fractions [0,1] are resolution-independent."""
    k_dub, k_ref, conf = _match(ref_gray, dub_gray)
    pd = (k_dub / np.array([LOFTR_W, LOFTR_H], np.float32)).astype(np.float32)
    pr = (k_ref / np.array([LOFTR_W, LOFTR_H], np.float32)).astype(np.float32)
    sel = conf > LOFTR_CONF
    if int(sel.sum()) < 12:
        return None
    M, inl = cv2.estimateAffine2D(pd[sel], pr[sel], method=cv2.RANSAC, ransacReprojThreshold=0.01)
    if M is None:
        return None
    sx = float(np.hypot(M[0, 0], M[1, 0])); sy = float(np.hypot(M[0, 1], M[1, 1]))
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    Mi = cv2.invertAffineTransform(M)
    dr = corners @ M[:, :2].T + M[:, 2]        # dub in fractions of the ref
    rd = corners @ Mi[:, :2].T + Mi[:, 2]      # ref in fractions of the dub

    def bb(fr):
        return (float(fr[:, 0].min()), float(fr[:, 1].min()), float(fr[:, 0].max()), float(fr[:, 1].max()))
    return dict(sx=sx, sy=sy, n_inl=int(inl.sum()), dub_in_ref=bb(dr), ref_in_dub=bb(rd))


def _consensus_crop(recs, ref_wh, dub_wh):
    """Median of the fractional bboxes across anchors -> crop_ref/crop_dub (native
    coordinates, direction picked automatically). The side that gets cropped is whichever
    one shows the OTHER side's full extent as less than the whole frame (i.e. it is the
    wider one). The decision is a consensus across anchors, not a single best one: a
    high-inlier anchor can still be misaligned in fps."""
    Wr, Hr = ref_wh; Wd, Hd = dub_wh
    dr = np.median(np.array([r["dub_in_ref"] for r in recs]), axis=0)
    rd = np.median(np.array([r["ref_in_dub"] for r in recs]), axis=0)

    def crop(b, W, H):
        x0 = max(0.0, b[0]); y0 = max(0.0, b[1]); x1 = min(1.0, b[2]); y1 = min(1.0, b[3])
        cw = _even((x1 - x0) * W); ch = _even((y1 - y0) * H)
        return "%d:%d:%d:%d" % (cw, ch, _even(x0 * W), _even(y0 * H)), float((x1 - x0) * (y1 - y0))
    crop_ref, area_ref = crop(dr, Wr, Hr)
    crop_dub, area_dub = crop(rd, Wd, Hd)
    return dict(crop_ref=crop_ref, crop_dub=crop_dub, area_ref=area_ref, area_dub=area_dub,
                sx=float(np.median([r["sx"] for r in recs])),
                sy=float(np.median([r["sy"] for r in recs])), n=len(recs))


def _mirror_crop(crop: str, width: int) -> str:
    """Crop box found on mirrored frames -> the same box in native coordinates."""
    w, h, x, y = map(int, crop.split(":"))
    return "%d:%d:%d:%d" % (w, h, _even(width - x - w), y)


def consensus_G(ref: Path, dub: Path, on_prog=None, flip: bool = False):
    """FULL pipeline, step 1 + step 2, producing crops ready for build_srm.
    Returns dict(crop_ref, crop_dub, sx, sy, n_in), or None if the geometry could not be
    recovered. crop_ref = 'W:H:X:Y' of the reference area visible in the dub (ROI, native
    reference coordinates); crop_dub = 'W:H:X:Y' of the dub's border crop. Both feed
    build_srm(crop=...)."""
    if not (_HAS_CV2 and _HAS_LOFTR):
        return None
    good, _, _ = _find_anchors(ref, dub, part(on_prog, 0.0, 0.6), flip=flip)   # STEP 1: synchronization (anchor times)
    if not good:
        return None
    Wr, Hr = probe_wh(ref); Wd, Hd = probe_wh(dub)
    raw_ref = "%d:%d:0:0" % (Wr, Hr); raw_dub = "%d:%d:0:0" % (Wd, Hd)
    recs = []                                        # STEP 2: LoFTR affine on RAW frames (letterbox bars are part of the affine)
    lp = part(on_prog, 0.6, 1.0)
    for k, (tc, td, _, _) in enumerate(good):
        rg = _decode_frame(ref, tc, raw_ref); dg = _decode_frame(dub, td, raw_dub, flip=flip)
        if lp is not None:
            lp((k + 1) / len(good))
        if rg is None or dg is None:
            continue
        r = _affine_frac(rg, dg)
        if r is not None and r["n_inl"] >= AFFINE_MIN_INL:
            recs.append(r)
    if len(recs) < 2:
        return None
    con = _consensus_crop(recs, (Wr, Hr), (Wd, Hd))   # consensus median across anchors
    crop_dub = _mirror_crop(con["crop_dub"], Wd) if flip else con["crop_dub"]   # box back to native frames
    return dict(crop_ref=con["crop_ref"], crop_dub=crop_dub,
                sx=con["sx"], sy=con["sy"], n_in=con["n"], flip=bool(flip))
