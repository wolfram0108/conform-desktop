"""On-disk SRM cache (for the REFERENCE). Stored in a subdirectory of the series
directory (`<series>/_conform_cache/`), keyed by file content (name+size+mtime).

Why: the reference is reused across dubs (within a series) AND across RUNS (a new
dub arrives -- the reference is served from cache without re-decoding). Dubs are
not cached.

Format: features are stored raw as `.f16` (flat float16, shape (N, D)), metadata as
`.json` (N, fps). Loading returns an **np.memmap** (read-only) -- the reference is
NOT loaded into RAM in full (matters for long files). Old `.npz` caches are not
read -- they are simply rebuilt.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
from loguru import logger

from track_muxer.conform.features import GH, GW, probe_video_duration, vfr_time_axis
from track_muxer.conform.models import SrmFeatures

_CACHE_AUDIT = bool(os.environ.get("TM_CACHE_AUDIT"))


def _a(stage: str, hit: bool, name: str = "") -> None:
    """Cache-reuse audit (env TM_CACHE_AUDIT=1): the code itself logs whether it took
    the cache here or built on GPU. Enabled only for test runs; silent by default in production."""
    if _CACHE_AUDIT:
        print(f"[КЭШ] {stage:16s} {'РЕЮЗ ✓' if hit else 'промах → строю/декодю'}  {name}", flush=True)

D = GW * GH * 2          # frame vector dimensionality (rk+rd)

# Stage versions for checkpoint invalidation (tmp mode). Bump when a STAGE's code changes -- a
# stale checkpoint then rebuilds while the heavier stages above it are served from cache; this is
# not user-controlled. EMB_VER is the SRM/embedding stage (features.build_srm), covering the SRM
# cache for both the reference and the dub.
EMB_VER = 1
EXT_VER = 3              # audio extraction (_extract_wav[_mmap]) -- the CK2 audio cache
#   v3: audio is laid on the video axis by the container delay (features.probe_av_delay);
#   v2 caches of files with video_start != audio_start hold the audio shifted by that delay.
#   v2: audio decoding always goes through `aresample=async=1:first_pts=0` (see align._extract_base).
#   On clean input the data is bit-identical to v1, but a v1-built cache of a file with a PTS gap
#   is out of sync.
DSP_VER = 3             # DSP-48 benv of the REFERENCE (coarse_dtw) -- the CK4 cache. Bump when benv/bands/resampling changes.
#   v3: reference audio decode now applies the container delay (see EXT_VER v3).
#   v2: benv is computed from the reference audio, and its decode changed (see EXT_VER).


def cache_key(video: Path) -> str:
    st = Path(video).stat()
    return f"{Path(video).stem}__{st.st_size}__{int(st.st_mtime)}"


def _crop_tag(crop: str | None) -> str:
    """Key suffix for a CROPPED SRM (geometry correction): None -> '' (the plain cache)."""
    return ("__c" + crop.replace(":", "_")) if crop else ""


def srm_file(cache_dir: Path | str, video: Path | str, crop: str | None = None) -> Path:
    """Path to the raw features file (.f16) -- build_srm streams into it directly under low_mem.
    When crop is given, uses a separate key for the cropped SRM (geometry mode), not overlapping the plain one."""
    return Path(cache_dir) / (cache_key(Path(video)) + _crop_tag(crop) + ".f16")


def _meta_file(cache_dir: Path | str, video: Path | str, crop: str | None = None) -> Path:
    return Path(cache_dir) / (cache_key(Path(video)) + _crop_tag(crop) + ".json")


def load_srm(cache_dir: Path | str, video: Path | str, crop: str | None = None) -> SrmFeatures | None:
    """Reference/dub from the cache as a memmap (read-only). None if the cache is
    missing or broken. Logs the reuse for the audit. When crop is given, loads the cropped SRM (geometry mode)."""
    r = _load_srm(cache_dir, video, crop)
    _a("CK1 SRM" + (" geom" if crop else ""), r is not None, Path(video).name)
    return r


def _load_srm(cache_dir: Path | str, video: Path | str, crop: str | None = None) -> SrmFeatures | None:
    try:
        sp = srm_file(cache_dir, video, crop); mp = _meta_file(cache_dir, video, crop)
    except OSError:
        return None
    if not (sp.exists() and mp.exists()):
        return None
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
        # Invalidation by embedding version: a missing field means version 1 (how existing caches
        # were built before checkpoints existed); those stay valid, and bumping EMB_VER rebuilds.
        if int(meta.get("emb_ver", 1)) != EMB_VER:
            return None
        n = int(meta["n"]); d = int(meta.get("d", D))
        if n <= 0:                       # empty/broken SRM cache (the build failed, left n=0/0 bytes) ->
            return None                  # invalid -> conform rebuilds the SRM (otherwise len/fps=0 -> div0)
        # fps is RE-DERIVED from the file, not trusted from meta: old caches store fps from an
        # inflated container duration (a broken Segment Duration makes fps lie). Frames don't
        # depend on fps, so the SRM is not rebuilt, only fps is refreshed; a failed probe falls
        # back to the stored value.
        fps = float(meta.get("fps", 0.0))
        try:
            vd = probe_video_duration(Path(video))
            if vd and n:
                fps = n / vd
        except Exception:  # noqa: BLE001 -- probe unavailable -> keep the value from meta
            pass
        arr = np.memmap(sp, dtype=np.float16, mode="r", shape=(n, d)) if n else np.zeros((0, d), np.float16)
        # The VFR axis is NOT cached -- it is re-probed from the source on every load (0.05-1.2s,
        # a batch pass without decoding; the SRM decode itself, which the cache saves, is 40-60s).
        # CFR yields None (no cost beyond that same probe on the vast majority of files). As in
        # build_srm: with a live axis, fps is the average density over the axis.
        try:
            pts_ax = vfr_time_axis(Path(video), n)
        except Exception:  # noqa: BLE001 -- probe unavailable -> conservatively no axis
            pts_ax = None
        if pts_ax is not None and pts_ax[-1] > 0:
            fps = (n - 1) / float(pts_ax[-1])
        return SrmFeatures(srm=arr, fps=fps, src=Path(video), pts=pts_ax)
    except Exception as e:  # noqa: BLE001
        logger.warning("conform cache load failed {}: {}", sp, e)
        return None


def save_meta(cache_dir: Path | str, video: Path | str, n: int, fps: float, d: int = D,
              crop: str | None = None) -> None:
    """Write the metadata (after build_srm has already streamed the .f16 directly into the cache)."""
    if int(n) <= 0:                      # do NOT cache an empty SRM (the build failed) -- otherwise poison -> div0
        return
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        _meta_file(cache_dir, video, crop).write_text(
            json.dumps({"n": int(n), "d": int(d), "fps": float(fps), "emb_ver": EMB_VER}),
            encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.warning("conform cache meta save failed: {}", e)


def save_srm(cache_dir: Path | str, video: Path | str, feats: SrmFeatures) -> None:
    """Save already-built (in-RAM) features: raw .f16 + metadata. For the in-RAM path."""
    try:
        arr = np.asarray(feats.srm, np.float16)
        if arr.shape[0] <= 0:            # do not cache an empty SRM (the build failed) -- otherwise poison -> div0
            return
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        arr.tofile(srm_file(cache_dir, video))
        save_meta(cache_dir, video, arr.shape[0], feats.fps, arr.shape[1] if arr.ndim == 2 else D)
    except Exception as e:  # noqa: BLE001
        logger.warning("conform cache save failed: {}", e)


# ── CK2: extracted-audio cache (raw int16 [N,C], as in _extract_wav_mmap) ──

def _atr_tag(atrack: int) -> str:
    """Key suffix for a non-zero audio track (multi-track support).
    atrack=0 -> empty (old keys stay valid bit-for-bit)."""
    return f"__a{int(atrack)}" if atrack else ""


def audio_raw(cache_dir: Path | str, video: Path | str, atrack: int = 0) -> Path:
    """Path to the raw audio (.raw, int16 [N*C]) -- _extract_wav_mmap(dest=) writes into it directly."""
    return Path(cache_dir) / (cache_key(Path(video)) + _atr_tag(atrack) + "__audio.raw")


def _audio_meta(cache_dir: Path | str, video: Path | str, atrack: int = 0) -> Path:
    return Path(cache_dir) / (cache_key(Path(video)) + _atr_tag(atrack) + "__audio.json")


def load_audio_mmap(cache_dir: Path | str, video: Path | str, channels: int,
                    *, ext_ver: int = EXT_VER, atrack: int = 0):
    """CK2: int16 memmap [N,channels] from the cache (read-only), or None. Key = file+EXT_VER+channels+track."""
    r = _load_audio_mmap(cache_dir, video, channels, ext_ver=ext_ver, atrack=atrack)
    _a("CK2 аудио дубль", r is not None, Path(video).name)
    return r


def _load_audio_mmap(cache_dir: Path | str, video: Path | str, channels: int,
                     *, ext_ver: int = EXT_VER, atrack: int = 0):
    try:
        raw = audio_raw(cache_dir, video, atrack); mp = _audio_meta(cache_dir, video, atrack)
    except OSError:
        return None
    if not (raw.exists() and mp.exists()):
        return None
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
        if int(m.get("ext_ver", 1)) != ext_ver or int(m.get("channels", 0)) != int(channels):
            return None
        a = np.memmap(raw, dtype=np.int16, mode="r")
        return a[: (a.size // channels) * channels].reshape(-1, channels)
    except Exception as e:  # noqa: BLE001
        logger.warning("conform audio cache load failed {}: {}", raw, e)
        return None


def save_audio_meta(cache_dir: Path | str, video: Path | str, channels: int,
                    *, ext_ver: int = EXT_VER, atrack: int = 0) -> None:
    """Write the CK2 metadata (the raw file has already been written by _extract_wav_mmap(dest=audio_raw(...)))."""
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        _audio_meta(cache_dir, video, atrack).write_text(
            json.dumps({"channels": int(channels), "ext_ver": int(ext_ver)}), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.warning("conform audio cache meta save failed: {}", e)


# ── CK4: DSP-48 benv cache of the REFERENCE (coarse_dtw; the reference is reused across the episode's dubs) ──

def dsp_ref_path(cache_dir: Path | str, video: Path | str, atrack: int = 0) -> Path:
    """Path to the reference benv cache (.npy, f32 [48,Nf]) with DSP_VER in the name
    (bumping it forces a rebuild). benv is computed from the reference AUDIO, so the
    reference track is part of the key (atrack=0 -> the legacy name)."""
    return Path(cache_dir) / (cache_key(Path(video)) + _atr_tag(atrack) + f"__dsp{DSP_VER}.npy")


# ── CK-geom: cache of the consensus_G RESULT (per-pair crop/scale). LoFTR-geom is EXPENSIVE
#    (decode+neural matcher) and flaky (anchors/GPU), so it must not rerun on every conform. Key =
#    dub (unique per pair) + a check that the reference matches. A hit crops WITHOUT rerunning geom
#    (and without even kornia -- it is only needed to COMPUTE the crop, not to apply it). Bump
#    GEOM_VER when the geom algorithm changes. ──
GEOM_VER = 1


def _geom_meta(cache_dir: Path | str, dub_video: Path | str) -> Path:
    return Path(cache_dir) / (cache_key(Path(dub_video)) + "__geom.json")


def load_geom(cache_dir: Path | str, ref_video: Path | str, dub_video: Path | str):
    """Geometry result (dict crop_ref/crop_dub/sx/sy/n_in) from the cache, or None
    (missing, version mismatch, or the reference changed)."""
    try:
        p = _geom_meta(cache_dir, dub_video)
    except OSError:
        return None
    if not p.exists():
        _a("CK-geom", False, Path(dub_video).name)
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if int(d.get("ver", 0)) != GEOM_VER or d.get("ref_key") != cache_key(Path(ref_video)):
            return None
        _a("CK-geom", True, Path(dub_video).name)
        return d["G"]
    except Exception as e:  # noqa: BLE001
        logger.warning("geom cache load failed {}: {}", p, e)
        return None


def save_geom(cache_dir: Path | str, ref_video: Path | str, dub_video: Path | str, G: dict) -> None:
    """Save the pair's geometry result (after a successful consensus_G)."""
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        _geom_meta(cache_dir, dub_video).write_text(
            json.dumps({"ver": GEOM_VER, "ref_key": cache_key(Path(ref_video)), "G": G}),
            encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.warning("geom cache save failed: {}", e)


def clear_dir(cache_dir: Path | str) -> None:
    """Remove the whole cache subdirectory (when the "keep reference cache" flag is off)."""
    try:
        d = Path(cache_dir)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("conform cache clear failed: {}", e)
