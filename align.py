"""One PAIR ref↔dub → an aligned FLAC (conform).

Pipeline (single path, no legacy toggles):
  SRM matching (coarse_robust → band_align, the first window with a free left end) → pred;
  _level_decide: parses edits from the offset LEVEL SHIFT (restores blind drop-syn, resolves cuts);
  the VIDEO map of the vision layer (vision_detect.build_curve → tg_s, WITHOUT monotonization) is
  the ONLY source of structure; audio is resampled onto the REF grid; SILENCE fills cuts [t_end,t_nxt];
  the audio layer band/muq (anchor) runs ON TOP (it builds no anchors in vision silence); AFTER that,
  fill_silence fills the dub's silence with the synchronized reference; output is written as FLAC, compression level 8.
In addition (does NOT affect the wav) — a quality passport (PairResult).
"""

from __future__ import annotations

import subprocess
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from track_muxer.conform import cache as cache_mod
from track_muxer.conform.config import FFMPEG, FFPROBE
from track_muxer.conform import procreg
from track_muxer.conform import tmpfiles
from track_muxer.conform.memlog import memlog
from track_muxer.conform.decode_backend import decode_backend
from track_muxer.conform.interp_backend import warp_interp        # GPU/CPU audio warp (resample onto the reference grid)
from track_muxer.conform.features import (
    build_srm,
    probe_audio_channels,
    probe_av_delay,
    av_delay_filters,
    probe_duration,
    probe_has_video,
    probe_resolution,
)
from track_muxer.conform.kernel.band_align import band_align
from track_muxer.conform.kernel.coarse import coarse_robust, coarse_windowed
from track_muxer.conform.kernel.orient import mirror_srm, orientation_probe
from track_muxer.conform.models import PairResult, SrmFeatures
from track_muxer.conform.progress import Reporter, part
from track_muxer.conform.trace import Trace
from track_muxer.conform.vision_detect import build_map as _vision_build_map, global_trend as _vision_global_trend, vision_ow as _vision_ow, detelecine as _detelecine, is_baked_telecine as _is_telecine
from track_muxer.conform.anchor.params import FRAME as VFRAME, make_T as _make_T  # ms per frame + anchor grid T (cut silence, unified plot)

# conform v8 constants: algorithm calibration, do not change
SR = 44100
DT = 0.005
ABORT_ASSIGNED_PCT = 60.0  # assigned% below this means a foreign video (dub of another episode):
                           # fail the pair right after matching, before resampling and GPU audio
AUDIO_RESID_MAX_MS = 80.0       # project criterion: a place is out of sync beyond ±80 ms (2 frames)
AUDIO_COVERAGE_BLIND = 0.15     # below: the files share almost no sound, the layer measured nothing
AUDIO_COVERAGE_LOW = 0.5        # below: the layer had support on less than half of the track
AUDIO_EXCESS_WARN_MS = 2000.0   # cut movement that cancelled out (Σ|steps| − |net|): the layer went back and forth
AUDIO_EXCESS_CRIT_MS = 10000.0  # population: healthy ≤ 0.4 s (95th pct), blind chase ≥ 12 s; red above this
FREEZE_MIN_S = 10.0             # identical frames this long are a frozen picture (broken encode), not a scene
FREEZE_COS = 0.999              # SRM cosine of frames ~0.5 s apart that only identical decoded frames reach
FREEZE_GAP_S = 1.0              # a frozen run survives cadence dips shorter than this
MIRROR_RATIO = 2.0         # the mirrored frame sample must beat the plain one by this factor to switch orientation
GEOM_GATE = 50             # coarse pass found < N anchors: vision is blind (crop/zoom/anamorph/bars),
                           # so run the geometry pass before the foreign-video cutoff
COARSE_SAME_MIN_FRAC = 0.10  # share of thinned (K=8) frames in the monotone coarse chain that still
                           # means the same episode despite low assigned%
CUT_MIN_S = 0.3          # a cut is longer than 0.3 s
FADE = int(0.010 * SR)
FILL_MIN_S = 1.0         # cuts longer than 1 s are filled with the reference, shorter ones get silence
XFADE = int(0.030 * SR)  # crossfade dub<->original at fill seams (30 ms)
# Final fill of dub silence with the reference (after band/muq, once the track is in sync):
SIL_FILL_DB = -90.0      # dub silence threshold, dBFS: below -80 is the plateau of real gaps, above is
                         # quiet content; -90 sits mid-plateau, far from the -40..-60 edge
SIL_FILL_WIN_S = 0.02    # RMS window (20 ms)
SIL_FILL_MIN_S = 0.15    # shortest silence zone; shorter dips are left alone
EDGE_MIN_FR = 24         # edge re-pass only if more than ~1 s is dropped or uncovered
EDGE_COS_MIN = 0.5       # a recovered edge match is kept only above this cosine
DSYN = 0.30
MATCH_THR = 0.30
# level_edits layer: edits are decided by the offset LEVEL shift instead of restore_blind+R
LEVEL_EDIT_S = 1.0       # offset level shift above this is a real edit, otherwise a blind zone
LEVEL_MIN_INS_S = 2.0    # an insert is a solid dub block longer than this, shorter is jitter
LEVEL_WIN_S = 4.0        # median window of the offset level before/after a dropped run
# Level-aware cut detection: a transient offset outlier on static content is not a cut
LEVEL_CUT_COALESCE_S = 3.0  # reference gaps closer than this merge into one cluster
LEVEL_CUT_RECOVER_S = 8.0   # window to look for the level RETURNING (a sawtooth outlier) vs holding (a real cut)
# Creep detector: a zone of ASSIGNED frames whose content is foreign to the reference.
# Thresholds sit in the gap measured on 8 pairs: healthy zones have cos>=0.22 and last <=1.0 s,
# the defect has cos<=0.02 and lasts 11.3 s; both keep a >x1.5 margin to either side.
CREEP_COS = 0.15         # median cos of an assigned zone below this means the frames are foreign
CREEP_MIN_S = 3.0        # a zone longer than this is dropped (shorter is left alone; _level_decide is the arbiter)
CREEP_BRIDGE_S = 1.0     # bridges over already-dropped frames inside a zone


def _extract_base(video: Path, ffmpeg: str, channels: int = 2,
                  atrack: int = 0, delay_s: float = 0.0) -> list[str]:
    """Shared arguments for decoding audio → PCM s16le @ SR, `channels` channels. ffmpeg decodes
    any input codec (AAC/FLAC/AC3/PCM); `-ar SR` resamples ONLY when the input rate differs
    (44.1 kHz input → no-op). `-ac channels`: 2 (a downmix) for the reference/analysis, and the
    dub's own native channel count for the OUTPUT dub (2.0/5.1/7.1 → no-op remix, layout kept).

    `aresample=async=1:first_pts=0` is always applied — the decision is data-driven, not a
    switch. Justified by measurement:
      • It fixes PTS gaps in the audio. A container gap is a break in the timestamps, not
        silence (`silencedetect` cannot see it): plain decoding concatenates the PCM straight
        through and every later sample shifts. Measured on a stand with a 3.02 s gap: without
        the filter, 6 of 11 windows fall outside ±80 ms (worst case 3310 ms); with the filter,
        0 of 13.
      • It is safe: on a clean input it is bit-exact (md5 matched on synthetic input and on a
        real reference/dub pair — the output was identical, residual 1.1489566 in both runs);
        across a stand of 10 dubs it changed none of the other cases.
      • `async=1` only fills or trims, it never stretches (measured: 440/880 Hz tones were kept
        exact; stretching only starts above async=1).
    atrack is the file's audio-track index, for files carrying more than one audio track; 0
    selects the first.

    delay_s — container delay video_start − audio_start (features.probe_av_delay): the audio is
    laid on the VIDEO axis because SRM takes frame time from index 0. 0 → command unchanged."""
    # Delay filters go before aresample: first_pts=0 would otherwise refill the trimmed head.
    af = ",".join([*av_delay_filters(delay_s), "aresample=async=1:first_pts=0"])
    return [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-map", f"0:a:{int(atrack)}", "-af", af,
            "-ac", str(channels), "-ar", str(SR), "-f", "s16le"]


def _pump_progress(stream, dur, reporter) -> None:
    """Feed ffmpeg `-progress` lines (out_time_us) into the stage reporter; with no reporter
    just drain the pipe so ffmpeg never blocks on a full buffer. Accepts bytes and str."""
    for raw in stream:
        line = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw
        if reporter is not None and dur > 0 and line.startswith("out_time_us="):
            try:
                us = int(line.split("=", 1)[1])
            except ValueError:
                continue
            reporter(min(us / 1e6 / dur, 0.999))


def _decode_audio(video: Path, ffmpeg: str, *, channels: int = 2,
                 atrack: int = 0, reporter: Reporter | None = None,
                 delay_s: float | None = None) -> np.ndarray:
    """Decode audio (`channels` channels) to float32 (ENTIRELY in RAM) straight from the ffmpeg
    pipe — no temporary file on disk. PCM s16le → stdout; progress comes over stderr
    (`-progress pipe:2`), read on a separate thread (otherwise either pipe fills and blocks).
    Returns (N, channels) float32 (int16 range).
    delay_s=None → the container delay is probed here; callers that trace it pass it in."""
    if delay_s is None:
        delay_s = probe_av_delay(video, FFPROBE, atrack=atrack)
    dur = (probe_duration(video, FFPROBE) or 0.0) if reporter is not None else 0.0
    cmd = [*_extract_base(video, ffmpeg, channels, atrack, delay_s),
           "-progress", "pipe:2", "-nostats", "pipe:1"]
    proc = procreg.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    th = threading.Thread(target=_pump_progress, args=(proc.stderr, dur, reporter), daemon=True)
    th.start()
    buf = proc.stdout.read()
    proc.wait(); procreg.done(proc); th.join(timeout=2)
    if proc.returncode not in (0, None):
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    a = np.frombuffer(buf, np.int16)
    return a[: (a.size // channels) * channels].reshape(-1, channels).astype(np.float32)


def _decode_audio_mmap(video: Path, ffmpeg: str, *, channels: int = 2,
                      atrack: int = 0, reporter: Reporter | None = None,
                      dest: Path | None = None, delay_s: float | None = None,
                      scratch: tmpfiles.Workspace | None = None):
    """low_mem: ffmpeg decodes PCM s16le (`channels` channels) straight into the working file
    `a.raw`, with no WAV wrapper and no re-reading, and `np.memmap` opens it: RAM does not grow
    with the length of the track. PCM goes to the file, so stdout stays free for
    `-progress pipe:1`. Values are bit for bit those of _decode_audio. -> int16 memmap [N, channels].

    The raw file lives in `scratch` and goes away with its owner; with `dest` (the audio
    checkpoint) it is written straight there and outlives the run."""
    if dest is not None:
        dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
        raw = dest
    else:
        raw = scratch.sub("extract") / "a.raw"
    if delay_s is None:
        delay_s = probe_av_delay(video, FFPROBE, atrack=atrack)
    dur = (probe_duration(video, FFPROBE) or 0.0) if reporter is not None else 0.0
    cmd = [*_extract_base(video, ffmpeg, channels, atrack, delay_s),
           "-progress", "pipe:1", "-nostats", str(raw)]
    proc = procreg.popen(cmd, stdout=subprocess.PIPE, text=True)
    _pump_progress(proc.stdout, dur, reporter)   # drains stdout to the end
    proc.wait(); procreg.done(proc)
    if proc.returncode not in (0, None):
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    a = np.memmap(raw, dtype=np.int16, mode="r")
    return a[: (a.size // channels) * channels].reshape(-1, channels)


def decode_reference_audio(video: Path, ffmpeg: str, *, atrack: int, reporter: Reporter | None,
                           low_mem: bool, scratch: tmpfiles.Workspace | None):
    """Reference sound, stereo, sample values as int16 range. In low_mem it is a file-backed int16
    array in `scratch`, otherwise float32 in RAM; the values are the same."""
    if low_mem:
        return _decode_audio_mmap(video, ffmpeg, atrack=atrack, reporter=reporter, scratch=scratch)
    return _decode_audio(video, ffmpeg, atrack=atrack, reporter=reporter)


_REF_COPY_BLK = 60 * SR          # samples copied at once: memory stays O(block) whatever the duration


def _reference_buffer(ref_audio, n_out: int, *, low_mem: bool, scratch: tmpfiles.Workspace | None):
    """Reference sound on the pair's output grid: (n_out, 2) float32, silence past its end.

    `ref_audio` is the sound the episode decoded once for all its dubs. None means the reference has
    no usable sound: the audio layer is then impossible and the pair goes on without it.
    """
    if ref_audio is None:
        return None
    try:
        ra = ref_audio
        mr = min(len(ra), n_out)
        if low_mem:
            buf = np.memmap(scratch.sub("refbuf") / "ref.f32", dtype=np.float32, mode="w+", shape=(n_out, 2))
        else:
            buf = np.zeros((n_out, 2), np.float32)
        for s in range(0, mr, _REF_COPY_BLK):
            e = min(s + _REF_COPY_BLK, mr)
            buf[s:e] = ra[s:e]
        if low_mem and mr < n_out:
            buf[mr:] = 0.0
        return buf
    except Exception:  # noqa: BLE001 — no usable reference sound
        return None


def _write_audio_streamed(path: Path, out, ffmpeg: str, *, layout: str | None = None, on_prog=None) -> None:
    """Multichannel int16 → FLAC (lossless, level 8) STREAMED through ffmpeg, with no intermediate
    WAV: int16 chunks on stdin (`-f s16le`) → `-c:a flac -compression_level 8`.
    out is (N,C) float32 (memmap or ndarray); the whole array is never held in RAM at once, and the
    channel count C is read from out. layout, when given, carries the dub's channel layout
    (5.1/7.1) so the output gets the correct channel tag. Samples are bit-exact to what an
    intermediate WAV would hold (the same int16 conversion); FLAC decodes them losslessly.

    Level 8 (not 12): encoding is ×6 faster for about +1.6% size at the SAME decoded track (FLAC
    is lossless: the PCM is bit-exact regardless of level — the level only trades off speed
    against size)."""
    ch = out.shape[1] if out.ndim == 2 else 1
    lay = ["-channel_layout", layout] if layout else []
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "s16le", "-ar", str(SR), "-ac", str(ch), *lay, "-i", "pipe:0",
           "-c:a", "flac", "-compression_level", "8", str(path)]
    # Capture stderr: if the receiver dies mid-write, writing to its pipe raises a bare
    # BrokenPipeError with no reason. The real reason lives in ffmpeg's own stderr and is
    # lost for good unless it is read here.
    n = len(out)
    logger.info("запись {}: {} сэмплов × {} кан. ({:.1f} мин, ~{:.1f} ГБ int16), раскладка {}",
                path.name, n, ch, n / SR / 60, n * ch * 2 / 2**30, layout or "по числу каналов")
    proc = procreg.popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    BLK = 1 << 20
    broken = None
    try:
        for s in range(0, n, BLK):
            proc.stdin.write(np.ascontiguousarray(out[s:s + BLK]).astype(np.int16).tobytes())
            if on_prog is not None:
                on_prog(min(1.0, (s + BLK) / max(1, n)))
    except (BrokenPipeError, OSError) as e:
        broken = e                      # the receiver closed the pipe; its actual complaint is read from stderr below
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
    err = (proc.stderr.read() or b"").decode("utf-8", "replace").strip() if proc.stderr else ""
    rc = proc.wait(); procreg.done(proc)
    if broken is not None or rc != 0:
        written = path.stat().st_size if path.exists() else 0
        raise RuntimeError(
            f"запись {path.name} не удалась: код {rc}"
            + (f", обрыв канала ({broken})" if broken is not None else "")
            + f", записано {written / 2**20:.1f} МБ, ожидалось сэмплов {n}"
            + (f"; ffmpeg: {err[-2000:]}" if err else "; ffmpeg промолчал"))


def _fill_silence_from_ref(out, ref_buf, *, sr=SR, sil_db=SIL_FILL_DB,
                           win_s=SIL_FILL_WIN_S, min_dur_s=SIL_FILL_MIN_S, xfade=XFADE):
    """FINAL pass: fill the dub's SILENCE with the REFERENCE's sound. Called ONLY AFTER the full
    audio alignment (band/muq) — once the dub is already in sync with the reference (on an
    unsynced track, inserting the reference would create a false shift and manufacture cuts for
    the audio analysis, which is why filling from the reference by video cuts before analysis is
    forbidden). By this point the analysis is done, and the reference sits 1-to-1 on the REF grid
    and in sync.

    Zones where the dub is silent (RMS < sil_db) AND the reference has sound (RMS >= sil_db) are
    replaced with the reference, with an XFADE crossfade at the seams. Where BOTH are silent,
    silence stays. The dub's own sound is never lost (only gaps are touched). Mutates out in
    place. -> seconds filled."""
    n = out.shape[0]; win = int(win_s * sr)
    if win < 1 or n < 2 * win:
        return 0.0
    om = out.mean(1) if out.ndim == 2 else out
    rm = ref_buf.mean(1) if ref_buf.ndim == 2 else ref_buf
    k = min(len(om), len(rm)) // win
    if k < 1:
        return 0.0

    def rms_db(sig):
        s = np.asarray(sig[:k * win], np.float64).reshape(k, win)
        return 20.0 * np.log10(np.sqrt((s * s).mean(1)) / 32768.0 + 1e-12)   # dBFS (int16 fullscale)

    do = rms_db(om); dr = rms_db(rm)
    fill = (do < sil_db) & (dr >= sil_db)             # dub is silent, reference has sound
    minw = max(1, int(round(min_dur_s / win_s)))
    filled = 0; i = 0
    while i < k:
        if not fill[i]:
            i += 1; continue
        j = i
        while j < k and fill[j]:
            j += 1
        if j - i >= minw:
            s0 = i * win; s1 = min(j * win, n, ref_buf.shape[0])
            if s1 > s0:
                dub = out[s0:s1].copy(); rb = ref_buf[s0:s1]
                if rb.ndim == 2 and out.ndim == 2 and rb.shape[1] != out.shape[1]:
                    # dub is multichannel (5.1/7.1), reference is stereo: fill the gap with a
                    # mono downmix of the reference broadcast to every dub channel
                    rb = np.broadcast_to(rb.mean(1, keepdims=True), (rb.shape[0], out.shape[1]))
                out[s0:s1] = rb
                X = min(xfade, (s1 - s0) // 2)
                if X >= 1:
                    w = np.linspace(0.0, 1.0, X)[:, None]
                    out[s0:s0 + X] = dub[:X] * (1 - w) + rb[:X] * w          # dub to reference
                    out[s1 - X:s1] = rb[-X:] * (1 - w) + dub[-X:] * w        # reference to dub
                filled += s1 - s0
        i = j
    return filled / sr


def _recover_side(sub_syn, sub_ref, pred, syn_off, ref_off):
    """One extra pass: a dub sub-chunk × a reference sub-chunk → band_align → write the matches
    that come back (cos > EDGE_COS_MIN) into pred. Returns: the number of frames written in."""
    if len(sub_syn) < 5 or len(sub_ref) < 5:
        return 0
    off_sub, nrel, _, _ = coarse_robust(sub_syn, sub_ref)
    if nrel < 5:
        return 0
    pred_sub = band_align(sub_syn, sub_ref, off_sub, affine=True, DSYN=DSYN,
                          MATCH_THR=MATCH_THR)
    rec = np.where(pred_sub >= 0)[0]
    if not len(rec):
        return 0
    cosr = np.einsum("ij,ij->i", sub_syn[rec].astype(np.float32),
                     sub_ref[pred_sub[rec]].astype(np.float32))
    good = rec[cosr > EDGE_COS_MIN]
    for k in good:
        pred[syn_off + int(k)] = ref_off + int(pred_sub[int(k)])
    return int(len(good))


def _recover_edges(syn, refv, pred):
    """Extra passes at the edges (post-processing that only touches the edges, never the matching
    core). free_start can throw out a MATCHING start/end of the dub (e.g. a shared opening with a
    different lead-in on BD vs WEB). Take the discarded dub chunk against the uncovered reference
    chunk, run band_align on that sub-problem and write back the matches that come back. Mutates
    pred. Returns: the number of frames recovered. Runs ONLY when there is a sizeable drop AND an
    uncovered reference chunk (ordinary tracks with ads at the start do not trigger it: once the ad
    is dropped, the reference is covered from ~0)."""
    asg = np.where(pred >= 0)[0]
    if len(asg) < 2:
        return 0
    n = len(syn); rn = len(refv)
    rec = 0
    a0 = int(asg[0]); r0 = int(pred[a0])                  # left edge
    if a0 > EDGE_MIN_FR and r0 > EDGE_MIN_FR:
        rec += _recover_side(syn[:a0], refv[:r0], pred, 0, 0)
    aN = int(asg[-1]); rN = int(pred[aN])                 # right edge
    if (n - 1 - aN) > EDGE_MIN_FR and (rn - 1 - rN) > EDGE_MIN_FR:
        rec += _recover_side(syn[aN + 1:], refv[rN + 1:], pred, aN + 1, rN + 1)
    return rec


def _creep_drop(syn, refv, pred, fps_dub):
    """"Creep" detector (a post-pass — never touches the band_align core).

    The defect: an insert in the dub (a repeat of its own content with overlaid text) gets
    assigned by Drop-DTW as a "creeping" run instead of being dropped — the twin frames sit AHEAD
    of it in the reference, monotonicity forbids assigning them there, and the familiar content
    blocks a clean drop. The signal is glaring: the assigned frames are, by content, FOREIGN to
    their reference partners (cos≈0.00-0.02 for 11.5 s in a row), while honest matches keep
    cos≥0.22 even in dark scenes and never sag for longer than 1 s.

    Mechanics: a sliding median (~1 s window) of the cos of assigned frames; connected zones below
    CREEP_COS (bridged over already-dropped frames up to CREEP_BRIDGE_S) that last ≥ CREEP_MIN_S
    → pred=-1. What happens next is decided by _level_decide: an offset level shift across the
    zone → a real insert (stays dropped, the reference gap gets filled with the original); no
    shift → restored by interpolation (the map is unchanged) — a false positive is harmless.

    Mutates pred. Returns: [(j0, j1, mean_cos, slope)…] of the dropped zones (telemetry)."""
    asgm = pred >= 0
    asg = np.where(asgm)[0]
    if len(asg) < 2:
        return []
    cosv = np.full(len(pred), np.nan, np.float32)
    CH = 4096
    for i in range(0, len(asg), CH):
        sl = asg[i:i + CH]
        a = syn[sl].astype(np.float32); b = refv[pred[sl]].astype(np.float32)
        cosv[sl] = np.einsum("ij,ij->i", a, b) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    W = max(3, int(fps_dub) | 1)
    half = W // 2
    sw = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([np.full(half, np.nan, np.float32), cosv,
                        np.full(half, np.nan, np.float32)]), W)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN windows occur on drop runs
        med = np.nanmedian(sw[:len(pred)], axis=1)
    low = asgm & (med < CREEP_COS)
    BR = int(CREEP_BRIDGE_S * fps_dub)
    zones: list[list[int]] = []
    for s, e in _runs1d(np.where(low)[0]):
        # bridge: merge with the previous zone when no assigned frames sit between them
        if zones and s - zones[-1][1] <= BR and not asgm[zones[-1][1] + 1:s].any():
            zones[-1][1] = e
        else:
            zones.append([s, e])
    out = []
    MIN = CREEP_MIN_S * fps_dub
    for s, e in zones:
        if e - s + 1 < MIN:
            continue
        slope = (int(pred[e]) - int(pred[s])) / max(1, e - s)
        out.append((int(s), int(e), float(np.nanmean(cosv[s:e + 1])), float(slope)))
        pred[s:e + 1] = -1
    return out


def _runs1d(idx):
    """Runs of consecutive indices → [(s,e)…]."""
    out = []
    if not len(idx):
        return out
    s = p = int(idx[0])
    for j in idx[1:]:
        if j == p + 1:
            p = int(j)
        else:
            out.append((s, p)); s = p = int(j)
    out.append((s, p))
    return out


def _level_decide(pred, fps_ref, fps_dub, n_ref, n_syn, *,
                  edit_s=LEVEL_EDIT_S, min_ins_s=LEVEL_MIN_INS_S, win_s=LEVEL_WIN_S,
                  cut_min_s=CUT_MIN_S):
    """A single robust rule that decides edits from the offset LEVEL shift.

    Idea: a real edit = a STANDING shift of the offset LEVEL; a blind zone = the level holds.
    For each dropped dub block (drop-syn), compare the offset median BEFORE and AFTER (window
    win_s). If |Δ| ≤ edit_s OR the block is shorter than min_ins_s (jitter) → a BLIND zone →
    restore by interpolation (this also covers the matching reference gap). Otherwise → a real
    insert → leave it dropped. Afterward: any remaining reference gap > cut_min_s is a real CUT
    (proven: a reference gap during continuous dub audio always shifts the level, so no blind
    gaps remain). Uncovered edge zones of the reference (> FILL_MIN_S) are cuts too (anti-freeze).

    One physical threshold, in ABSOLUTE frames rather than a ratio (a shift of ~1 s, a block of
    ~2 s) — more robust than a ratio-based threshold.
    Returns: (pred, cut_intervals[(R1,R2)…], n_restored)."""
    pred = pred.copy()
    asg = np.where(pred >= 0)[0]
    if len(asg) < 2:
        return pred, [], 0
    off_asg = pred[asg].astype(np.float64) - asg
    W = max(1, int(win_s * fps_dub))
    EDIT = edit_s * fps_dub
    MIN_INS = min_ins_s * fps_dub

    def lvl(frame, before):
        m = ((asg < frame) & (asg >= frame - W)) if before else ((asg > frame) & (asg <= frame + W))
        return float(np.median(off_asg[m])) if m.any() else None

    n_restored = 0
    for a, b in _runs1d(np.where(pred == -1)[0]):
        prev = asg[asg < a]; nxt = asg[asg > b]
        if not len(prev) or not len(nxt):
            continue
        lb = lvl(a, True); la = lvl(b, False)
        if lb is None or la is None:
            continue
        if not (abs(la - lb) > EDIT and (b - a + 1) > MIN_INS):    # a blind gap or jitter: restore it
            p0, p1 = int(prev[-1]), int(nxt[0])
            pred[a:b + 1] = np.round(np.interp(
                np.arange(a, b + 1), [p0, p1], [pred[p0], pred[p1]])).astype(np.int64)
            n_restored += b - a + 1
        # otherwise it is a real insert: leave it dropped (its audio drops out, filled later with the reference)

    # --- CUTS: a reference gap is a cut only when the offset level HOLDS (rules out a sawtooth) ---
    # Neighbouring gaps merge into one cluster; if the offset level RETURNS (an outlier on static
    # content), restore it by interpolation (dub audio kept, no silence). If the level holds, it
    # is a real cut: the cluster collapses into ONE interval, filled with the reference when it
    # exceeds a second, rather than a sawtooth of silences under a second.
    asg = np.where(pred >= 0)[0]
    ratio = fps_ref / fps_dub
    off = pred[asg].astype(np.float64) - asg * ratio
    Wc = max(1, int(win_s * fps_dub)); EDIT_R = edit_s * fps_ref
    cand = [i for i in range(len(asg) - 1)
            if int(pred[asg[i + 1]]) - int(pred[asg[i]]) > cut_min_s * fps_ref]
    COAL = LEVEL_CUT_COALESCE_S * fps_dub
    clusters: list[list[int]] = []
    for i in cand:
        if clusters and (asg[i] - asg[clusters[-1][-1] + 1]) <= COAL:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    cut_intervals: list[tuple[int, int]] = []
    for cl in clusters:
        i0, i1 = cl[0], cl[-1]
        bef = off[(asg < asg[i0]) & (asg >= asg[i0] - Wc)]
        lb = float(np.median(bef)) if len(bef) else float(off[i0])
        recover = None; lim = asg[i1] + LEVEL_CUT_RECOVER_S * fps_dub
        for k in range(i1 + 1, len(asg)):
            if asg[k] > lim:
                break
            if abs(off[k] - lb) <= EDIT_R:
                recover = k; break
        if recover is not None:                       # an outlier that returns (a sawtooth): restore it
            d0, d1 = int(asg[i0]), int(asg[recover])
            pred[d0:d1 + 1] = np.round(np.interp(np.arange(d0, d1 + 1), [d0, d1],
                                                 [pred[d0], pred[d1]])).astype(np.int64)
            n_restored += d1 - d0 + 1
        else:                                          # a holding shift: one cut per cluster
            cut_intervals.append((int(pred[asg[i0]]) + 1, int(pred[asg[i1 + 1]])))
    asg = np.where(pred >= 0)[0]                       # recomputed after the restores above
    # uncovered edge zones of the reference are cuts too; filling in the very start is left untouched here
    r_first = int(pred[int(asg[0])]); r_last = int(pred[int(asg[-1])])
    if r_first > FILL_MIN_S * fps_ref:
        cut_intervals.insert(0, (0, r_first))
    if (n_ref - r_last) > FILL_MIN_S * fps_ref:
        cut_intervals.append((r_last, n_ref))
    return pred, sorted(set(cut_intervals)), n_restored


def _cos_anchors(syn, refv, pred, asg):
    """cos of every assigned frame (asg) to its reference partner (SRM L2-normalized to ~1) — the
    weight used by the vision analyzer (the video map) and the unified plot."""
    cos = np.empty(len(asg), np.float32)
    CH = 4096
    for i in range(0, len(asg), CH):
        sl = asg[i:i + CH]
        a = syn[sl].astype(np.float32); b = refv[pred[sl]].astype(np.float32)
        cos[i:i + len(sl)] = np.einsum("ij,ij->i", a, b) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    return cos


# ── tmp-mode checkpoints: stage versions + the align stamp (CK3 validity) ──
ALIGN_VER = 3   # matching + vision map (band_align/_level_decide/vision_detect); bump when they change


def _align_stamp(dub_audio, ref, fps_ref, fps_tst) -> str:
    """CK3 stamp (the alignment result): changes when the input (files), versions (EMB/ALIGN), or
    options affecting the matching+map change. A match → __vision.npz can be reused."""
    import json as _j
    return _j.dumps({
        "emb": cache_mod.EMB_VER, "align": ALIGN_VER,
        "dub": cache_mod.cache_key(Path(dub_audio)),
        "ref": cache_mod.cache_key(Path(ref.src)) if ref.src else "",
        "fr": round(float(fps_ref), 6), "ft": round(float(fps_tst), 6),
    }, sort_keys=True)


def _load_align_ckpt(out_path: Path, n_syn: int, stamp: str):
    """CK3: returns (pred, asg, cut_intervals) from `<stem>__vision.npz` when the stamp matches,
    otherwise None. pred is rebuilt as pred[asg]=pred_asg. The vision map (grid/tg_s/vision_cuts)
    is NOT cached — it is cheap to recompute from pred/asg via build_map below (bit-exact)."""
    p = Path(out_path).parent / "_plots" / (Path(out_path).stem + "__vision.npz")
    if not p.exists():
        return None
    try:
        # Numbers and one string: nothing in the file needs object loading, so none is allowed.
        with np.load(str(p)) as d:
            if "stamp" not in d.files or str(d["stamp"]) != stamp or "cut_intervals" not in d.files:
                return None
            asg = np.asarray(d["asg"], np.int64)
            pred = np.full(int(n_syn), -1, np.int64)
            if len(asg):
                pred[asg] = np.asarray(d["pred_asg"], np.int64)
            ci = [(int(a), int(b)) for a, b in np.asarray(d["cut_intervals"], np.int64).reshape(-1, 2)]
        return pred, asg, ci
    except Exception:  # noqa: BLE001 — a broken cache entry triggers a recompute
        return None


@dataclass
class _Match:
    """Features and everything measured on them. Replaced as one unit whenever the features
    change (detelecine, geom rebuild), so no decision can read a value from an older state."""

    refv: np.ndarray
    syn: np.ndarray
    fps_ref: float
    fps_dub: float
    ax_ref: np.ndarray | None
    ax_dub: np.ndarray | None
    tc_ref: dict
    tc_dub: dict
    state: str                                   # "plain" | "detelecine" | "mirror" | "geom"
    off0: np.ndarray | None = None
    nrel: int = 0
    n_keep: int = -1                             # -1: no coarse pass on this state (CK3 reuse)
    chain: np.ndarray | None = None
    mirror: bool = False                         # dub features are those of the horizontally mirrored frames

    @property
    def n(self) -> int:
        return len(self.syn)

    @property
    def strong_thr(self) -> float:
        return COARSE_SAME_MIN_FRAC * self.n / 8  # K=8 is the thinning step of coarse._anchors

    @property
    def coarse_strong(self) -> bool:
        return self.n_keep >= self.strong_thr


def _frozen_runs(syn, fps: float, ax=None) -> list[tuple[float, float]]:
    """Runs of identical consecutive frames longer than FREEZE_MIN_S, in seconds. Probes the rows the
    coarse pass thins to (every K-th frame, so they are already in the page cache after it) and
    compares each with the probe two steps later; a telecine cadence dents single probes, so a run
    survives dips up to FREEZE_GAP_S."""
    K = 8; n = len(syn)
    idx = np.arange(0, n - 2 * K, K)
    if len(idx) < 4:
        return []
    a = np.asarray(syn[idx]).astype(np.float32); b = np.asarray(syn[idx + 2 * K]).astype(np.float32)
    cos = np.einsum("ij,ij->i", a, b)
    t = (ax[idx] if ax is not None else idx / fps).astype(np.float64)
    runs: list[tuple[float, float]] = []; start = None; last_hi = None
    for ti, c in zip(t, cos):
        if c >= FREEZE_COS:
            start = ti if start is None else start; last_hi = ti
        elif start is not None and ti - last_hi > FREEZE_GAP_S:
            if last_hi - start >= FREEZE_MIN_S:
                runs.append((float(start), float(last_hi)))
            start = None
    if start is not None and last_hi - start >= FREEZE_MIN_S:
        runs.append((float(start), float(last_hi)))
    return runs


def _run_coarse(m: _Match, low_mem: bool, rep: Reporter | None, trace: Trace) -> _Match:
    """Coarse pass on the given state; the chain and n_keep stay bound to that state."""
    if rep is not None:
        rep.mark(0.0)
    m.off0, m.nrel, m.n_keep, m.chain = (coarse_windowed(m.syn, m.refv, on_prog=rep) if low_mem
                                         else coarse_robust(m.syn, m.refv))
    trace.event("coarse", state=m.state, n_dub=m.n, n_ref=len(m.refv), anchors=int(m.nrel),
                n_keep=int(m.n_keep), chain_len=int(len(m.chain)),
                off0_median_s=float(np.median(m.off0)) / m.fps_ref if m.fps_ref else 0.0)
    return m


def _require_scratch(low_mem: bool, scratch: tmpfiles.Workspace | None) -> None:
    """low_mem keeps its buffers in files, and files need an owner that removes them."""
    if low_mem and scratch is None:
        raise ValueError("low_mem requires a scratch workspace")


def _mirrored_copy(v: np.ndarray, scratch: tmpfiles.Workspace | None) -> np.ndarray:
    """SRM of the mirrored frames; on disk in `scratch` like the other rebuilt features."""
    if scratch is not None:
        out = np.memmap(scratch.sub("mirror") / "f.f16", dtype=np.float16, mode="w+", shape=v.shape)
    else:
        out = np.empty(v.shape, np.float16)
    return mirror_srm(v, out=out)


def _try_mirror(m: _Match, low_mem: bool, scratch: tmpfiles.Workspace | None,
                rep: Reporter | None, trace: Trace) -> _Match | None:
    """Orientation second chance, before geometry: a frame sample is matched in both orientations
    (seconds, no decode); when the mirrored sample clearly wins, the coarse pass is redone on the
    mirrored features. Returns the new state or None; the decision is traced either way."""
    pr = orientation_probe(m.syn, m.refv)
    need = COARSE_SAME_MIN_FRAC * pr["sample"]
    win = pr["mirror"] >= need and pr["mirror"] > MIRROR_RATIO * max(pr["plain"], 1)
    trace.decide("mirror", state=m.state, inputs=dict(pr, n_keep=int(m.n_keep)),
                 thresholds={"min_strong": need, "MIRROR_RATIO": MIRROR_RATIO},
                 verdict="applied" if win else "none")
    if not win:
        return None
    mm = _Match(m.refv, _mirrored_copy(m.syn, scratch), m.fps_ref, m.fps_dub,
                m.ax_ref, m.ax_dub, m.tc_ref, m.tc_dub, "mirror", mirror=True)
    return _run_coarse(mm, low_mem, rep, trace)


def _detelecine_state(m: _Match, scratch: tmpfiles.Workspace | None, trace: Trace) -> _Match:
    """Thin the baked 3:2 cadence on both sides; a side with a live VFR axis is left alone
    because thinning would break its 1:1 index-to-time stitching."""
    if m.ax_ref is None:
        m.refv, m.fps_ref, m.tc_ref = _detelecine(m.refv, m.fps_ref, scratch=scratch)
    if m.ax_dub is None:
        m.syn, m.fps_dub, m.tc_dub = _detelecine(m.syn, m.fps_dub, scratch=scratch)
    m.state = "detelecine"
    m.off0 = None; m.nrel = 0; m.n_keep = -1; m.chain = None
    trace.event("detelecine", state=m.state, dropped_ref=int(m.tc_ref["dropped"]),
                dropped_dub=int(m.tc_dub["dropped"]), fps_ref=m.fps_ref, fps_dub=m.fps_dub, n_dub=m.n)
    return m


def _try_geom(ref, dub_audio, fps_ref, fps_dub, low_mem, cache_dir, keep_tmp,
              ffmpeg, rep: Reporter | None, trace: Trace, mirror: bool = False,
              scratch: tmpfiles.Workspace | None = None) -> tuple[_Match | None, dict | None]:
    """Geometry pass for a pair vision cannot match (crop/zoom/anamorph/bars): consensus_G finds
    the global transform, both SRM are rebuilt with `crop` (a second decode, only for such pairs)
    and the coarse pass runs on the rebuilt features. Returns the new state and G, or (None, None)
    with the reason recorded in the trace."""
    from track_muxer.conform import geom
    if ref.src is None:
        trace.decide("geom", state="geom", verdict="none", inputs={"reason": "reference has no source file"})
        return None, None
    # The LoFTR consensus is expensive and not deterministic, so a cached G is reused as is.
    G = (cache_mod.load_geom(cache_dir, Path(ref.src), Path(dub_audio))
         if (keep_tmp and cache_dir is not None) else None)
    if G is not None and bool(G.get("flip", False)) != mirror:   # cached for the other orientation
        G = None
    source = "cache"
    if G is None:
        if not geom.available():
            trace.decide("geom", state="geom", verdict="none", inputs={"reason": "geom backend unavailable"})
            return None, None
        if rep is not None:
            rep.mark(0.0, "оценка кадрирования и масштаба")
        G = geom.consensus_G(Path(ref.src), Path(dub_audio), on_prog=part(rep, 0.0, 0.40), flip=mirror)
        source = "computed"
        if G is None:
            trace.decide("geom", state="geom", verdict="none", inputs={"reason": "no consensus transform"})
            return None, None
        if keep_tmp and cache_dir is not None:
            cache_mod.save_geom(cache_dir, Path(ref.src), Path(dub_audio), G)
    trace.decide("geom", state="geom", verdict="applied", source=source,
                 inputs={"sx": G["sx"], "sy": G["sy"], "n_in": G["n_in"],
                         "crop_ref": G["crop_ref"], "crop_dub": G["crop_dub"], "mirror": mirror})
    if rep is not None:
        rep.mark(0.40, "пересчёт признаков кадров")

    def _srm(video: Path, fps: float, crop: str, sub: Reporter | None) -> SrmFeatures:
        # tmp checkpoint of the cropped SRM (CK1 geom): a repeat blind pair skips the decode
        h = probe_resolution(video)
        if low_mem and keep_tmp and cache_dir is not None:
            cached = cache_mod.load_srm(cache_dir, video, crop=crop)
            if cached is not None:
                return cached
            mp = cache_mod.srm_file(cache_dir, video, crop=crop)
            with decode_backend(h, ffmpeg) as _be:
                f = build_srm(video, fps, ffmpeg=ffmpeg, crop=crop, mmap_path=mp, backend=_be, reporter=sub)
            cache_mod.save_meta(cache_dir, video, len(f.srm), f.fps, crop=crop)
            return f
        mp = (scratch.sub("geomsrm") / "f.f16") if scratch is not None else None
        with decode_backend(h, ffmpeg) as _be:
            return build_srm(video, fps, ffmpeg=ffmpeg, crop=crop, mmap_path=mp, backend=_be, reporter=sub)

    ref2 = _srm(Path(ref.src), fps_ref, G["crop_ref"], part(rep, 0.40, 0.70))
    dub2 = _srm(Path(dub_audio), fps_dub, G["crop_dub"], part(rep, 0.70, 0.96))
    # Cropping keeps the frame set, so the VFR axes of the rebuilt SRM are the source's own.
    ax_ref2 = ref2.pts if (ref2.pts is not None and len(ref2.pts) == len(ref2.srm)) else None
    ax_dub2 = dub2.pts if (dub2.pts is not None and len(dub2.pts) == len(dub2.srm)) else None
    _no_tc = {"telecine": False, "tele_score": 0.0, "argmax": 0, "dropped": 0}
    syn2 = _mirrored_copy(dub2.srm, scratch) if mirror else dub2.srm
    m = _Match(ref2.srm, syn2, fps_ref, fps_dub, ax_ref2, ax_dub2, dict(_no_tc), dict(_no_tc), "geom",
               mirror=mirror)
    if m.ax_ref is None:
        m.refv, m.fps_ref, m.tc_ref = _detelecine(m.refv, m.fps_ref, scratch=scratch)
    if m.ax_dub is None:
        m.syn, m.fps_dub, m.tc_dub = _detelecine(m.syn, m.fps_dub, scratch=scratch)
    if rep is not None:
        rep.mark(0.96, "повторное грубое соответствие")
    return _run_coarse(m, low_mem, part(rep, 0.96, 1.0), trace), G


def _edge_silence(out, grid, tg_s, n_aud, sr, fill_spans, fade):
    """An EDGE cut = the dub is physically ABSENT at an edge of the reference timeline: HEAD (the
    dub starts later than the reference → tg_s<0) / TAIL (the dub is shorter than the reference →
    tg_s runs past the dub's end). Outside its body, build_curve extrapolates a LINE (curve=base)
    → warp clamps the dub's first/last sample into a CONSTANT rather than silence. That constant
    anomaly breaks band across the WHOLE track (normalization/segmentation are global).
    We zero it out with SILENCE (+fade) and record it in fill_spans as an internal cut: band
    ignores the zone (vision_spans), and _fill_silence_from_ref fills it with the reference's
    synchronized original sound. Only CONTINUOUS edges are handled here (internal cuts are
    silenced by the caller via vision_cuts). tg_s at the edge is monotone (the base line)."""
    tg = np.asarray(tg_s, float); n_out = out.shape[0]; dur_dub = n_aud / sr
    EDGE_MIN_S = 0.3                                              # an edge under 0.3 s is rounding/sub-frame noise, left alone

    def _zero(a_s, b_s):
        if b_s - a_s < EDGE_MIN_S:                               # a micro-edge (dub length matches the reference almost exactly): no-op
            return
        s1 = max(0, int(a_s * sr)); s2 = min(n_out, int(b_s * sr))
        if s2 - s1 <= 0:
            return
        out[s1:s2] = 0.0
        if s1 - fade >= 0:
            out[s1 - fade:s1] *= np.linspace(1, 0, fade)[:, None]
        if s2 + fade <= n_out:
            out[s2:s2 + fade] *= np.linspace(0, 1, fade)[:, None]
        fill_spans.append((float(a_s), float(b_s)))

    if len(tg) >= 2 and tg[0] < 0 and (tg >= 0.0).any():          # HEAD: the dub does not exist before grid[k]
        _zero(0.0, float(grid[int(np.argmax(tg >= 0.0))]))
    if len(tg) >= 2 and tg[-1] > dur_dub and (tg <= dur_dub).any():  # TAIL: the dub does not exist after grid[k]
        k = len(tg) - 1 - int(np.argmax((tg <= dur_dub)[::-1]))
        _zero(float(grid[k]), float(grid[-1]))


def conform_features(
    ref: SrmFeatures,
    dub: SrmFeatures,
    dub_audio: Path,
    out_path: Path,
    *,
    # Task parameters: a choice of the user about the method or the content of the result.
    # Nothing here is tuned to a particular track: each value has to give a result in sync.
    audio_method: str = "band",     # meter of the audio layer: "band" (48-band DSP) or "muq" (neural embedder)
    drift_speed_pct: float = 1.25,  # how fast the laid sound may follow a drift, % per second
    fill_silence: bool = True,      # fill the silence left in the dub with the reference sound, after the audio layer
    ref_audio: np.ndarray | None = None,
    ref_atrack: int = 0,            # audio track index of the reference (sound baseline for band/fill)
    dub_atrack: int = 0,            # audio track index of the dub (the track being aligned)
    ffmpeg: str = FFMPEG,
    progress=None,
    dub_name: str | None = None,
    progress_meta: tuple[int, int, str] = (0, 0, ""),
    low_mem: bool = False,
    cache_dir: Path | str | None = None,   # checkpoints CK2/CK3 (keep_tmp mode); None = off
    keep_tmp: bool = False,
    trace: Trace | None = None,     # decision trace of the pair; created here when the caller has none
    scratch: tmpfiles.Workspace | None = None,   # owner of the pair's temporary files; required with low_mem
) -> PairResult:
    """Align the dub (dub features + its audio dub_audio) onto the ref timeline → out_path.

    keep_tmp + cache_dir: checkpoints in `cache_dir` (audio CK2, alignment CK3); a rerun skips the
    audio decode and the matching. The SRM of the dub (CK1) is kept by conform_pair."""
    t0 = time.perf_counter()
    _require_scratch(low_mem, scratch)

    def _rep(stage: str, detail: str = "") -> Reporter | None:
        return Reporter.of(progress, stage, progress_meta, detail)

    name = dub_name or (dub.src.name if dub.src else dub_audio.name)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fps_ref, fps_tst = ref.fps, dub.fps            # frame rates come from the files
    fps0_ref, fps0_tst = fps_ref, fps_tst   # the originals, before detelecine, for the geometry second chance below
    # SRM stays f16 (as in the cache); upcast to f32 only on the slices being multiplied
    # (band_align/_recover/_metrics), so a full f32 copy is never materialized.
    syn = dub.srm
    refv = ref.srm
    if len(refv) == 0 or not (fps_ref and fps_ref > 0):   # an empty/broken reference SRM means its build failed; raise a clear
        raise ValueError(f"SRM рефа пуст/битый (кадров={len(refv)}, fps={fps_ref}) — пересоберите кэш рефа")  # error, not a division by zero
    if len(syn) == 0 or not (fps_tst and fps_tst > 0):
        raise ValueError(f"SRM дубля пуст/битый (кадров={len(syn)}, fps={fps_tst})")
    N = len(syn)

    # ── The real frame time axis (SrmFeatures.pts) is set only for VFR; otherwise None keeps
    #    the index/fps path bit-exact. Its length must match the SRM (1:1 index-to-index stitching). ──
    ax_ref = ref.pts if (ref.pts is not None and len(ref.pts) == len(refv)) else None
    ax_dub = dub.pts if (dub.pts is not None and len(dub.pts) == len(syn)) else None

    # ── Baked 3:2 telecine (NTSC rips bake a 23.976 film into 29.97 without IVTC, which makes
    #    the SRM noisy and the anchors jitter). Detected cheaply (a fixed sample) for the
    #    CK3 gate and the quality passport; the cadence thinning itself happens in the
    #    matching below (a no-op, bit-exact, on non-telecine input). A stream with a live VFR
    #    axis is never checked for telecine: its period-5 autocorrelation assumes an even grid
    #    by construction, and thinning would break the 1:1 index stitching. ──
    _tcr0 = _is_telecine(refv, fps_ref) if ax_ref is None else (False, 0.0, 0)
    _tcd0 = _is_telecine(syn, fps_tst) if ax_dub is None else (False, 0.0, 0)
    tc_ref = {"telecine": _tcr0[0], "tele_score": _tcr0[1], "argmax": _tcr0[2], "dropped": 0}
    tc_dub = {"telecine": _tcd0[0], "tele_score": _tcd0[1], "argmax": _tcd0[2], "dropped": 0}
    _is_tc = _tcr0[0] or _tcd0[0]

    # CK3 (tmp mode): the matching result (pred/asg/cut_intervals) comes from __vision.npz when
    # the stamp matches, skipping the costly coarse/band_align/_level_decide. The vision map
    # (build_map below) is cheap to rebuild from pred/asg, bit-exact.
    # The telecine CK3 track is not reused: decimation shifts reference indices, cached pred is stale.
    trace = trace if trace is not None else Trace(name)
    trace.event("telecine", state="plain", ref_fps=fps_ref, dub_fps=fps_tst, n_ref=len(refv), n_dub=N,
                ref=tc_ref, dub=tc_dub, vfr_ref=ax_ref is not None, vfr_dub=ax_dub is not None)
    n_restore = 0; n_recovered = 0; n_blind = 0
    geom_info = None                                  # G of the geometry pass when it was applied
    creep_zones: list[tuple[int, int, float, float]] = []
    cut_intervals: list[tuple[int, int]] = []
    m = _Match(refv, syn, fps_ref, fps_tst, ax_ref, ax_dub, tc_ref, tc_dub, "plain")
    freeze_runs: list[tuple[float, float]] = []
    _astamp = _align_stamp(dub_audio, ref, fps_ref, fps_tst) if keep_tmp else ""
    _ck3 = (_load_align_ckpt(out_path, N, _astamp)
            if (keep_tmp and cache_dir is not None and not _is_tc) else None)
    cache_mod._a("CK3 зрение", _ck3 is not None, out_path.stem)        # reuses the matching result, skipping the GPU matching pass
    trace.decide("ck3_reuse", state="cache", source="cache",
                 inputs={"checkpoints": bool(keep_tmp and cache_dir is not None), "telecine": _is_tc},
                 verdict="hit" if _ck3 is not None else "miss")
    _band = None

    def _match(m: _Match, detail: str):
        """Band + edge/creep/level passes on one state; the chain is the one coarse found on it."""
        nonlocal _band
        _band = _rep("band", detail)
        if _band is not None:
            _band.mark(0.0)
        pred = band_align(m.syn, m.refv, m.off0, affine=True, DSYN=DSYN, MATCH_THR=MATCH_THR,
                          on_prog=part(_band, 0.0, 0.85), chain_aj=m.chain,
                          chain_off=(m.off0[m.chain] if m.chain is not None else None))
        if _band is not None:
            _band.mark(0.85, "разбор уровней и краёв")
        raw = int((pred >= 0).sum())
        n_rec = _recover_edges(m.syn, m.refv, pred)
        creep = _creep_drop(m.syn, m.refv, pred, m.fps_dub)
        pred, cuts, n_res = _level_decide(pred, m.fps_ref, m.fps_dub, len(m.refv), m.n)
        asg = np.where(pred >= 0)[0]
        trace.event("band", state=m.state, chain_len=int(len(m.chain)) if m.chain is not None else 0,
                    assigned_raw_pct=100.0 * raw / max(1, m.n), edge_recovered=int(n_rec),
                    creep_zones=len(creep), cuts=len(cuts), restored=int(n_res),
                    assigned_pct=100.0 * len(asg) / max(1, m.n))
        return pred, asg, n_rec, creep, cuts, n_res

    if _ck3 is not None:
        pred, asg, cut_intervals = _ck3
    else:
        m = _run_coarse(m, low_mem, _rep("coarse", "оценка общего смещения"), trace)
        freeze_runs = _frozen_runs(m.syn, m.fps_dub, m.ax_dub)      # after coarse: its thinned rows are cached
        trace.event("video_freeze", state=m.state, runs=[(round(a, 2), round(b, 2)) for a, b in freeze_runs],
                    total_s=round(sum(b - a for a, b in freeze_runs), 2))
        # Too few anchors means vision is blind (crop/zoom/anamorph/bars): geometry before anything else.
        blind = m.n_keep < GEOM_GATE
        trace.decide("geom_gate", state=m.state, inputs={"n_keep": m.n_keep},
                     thresholds={"GEOM_GATE": GEOM_GATE},
                     verdict="geom" if blind else ("detelecine" if _is_tc else "band"))
        if blind:
            # Orientation is checked before geometry: the probe costs seconds and needs no decode.
            mm = _try_mirror(m, low_mem, scratch, _rep("coarse", "проверка ориентации кадра"), trace)
            if mm is not None:
                m = mm
                if _is_tc:
                    m = _detelecine_state(m, scratch, trace)
                    m = _run_coarse(m, low_mem, _rep("coarse", "повторная оценка после прореживания каденса"), trace)
                blind = m.n_keep < GEOM_GATE
        if blind:
            gm, G = _try_geom(ref, dub_audio, fps_ref, fps_tst, low_mem,
                              cache_dir, keep_tmp, ffmpeg, _rep("geom"), trace, mirror=m.mirror,
                              scratch=scratch)
            if G is not None:
                m, geom_info = gm, G
        elif _is_tc and m.state == "plain":
            m = _detelecine_state(m, scratch, trace)
            m = _run_coarse(m, low_mem, _rep("coarse", "повторная оценка после прореживания каденса"), trace)
        pred, asg, n_recovered, creep_zones, cut_intervals, n_restore = _match(
            m, "сопоставление кадров в полосе поиска")

    assigned_pct = 100.0 * len(asg) / max(1, m.n)
    # A low assigned share is only a proxy for a foreign video: a soft zoom leaves enough coarse
    # anchors to pass GEOM_GATE yet starves the band, so geometry gets a second chance first.
    second = assigned_pct < ABORT_ASSIGNED_PCT and not m.coarse_strong and geom_info is None
    trace.decide("geom_second_chance", state=m.state,
                 inputs={"assigned_pct": assigned_pct, "n_keep": m.n_keep, "geom_used": geom_info is not None},
                 thresholds={"ABORT_ASSIGNED_PCT": ABORT_ASSIGNED_PCT, "strong_thr": m.strong_thr},
                 verdict="try" if second else "skip")
    if second and not m.mirror:
        mm = _try_mirror(m, low_mem, scratch, _rep("coarse", "проверка ориентации кадра"), trace)
        if mm is not None:
            m = mm
            if _is_tc:
                m = _detelecine_state(m, scratch, trace)
                m = _run_coarse(m, low_mem, _rep("coarse", "повторная оценка после прореживания каденса"), trace)
            pred, asg, n_recovered, creep_zones, cut_intervals, n_restore = _match(
                m, "сопоставление кадров после отражения")
            assigned_pct = 100.0 * len(asg) / max(1, m.n)
            second = assigned_pct < ABORT_ASSIGNED_PCT and not m.coarse_strong
    if second:
        # fps0: the current ones may already be thinned by detelecine, and the geometry pass
        # rebuilds the SRM from the files and thins on its own.
        gm, G = _try_geom(ref, dub_audio, fps0_ref, fps0_tst, low_mem,
                          cache_dir, keep_tmp, ffmpeg, _rep("geom"), trace, mirror=m.mirror,
                          scratch=scratch)
        if G is not None:
            m, geom_info = gm, G
            pred, asg, n_recovered, creep_zones, cut_intervals, n_restore = _match(
                m, "сопоставление кадров после коррекции")
            assigned_pct = 100.0 * len(asg) / max(1, m.n)
    refv, syn, fps_ref, fps_tst, N = m.refv, m.syn, m.fps_ref, m.fps_dub, m.n
    ax_ref, ax_dub, tc_ref, tc_dub, n_keep = m.ax_ref, m.ax_dub, m.tc_ref, m.tc_dub, m.n_keep
    _geom_kw = dict(mirror_used=bool(m.mirror), geom_used=geom_info is not None,
                    geom_n_in=int(geom_info["n_in"]) if geom_info else 0,
                    geom_sx=float(geom_info["sx"]) if geom_info else 0.0,
                    geom_sy=float(geom_info["sy"]) if geom_info else 0.0,
                    telecine_ref=bool(tc_ref["telecine"]), telecine_dub=bool(tc_dub["telecine"]),
                    tele_score=round(max(tc_ref["tele_score"], tc_dub["tele_score"]), 3),
                    tc_dropped=int(tc_ref["dropped"]) + int(tc_dub["dropped"]))
    # Foreign video only when both signals agree; a strong coarse chain with a low assigned share
    # is a low-cosine encode of the same episode and proceeds with a warning.
    low_cos_proceed = assigned_pct < ABORT_ASSIGNED_PCT and m.coarse_strong
    foreign = assigned_pct < ABORT_ASSIGNED_PCT and not m.coarse_strong
    trace.decide("foreign_gate", state=m.state,
                 inputs={"assigned_pct": assigned_pct, "n_keep": m.n_keep, "n_dub": m.n},
                 thresholds={"ABORT_ASSIGNED_PCT": ABORT_ASSIGNED_PCT, "strong_thr": m.strong_thr},
                 verdict="abort" if foreign else ("proceed_low_cos" if low_cos_proceed else "proceed"))
    if foreign:
        return PairResult(
            dub=name, out_path=None, ok=False,
            error=f"чужое видео: сопоставлено {assigned_pct:.0f}% (< {ABORT_ASSIGNED_PCT:.0f}%) — выравнивание прервано",
            fps_ref=fps_ref, fps_dub=fps_tst, n_frames=N,
            duration_s=len(refv) / fps_ref, assigned_pct=assigned_pct,
            elapsed_s=time.perf_counter() - t0, trace=trace.to_list(), **_geom_kw)

    # --- The vision layer builds the time map (the only path): detects steps and keeps a
    #     piecewise curve instead of a ramp. tg_s steps at the cuts (no maximum.accumulate);
    #     the cuts are silenced further below. ---
    if _band is not None:
        _band.mark(0.88, "построение карты соответствия")
    # Reference length: on a VFR axis it is the last frame's time plus one average frame (index/fps lies on VFR).
    dur_ref = (float(ax_ref[-1]) + 1.0 / fps_ref) if ax_ref is not None else len(refv) / fps_ref
    cos_asg = _cos_anchors(syn, refv, pred, asg)   # anchor cosine: weight for the vision layer and the unified plot
    grid, tg_s, vision_cuts, _vo, _vw, _vcurve = _vision_build_map(
        pred, asg, cos_asg, fps_ref, fps_tst, dur_ref, DT, ax_ref=ax_ref, ax_dub=ax_dub)

    # --- resample the audio onto the REF grid (low_mem: dub audio and out via memmap, resampled in blocks) ---
    # The dub channel layout (2.0/5.1/7.1) carries over to the output: decode into the native
    # channel count, size `out` for C channels, and warp each channel. Analysis (band/muq) works
    # on the mono downmix; one warp is applied to all channels, so inter-channel phase is kept.
    n_out = int(dur_ref * SR)
    _extract = _rep("extract", "декодирование звука озвучки")
    if _extract is not None:
        _extract.mark(0.0)
    dub_ch, dub_layout = probe_audio_channels(dub_audio, FFPROBE, atrack=dub_atrack)
    dub_delay = probe_av_delay(dub_audio, FFPROBE, atrack=dub_atrack)
    memlog('перед декодом аудио озвучки')
    if low_mem:
        # CK2: dub audio comes from the cache (skipping the decode), or is decoded straight into the cache (tmp mode)
        aud = (cache_mod.load_audio_mmap(cache_dir, dub_audio, dub_ch, atrack=dub_atrack)
               if (keep_tmp and cache_dir is not None) else None)
        _aud_src = "cache" if aud is not None else "decoded"
        if aud is None:
            dest = (cache_mod.audio_raw(cache_dir, dub_audio, dub_atrack)
                    if (keep_tmp and cache_dir is not None) else None)
            aud = _decode_audio_mmap(dub_audio, ffmpeg, channels=dub_ch,
                                     atrack=dub_atrack, reporter=_extract,
                                     dest=dest, delay_s=dub_delay, scratch=scratch)  # int16-memmap [N,C]
            if dest is not None:
                cache_mod.save_audio_meta(cache_dir, dub_audio, dub_ch, atrack=dub_atrack)   # CK2 metadata (+EXT_VER)
        n_aud = len(aud)
        trace.event("dub_audio", state="audio", source=_aud_src, channels=dub_ch, layout=dub_layout,
                    atrack=dub_atrack, seconds=n_aud / SR, container_delay_s=dub_delay)
        out = np.memmap(scratch.sub("out") / "out.f32", dtype=np.float32, mode="w+", shape=(n_out, dub_ch))
        BLK = 30 * SR                                    # a 30 s block: both the index math and the audio read work block by block
        _resample = _rep("resample", "перекладка звука на таймлайн референса")
        for s1 in range(0, n_out, BLK):
            s2 = min(s1 + BLK, n_out)
            if _resample is not None:
                _resample(s2 / max(1, n_out))
            src = np.interp(np.arange(s1, s2) / SR, grid, tg_s) * SR
            # The [a1:a2] band covers src. Clamping to [0, n_aud] is required: the map (vision or
            # any other) can point past the dub audio's end (a dub shorter than the reference
            # sends tail blocks of src off its end), and without the clamp a1>=a2 gives an empty
            # xp and a ValueError. np.interp itself clamps src to the band's edges, so past the
            # dub's end the edge sample plays (bit-exact with the non-low_mem branch).
            a1 = min(max(0, int(np.floor(src.min())) - 1), n_aud - 1)
            a2 = max(min(n_aud, int(np.ceil(src.max())) + 2), a1 + 1)
            for ch in range(dub_ch):
                out[s1:s2, ch] = warp_interp(aud[a1:a2, ch], src - a1)   # xp=arange(a1,a2), so src is shifted by -a1; GPU/CPU
        del aud
        memlog('после ресэмпла звука по зрению')
    else:
        aud = _decode_audio(dub_audio, ffmpeg, channels=dub_ch, atrack=dub_atrack,
                           reporter=_extract, delay_s=dub_delay)
        n_aud = len(aud)
        trace.event("dub_audio", state="audio", source="decoded", channels=dub_ch, layout=dub_layout,
                    atrack=dub_atrack, seconds=n_aud / SR, container_delay_s=dub_delay)
        t_out = np.arange(n_out) / SR
        t_syn_at = np.interp(t_out, grid, tg_s)
        src = t_syn_at * SR
        out = np.empty((n_out, dub_ch), np.float32)
        _resample = _rep("resample", "перекладка звука на таймлайн референса")
        if _resample is not None:
            _resample.mark(0.5)
        for ch in range(dub_ch):
            out[:, ch] = warp_interp(aud[:, ch], src)        # sg=arange(len(aud)) → grid; GPU/CPU
        del aud, t_out, t_syn_at, src   # free the large index arrays right after resampling

    # --- Silence in the cuts comes from ONE source: the vision layer's own cuts (build_curve).
    #     Neither cut_intervals (_level_decide) nor monotonizing take part here (a single, proven
    #     detector). A negative-delta cut (the reference has a piece the dub does not) becomes
    #     silence exactly |delta|/fps wide, and the dub stays continuous after it (tg_s is not
    #     monotonized). A positive-delta cut (the dub has extra content) becomes a narrow splice
    #     seam. band/muq run on top of this silence; _fill_silence_from_ref fills it with the
    #     reference afterwards.
    fill_spans: list[tuple[float, float]] = []           # cuts (no dub content) for the plot and telemetry
    for tc, v, te, tn in vision_cuts:                   # a cut is the span BETWEEN anchors [te, tn]
        if v >= 0:
            continue
        a_s = float(te); b_s = float(tn)
        s1 = max(0, int(a_s * SR)); s2 = min(n_out, int(b_s * SR))
        if s2 - s1 <= 0:
            continue
        out[s1:s2] = 0.0
        if s1 - FADE >= 0:
            out[s1 - FADE:s1] *= np.linspace(1, 0, FADE)[:, None]
        if s2 + FADE <= n_out:
            out[s2:s2 + FADE] *= np.linspace(0, 1, FADE)[:, None]
        fill_spans.append((a_s, b_s))
    for tc, v, te, tn in vision_cuts:                   # an insert: a positive-delta cut (extra dub content) becomes a narrow seam
        if v > 0:
            a_s, b_s = float(max(0.0, tc - 0.15)), float(tc + 0.15)
            out[int(a_s * SR):min(n_out, int(b_s * SR))] = 0.0
            fill_spans.append((a_s, b_s))             # recorded so the plot matches exactly what was silenced
    # An edge cut (head/tail, where the dub does not exist) gets silence plus a fill_span instead
    # of a clamped constant (build_curve extrapolates a line past the body, warp clamps it, and
    # that constant anomaly would break band across the whole track).
    _edge_silence(out, grid, tg_s, n_aud, SR, fill_spans, FADE)
    fill_spans.sort()

    # The unified plot (vision + audio) is built further below, after the audio layer, as one plot for both.

    # --- reference audio on the REF grid (needed by the band/muq audio layer and by the final silence fill) ---
    # ref_audio is reused when the caller already extracted it once for the episode; otherwise it is extracted here.
    ref_buf = _reference_buffer(ref_audio, n_out, low_mem=low_mem, scratch=scratch)

    # --- The band/muq audio layer refines the track after the vision resample. It sees the
    #     reference/silence already in the cuts, not the raw dub, so it never mistakes an edge cut
    #     for a real offset. ---
    audio_resid_ms = 0.0
    audio_info: dict = {}
    anchor_on = ref_buf is not None                 # no reference sound: nothing to listen against

    # --- the actual cuts are silence, filled with the synchronized reference in the final pass below ---
    real_cuts_s = sum(b - a for a, b in fill_spans)      # the cuts are already silenced above (build_curve)
    n_filled = 0                                 # reassigned below: seconds of silence filled with the reference

    # --- Studio A/V desync detector (read-only, does not touch the wav) runs on the video-laid
    #     out BEFORE the band refinement. It must run before band: band's own cuts smear the
    #     desync, so measuring after it goes blind (MAD rises, the danger flag fades). Here out
    #     is a clean video alignment with a stable signal (band's own gcc witness, MAD near 0). A
    #     large and stable PHAT residual for M&E means the source audio is offset from its own video
    #     (a studio dub: the original track a second or two off, the voiceover recorded to
    #     picture) — a source defect that makes the track unfit to proceed. Neither the alignment
    #     nor band is touched here; this only raises a red flag.
    memlog('перед буфером рефа')
    trace.event("vision_map", state=m.state, cuts=len(vision_cuts), silenced_spans=len(fill_spans),
                silenced_s=sum(b - a for a, b in fill_spans), dur_ref_s=dur_ref, n_anchors=int(len(asg)))
    dv = None
    if ref_buf is not None:
        try:
            from .anchor import apply as _anchor_det
            dv = _anchor_det.detect_av_desync(out, ref_buf, sr_audio=SR)
            # Diagnostics only: a constant offset is what the audio layer removes; the track verdict
            # is taken from what remains after it (_audio_verdict), never from this pre-layer value.
            trace.decide("av_desync", state="audio", verdict="offset" if (dv and dv["danger"]) else "ok",
                         inputs={k: dv[k] for k in ("lag_ms", "mad_ms", "n_windows") if dv and k in dv})
        except Exception as e:  # noqa: BLE001 — the detector must not crash conform
            trace.decide("av_desync", state="audio", verdict="error", inputs={"error": str(e)[:200]})
    else:
        trace.decide("av_desync", state="audio", verdict="skipped", inputs={"reason": "no reference audio supplied"})

    # --- The Band/MuQ audio layer runs after the cuts are silenced: it sees the reference/silence
    #     already in the cut, not the raw dub, so it never mistakes an edge cut for a real offset. Warps out.
    memlog('перед аудио-слоем')
    _audio = _rep("audio", "звуковой анализ") if anchor_on else None
    if anchor_on:
        from .anchor import apply as _anchor          # a lazy import: GPU and the optional transformers dependency load only when this layer runs
        if _audio is not None:
            _audio.mark(0.0)
        # CK4: the reference envelope cache (coarse_dtw) is reused across the episode's dubs and reruns (keep_tmp)
        _dsp_cache = (cache_mod.dsp_ref_path(cache_dir, ref.src, ref_atrack)
                      if (keep_tmp and cache_dir is not None and ref.src is not None) else None)
        audio_resid_ms = _anchor.audio_anchor(
            out, ref_buf, fps_ref, method=audio_method,
            drift_speed_pct=drift_speed_pct, info=audio_info, progress=part(_audio, 0.0, 0.70),
            plot_dir=out_path.parent / "_plots", plot_stem=out_path.stem, render_own=False,
            vision_spans=fill_spans, dsp_cache=_dsp_cache)   # the vision layer's silence zones plus the CK4 reference envelope cache
            # render_own=False: band does not draw its own plot; conform builds the unified vision+audio plot below
        trace.event("audio_layer", state="audio", method=audio_method,
                    resid_ms=audio_resid_ms, cuts=int(audio_info.get("audio_cuts", 0)),
                    max_step_ms=float(audio_info.get("audio_max_step_ms", 0.0)),
                    coverage=float(audio_info.get("audio_coverage", 0.0)),
                    span_ms=float(audio_info.get("audio_span_ms", 0.0)),
                    drift_ms=float(audio_info.get("audio_drift_ms", 0.0)),
                    excess_ms=_audio_excess_ms(audio_info),
                    global_offset_ms=float(audio_info.get("audio_global_offset_ms", 0.0)),
                    structure=audio_info.get("audio_structure"))
    else:
        trace.event("audio_layer", state="audio", method="none",
                    reason="no reference audio supplied")

    # --- The final fill of dub silence with the reference runs only after band/muq, once the
    #     track is in sync with the reference. In the dub's gaps (cuts, excisions, edges) where the reference
    #     has sound, the synchronized reference is inserted. Skipped when the audio layer is off,
    #     because inserting into an unsynchronized track would create a desync. ---
    memlog('после аудио-слоя')
    if _audio is not None:
        _audio.mark(0.70, "заполнение тишины рефом")
    ref_filled_s = 0.0
    if fill_silence and anchor_on and ref_buf is not None:
        ref_filled_s = _fill_silence_from_ref(out, ref_buf)
        audio_info["ref_filled_s"] = round(ref_filled_s, 1)
    n_filled = int(round(ref_filled_s))          # PairResult.filled_cuts: seconds of silence filled with the reference
    trace.event("fill_silence", state="audio", enabled=bool(fill_silence and anchor_on and ref_buf is not None),
                filled_s=ref_filled_s)

    # --- the quality passport and warnings (read-only, do not touch the wav); computed before the plot, whose header shows the verdict ---
    metrics = _metrics(syn, refv, pred, asg, fps_ref, fps_tst)
    warns = _warnings(asg=asg, n_syn=N, pred=pred, fps_ref=fps_ref, fps_dub=fps_tst,
                      dur_ref=dur_ref, real_cuts_s=real_cuts_s, metrics=metrics,
                      creep_zones=creep_zones, freeze_runs=freeze_runs)
    critical: list[str] = []
    if anchor_on and audio_info:
        critical, audio_warns = _audio_verdict(audio_info, audio_resid_ms)
        warns += audio_warns
    verdict = "critical" if critical else ("warn" if warns else "ok")
    trace.decide("verdict", state="audio", verdict=verdict, inputs={"critical": critical, "warnings": warns})

    # --- The unified alignment plot: vision (always, the video alignment) plus audio (when the anchor
    #     layer ran). Read-only: a failure here never crashes conform. One panel with vision only,
    #     two panels with vision and audio. ---
    if _audio is not None:
        _audio.mark(0.78, "графики укладки")
    unified_plots: list[dict] = []
    try:
        from .anchor import plots_unified as _pu
        T_v = _make_T(dur_ref)                                        # the plot grid spans the true reference duration
        o_v, w_v = _vision_ow(pred, asg, cos_asg, fps_ref, fps_tst, T=T_v,
                              ax_ref=ax_ref, ax_dub=ax_dub)
        t_ref_v = (ax_ref[pred[asg]] if ax_ref is not None
                   else pred[asg].astype(np.float64) / fps_ref)
        t_dub_v = (ax_dub[asg] if ax_dub is not None
                   else asg.astype(np.float64) / fps_tst)
        shift_v = (t_dub_v - t_ref_v) * fps_ref
        shift_T = np.interp(T_v, grid, (tg_s - grid) * fps_ref)       # the production time map converted to a shift in frames on grid T
        cuts_g = [(float(tc), float(v)) for tc, v, te, tn in vision_cuts]   # the vision layer's own cuts (time, delta)
        # In the plot, the alignment line breaks at each cut (a gap stays a gap, not a vertical jump)
        # and dashed verticals mark the [te, tn] bounds. shift_T_plot is a plot-only copy with NaN
        # at the cuts; curve_fr in the npz and tg_s stay clean, since a NaN there would break them.
        shift_T_plot = shift_T.copy(); cut_marks: list[float] = []
        for tc, v, te, tn in vision_cuts:
            in_cut = (T_v >= te) & (T_v <= tn)
            if in_cut.any():
                shift_T_plot[in_cut] = np.nan
            else:                                                     # a narrow insert between grid nodes: mark the nearest node
                shift_T_plot[int(np.argmin(np.abs(T_v - tc)))] = np.nan
            cut_marks += [float(te), float(tn)]
        sa, sb = _vision_global_trend(o_v, w_v, T_v, fps_ref)          # the same scale the time map itself uses
        vision_d = dict(o=o_v, w=w_v, curve=shift_T_plot, cuts=cuts_g, cut_marks=cut_marks,
                        T=np.asarray(T_v, np.float32),
                        t_ref=t_ref_v.astype(np.float32), shift_fr=shift_v.astype(np.float32),
                        cos=np.asarray(cos_asg, np.float32),
                        scale_a=float(sa), scale_b=float(sb),         # the plot baseline (drift shape) with the sawtooth removed
                        fill_spans=fill_spans,                        # cuts (no dub content) exactly matching what was silenced in out
                        head_s=float(t_ref_v.min()) if len(t_ref_v) else 0.0, dur=dur_ref)
        # The raw vision dump (.npz + .json) next to the plot holds every value of the top panel,
        # the production warp map (grid to tg_s), and the raw video anchors, so the vision alignment
        # can be diagnosed without regenerating it (matching band's own __raw.npz). Must not crash conform.
        try:
            import json as _json
            _pd = out_path.parent / "_plots"; _pd.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                _pd / f"{out_path.stem}__vision.npz",
                T=np.asarray(T_v, np.float32), o=o_v.astype(np.float32), w=w_v.astype(np.float32),
                curve_fr=shift_T.astype(np.float32),               # the production vision map: shift in frames on T
                cuts=np.asarray(cuts_g, np.float64).reshape(-1, 2),  # (tc in seconds, v in frames)
                t_ref=t_ref_v.astype(np.float32),                   # reference time of the raw anchors, seconds
                shift_fr=shift_v.astype(np.float32),                # shift of the raw anchors, reference frames
                cos=np.asarray(cos_asg, np.float32),                # anchor confidence (cosine)
                asg=np.asarray(asg, np.int32),                      # dub frames that got an anchor
                pred_asg=np.asarray(pred[asg], np.int32),           # their matched reference frames (pred)
                grid=np.asarray(grid, np.float32),                  # resample grid, seconds (reference)
                tg_s=np.asarray(tg_s, np.float64),                  # the applied map: dub time, seconds
                fill_spans=(np.asarray(fill_spans, np.float64).reshape(-1, 2)
                            if fill_spans else np.zeros((0, 2), np.float64)),  # cuts (no dub content) = the silenced spans
                cut_intervals=(np.asarray(cut_intervals, np.int64).reshape(-1, 2)
                               if cut_intervals else np.zeros((0, 2), np.int64)),  # legacy _level_decide (CK3/legacy auto)
                stamp=np.asarray(_astamp),                          # CK3 stamp (validity for reuse)
                fps_ref=np.float64(fps_ref), fps_dub=np.float64(fps_tst),
                scale_a=np.float64(sa), scale_b=np.float64(sb),     # the actual scale (median of the slopes)
                dur_ref=np.float64(dur_ref))
            (_pd / f"{out_path.stem}__vision.json").write_text(_json.dumps({
                "fps_ref": float(fps_ref), "fps_dub": float(fps_tst), "dur_ref": float(dur_ref),
                "scale_pct": float(sa) / max(float(fps_ref), 1e-6) * 100.0,  # dub scale, percent per second
                "n_anchors": int(len(asg)),
                "median_shift_fr": float(np.median(shift_v)) if len(shift_v) else 0.0,
                "cuts": [[float(tc), float(v)] for tc, v in cuts_g],
                "npz": f"{out_path.stem}__vision.npz",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001 — the vision dump must not crash conform
            logger.exception("conform: дамп зрения {} не записан", out_path.name)
        audio_d = audio_info.get("band_layers")                       # None means vision only (one panel)
        passport = _pair_passport(
            vision_cuts=vision_cuts, fill_spans=fill_spans, freeze_runs=freeze_runs, audio_info=audio_info,
            shift_T=shift_T, T_v=T_v, scale_a=sa, fps_ref=fps_ref, assigned_pct=assigned_pct,
            metrics=metrics, tc_ref=tc_ref, tc_dub=tc_dub, geom_info=geom_info, mirror=bool(m.mirror),
            container_delay=dub_delay, anchor_on=anchor_on, audio_resid_ms=audio_resid_ms,
            av_desync=dv, verdict=verdict, verdict_text=(critical or warns or [""])[0])
        unified_plots = _pu.render_unified(out_path.parent / "_plots", out_path.stem,
                                           vision=vision_d, audio=audio_d, title=out_path.stem,
                                           passport=passport)
    except Exception as e:  # noqa: BLE001 — the plots must not crash conform
        logger.exception("conform: графики укладки {} не построены", out_path.name)
        trace.event("plots", state="output", error=str(e)[:200])
        unified_plots = []

    _write = _rep("write", out_path.name)
    if _write is not None:
        _write.mark(0.0)
    memlog('перед записью файла')
    _write_audio_streamed(out_path, out, ffmpeg, layout=dub_layout, on_prog=_write)  # FLAC in the dub's channel layout
    trace.event("write", state="output", path=out_path, layout=dub_layout, seconds=n_out / SR,
                bytes=out_path.stat().st_size if out_path.exists() else 0)

    # The vision layout is the pair's time map; stored so subtitles can follow the audio.
    try:
        from .subs_transfer import save_time_map
        save_time_map(out_path, _make_T(dur_ref), _vcurve, vision_cuts, fps_ref, dur_ref)
        trace.event("time_map", state="output", cuts=len(vision_cuts))
    except Exception as e:  # noqa: BLE001
        trace.event("time_map", state="output", error=str(e)[:200])

    if low_cos_proceed:                               # assigned% is below the threshold, but the coarse pass confirmed the same episode
        warns.insert(0, f"низкий косинус энкода (назн {assigned_pct:.0f}% < {ABORT_ASSIGNED_PCT:.0f}%): "
                        f"отсечка «чужое видео» НЕ применена — грубый проход подтвердил серию "
                        f"(n_keep={n_keep}); ПРОВЕРИТЬ вручную")
    return PairResult(
        dub=name, out_path=out_path, ok=True,
        fps_ref=fps_ref, fps_dub=fps_tst, n_frames=N, duration_s=dur_ref,
        assigned_pct=100.0 * len(asg) / max(1, N),
        slope=metrics["slope"], cos_median=metrics["cos_median"],
        monotonic_violations=metrics["mono"],
        real_cuts=fill_spans, real_cuts_s=real_cuts_s,
        filled_cuts=n_filled,
        blind_zones=n_blind, blind_restored=n_restore, edge_recovered=n_recovered,
        dropped_intro_s=metrics["intro_s"], audio_resid_ms=audio_resid_ms,
        audio_cuts=int(audio_info.get("audio_cuts", 0)),
        audio_max_step_ms=float(audio_info.get("audio_max_step_ms", 0.0)),
        audio_coverage=float(audio_info.get("audio_coverage", 0.0)),
        audio_span_ms=float(audio_info.get("audio_span_ms", 0.0)),
        plots=unified_plots,
        warnings=warns,
        critical=critical,
        elapsed_s=time.perf_counter() - t0, trace=trace.to_list(), **_geom_kw,
    )


def _metrics(syn, refv, pred, asg, fps_ref, fps_tst) -> dict:
    if len(asg) == 0:
        return {"slope": 0.0, "cos_median": 0.0, "mono": 0, "intro_s": 0.0}
    cosp = np.empty(len(asg), np.float32)
    CH = 4096
    pr = pred[asg]
    for i in range(0, len(asg), CH):
        sl = asg[i:i + CH]
        cosp[i:i + len(sl)] = np.einsum("ij,ij->i", syn[sl].astype(np.float32),
                                        refv[pred[sl]].astype(np.float32))
    A = np.vstack([asg, np.ones(len(asg))]).T
    slope = float(np.linalg.lstsq(A, pr, rcond=None)[0][0])
    mono = int((np.diff(pr) < 0).sum())
    k0 = int(asg[0])                                # frames of the dub dropped before the first match
    intro = k0 if (pred[:k0] == -1).all() else 0
    return {"slope": slope, "cos_median": float(np.median(cosp)),
            "mono": mono, "intro_s": intro / fps_tst}


def _mmss(t: float) -> str:
    t = max(0, int(t)); return f"{t // 60}:{t % 60:02d}"


def _pair_passport(*, vision_cuts, fill_spans, freeze_runs, audio_info, shift_T, T_v, scale_a, fps_ref,
                   assigned_pct, metrics, tc_ref, tc_dub, geom_info, mirror, container_delay, anchor_on,
                   audio_resid_ms, av_desync, verdict, verdict_text) -> dict:
    """One passport of the pair for both chart renders: zones and events with their kind and size,
    layout segments whose speed differs from the global scale, header metrics of vision and hearing,
    and the verdict. Numbers only; labels and units come from the chart's term catalog."""
    st = audio_info.get("audio_structure") or {}
    zones = ([{"kind": "vision_cut", "a": float(a), "b": float(b)} for a, b in fill_spans]
             + [{"kind": "video_freeze", "a": float(a), "b": float(b)} for a, b in freeze_runs]
             + [{"kind": "audio_gap", "a": float(a), "b": float(b)} for a, b in st.get("gaps", [])]
             + [{"kind": "dtw_cut", "a": float(a), "b": float(b)}
                for a, b in (audio_info.get("band_layers") or {}).get("dtw_cut_zones", [])])
    events = ([{"kind": "vision_step", "t": float(tc), "value": float(v) / fps_ref} for tc, v, _te, _tn in vision_cuts]
              + [{"kind": "audio_cut", "t": float(t), "value": float(v) * VFRAME} for t, v in audio_info.get("cuts", [])]
              + [{"kind": "audio_jump", "t": float(t), "value": float(d)} for t, d in st.get("jumps", [])]
              + [{"kind": "dtw_insert", "t": float(tc), "value": float(ln)}
                 for tc, ln in (audio_info.get("band_layers") or {}).get("dtw_inserts", [])])
    # Layout speed per segment between vision steps: slope of the applied map against the global scale.
    segments = []
    bounds = [0.0] + sorted(float(tc) for tc, *_ in vision_cuts) + [float(T_v[-1])]
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = (T_v >= a) & (T_v <= b) & np.isfinite(shift_T)
        if m.sum() >= 10 and b - a >= 20.0:
            slope = float(np.polyfit(T_v[m], shift_T[m], 1)[0])          # frames of layout per second
            segments.append({"a": a, "b": b, "speed_pct": (slope - scale_a) / fps_ref * 100.0})
    vision_line = [
        {"key": "assigned", "value": float(assigned_pct)},
        {"key": "cos", "value": float(metrics["cos_median"])},
        {"key": "scale", "value": float(scale_a) / max(float(fps_ref), 1e-6) * 100.0},
        {"key": "telecine", "tpl": "value.telecine",
         "value": {"ref": float(tc_ref["tele_score"]), "dub": float(tc_dub["tele_score"])}},
        {"key": "tc_dropped", "value": int(tc_ref["dropped"]) + int(tc_dub["dropped"])},
        {"key": "geom", "tpl": "value.geom",
         "value": ({"sx": float(geom_info["sx"]), "sy": float(geom_info["sy"]), "n": int(geom_info["n_in"])}
                   if geom_info else None)},
        {"key": "mirror", "value": bool(mirror)},
        {"key": "container_delay", "value": float(container_delay)},
        {"key": "vision_cuts", "tpl": "value.count_dur",
         "value": {"n": len(fill_spans), "dur": float(sum(b - a for a, b in fill_spans))}},
        {"key": "freezes", "tpl": "value.count_dur",
         "value": {"n": len(freeze_runs), "dur": float(sum(b - a for a, b in freeze_runs))}},
    ]
    audio_line = []
    if anchor_on and audio_info:
        audio_line = [
            {"key": "method", "value": str(audio_info.get("anchor_method", ""))},
            {"key": "structure", "tpl": "value.structure",
             "value": ({"plateaus": len(st["plateaus"]), "jumps": len(st["jumps"])} if st else None)},
            {"key": "resid", "value": float(abs(audio_resid_ms))},
            {"key": "cuts", "value": int(audio_info.get("audio_cuts", 0))},
            {"key": "max_step", "value": float(audio_info.get("audio_max_step_ms", 0.0))},
            {"key": "coverage", "value": float(audio_info.get("audio_coverage", 0.0)) * 100.0},
            {"key": "excess", "value": _audio_excess_ms(audio_info) / 1000.0},
            {"key": "filled", "value": float(audio_info.get("ref_filled_s", 0.0))},
            {"key": "av_offset", "tpl": "value.av_offset",
             "value": ({"lag": float(av_desync["lag_ms"]), "mad": float(av_desync["mad_ms"])} if av_desync else None)},
        ]
    return {"zones": zones, "events": events, "segments": segments,
            "header": [vision_line, audio_line], "verdict": verdict, "verdict_text": verdict_text}


def _audio_excess_ms(audio_info: dict) -> float:
    """Cut movement that cancelled itself out: Σ|steps| − |Σsteps|. Real audio edits accumulate in one
    direction; a layer chasing a blind measurement goes back and forth and this grows to seconds."""
    return max(0.0, float(audio_info.get("audio_sum_ms", 0.0)) - abs(float(audio_info.get("audio_net_ms", 0.0))))


def _audio_verdict(audio_info: dict, resid_ms: float) -> tuple[list[str], list[str]]:
    """Track verdict from what remains AFTER the audio layer -> (critical, warnings). One chokepoint
    for both modes: constant offsets the layer removed never count, only its result does."""
    crit: list[str] = []; warn: list[str] = []
    resid = abs(float(resid_ms))
    cov = float(audio_info.get("audio_coverage", 0.0))
    cuts = int(audio_info.get("audio_cuts", 0)); excess = _audio_excess_ms(audio_info)
    drift = float(audio_info.get("audio_drift_ms", 0.0))
    if cov < AUDIO_COVERAGE_BLIND:
        crit.append(f"у файлов почти нет общего звука (опора {cov * 100:.0f}% длительности) — "
                    f"синхронность выхода не измерена")
    elif cov < AUDIO_COVERAGE_LOW:
        warn.append(f"звуковое измерение имело опору лишь на {cov * 100:.0f}% длительности")
    if resid > AUDIO_RESID_MAX_MS:
        crit.append(f"остаток после доводки {resid:.0f} мс — больше допустимых ±{AUDIO_RESID_MAX_MS:.0f} мс")
    if excess > AUDIO_EXCESS_CRIT_MS:
        crit.append(f"слух метался: {cuts} резов, взаимно погашено {excess / 1000:.1f} с хода — "
                    f"синхронность выхода не гарантирована")
    elif excess > AUDIO_EXCESS_WARN_MS:
        warn.append(f"резы туда-обратно: {cuts} резов, взаимно погашено {excess / 1000:.1f} с хода — проверить")
    if abs(drift) > 1000.0:
        warn.append(f"сдвиг звука уходит на {drift / 1000:+.1f} с — вероятно, дефект исходного файла")
    st = audio_info.get("audio_structure")
    if st:
        gap_s = sum(b - a for a, b in st["gaps"])
        jumps = ", ".join(f"{j[1]:+.1f} с на {j[0]:.0f} с" for j in st["jumps"])
        warn.append(f"структура звука: {len(st['plateaus'])} плато" + (f", скачки {jumps}" if jumps else "")
                    + (f"; звука озвучки нет {gap_s:.0f} с — залито рефом" if gap_s else ""))
    return crit, warn


def _warnings(*, asg, n_syn, pred, fps_ref, fps_dub, dur_ref, real_cuts_s,
              metrics, creep_zones=(), freeze_runs=()) -> list[str]:
    """Heuristic "pay attention" warnings (do NOT affect the wav). Tolerant of false positives: it
    is better to over-warn. Each one carries a reason label, so a false one is easy to dismiss."""
    w: list[str] = []
    if len(asg) < 2:
        return ["почти нет сопоставленных кадров видео"]
    for j0, j1, cz, sl in creep_zones:
        w.append(f"налипшая вставка выброшена: {_mmss(j0 / fps_dub)}–{_mmss(j1 / fps_dub)} "
                 f"дубля (~{(j1 - j0 + 1) / fps_dub:.0f}с, cos {cz:.2f}, ход {sl:.2f}×)")
    for a, b in freeze_runs:
        w.append(f"кадр озвучки не меняется {_mmss(a)}–{_mmss(b)} ({b - a:.0f}с) — замершая картинка или статичная заставка")
    exp = (fps_ref / fps_dub) if fps_dub else 1.0
    # — geometry —
    if metrics["slope"] and abs(metrics["slope"] - exp) > 0.05:
        w.append(f"ход времени {metrics['slope']:.3f} вместо ожидаемого {exp:.3f}")
    if metrics["mono"] > max(20, int(0.01 * len(asg))):
        w.append(f"нарушений монотонности {metrics['mono']}")
    # — video coverage —
    ap = 100.0 * len(asg) / max(1, n_syn)
    if ap < 85.0:
        w.append(f"сопоставлено лишь {ap:.0f}% кадров")
    d = np.diff(asg)
    if len(d):
        i = int(np.argmax(d)); gap_s = (int(d[i]) - 1) / fps_dub
        if gap_s > max(15.0, 0.06 * dur_ref):
            w.append(f"крупный неразмеченный участок ~{gap_s:.0f}с (≈{_mmss(pred[asg[i]] / fps_ref)}) — вставка или сбой")
    # — fill / intro —
    ff = real_cuts_s / max(1e-6, dur_ref)
    if ff > 0.15:
        w.append(f"звуком референса заполнено {ff * 100:.0f}% длительности")
    if metrics["intro_s"] > 5.0:
        w.append(f"выпало/залито начало ~{metrics['intro_s']:.0f}с")
    return w


def _align_audio_only(
    ref: SrmFeatures,
    dub_audio: Path,
    out_path: Path,
    *,
    ffmpeg: str = FFMPEG,
    progress=None,
    progress_meta: tuple[int, int, str] = (0, 0, ""),
    ref_audio=None,
    low_mem: bool = False,
    cache_dir: Path | str | None = None,
    keep_tmp: bool = False,
    dub_name: str | None = None,
    audio_method: str = "band",      # task parameters, as in conform_features
    drift_speed_pct: float = 1.25,
    fill_silence: bool = True,
    ref_atrack: int = 0,             # audio track of the reference (the sound baseline)
    dub_atrack: int = 0,             # audio track of the dub (a bare audio file is usually single-track, but mka can carry several)
    should_stop=None,
    trace: Trace | None = None,
    scratch: tmpfiles.Workspace | None = None,   # owner of the pair's temporary files; required with low_mem
) -> PairResult:
    """AUDIO-ONLY conform: a dub
    with NO video stream (bare flac/mka/mp3/aac…). There is no vision by construction — the dub is
    laid on the reference timeline AS IS (identity, from zero), after which the standard audio
    layer removes ALL of the desync through exactly the same pipeline as after vision:
      the map of the audio structure (constant shifts and jumps, far beyond the band window) →
      `coarse_dtw` (inserts and cuts outside the ±2.5 s band window) → the band refinement (drift and
      cuts up to 2.5 s) → the silence filled with the reference.
    Plots: band draws ITS OWN track (`render_own=True`, `<stem>__track.png/html` in `_plots`) —
    no vision panel is built (nothing to show there).

    Limits of the method (by construction, only warned about, not fixed): a nonlinear video scale
    (PAL speed-up and the like, which vision would read from the data — audio here has nowhere to
    take it from, only the raw audio rate) and structural mismatches beyond the reach of the
    structure map and the DTW. PairResult.mode="audio"; the vision fields (assigned/cos/slope) are
    meaningless and stay 0."""
    t0 = time.perf_counter()
    _require_scratch(low_mem, scratch)

    def _rep(stage: str, detail: str = "") -> Reporter | None:
        return Reporter.of(progress, stage, progress_meta, detail)

    dub_audio = Path(dub_audio)
    name = dub_name or dub_audio.name
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace = trace if trace is not None else Trace(name)

    fps_ref = ref.fps
    dur_ref = ref.duration_s
    if not (dur_ref and dur_ref > 0 and fps_ref and fps_ref > 0):
        raise ValueError(f"реф пуст/битый (dur={dur_ref}, fps={fps_ref}) — аудио-only без длины рефа невозможен")
    n_out = int(dur_ref * SR)
    dub_ch, dub_layout = probe_audio_channels(dub_audio, FFPROBE, atrack=dub_atrack)
    dub_delay = probe_av_delay(dub_audio, FFPROBE, atrack=dub_atrack)
    trace.event("dub_audio", state="audio-only", channels=dub_ch, layout=dub_layout, atrack=dub_atrack,
                dur_ref_s=dur_ref, container_delay_s=dub_delay)

    # --- decode the dub and lay it on the reference timeline as is (anything past the reference
    #     length is cut, anything short of it is silence; the audio layer sorts out shifts and cuts) ---
    if low_mem:
        aud = (cache_mod.load_audio_mmap(cache_dir, dub_audio, dub_ch, atrack=dub_atrack)
               if (keep_tmp and cache_dir is not None) else None)
        if aud is None:
            dest = (cache_mod.audio_raw(cache_dir, dub_audio, dub_atrack)
                    if (keep_tmp and cache_dir is not None) else None)
            aud = _decode_audio_mmap(dub_audio, ffmpeg, channels=dub_ch, atrack=dub_atrack,
                                     reporter=_rep("extract", "декодирование звука озвучки"),
                                     dest=dest, delay_s=dub_delay, scratch=scratch)
            if dest is not None:
                cache_mod.save_audio_meta(cache_dir, dub_audio, dub_ch, atrack=dub_atrack)   # CK2 metadata (+EXT_VER)
        out = np.memmap(scratch.sub("out") / "out.f32", dtype=np.float32, mode="w+", shape=(n_out, dub_ch))
        m = min(len(aud), n_out)
        BLK = 60 * SR                                    # processed in blocks so RAM stays O(block), like the rest of the low_mem path
        for s1 in range(0, m, BLK):
            out[s1:min(s1 + BLK, m)] = aud[s1:min(s1 + BLK, m)]
        if m < n_out:
            out[m:] = 0.0
        del aud
    else:
        aud = _decode_audio(dub_audio, ffmpeg, channels=dub_ch, atrack=dub_atrack,
                            reporter=_rep("extract", "декодирование звука озвучки"), delay_s=dub_delay)
        out = np.zeros((n_out, dub_ch), np.float32)
        m = min(len(aud), n_out)
        out[:m] = aud[:m]
        del aud

    # --- reference audio (as in the vision path: reused from the episode, or extracted from ref.src) ---
    ref_buf = _reference_buffer(ref_audio, n_out, low_mem=low_mem, scratch=scratch)

    anchor_on = ref_buf is not None                 # no reference sound: nothing to listen against
    audio_resid_ms = 0.0
    audio_info: dict = {}
    _audio = _rep("audio", "звуковой анализ (файл без видеоряда)") if anchor_on else None
    if anchor_on:
        from .anchor import apply as _anchor
        if _audio is not None:
            _audio.mark(0.0)
        _dsp_cache = (cache_mod.dsp_ref_path(cache_dir, ref.src, ref_atrack)
                      if (keep_tmp and cache_dir is not None and ref.src is not None) else None)
        audio_resid_ms = _anchor.audio_anchor(
            out, ref_buf, fps_ref, method=audio_method,
            drift_speed_pct=drift_speed_pct, info=audio_info, progress=part(_audio, 0.0, 0.70),
            plot_dir=out_path.parent / "_plots", plot_stem=out_path.stem,
            render_own=True,                       # band draws its own plot here, since this mode has no vision panel
            vision_spans=None, dsp_cache=_dsp_cache,
            layout_map=True)                       # no vision: the audio layout is the pair's only time map
        trace.event("audio_layer", state="audio-only", method=audio_method,
                    resid_ms=audio_resid_ms, cuts=int(audio_info.get("audio_cuts", 0)),
                    max_step_ms=float(audio_info.get("audio_max_step_ms", 0.0)),
                    coverage=float(audio_info.get("audio_coverage", 0.0)),
                    span_ms=float(audio_info.get("audio_span_ms", 0.0)),
                    drift_ms=float(audio_info.get("audio_drift_ms", 0.0)))
    else:
        trace.event("audio_layer", state="audio-only", method="none",
                    reason="no reference audio supplied")

    if _audio is not None:
        _audio.mark(0.70, "заполнение тишины рефом")
    ref_filled_s = 0.0
    if fill_silence and anchor_on and ref_buf is not None:
        ref_filled_s = _fill_silence_from_ref(out, ref_buf)
        audio_info["ref_filled_s"] = round(ref_filled_s, 1)
    trace.event("fill_silence", state="audio-only", enabled=bool(fill_silence and anchor_on and ref_buf is not None),
                filled_s=ref_filled_s)

    # --- warnings (read-only): this vision-less mode must openly flag its own blind spot ---
    warns: list[str] = []; critical: list[str] = []
    if not anchor_on:
        warns.append("аудио-only БЕЗ аудио-слоя (у рефа нет аудио): "
                     "дорожка уложена КАК ЕСТЬ, выравнивание не выполнялось")
    else:
        critical, warns = _audio_verdict(audio_info, audio_resid_ms)
    trace.decide("verdict", state="audio-only", verdict="critical" if critical else ("warn" if warns else "ok"),
                 inputs={"critical": critical, "warnings": warns})

    _write = _rep("write", out_path.name)
    if _write is not None:
        _write.mark(0.0)
    _write_audio_streamed(out_path, out, ffmpeg, layout=dub_layout, on_prog=_write)
    trace.event("write", state="output", path=out_path, layout=dub_layout, seconds=n_out / SR,
                bytes=out_path.stat().st_size if out_path.exists() else 0)

    # Subtitles follow what they are tied to. With a video they are tied to the picture and follow
    # the vision map alone: the audio layer repairs the dub's sound against its own picture, and
    # moving the cues with it would undo their sync. Without a video they can only be tied to the
    # sound, so the audio layout is the pair's time map; with no audio layer the dub lies from zero.
    try:
        from .subs_transfer import identity_layout, save_time_map
        tm = audio_info.get("time_map") if anchor_on else None
        if tm is not None:
            save_time_map(out_path, tm["T"], tm["shift_s"] * fps_ref, tm["cuts"], fps_ref, dur_ref)
        else:
            save_time_map(out_path, *identity_layout(dur_ref), fps_ref, dur_ref)
        trace.event("time_map", state="output", cuts=len(tm["cuts"]) if tm is not None else 0)
    except Exception as e:  # noqa: BLE001 — subtitles must not take a finished pair down
        trace.event("time_map", state="output", error=str(e)[:200])

    return PairResult(
        dub=name, out_path=out_path, ok=True, mode="audio",
        fps_ref=fps_ref, duration_s=dur_ref,
        filled_cuts=int(round(ref_filled_s)),
        audio_resid_ms=audio_resid_ms,
        audio_cuts=int(audio_info.get("audio_cuts", 0)),
        audio_max_step_ms=float(audio_info.get("audio_max_step_ms", 0.0)),
        audio_coverage=float(audio_info.get("audio_coverage", 0.0)),
        audio_span_ms=float(audio_info.get("audio_span_ms", 0.0)),
        plots=list(audio_info.get("plots", [])),
        warnings=warns, critical=critical,
        elapsed_s=time.perf_counter() - t0, trace=trace.to_list(),
    )


def conform_pair(
    ref: SrmFeatures,
    dub_video: Path | str,
    out_path: Path | str,
    *,
    ffmpeg: str = FFMPEG,
    progress=None,
    should_stop=None,
    progress_meta: tuple[int, int, str] = (0, 0, ""),
    low_mem: bool = False,
    cache_dir: Path | str | None = None,
    keep_tmp: bool = False,
    **opts,
) -> PairResult:
    """Convenience wrapper: build the dub's features and align them onto ref → out_path.

    low_mem=True → the dub's SRM is built STREAMED to disk (`_tmp/.../dub.f16`, memmap), never
    accumulating in RAM; the file is removed after the pair. The result is bit-exact to the in-RAM
    path.

    keep_tmp=True + cache_dir → CHECKPOINT CK1: the dub's SRM is cached in `cache_dir`
    (`epXX/_conform_cache`, like the reference) and reused from there on a rerun — skipping the
    costly GPU decode. Key = file + EMB_VER (cache.load_srm)."""
    dub_video = Path(dub_video)
    out_path = Path(out_path)
    trace = Trace(dub_video.name)
    # The pair's temporary files have one owner: removed here as a whole, after every frame that
    # held them open has returned, whether the pair succeeded, was rejected or crashed. Making the
    # owner touches no disk, so nothing can fail before the chokepoint below.
    scratch = tmpfiles.Workspace(dub_video.parent, "pair") if low_mem else None
    # One chokepoint for every outcome: the trace reaches the result and the disk whether the
    # pair succeeded, was rejected by a gate or crashed; a crash alone must not lose the record.
    try:
        try:
            res = _conform_pair(ref, dub_video, out_path, trace, ffmpeg=ffmpeg,
                                progress=progress, should_stop=should_stop, progress_meta=progress_meta,
                                low_mem=low_mem, cache_dir=cache_dir, keep_tmp=keep_tmp,
                                scratch=scratch, **opts)
        except Exception as e:  # noqa: BLE001 — one pair must not take the episode down
            logger.exception("conform: озвучка {} упала на серии {}", dub_video.name, ref.src)
            trace.event("exception", state="error", type=type(e).__name__, error=str(e)[:500])
            res = PairResult(dub=dub_video.name, out_path=None, ok=False, error=str(e), trace=trace.to_list())
    finally:
        if scratch is not None:
            scratch.close()
    trace.save(out_path)
    return res


def _conform_pair(ref, dub_video: Path, out_path: Path, trace: Trace, *, ffmpeg, progress,
                  should_stop, progress_meta, low_mem, cache_dir, keep_tmp, scratch, **opts) -> PairResult:
    # A dub without a video stream cannot be matched by vision; the audio-only path lays it down
    # as is and lets the audio layer do all the alignment. Decided from the file, never a switch.
    if not probe_has_video(dub_video):
        trace.decide("mode", state="probe", verdict="audio", inputs={"has_video": False})
        return _align_audio_only(ref, dub_video, out_path, ffmpeg=ffmpeg,
                                 progress=progress, progress_meta=progress_meta,
                                 low_mem=low_mem, cache_dir=cache_dir, keep_tmp=keep_tmp,
                                 dub_name=dub_video.name, should_stop=should_stop, trace=trace,
                                 scratch=scratch, **opts)
    trace.decide("mode", state="probe", verdict="av", inputs={"has_video": True})
    _decode = Reporter.of(progress, "decode", progress_meta, "разбор файла")
    if _decode is not None:
        _decode.mark(0.0)
    # CK1: try to take the dub's SRM from the episode cache, skipping the GPU decode
    dub = cache_mod.load_srm(cache_dir, dub_video) if (keep_tmp and cache_dir is not None) else None
    trace.decide("ck1_srm", state="cache", source="cache", verdict="hit" if dub is not None else "miss",
                 inputs={"checkpoints": bool(keep_tmp and cache_dir is not None)})
    if dub is not None:
        return conform_features(ref, dub, dub_video, out_path, ffmpeg=ffmpeg,
                                progress=progress, dub_name=dub_video.name,
                                progress_meta=progress_meta, low_mem=low_mem,
                                cache_dir=cache_dir, keep_tmp=keep_tmp, trace=trace,
                                scratch=scratch, **opts)
    # No valid checkpoint: the SRM is built straight into the cache when checkpoints are kept,
    # otherwise into the pair's scratch.
    mp = None; into_cache = False
    if keep_tmp and cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        mp = cache_mod.srm_file(cache_dir, dub_video); into_cache = True
    elif scratch is not None:
        mp = scratch.sub("srm") / "dub.f16"
    with decode_backend(probe_resolution(dub_video), ffmpeg) as _be:   # 1080p+ uses GPU (the NVDEC ceiling), otherwise CPU
        dub = build_srm(dub_video, ffmpeg=ffmpeg, reporter=_decode,
                        should_stop=should_stop, mmap_path=mp, backend=_be)
    trace.event("srm_dub", state="plain", source="decoded", frames=len(dub.srm), fps=dub.fps,
                vfr=dub.pts is not None, cached=into_cache)
    if into_cache:
        cache_mod.save_meta(cache_dir, dub_video, len(dub.srm), dub.fps)   # CK1 metadata (+EMB_VER)
    return conform_features(ref, dub, dub_video, out_path, ffmpeg=ffmpeg,
                            progress=progress, dub_name=dub_video.name,
                            progress_meta=progress_meta, low_mem=low_mem,
                            cache_dir=cache_dir, keep_tmp=keep_tmp, trace=trace,
                            scratch=scratch, **opts)
