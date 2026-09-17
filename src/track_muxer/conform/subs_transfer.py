"""Carry a video's sidecar subtitles onto the reference timeline with the pair's vision map.

The map is stored next to the pair's output, so subtitles can be transferred again later
(new sidecars, a finished pair) without repeating the matching.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from track_muxer.conform.features import FFPROBE, probe_video_duration, probe_video_start
from track_muxer.conform.subtitles.clean import clean_cues
from track_muxer.conform.subtitles.retime import TimeMap, build_time_map, retime_cues
from track_muxer.conform.subtitles.sidecars import text_track_sidecars, text_track_tail
from track_muxer.conform.subtitles.srt import dump_srt
from track_muxer.conform.subtitles.vtt import neutral_cues, parse_vtt

MAP_DIR = "_timemap"
# Source format -> reader giving (format-neutral cues, cue settings dropped).
# A format without a reader is reported, never guessed at.
_READERS = {".vtt": lambda text: neutral_cues(parse_vtt(text))}
# The format is a detail of writing: one place decides what lands next to the aligned audio.
OUT_EXT, _WRITE = ".srt", dump_srt


def map_path(out_path: Path) -> Path:
    return out_path.parent / MAP_DIR / f"{out_path.stem}.npz"


def save_time_map(out_path: Path, T, curve, cuts, fps_ref: float, dur_ref: float) -> Path:
    p = map_path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    part = p.with_name(p.name + ".part.npz")
    np.savez_compressed(
        part, T=np.asarray(T, np.float64), curve=np.asarray(curve, np.float64),
        cuts=np.asarray(cuts, np.float64).reshape(-1, 4),
        fps_ref=np.float64(fps_ref), dur_ref=np.float64(dur_ref))
    part.replace(p)
    return p


def load_time_map(out_path: Path) -> TimeMap | None:
    p = map_path(out_path)
    if not p.exists():
        return None
    with np.load(p) as z:
        return build_time_map(z["T"], z["curve"], [tuple(c) for c in z["cuts"]],
                              float(z["fps_ref"]), float(z["dur_ref"]))


def identity_time_map(fps: float, dur: float) -> TimeMap:
    """The reference's own subtitles already sit on its timeline."""
    return build_time_map(np.array([0.0, dur]), np.zeros(2), [], fps, dur)


def transfer_sidecars(video: Path, out_dir: Path, out_stem: str, tmap: TimeMap,
                      ffprobe: str = FFPROBE) -> list[dict]:
    """Every sidecar of `video` -> `out_dir/<out_stem><sidecar tail>`, on the reference timeline.

    Never raises: subtitles must not take a finished pair down.
    """
    report: list[dict] = []
    try:
        sidecars = text_track_sidecars(video)
    except OSError as e:
        logger.warning("subtitles: cannot list sidecars of {}: {}", video.name, e)
        return report
    if not sidecars:
        return report
    # Sidecars are timed from the container start, the map from the first video frame.
    video_start = probe_video_start(video, ffprobe)
    # A cue past the end of its own video can never have been shown.
    dur_src = probe_video_duration(video, ffprobe) or float("inf")
    for src in sidecars:
        entry: dict = {"source": src.name, "file": None, "cues_in": 0, "cues_out": 0,
                       "dropped_drawings": 0, "dropped_empty": 0, "dropped_duplicates": 0,
                       "dropped_settings": 0, "error": None}
        report.append(entry)
        read = _READERS.get(src.suffix.lower())
        if read is None:
            entry["error"] = f"no reader for {src.suffix}"
            continue
        try:
            cues_in, n_settings = read(src.read_text(encoding="utf-8-sig"))
            cues, dropped = clean_cues(cues_in)
            cues = [c.at(c.start - video_start, c.end - video_start) for c in cues]
            moved = retime_cues(cues, tmap, dur_src)
            tail = text_track_tail(video, src)
            dst = out_dir / (out_stem + tail[:len(tail) - len(src.suffix)] + OUT_EXT)
            part = dst.with_name(dst.name + ".part")
            part.write_text(_WRITE(moved), encoding="utf-8")
            part.replace(dst)
            entry.update(file=dst.name, cues_in=len(cues_in), cues_out=len(moved),
                         dropped_settings=n_settings, **dropped)
        except Exception as e:  # noqa: BLE001
            entry["error"] = str(e)[:200]
            logger.warning("subtitles: {} not transferred: {}", src.name, e)
    return report
