"""The audio-conform module: aligns a dub to the reference's timeline.

Layers:
  • kernel/   — the numeric CORE: thresholds and dynamics tuned together, a change to one changes every output;
  • features  — video → SRM vectors (decode plus convolution);
  • align     — one PAIR ref↔dub → the output audio file;
  • episode   — one EPISODE (the reference decoded once, plus the list of dubs);
  • cache     — reuse of SRM data (reference held in RAM, disk optional);
  • models    — dataclasses for the result and progress;
  • config    — paths (ffmpeg/ffprobe, cache).

The queue and the HTTP API sit on top as separate layers.
"""

from __future__ import annotations

from track_muxer.conform.models import (
    EpisodeResult,
    PairResult,
    Progress,
    SrmFeatures,
)

__all__ = [
    "EpisodeResult",
    "PairResult",
    "Progress",
    "SrmFeatures",
    "build_srm",
    "conform_episode",
    "conform_features",
    "conform_pair",
]


def __getattr__(name: str):  # noqa: D401 — lazy loading of heavy layers (torch/numba/scipy)
    if name == "build_srm":
        from track_muxer.conform.features import build_srm
        return build_srm
    if name in ("conform_pair", "conform_features"):
        from track_muxer.conform import align
        return getattr(align, name)
    if name == "conform_episode":
        from track_muxer.conform.episode import conform_episode
        return conform_episode
    raise AttributeError(name)
