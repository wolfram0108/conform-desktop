"""Paths and environment constants for conform.

ffmpeg/ffprobe: by default, the project's ROOT ffmpeg.exe (the same binary the
reference build was validated against — needed for bit-exact acceptance). If
none is found at the root, PATH is used instead. Overridden by the env vars
TM_FFMPEG / TM_FFPROBE.
"""

from __future__ import annotations

import os
from pathlib import Path

# .../src/track_muxer/conform/config.py -> parents[3] = the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve(env: str, exe: str) -> str:
    v = os.environ.get(env)
    if v:
        return v
    root_bin = PROJECT_ROOT / exe
    if root_bin.exists():
        return str(root_bin)
    return exe.rsplit(".", 1)[0]          # from PATH ("ffmpeg" / "ffprobe")


FFMPEG = _resolve("TM_FFMPEG", "ffmpeg.exe")
FFPROBE = _resolve("TM_FFPROBE", "ffprobe.exe")


# ── Hybrid CPU/GPU video decode ──
# 1080p+ decodes on GPU (NVDEC) while active CUDA sessions stay below CUDA_MAX (a ceiling against
# splitting one chip: 4 CUDA sessions cost +3%, 2 CUDA + 2 CPU costs +13%); below 1080p always CPU
# (CPU is faster there). No CUDA in ffmpeg -> everything falls back to CPU. GPU decode is bit-exact
# with CPU (scaling runs on CPU).
CUDA_MAX = int(os.environ.get("TM_CUDA_MAX", "2"))     # max concurrent GPU decodes (NVDEC ceiling)
HYBRID_MIN_GPU_H = 1080                                # frame height above which GPU is tried
