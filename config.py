"""Пути и константы окружения для conform.

ffmpeg/ffprobe: по умолчанию КОРНЕВОЙ ffmpeg.exe проекта (тот же бинарь, что
строил эталонный conform v8 — важно для бит-в-бит приёмки). Если в корне нет —
из PATH. Переопределяется env TM_FFMPEG / TM_FFPROBE.
"""

from __future__ import annotations

import os
from pathlib import Path

# .../src/track_muxer/conform/config.py → parents[3] = корень репозитория
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve(env: str, exe: str) -> str:
    v = os.environ.get(env)
    if v:
        return v
    root_bin = PROJECT_ROOT / exe
    if root_bin.exists():
        return str(root_bin)
    return exe.rsplit(".", 1)[0]          # из PATH ("ffmpeg" / "ffprobe")


FFMPEG = _resolve("TM_FFMPEG", "ffmpeg.exe")
FFPROBE = _resolve("TM_FFPROBE", "ffprobe.exe")

# Каталог опционального дискового SRM-кэша (по умолчанию ВЫКЛ — ~1.26 ГБ/файл).
DEFAULT_CACHE_DIR = Path(os.environ.get("TM_CONFORM_CACHE", PROJECT_ROOT / "_conform_cache"))

# ── Hybrid CPU/GPU video decode ──
# 1080+ декодируется на GPU (NVDEC) пока активных cuda-сессий < CUDA_MAX (потолок против деления
# одного чипа: 4 cuda = +3%, 2 cuda+2 cpu = +13%); <1080 — всегда CPU (там CPU быстрее GPU).
# Нет CUDA в ffmpeg → всё на CPU (fallback). GPU-декод бит-в-бит идентичен CPU (scale на CPU).
CUDA_MAX = int(os.environ.get("TM_CUDA_MAX", "2"))     # макс. одновременных GPU-декодов (потолок NVDEC)
HYBRID_MIN_GPU_H = 1080                                # высота кадра, с которой пробуем GPU
