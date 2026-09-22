# -*- coding: utf-8 -*-
"""Choosing the video decode backend: 1080+ -> GPU (NVDEC) with a ceiling on concurrent sessions,
<1080 -> CPU.

Auto-detected from the hardware: no cuda-hwaccel in ffmpeg -> ALWAYS CPU (the fallback is
mandatory, so a consumer without a GPU is never left without a decode path). The CUDA_MAX ceiling
protects the single NVDEC unit from oversubscription: measured that 4 cuda sessions on one chip
gain only +3%, while 2 cuda + 2 cpu gain +13% (NVDEC at half load plus a free CPU). GPU decode is
bit-exact with CPU (scale stays on the CPU, the decode is deterministic), so the SRM/wav cache
does NOT depend on the backend.

The active-cuda-decode counter is global and thread-safe (production runs episodes concurrently
on threads).
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

from track_muxer.conform.config import CUDA_MAX, FFMPEG, HYBRID_MIN_GPU_H
from track_muxer.conform import procreg

_lock = threading.Lock()
_cuda_active = 0
_cuda_ok: bool | None = None


def cuda_available(ffmpeg: str = FFMPEG) -> bool:
    """Whether WORKING cuda decode is available: ffmpeg built with cuda-hwaccel AND a GPU device
    actually present. Detected once, then cached. Error or no ffmpeg -> False.

    `-hwaccels` lists COMPILED-IN backends without checking for a device: a bundled ffmpeg with
    cuda support (a portable EXE carrying a cuda build) lists cuda on a machine without an NVIDIA
    GPU, but the real `-hwaccel cuda` then fails at cuInit (CUDA_ERROR_NO_DEVICE), crashing the
    decode of 1080p+ content. Cross-checking against torch.cuda.is_available() avoids this: it
    genuinely calls cuInit and sees the device. This separates the compile-time "can do cuda" from
    the runtime "device is present" (BOTH are required)."""
    global _cuda_ok
    if _cuda_ok is None:
        try:
            import torch
            if not torch.cuda.is_available():          # no real GPU device -> CPU decode
                _cuda_ok = False
                return _cuda_ok
            out = procreg.run([ffmpeg, "-hide_banner", "-hwaccels"],
                                 capture_output=True, text=True, timeout=15).stdout
            _cuda_ok = "cuda" in out.split()
        except Exception:  # noqa: BLE001 — no ffmpeg/torch, or a timeout -> CPU-only
            _cuda_ok = False
    return _cuda_ok


@contextmanager
def decode_backend(height: int, ffmpeg: str = FFMPEG):
    """Decode context for a frame of height `height`: yields 'cuda' | 'cpu'; the cuda slot is held
    until exit.

    1080+ -> 'cuda' while active cuda sessions < CUDA_MAX, otherwise 'cpu'. <1080 -> 'cpu'.
    No CUDA -> 'cpu'. The slot is held for the WHOLE decode (build_srm) and released in finally.
    """
    global _cuda_active
    use_cuda = False
    if height >= HYBRID_MIN_GPU_H and cuda_available(ffmpeg):
        with _lock:
            if _cuda_active < CUDA_MAX:
                _cuda_active += 1
                use_cuda = True
    try:
        yield "cuda" if use_cuda else "cpu"
    finally:
        if use_cuda:
            with _lock:
                _cuda_active -= 1
