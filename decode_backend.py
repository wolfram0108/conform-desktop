# -*- coding: utf-8 -*-
"""Выбор бэкенда декода видео: 1080+ → GPU (NVDEC) с потолком одновременных сессий, <1080 → CPU.

Авто по железу: нет cuda-hwaccel в ffmpeg → ВСЕГДА CPU (fallback обязателен — потребитель без
GPU не теряется). Потолок CUDA_MAX защищает единственный NVDEC от деления: замеры
(doc/reports/decode_bench/PLAN_hybrid_decode.md) показали, что 4 cuda-сессии на одном чипе дают
+3%, а 2 cuda + 2 cpu — +13% (NVDEC наполовину + свободный CPU). GPU-декод бит-в-бит идентичен
CPU (scale остаётся на CPU, декод детерминирован) → кэш SRM/wav от бэкенда НЕ зависят.

Счётчик активных cuda-декодов глобальный и потокобезопасный (прод параллелит серии потоками).
"""
from __future__ import annotations

import subprocess
import threading
from contextlib import contextmanager

from track_muxer.conform.config import CUDA_MAX, FFMPEG, HYBRID_MIN_GPU_H
from track_muxer.conform import procreg

_lock = threading.Lock()
_cuda_active = 0
_cuda_ok: bool | None = None


def cuda_available(ffmpeg: str = FFMPEG) -> bool:
    """Есть ли РАБОЧИЙ cuda-декод: ffmpeg собран с cuda-hwaccel И реально доступно GPU-устройство.
    Детект единожды, кэшируется. Ошибка/нет ffmpeg → False.

    ⚠ `-hwaccels` перечисляет СКОМПИЛИРОВАННЫЕ бэкенды, НЕ проверяя наличие устройства: bundled
    ffmpeg с cuda (наш случай — портативный EXE везёт cuda-сборку) на машине без NVIDIA покажет
    cuda в списке, но реальный `-hwaccel cuda` упадёт на cuInit (CUDA_ERROR_NO_DEVICE) → крах декода
    1080p+. Поэтому сверяем с torch.cuda.is_available() — он честно зовёт cuInit и видит устройство.
    Так compile-time «умеет cuda» отделено от runtime «устройство доступно» (нужны ОБА)."""
    global _cuda_ok
    if _cuda_ok is None:
        try:
            import torch
            if not torch.cuda.is_available():          # нет реального GPU-устройства → CPU-декод
                _cuda_ok = False
                return _cuda_ok
            out = procreg.run([ffmpeg, "-hide_banner", "-hwaccels"],
                                 capture_output=True, text=True, timeout=15).stdout
            _cuda_ok = "cuda" in out.split()
        except Exception:  # noqa: BLE001 — нет ffmpeg/torch/таймаут → CPU-only
            _cuda_ok = False
    return _cuda_ok


@contextmanager
def decode_backend(height: int, ffmpeg: str = FFMPEG):
    """Контекст декода кадра высоты `height`: yield 'cuda' | 'cpu'; cuda-слот удерживается до выхода.

    1080+ → 'cuda' пока активных cuda-сессий < CUDA_MAX, иначе 'cpu'. <1080 → 'cpu'.
    Нет CUDA → 'cpu'. Слот занимается на ВСЁ время декода (build_srm) и освобождается в finally.
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
