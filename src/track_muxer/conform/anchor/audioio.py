# -*- coding: utf-8 -*-
"""Загрузка аудио окнами через ffmpeg pipe (f32le mono) — единственное, что нужно
пайплайну якорей из исследовательского `lib_audio`. На этапе 2 (in-memory) прод-путь
эту функцию не вызывает: массивы приходят уже декодированными из conform."""
import subprocess
import numpy as np


def load_window(path, t0, dur, sr, ffmpeg="ffmpeg"):
    """[t0, t0+dur) из аудио → float32 mono @ sr, в [-1,1] (ffmpeg сам ресемплит/микширует)."""
    cmd = [ffmpeg, "-v", "error", "-ss", f"{t0:.6f}", "-t", f"{dur:.6f}",
           "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", str(sr),
           "-f", "f32le", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()
