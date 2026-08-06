"""Линейная интерполяция аудио по дробным точкам (варп/ресэмпл) — GPU/CPU бэкенд.

Эквивалент `np.interp(src, np.arange(len(y)), y)` с клампом краёв (без экстраполяции).
В цикле conform этих варпов несколько на ПОЛНОЙ длине (@44.1к × каналы): выходной ресэмпл
аудио на сетку рефа (align) + варпы band (_warp_by_off0/_warp_piecewise). На CPU (`np.interp`)
это ~16–22с; на GPU те же линейные lerp'ы — ×10 (замер: doc/reports/decode_bench).

GPU-ветка считает в float64 ТОЙ ЖЕ формулой, что np.interp (y0 + f·(y1−y0)) → бит-в-бит с CPU
(сложение/умножение IEEE коммутативны точно; деление на шаг сетки =1.0 точное). CPU-fallback
(`np.interp`) — тоже бит-в-бит со старым кодом. fp64 на варпе memory-bound → почти без замедления.

ЗАКОН ДЛИТЕЛЬНОСТИ: GPU-ветка БЛОЧНАЯ (по `block` точек) — VRAM const, не растёт с длиной файла.
ЗАКОН GPU-FIRST: CUDA есть → GPU, иначе CPU (потребитель без GPU не теряется).
"""

from __future__ import annotations

import numpy as np
import torch

# Окно блока (точек src). ~90с@44.1к; y-окно блока + src ≈ десятки МБ → VRAM const на любой длине.
WARP_BLOCK = 4_000_000


def cuda_ok() -> bool:
    return torch.cuda.is_available()


def warp_interp(y, src, *, block: int = WARP_BLOCK) -> np.ndarray:
    """Линейная интерполяция значений `y` (на целочисл. сетке 0..len(y)-1) в дробных точках `src`.
    Клампит края как np.interp (без экстраполяции). GPU блочно при CUDA, иначе CPU (бит-в-бит).
    Возврат: float32 формы `src`."""
    y = np.ascontiguousarray(y, dtype=np.float32)
    src = np.asarray(src, dtype=np.float64)
    n = int(y.shape[0])
    if n < 2 or not cuda_ok():
        return np.interp(src, np.arange(n), y).astype(np.float32)   # CPU-fallback — бит-в-бит

    out = np.empty(src.shape, dtype=np.float32)
    for s0 in range(0, len(src), block):
        sb = src[s0:s0 + block]
        lo = min(max(0, int(np.floor(sb.min()))), n - 2)    # y-окно блока; clamp lo≤n-2 → срез НЕ пустой
        hi = max(min(n, int(np.ceil(sb.max())) + 2), lo + 2)  # весь блок за концом аудио → края (как np.interp), без краша
        # float64 ТОЙ ЖЕ формулой, что np.interp → бит-в-бит с CPU; финальный каст float32 в конце
        yt = torch.from_numpy(y[lo:hi]).to("cuda", torch.float64)
        st = torch.from_numpy(sb).to("cuda")                 # src уже float64
        m = int(yt.shape[0])
        idx = torch.clamp(torch.floor(st).long() - lo, 0, m - 2)
        f = (st - lo - idx.double()).clamp_(0.0, 1.0)        # доля в float64; clamp = без экстрапол.
        y0 = yt[idx]
        out[s0:s0 + block] = (y0 + f * (yt[idx + 1] - y0)).to(torch.float32).cpu().numpy()
    return out
