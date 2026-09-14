"""Грубый проход: опора по КОНСЕНСУСУ надёжных кадров. Перенос 1-в-1 из
research/dropdtw_conform/_coarse_hough_srm.py — функции lis_nd, coarse_robust
БЕЗ изменений логики/констант.

Единственная правка против оригинала: DEV выбирается автоматически
(cuda при наличии, иначе cpu) вместо жёсткого "cuda". На машине с CUDA
(как при сборке эталонного v8) поведение идентично. Убраны тестовые
coarse_offset/evaluate/main и привязка к датасету.
"""

from __future__ import annotations

import numpy as np
import torch

K = 8; W = 80; VHI = 0.60; CMIN = 0.12
CWIN = 8000; COV = 2000; CSR = 4000     # оконный coarse: окно / перекрытие / полуполоса поиска ref (кадры)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def lis_nd(a):
    """индексы самой длинной НЕубывающей подпоследовательности a."""
    n = len(a)
    if n == 0: return []
    tails = []; back = [-1] * n
    for i in range(n):
        x = a[i]; lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if a[tails[mid]] <= x: lo = mid + 1
            else: hi = mid
        if lo > 0: back[i] = tails[lo - 1]
        if lo == len(tails): tails.append(i)
        else: tails[lo] = i
    k = tails[-1]; out = []
    while k != -1: out.append(k); k = back[k]
    return out[::-1]


def _anchors(srm_s, srm_r):
    """G=cs@cr.T → НАДЁЖНЫЕ якоря (synth_кадр, ref_кадр) ЛОКАЛЬНО. Ядро coarse_robust
    (тот же критерий VHI/CMIN) — переиспользуется и полным, и оконным проходом."""
    cs = torch.from_numpy(srm_s[::K].astype(np.float32)).to(DEV)
    cr = torch.from_numpy(srm_r[::K].astype(np.float32)).to(DEV)
    Nr = cr.shape[0]
    if cs.shape[0] < 2 or Nr < 2:
        return np.array([], np.int64), np.array([], np.int64)
    G = cs @ cr.T                                  # Ns×Nr
    bestv, best = G.max(1)
    offs = torch.arange(-W, W + 1, device=DEV)
    idx = (best.unsqueeze(1) + offs).clamp(0, Nr - 1)
    G.scatter_(1, idx, -2.0)                        # занулить окрестность best
    second = G.max(1).values
    rel = ((bestv > VHI) & ((bestv - second) > CMIN)).cpu().numpy()
    bestc = best.cpu().numpy()
    return np.where(rel)[0] * K, bestc[rel] * K     # synth-кадры якорей, их ref


def coarse_robust(srm_s, srm_r):
    """ПОЛНЫЙ грубый проход (вся матрица G). Для длинных файлов память/время растут —
    тогда coarse_windowed (тот же результат на проверенных кейсах, ограничен полосой)."""
    aj, ar = _anchors(srm_s, srm_r)
    nrel = len(aj)
    if nrel < 5:
        return np.zeros(len(srm_s), np.int64), 0, 0, np.array([], np.int64)
    keep = lis_nd(ar.tolist())                      # монотонная по ref цепочка
    aj2 = aj[keep]; ar2 = ar[keep]
    off = np.interp(np.arange(len(srm_s)), aj2, ar2 - aj2).astype(np.int64)
    return off, nrel, len(keep), aj2                # aj2 = synth-кадры цепочки (для краевого спасения band)


def coarse_windowed(srm_s, srm_r, win=CWIN, ov=COV, sr=CSR, on_prog=None):
    """ОКОННЫЙ грубый проход: окна по synth с перекрытием, seed offset ПЕРЕТЕКАЕТ из окна
    в окно, поиск ref только в ПОЛОСЕ ±sr вокруг seed. Память и время ограничены окном
    (не растут с длиной файла). Якорный критерий тот же (_anchors). Возврат как coarse_robust.
    Бит-в-бит совпал с полным проходом на 26 реальных/синтетических кейсах.
    on_prog(frac) — optional per-window progress hook."""
    N = len(srm_s); Rn = len(srm_r)
    gj_all, gr_all = [], []
    seed = 0; a = 0
    while a < N:
        b = min(a + win, N)
        r1 = max(0, a + seed - sr); r2 = min(Rn, b + seed + sr)
        aj, ar = _anchors(srm_s[a:b], srm_r[r1:r2])
        if on_prog is not None:
            on_prog(b / N)
        if len(aj):
            gj = aj + a; gr = ar + r1
            gj_all.append(gj); gr_all.append(gr)
            seed = int(np.median(gr - gj))          # seed следующего окна — из этого
        if b >= N:
            break
        a += win - ov
    if not gj_all:
        return np.zeros(N, np.int64), 0, 0, np.array([], np.int64)
    gj = np.concatenate(gj_all); gr = np.concatenate(gr_all)
    o = np.argsort(gj, kind="stable"); gj, gr = gj[o], gr[o]
    uj, ui = np.unique(gj, return_index=True); ur = gr[ui]   # дедуп по synth-кадру (перекрытия)
    keep = lis_nd(ur.tolist())                                # монотонная по ref цепочка
    uj, ur = uj[keep], ur[keep]
    off = np.interp(np.arange(N), uj, ur - uj).astype(np.int64)
    return off, len(gj), len(keep), uj                        # uj = synth-кадры цепочки (для краевого спасения band)
