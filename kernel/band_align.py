"""Следящая полоса + Drop-DTW на целом видео. Функции coarse_offset, band_align,
band_align_local_affine, cluster перенесены из исследовательского прототипа БЕЗ изменений
логики/констант.

Убран тестовый main() (датасет/matplotlib). Импорты движка — относительные.
"""

from __future__ import annotations

import numpy as np
import torch

from track_muxer.conform.kernel.dropdtw import (
    backtrack,
    backtrack_affine,
    drop_dtw,
    drop_dtw_affine,
    drop_dtw_affine_guard,
    drop_dtw_affine_guard_struct,
    runs_mask,
)

_BAND_CUDA = torch.cuda.is_available()       # GPU-бэкенд стоимости band; нет CUDA → CPU-эталон (fallback)


def _chunk_cost(ref_w, syn_w):
    """Стоимость C = 1 − cos одного КУСКА. CUDA → матмул на GPU (срез куска грузим на VRAM и
    освобождаем → память O(чанка), НЕ весь SRM → закон неограниченной длительности соблюдён).
    Нет CUDA → CPU-numpy (эталон, бит-в-бит как было). ref_w/syn_w — float32 numpy срезы куска.
    Возврат — np.float64 (как раньше), для numba-DTW. (GPU float32-матмул может отличаться от CPU
    на ~1 LSB — это и есть GPU-эталон, на котором валидирован ep47-фикс; алгоритм идентичен.)"""
    if _BAND_CUDA:
        rt = torch.from_numpy(ref_w).cuda(); st = torch.from_numpy(syn_w).cuda()
        C = (1.0 - rt @ st.t()).double().cpu().numpy()
        del rt, st
        return C
    return (1.0 - ref_w @ syn_w.T).astype(np.float64)

K = 8            # прорежение для грубого прохода
MARG = 120       # ПОЛУШИРИНА полосы (кадры) вокруг черновой линии
CHUNK = 4000     # длина куска (synth-кадры)
OVERLAP = 700    # перекрытие кусков (> макс правки ~500, чтобы правка целиком попадала в чей-то центр)
DROP = 0.20

# Краевое спасение по ПЛОТНОСТИ якорей (идея пользователя): band роняет яркую голову/край с низким
# абс. cos, хотя там РЕАЛЬНЫЙ синхронный контент (кейс ep47 — голова синхронна, но band её выбросил).
# Признак реального контента = ПЛОТНАЯ цепочка монотонных якорей грубого прохода рядом (а не cos).
# Действует ТОЛЬКО на КРАЯХ (первый/последний кусок) — там живёт баг; в теле сброс нужен как клапан
# (на сильно-растянутых дублях защита тела даёт ложные провалы — проверено).
OC_WIN = 60       # окно плотности, кадры (±2.5с)
OC_MIN_N = 3      # мин якорей цепочки в окне → кадр «на плотной цепочке» → защищён от выброса
OC_OFF_TOL = 20   # |скачок сдвига| цепочки > этого = РЕАЛЬНЫЙ вырез/вставка → зону вокруг НЕ защищать


def onchain_density(chain_aj, chain_off, N, win=OC_WIN, min_n=OC_MIN_N, off_tol=OC_OFF_TOL):
    """bool[N]: кадр k защищён от выброса band, если рядом (±win) ПЛОТНО (≥min_n) лежат якоря
    МОНОТОННОЙ цепочки (LIS грубого прохода) И сдвиг цепочки там СОГЛАСОВАН (нет скачка-выреза рядом).
    chain_aj — synth-кадры якорей (СОРТ по возрастанию), chain_off — их сдвиг (ar−aj).
    Зачем согласованность: дубль через ВЫРЕЗ непрерывен по своим кадрам (gap только в РЕФЕ) → одна
    плотность слепа к вырезу и оживила бы его (кейс ep27 OP-вырез). Реальный вырез/вставка = СКАЧОК
    сдвига цепочки > off_tol → зона ±win вокруг НЕ защищается. Голова ep47 = сдвиг ровный → защита."""
    aj = np.asarray(chain_aj, np.int64); of = np.asarray(chain_off, np.float64)
    if len(aj) < min_n:
        return np.zeros(N, np.bool_)
    k = np.arange(N)
    lo = np.searchsorted(aj, k - win, side="left")
    hi = np.searchsorted(aj, k + win, side="right")
    dense = (hi - lo) >= min_n
    # СОБЫТИЯ цепочки (вырез/вставка) = СКАЧОК сдвига. Защищаем ТОЛЬКО голову (до 1-го события) и
    # хвост (после последнего) — там живёт баг ep47. Середину (между событиями) НЕ трогаем: иначе
    # density перекрывает переход через вырез и band не делает прыжок (кейс ep27 OP-вырез, −81с).
    steps = np.where(np.abs(np.diff(of)) > off_tol)[0]
    if len(steps):
        first_ev = (int(aj[steps[0]]) + int(aj[steps[0] + 1])) // 2
        last_ev = (int(aj[steps[-1]]) + int(aj[steps[-1] + 1])) // 2
    else:
        first_ev, last_ev = N, 0                              # событий нет → вся дорожка = голова/хвост
    edge = (k <= first_ev) | (k >= last_ev)
    return dense & edge


def coarse_offset(syn, ref):
    """Грубый проход на прорежённых кадрах -> offset(synth) для всех кадров (интерполяция по якорям)."""
    cs = syn[::K].astype(np.float32); cr = ref[::K].astype(np.float32)
    C = (1.0 - cr @ cs.T).astype(np.float64)
    D, B = drop_dtw(C, DROP)
    pred, _, _ = backtrack(D, B, 0)              # pred[m] = грубый ref-индекс для прорежённого synth m
    ak, ao = [], []
    for m in range(len(pred)):
        if pred[m] >= 0:
            ak.append(m * K); ao.append(pred[m] * K - m * K)
    ak = np.array(ak); ao = np.array(ao)
    off = np.interp(np.arange(len(syn)), ak, ao).astype(np.int64)
    return off, len(ak)


def band_align(syn, ref, off, affine=False, OPEN=0.20, EXT=0.02, DSYN=0.20, MATCH_THR=None,
               free_start=False, on_prog=None, chain_aj=None, chain_off=None):
    """Следящая полоса: на каждый кусок узкое ref-окно вокруг черновой линии, Drop-DTW, склейка.
    affine=True -> аффинный штраф за выброс ref (острый шов).
    MATCH_THR задан -> guard: выброс synth запрещён для кадров с хорошим матчем (base_k<=MATCH_THR).
    free_start=True -> свободный ЛЕВЫЙ конец по synth для ПЕРВОГО окна: реклама/заставка начала выпадает
    сама (не прибиваем первый кадр к ref). Опора — найденные ЯКОРЯ (грубый проход), концы свободны.
    chain_aj задан -> КРАЕВОЕ СПАСЕНИЕ ПО ПЛОТНОСТИ ЯКОРЕЙ: в первом/последнем куске кадр с плотной
    цепочкой якорей (onchain_density) НЕ выбрасывается, даже при низком cos (яркая синхронная голова,
    кейс ep47). В теле не действует. chain_aj=None -> поведение БИТ-В-БИТ как раньше.
    on_prog(frac) -> опц. коллбэк прогресса по чанкам (0..1) для веб-полосы."""
    N = len(syn); Rn = len(ref)
    onc = (onchain_density(chain_aj, chain_off, N)
           if (chain_aj is not None and chain_off is not None and MATCH_THR is not None) else None)
    pred_full = np.full(N, -2, np.int64)         # -2 = не назначен, -1 = выброшен (вставка/край)
    quality = np.full(N, -1, np.int64)           # «насколько кадр внутри куска» (для склейки)
    starts = list(range(0, N, CHUNK - OVERLAP))
    nst = max(1, len(starts))
    for ist, a in enumerate(starts):
        if on_prog is not None:
            on_prog(ist / nst)
        b = min(a + CHUNK - 1, N - 1)
        ks = np.arange(a, b + 1)
        refk = ks + off[ks]
        r1 = max(0, int(refk.min()) - MARG); r2 = min(Rn - 1, int(refk.max()) + MARG)
        if r2 - r1 < 50: continue
        syn_w = syn[a:b + 1].astype(np.float32); ref_w = ref[r1:r2 + 1].astype(np.float32)
        C = _chunk_cost(ref_w, syn_w)                # GPU если CUDA, иначе CPU (память O(чанка))
        if affine:
            if MATCH_THR is not None:
                if onc is not None and (a == 0 or b == N - 1):     # краевой кусок → плотность защищает голову/хвост
                    st = np.ascontiguousarray(onc[a:b + 1])
                    M, Dr, BM, BDr = drop_dtw_affine_guard_struct(
                        C, OPEN, EXT, DSYN, MATCH_THR, st, free_start and a == 0)
                else:                                              # тело (или chain_aj нет) → как прод (бит-в-бит)
                    M, Dr, BM, BDr = drop_dtw_affine_guard(C, OPEN, EXT, DSYN, MATCH_THR, free_start and a == 0)
            else:
                M, Dr, BM, BDr = drop_dtw_affine(C, OPEN, EXT, DSYN)
            pred, _, _ = backtrack_affine(M, Dr, BM, BDr, r1)
        else:
            D, B = drop_dtw(C, DROP); pred, _, _ = backtrack(D, B, r1)
        for idx, kk in enumerate(range(a, b + 1)):
            q = min(kk - a, b - kk)              # расстояние до края куска (больше = надёжнее)
            if q > quality[kk]:
                quality[kk] = q; pred_full[kk] = pred[idx]
        if b == N - 1: break
    if on_prog is not None:
        on_prog(1.0)
    return pred_full


def band_align_local_affine(syn, ref, off, OPEN=0.20, EXT=0.02, DSYN=0.20, PAD=80):
    """Плоская полоса (стабильная основа) + АФФИН ЛОКАЛЬНО вокруг скачков offset (швов).
    Скачок плоской линии = триггер (детектор не нужен). На логотипных ровных участках аффин не
    включается -> убегания нет. Короткое окно вокруг шва -> аффину негде убегать."""
    N = len(syn); Rn = len(ref)
    pred = band_align(syn, ref, off, affine=False).copy()
    matched = np.where(pred >= 0)[0]
    if len(matched) < 3:
        return pred
    offm = pred[matched] - matched                        # offset на сматченных кадрах
    of_full = np.interp(np.arange(N), matched, offm)      # заполненный offset
    W2 = 12
    hi = np.minimum(np.arange(N) + W2, N - 1); lo = np.maximum(np.arange(N) - W2, 0)
    d = of_full[hi] - of_full[lo]                         # изменение offset на ±W2
    cut_mask = d > 10                                     # устойчивый скачок ВВЕРХ = ВЫРЕЗ (не логотип, не вставка)
    windows = []                                          # связные зоны скачка -> окна ±PAD, слитые
    for s, e in runs_mask(cut_mask):
        c = (s + e) // 2
        a, b = max(0, c - PAD), min(N - 1, c + PAD)
        if windows and a <= windows[-1][1] + 1:
            windows[-1] = [windows[-1][0], max(windows[-1][1], b)]
        else:
            windows.append([a, b])
    for a, b in windows:
        refs = [pred[k] for k in range(a, b + 1) if pred[k] >= 0]
        if not refs: continue
        r1 = max(0, min(refs) - 40); r2 = min(Rn - 1, max(refs) + 40)
        if r2 - r1 < 50: continue
        C = (1.0 - ref[r1:r2 + 1].astype(np.float32) @ syn[a:b + 1].astype(np.float32).T).astype(np.float64)
        M, Dr, BM, BDr = drop_dtw_affine(C, OPEN, EXT, DSYN)
        lp, _, _ = backtrack_affine(M, Dr, BM, BDr, r1)
        # ПРИЁМКА без истины: аффин должен СОВПАСТЬ с плоским у КРАЁВ окна (там чистый контент).
        # Разошёлся у краёв -> аффин УБЕЖАЛ (логотип) -> отвергаем, оставляем плоский.
        ed = 25; w = b - a + 1
        fl = pred[a:b + 1]

        def disagree(sl):
            f = fl[sl]; l = lp[sl]; m = (f >= 0) & (l >= 0)
            return float(np.mean(np.abs(f[m] - l[m]))) if m.any() else 0.0
        if w > 2 * ed and disagree(slice(0, ed)) <= 5 and disagree(slice(w - ed, w)) <= 5:
            pred[a:b + 1] = lp
    return pred


def cluster(js):
    if not js: return []
    js = sorted(js); pl = []; s = p = js[0]
    for j in js[1:]:
        if j - p <= 2: p = j
        else: pl.append((s, p)); s = p = j
    pl.append((s, p)); return pl
