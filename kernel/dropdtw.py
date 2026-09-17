"""Drop-DTW: «выравнивание с пропусками» (numba). Функции движка перенесены из
исследовательского прототипа БЕЗ изменений.

Убраны (не нужны в боевом конвейере, тянули датасет/matplotlib):
  ref_window, case_of, probe_case, find_cuts, probe_cut, main.
Оставлены ровно те функции, что зовёт band_align: drop_dtw, backtrack,
runs_mask, runs_vals, drop_dtw_affine, backtrack_affine, drop_dtw_affine_guard.
"""

from __future__ import annotations

import numpy as np
from numba import njit

MS = 1001 / 24000 * 1000
DROPS = [0.15, 0.20, 0.25, 0.30, 0.40]


@njit
def drop_dtw(C, DROP):
    R, S = C.shape
    INF = 1e18
    D = np.full((R, S), INF, np.float64)
    B = np.zeros((R, S), np.int8)          # 0 диаг,1 вверх,2 вбок,3 выброс-syn,4 выброс-ref,5 старт
    for i in range(R):                     # СВОБОДНЫЙ старт: synth[0] на любой ref-строке за C
        D[i, 0] = C[i, 0]; B[i, 0] = 5
    for k in range(1, S):                  # ref-строка 0, движемся по synth
        warp = D[0, k - 1] + C[0, k]; drp = D[0, k - 1] + DROP
        if warp <= drp: D[0, k] = warp; B[0, k] = 2
        else:           D[0, k] = drp;  B[0, k] = 3
    for i in range(1, R):
        for k in range(1, S):
            c = C[i, k]
            best = D[i - 1, k - 1] + c; bp = 0
            v = D[i - 1, k] + c
            if v < best: best = v; bp = 1
            v = D[i, k - 1] + c
            if v < best: best = v; bp = 2
            v = D[i, k - 1] + DROP
            if v < best: best = v; bp = 3
            v = D[i - 1, k] + DROP
            if v < best: best = v; bp = 4
            D[i, k] = best; B[i, k] = bp
    return D, B


def backtrack(D, B, r1):
    R, S = B.shape
    i = int(np.argmin(D[:, S - 1])); k = S - 1     # СВОБОДНЫЙ конец: старт обратного хода = argmin посл. столбца
    pred = np.full(S, -1, np.int64)
    drop_syn = np.zeros(S, np.bool_); drop_ref = []
    while k > 0:
        bp = B[i, k]
        if bp == 0:   pred[k] = r1 + i; i -= 1; k -= 1
        elif bp == 1: i -= 1
        elif bp == 2: pred[k] = r1 + i; k -= 1
        elif bp == 3: drop_syn[k] = True; k -= 1
        elif bp == 4: drop_ref.append(r1 + i); i -= 1
        else: break
    pred[0] = r1 + i
    return pred, drop_syn, drop_ref


def runs_mask(mask):
    out = []; s = None
    for k in range(len(mask)):
        if mask[k] and s is None: s = k
        elif not mask[k] and s is not None: out.append((s, k - 1)); s = None
    if s is not None: out.append((s, len(mask) - 1))
    return out


def runs_vals(vals):
    if not vals: return []
    vs = sorted(set(vals)); out = []; s = p = vs[0]
    for v in vs[1:]:
        if v == p + 1: p = v
        else: out.append((s, p)); s = p = v
    out.append((s, p)); return out


@njit
def drop_dtw_affine(C, OPEN, EXT, DSYN):
    """2-состоянийная аффинная динамика. M = матч/варп (платит C), Dr = ВНУТРИ выброса ref
    (открыть=OPEN из M, продлить=EXT из Dr). Выброс synth (вставка) — плоский DSYN внутри M.
    Свободный конец по ref. Возврат M,Dr и backpointer'ы BM (move*2+prevstate), BDr (0 open/1 ext)."""
    R, S = C.shape
    INF = 1e18
    M = np.full((R, S), INF); Dr = np.full((R, S), INF)
    BM = np.full((R, S), -1, np.int8); BDr = np.zeros((R, S), np.int8)
    for i in range(R):                                   # свободный старт по ref
        M[i, 0] = C[i, 0]; BM[i, 0] = 8                  # 8 = старт
    for k in range(1, S):                                # строка ref 0
        c = C[0, k]; pm = M[0, k - 1]; pd = Dr[0, k - 1]
        base = pm if pm <= pd else pd; ps = 0 if pm <= pd else 1
        hw = c + base; ds = DSYN + base
        if hw <= ds: M[0, k] = hw; BM[0, k] = 1 * 2 + ps
        else:        M[0, k] = ds; BM[0, k] = 3 * 2 + ps
    for i in range(1, R):
        for k in range(1, S):
            c = C[i, k]
            best = INF; bm = -1
            # diag (i-1,k-1)
            pm = M[i - 1, k - 1]; pd = Dr[i - 1, k - 1]
            b = pm if pm <= pd else pd; ps = 0 if pm <= pd else 1
            v = c + b
            if v < best: best = v; bm = 0 * 2 + ps
            # hwarp (i,k-1)
            pmh = M[i, k - 1]; pdh = Dr[i, k - 1]
            bh = pmh if pmh <= pdh else pdh; psh = 0 if pmh <= pdh else 1
            v = c + bh
            if v < best: best = v; bm = 1 * 2 + psh
            # vwarp (i-1,k)
            pmv = M[i - 1, k]; pdv = Dr[i - 1, k]
            bv = pmv if pmv <= pdv else pdv; psv = 0 if pmv <= pdv else 1
            v = c + bv
            if v < best: best = v; bm = 2 * 2 + psv
            # drop-syn (i,k-1), плоский
            v = DSYN + bh
            if v < best: best = v; bm = 3 * 2 + psh
            M[i, k] = best; BM[i, k] = bm
            # Dr[i,k] из (i-1,k): открыть из M или продлить из Dr
            op = M[i - 1, k] + OPEN; ex = Dr[i - 1, k] + EXT
            if op <= ex: Dr[i, k] = op; BDr[i, k] = 0
            else:        Dr[i, k] = ex; BDr[i, k] = 1
    return M, Dr, BM, BDr


def backtrack_affine(M, Dr, BM, BDr, r1):
    R, S = M.shape
    i = 0; bestv = INF = 1e18
    for ii in range(R):
        v = M[ii, S - 1] if M[ii, S - 1] <= Dr[ii, S - 1] else Dr[ii, S - 1]
        if v < bestv: bestv = v; i = ii
    state = 0 if M[i, S - 1] <= Dr[i, S - 1] else 1
    k = S - 1
    pred = np.full(S, -1, np.int64); drop_syn = np.zeros(S, np.bool_); drop_ref = []
    started = False
    while k > 0:
        if state == 0:
            code = BM[i, k]
            if code == 8: break
            if code == 9:                       # СВОБОДНЫЙ СТАРТ (free_start): путь начинается с этого якоря,
                pred[k] = r1 + i                # ведущие synth-кадры [0..k-1] выброшены (реклама/заставка начала)
                for j in range(k): drop_syn[j] = True
                started = True; break
            move = code >> 1; ps = code & 1
            if move == 0:   pred[k] = r1 + i; i -= 1; k -= 1; state = ps
            elif move == 1: pred[k] = r1 + i; k -= 1; state = ps
            elif move == 2: pred[k] = r1 + i; i -= 1; state = ps
            elif move == 3: drop_syn[k] = True; k -= 1; state = ps
        else:
            drop_ref.append(r1 + i); code = BDr[i, k]; i -= 1; state = 0 if code == 0 else 1
            if i < 0: break
    if state == 0 and pred[0] < 0 and not started:   # без free_start — старое поведение (первый кадр прибит)
        pred[0] = r1 + i
    return pred, drop_syn, drop_ref


@njit
def drop_dtw_affine_guard(C, OPEN, EXT, DSYN, MATCH_THR, free_start=False):
    """Как drop_dtw_affine, НО выброс synth-кадра (вставка) РАЗРЕШЁН только если у кадра НЕТ хорошего
    матча: base_k = min по ref C[:,k] > MATCH_THR. Кадр с хорошим матчем (base_k <= MATCH_THR) защищён
    от выброса (drop-цена = INF) — лечит расширение drop-прогона на matchable кадры (хвосты cos=1.0).
    free_start=True: СВОБОДНЫЙ ЛЕВЫЙ КОНЕЦ по synth — путь может начаться с ЛЮБОГО якоря (клетки),
    выбросив ведущие synth-кадры БЕСПЛАТНО (реклама/заставка начала). Симметрично свободному концу справа.
    Старт = только если матч C[i,k] выгоднее накопленного пути слева (т.е. слева был мусор/вставка)."""
    R, S = C.shape
    INF = 1e18
    colmin = np.empty(S, np.float64)                 # лучший достижимый матч на каждый synth-кадр
    for k in range(S):
        m = C[0, k]
        for i in range(1, R):
            if C[i, k] < m: m = C[i, k]
        colmin[k] = m
    M = np.full((R, S), INF); Dr = np.full((R, S), INF)
    BM = np.full((R, S), -1, np.int8); BDr = np.zeros((R, S), np.int8)
    for i in range(R):
        M[i, 0] = C[i, 0]; BM[i, 0] = 8
    for k in range(1, S):
        c = C[0, k]; pm = M[0, k - 1]; pd = Dr[0, k - 1]
        base = pm if pm <= pd else pd; ps = 0 if pm <= pd else 1
        dsk = DSYN if colmin[k] > MATCH_THR else INF
        hw = c + base; ds = dsk + base
        mbest = hw; mbm = 1 * 2 + ps
        if ds < mbest: mbest = ds; mbm = 3 * 2 + ps
        cf = c + k * DSYN                                    # старт здесь: ведущие [0..k-1] выброшены по цене вставки
        if free_start and cf < mbest: mbest = cf; mbm = 9    # (штраф растёт с k → реклама выпадет, но не улетит далеко)
        M[0, k] = mbest; BM[0, k] = mbm
    for i in range(1, R):
        for k in range(1, S):
            c = C[i, k]
            best = INF; bm = -1
            pm = M[i - 1, k - 1]; pd = Dr[i - 1, k - 1]
            b = pm if pm <= pd else pd; ps = 0 if pm <= pd else 1
            v = c + b
            if v < best: best = v; bm = 0 * 2 + ps
            pmh = M[i, k - 1]; pdh = Dr[i, k - 1]
            bh = pmh if pmh <= pdh else pdh; psh = 0 if pmh <= pdh else 1
            v = c + bh
            if v < best: best = v; bm = 1 * 2 + psh
            pmv = M[i - 1, k]; pdv = Dr[i - 1, k]
            bv = pmv if pmv <= pdv else pdv; psv = 0 if pmv <= pdv else 1
            v = c + bv
            if v < best: best = v; bm = 2 * 2 + psv
            dsk = DSYN if colmin[k] > MATCH_THR else INF
            v = dsk + bh
            if v < best: best = v; bm = 3 * 2 + psh
            cf = c + k * DSYN                                # старт: ведущие [0..k-1] выброшены по цене вставки
            if free_start and cf < best: best = cf; bm = 9   # штраф k*DSYN ограничивает выброс рекламой, не улетает
            M[i, k] = best; BM[i, k] = bm
            op = M[i - 1, k] + OPEN; ex = Dr[i - 1, k] + EXT
            if op <= ex: Dr[i, k] = op; BDr[i, k] = 0
            else:        Dr[i, k] = ex; BDr[i, k] = 1
    return M, Dr, BM, BDr


@njit
def drop_dtw_affine_guard_struct(C, OPEN, EXT, DSYN, MATCH_THR, struct, free_start=False):
    """Как drop_dtw_affine_guard, НО выброс synth-кадра запрещён ещё и для СТРУКТУРНЫХ кадров
    (struct[k]=True — рядом ПЛОТНАЯ цепочка якорей грубого прохода → реальный контент, пусть и с
    низким cos: яркая голова/край). free_start не стартует, если в выбрасываемом префиксе есть
    структурный кадр. struct=все-False → БИТ-В-БИТ равен drop_dtw_affine_guard (безопасный фолбэк)."""
    R, S = C.shape
    INF = 1e18
    colmin = np.empty(S, np.float64)
    for k in range(S):
        m = C[0, k]
        for i in range(1, R):
            if C[i, k] < m:
                m = C[i, k]
        colmin[k] = m
    pre = np.zeros(S + 1, np.int64)                      # префиксная сумма структурных (гейт free_start)
    for k in range(S):
        pre[k + 1] = pre[k] + (1 if struct[k] else 0)
    M = np.full((R, S), INF); Dr = np.full((R, S), INF)
    BM = np.full((R, S), -1, np.int8); BDr = np.zeros((R, S), np.int8)
    for i in range(R):
        M[i, 0] = C[i, 0]; BM[i, 0] = 8
    for k in range(1, S):
        c = C[0, k]; pm = M[0, k - 1]; pd = Dr[0, k - 1]
        base = pm if pm <= pd else pd; ps = 0 if pm <= pd else 1
        dsk = DSYN if (colmin[k] > MATCH_THR and not struct[k]) else INF
        hw = c + base; ds = dsk + base
        mbest = hw; mbm = 1 * 2 + ps
        if ds < mbest:
            mbest = ds; mbm = 3 * 2 + ps
        cf = c + k * DSYN
        if free_start and pre[k] == 0 and cf < mbest:    # старт только если в префиксе нет структурных
            mbest = cf; mbm = 9
        M[0, k] = mbest; BM[0, k] = mbm
    for i in range(1, R):
        for k in range(1, S):
            c = C[i, k]
            best = INF; bm = -1
            pm = M[i - 1, k - 1]; pd = Dr[i - 1, k - 1]
            b = pm if pm <= pd else pd; ps = 0 if pm <= pd else 1
            v = c + b
            if v < best:
                best = v; bm = 0 * 2 + ps
            pmh = M[i, k - 1]; pdh = Dr[i, k - 1]
            bh = pmh if pmh <= pdh else pdh; psh = 0 if pmh <= pdh else 1
            v = c + bh
            if v < best:
                best = v; bm = 1 * 2 + psh
            pmv = M[i - 1, k]; pdv = Dr[i - 1, k]
            bv = pmv if pmv <= pdv else pdv; psv = 0 if pmv <= pdv else 1
            v = c + bv
            if v < best:
                best = v; bm = 2 * 2 + psv
            dsk = DSYN if (colmin[k] > MATCH_THR and not struct[k]) else INF
            v = dsk + bh
            if v < best:
                best = v; bm = 3 * 2 + psh
            cf = c + k * DSYN
            if free_start and pre[k] == 0 and cf < best:
                best = cf; bm = 9
            M[i, k] = best; BM[i, k] = bm
            op = M[i - 1, k] + OPEN; ex = Dr[i - 1, k] + EXT
            if op <= ex:
                Dr[i, k] = op; BDr[i, k] = 0
            else:
                Dr[i, k] = ex; BDr[i, k] = 1
    return M, Dr, BM, BDr


@njit
def drop_dtw_affine_guard_amerce(C, OPEN, EXT, DSYN, MATCH_THR, AMERCE, free_start=False):
    """Как drop_dtw_affine_guard, НО AMERCING (Herrmann&Webb): штраф ω за ВАРП-шаг матч-состояния.
    Плато постоянного лага = диагональ (move 0, без штрафа); лаг меняется через hwarp(move1)/vwarp(move2)
    — им +ω. DROP-ветви (drop_syn DSYN / drop_ref OPEN/EXT) = реальные события, ω их НЕ трогает.
    Подавляет ВАРП-сингулярности (ложные дипы пути); реальное событие (выигрыш ≫ω) берётся.
    Диаг-приор (тяга к тренду в слепых зонах) добавляется ВЫШЕ в матрицу C (банд-обёртка), не здесь."""
    R, S = C.shape
    INF = 1e18
    colmin = np.empty(S, np.float64)
    for k in range(S):
        m = C[0, k]
        for i in range(1, R):
            if C[i, k] < m: m = C[i, k]
        colmin[k] = m
    M = np.full((R, S), INF); Dr = np.full((R, S), INF)
    BM = np.full((R, S), -1, np.int8); BDr = np.zeros((R, S), np.int8)
    for i in range(R):
        M[i, 0] = C[i, 0]; BM[i, 0] = 8
    for k in range(1, S):
        c = C[0, k]; pm = M[0, k - 1]; pd = Dr[0, k - 1]
        base = pm if pm <= pd else pd; ps = 0 if pm <= pd else 1
        dsk = DSYN if colmin[k] > MATCH_THR else INF
        hw = c + base + AMERCE; ds = dsk + base               # hwarp += ω; drop-syn без ω
        mbest = hw; mbm = 1 * 2 + ps
        if ds < mbest: mbest = ds; mbm = 3 * 2 + ps
        cf = c + k * DSYN
        if free_start and cf < mbest: mbest = cf; mbm = 9
        M[0, k] = mbest; BM[0, k] = mbm
    for i in range(1, R):
        for k in range(1, S):
            c = C[i, k]
            best = INF; bm = -1
            pm = M[i - 1, k - 1]; pd = Dr[i - 1, k - 1]
            b = pm if pm <= pd else pd; ps = 0 if pm <= pd else 1
            v = c + b                                          # ДИАГ (лаг постоянен) — БЕЗ штрафа
            if v < best: best = v; bm = 0 * 2 + ps
            pmh = M[i, k - 1]; pdh = Dr[i, k - 1]
            bh = pmh if pmh <= pdh else pdh; psh = 0 if pmh <= pdh else 1
            v = c + bh + AMERCE                                # HWARP (лаг↓) += ω
            if v < best: best = v; bm = 1 * 2 + psh
            pmv = M[i - 1, k]; pdv = Dr[i - 1, k]
            bv = pmv if pmv <= pdv else pdv; psv = 0 if pmv <= pdv else 1
            v = c + bv + AMERCE                                # VWARP (лаг↑) += ω
            if v < best: best = v; bm = 2 * 2 + psv
            dsk = DSYN if colmin[k] > MATCH_THR else INF
            v = dsk + bh                                       # DROP-SYN (вставка) — БЕЗ ω
            if v < best: best = v; bm = 3 * 2 + psh
            cf = c + k * DSYN
            if free_start and cf < best: best = cf; bm = 9
            M[i, k] = best; BM[i, k] = bm
            op = M[i - 1, k] + OPEN; ex = Dr[i - 1, k] + EXT   # DROP-REF (вырез) — своя цена, без ω
            if op <= ex: Dr[i, k] = op; BDr[i, k] = 0
            else:        Dr[i, k] = ex; BDr[i, k] = 1
    return M, Dr, BM, BDr
