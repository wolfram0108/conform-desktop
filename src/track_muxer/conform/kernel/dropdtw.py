"""Drop-DTW, alignment with skips (numba): the plain kernel, the affine-gap kernel and its guarded
variants, each with its backtrack."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit
def drop_dtw(C, DROP):
    R, S = C.shape
    INF = 1e18
    D = np.full((R, S), INF, np.float64)
    B = np.zeros((R, S), np.int8)          # 0 diag, 1 up, 2 side, 3 drop-syn, 4 drop-ref, 5 start
    for i in range(R):                     # free start: synth[0] can match any ref row, cost from C
        D[i, 0] = C[i, 0]; B[i, 0] = 5
    for k in range(1, S):                  # ref row 0, moving along synth
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
    i = int(np.argmin(D[:, S - 1])); k = S - 1     # free end: backtrack starts at the argmin of the last column
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


@njit
def drop_dtw_affine(C, OPEN, EXT, DSYN):
    """Two-state affine dynamics. M = match/warp (pays C), Dr = INSIDE a ref drop (open=OPEN
    from M, extend=EXT from Dr). A synth drop (insertion) is a flat DSYN inside M. Free end
    on the ref side. Returns M, Dr and the backpointers BM (move*2+prevstate), BDr (0 open/1 ext)."""
    R, S = C.shape
    INF = 1e18
    M = np.full((R, S), INF); Dr = np.full((R, S), INF)
    BM = np.full((R, S), -1, np.int8); BDr = np.zeros((R, S), np.int8)
    for i in range(R):                                   # free start along ref
        M[i, 0] = C[i, 0]; BM[i, 0] = 8                  # 8 = start
    for k in range(1, S):                                # ref row 0
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
            # drop-syn (i,k-1), flat
            v = DSYN + bh
            if v < best: best = v; bm = 3 * 2 + psh
            M[i, k] = best; BM[i, k] = bm
            # Dr[i,k] from (i-1,k): open from M or extend from Dr
            op = M[i - 1, k] + OPEN; ex = Dr[i - 1, k] + EXT
            if op <= ex: Dr[i, k] = op; BDr[i, k] = 0
            else:        Dr[i, k] = ex; BDr[i, k] = 1
    return M, Dr, BM, BDr


def backtrack_affine(M, Dr, BM, BDr, r1):
    R, S = M.shape
    i = 0; bestv = 1e18
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
            if code == 9:                       # free start: the path begins at this anchor,
                pred[k] = r1 + i                # the leading synth frames [0..k-1] are dropped (an intro/ad at the start)
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
    if state == 0 and pred[0] < 0 and not started:   # with free_start off, the first frame is pinned
        pred[0] = r1 + i
    return pred, drop_syn, drop_ref


@njit
def drop_dtw_affine_guard(C, OPEN, EXT, DSYN, MATCH_THR, free_start=False):
    """Same as drop_dtw_affine, BUT dropping a synth frame (insertion) is ALLOWED only when
    the frame has NO good match: base_k = min over ref C[:,k] > MATCH_THR. A frame with a
    good match (base_k <= MATCH_THR) is protected from being dropped (drop cost = INF) --
    this stops a drop run from spreading onto matchable frames (cos=1.0 tails).
    free_start=True: a FREE LEFT END on the synth side -- the path may start at ANY anchor
    (cell), dropping the leading synth frames FOR FREE (an intro/bumper at the start).
    Symmetric to the free end on the right. It only starts there when matching C[i,k] beats
    the accumulated path from the left (i.e. the left side was junk/insertion)."""
    R, S = C.shape
    INF = 1e18
    colmin = np.empty(S, np.float64)                 # best achievable match for each synth frame
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
        cf = c + k * DSYN                                    # start here: the leading [0..k-1] frames are dropped at insert cost
        if free_start and cf < mbest: mbest = cf; mbm = 9    # (the penalty grows with k: an intro drops but cannot run far)
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
            cf = c + k * DSYN                                # start: the leading [0..k-1] frames are dropped at insert cost
            if free_start and cf < best: best = cf; bm = 9   # the k*DSYN penalty bounds the drop to an intro, it cannot run away
            M[i, k] = best; BM[i, k] = bm
            op = M[i - 1, k] + OPEN; ex = Dr[i - 1, k] + EXT
            if op <= ex: Dr[i, k] = op; BDr[i, k] = 0
            else:        Dr[i, k] = ex; BDr[i, k] = 1
    return M, Dr, BM, BDr


@njit
def drop_dtw_affine_guard_struct(C, OPEN, EXT, DSYN, MATCH_THR, struct, free_start=False):
    """Same as drop_dtw_affine_guard, BUT dropping a synth frame is also forbidden for
    STRUCTURAL frames (struct[k]=True: a DENSE chain of coarse-pass anchors nearby means
    real content even at low cos, e.g. a bright opening/edge). free_start does not start
    when the prefix being dropped contains a structural frame. struct=all-False makes this
    BIT-FOR-BIT identical to drop_dtw_affine_guard (a safe fallback)."""
    R, S = C.shape
    INF = 1e18
    colmin = np.empty(S, np.float64)
    for k in range(S):
        m = C[0, k]
        for i in range(1, R):
            if C[i, k] < m:
                m = C[i, k]
        colmin[k] = m
    pre = np.zeros(S + 1, np.int64)                      # prefix sum of structural frames (gates free_start)
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
        if free_start and pre[k] == 0 and cf < mbest:    # start only when the prefix has no structural frame
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
    """Same as drop_dtw_affine_guard, BUT with AMERCING (Herrmann & Webb): a penalty omega on
    a WARP step of the match state. A plateau of constant lag is the diagonal (move 0, no
    penalty); lag changes via hwarp (move 1) / vwarp (move 2), which get +omega. The DROP
    branches (drop_syn DSYN / drop_ref OPEN/EXT) are real events, and omega does not touch
    them. This suppresses WARP singularities (spurious dips in the path); a real event
    (gain much greater than omega) still wins. The diagonal prior (pull toward the trend in
    blind zones) is added higher up, in matrix C (the band wrapper), not here."""
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
        hw = c + base + AMERCE; ds = dsk + base               # hwarp += omega; drop-syn has no omega
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
            v = c + b                                          # diag (lag constant): no penalty
            if v < best: best = v; bm = 0 * 2 + ps
            pmh = M[i, k - 1]; pdh = Dr[i, k - 1]
            bh = pmh if pmh <= pdh else pdh; psh = 0 if pmh <= pdh else 1
            v = c + bh + AMERCE                                # hwarp (lag decreases) += omega
            if v < best: best = v; bm = 1 * 2 + psh
            pmv = M[i - 1, k]; pdv = Dr[i - 1, k]
            bv = pmv if pmv <= pdv else pdv; psv = 0 if pmv <= pdv else 1
            v = c + bv + AMERCE                                # vwarp (lag increases) += omega
            if v < best: best = v; bm = 2 * 2 + psv
            dsk = DSYN if colmin[k] > MATCH_THR else INF
            v = dsk + bh                                       # drop-syn (insert): no omega
            if v < best: best = v; bm = 3 * 2 + psh
            cf = c + k * DSYN
            if free_start and cf < best: best = cf; bm = 9
            M[i, k] = best; BM[i, k] = bm
            op = M[i - 1, k] + OPEN; ex = Dr[i - 1, k] + EXT   # drop-ref (cut): its own cost, no omega
            if op <= ex: Dr[i, k] = op; BDr[i, k] = 0
            else:        Dr[i, k] = ex; BDr[i, k] = 1
    return M, Dr, BM, BDr
