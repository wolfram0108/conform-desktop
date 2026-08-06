# -*- coding: utf-8 -*-
"""ОБЩЕЕ ядро детекта (одно на оба метода). Вход — (o,w) от любой карты.
Взвешенная (вес=качество^qpow) ломающаяся-прямая DP-сегментация → seglines + резы;
ПОСТФИЛЬТР по смещению хвоста (рез реален только если устойчивый уровень после ≠ до
на ≥ порога). Контент-агностично и симметрично. Знак: правее=+; кадр=41.708мс."""
import numpy as np
from .params import T as _DEFT, PEN, QPOW, SMAX, MSIZE_S, STEP, MIN_FR

# ---------- взвешенная DP-сегментация (ломающиеся прямые) ----------
def wfit(t, y, w):
    for _ in range(2):
        W = w.sum()
        if W < 1e-9: return 0.0, float(np.mean(y)), 99.0
        mt = np.sum(w*t)/W; my = np.sum(w*y)/W; den = np.sum(w*(t-mt)**2)
        a = np.sum(w*(t-mt)*(y-my))/den if den > 1e-9 else 0.0; b = my-a*mt
        r = y-(a*t+b); s = np.sqrt(np.sum(w*r*r)/W)
        w = w*np.exp(-(r/(2.5*s+1e-9))**2) if s > 1e-9 else w
    return a, b, float(s)

def _wtab(t, y, w):
    z = lambda a: np.concatenate([[0], np.cumsum(a)])
    return z(w), z(w*t), z(w*y), z(w*t*t), z(w*t*y), z(w*y*y)

def _wline(i, j, P):
    Sw, Swt, Swy, Swtt, Swty, Swyy = P
    sw = Sw[j]-Sw[i]; swt = Swt[j]-Swt[i]; swy = Swy[j]-Swy[i]
    swtt = Swtt[j]-Swtt[i]; swty = Swty[j]-Swty[i]
    if sw < 1e-9: return 0.0, 0.0
    den = sw*swtt-swt*swt
    if den <= 1e-9: return 0.0, swy/sw
    a = (sw*swty-swt*swy)/den; return a, (swy-a*swt)/sw

def _wsegment(t, y, w, pen, msize):
    n = len(t); Sw, Swt, Swy, Swtt, Swty, Swyy = _wtab(t, y, w)
    opt = np.full(n+1, np.inf); opt[0] = -pen; back = np.zeros(n+1, int)
    for j in range(msize, n+1):
        ii = np.arange(0, j-msize+1)
        sw = Sw[j]-Sw[ii]; swt = Swt[j]-Swt[ii]; swy = Swy[j]-Swy[ii]
        swtt = Swtt[j]-Swtt[ii]; swty = Swty[j]-Swty[ii]; swyy = Swyy[j]-Swyy[ii]
        den = sw*swtt-swt*swt; ok = den > 1e-9
        a = np.where(ok, (sw*swty-swt*swy)/np.where(ok, den, 1.0), 0.0)
        b = np.where(sw > 1e-9, (swy-a*swt)/np.where(sw > 1e-9, sw, 1.0), 0.0)
        sse = np.maximum(swyy-b*swy-a*swty, 0.0); cand = opt[ii]+sse+pen
        k = int(np.argmin(cand)); opt[j] = cand[k]; back[j] = ii[k]
    bnds = []; j = n
    while j > 0: i = back[j]; bnds.append((i, j)); j = i
    return bnds[::-1]

# ---------- величина у стыка (локально, наклон ограничен) ----------
def _lvl(t, y, w, tc, Smax):
    a, b, _ = wfit(t, y, w.copy()); a = float(np.clip(a, -Smax, Smax))
    W = w.sum(); b = np.sum(w*(y-a*t))/W if W > 1e-9 else float(np.mean(y-a*t))
    return a*tc+b

def local_value(o, w, tc, Wl=12.0, buf=5.0, Smax=SMAX, T=_DEFT):
    lm = (T >= tc-buf-Wl) & (T <= tc-buf); rm = (T >= tc+buf) & (T <= tc+buf+Wl)
    if lm.sum() < 4 or rm.sum() < 4: return None
    return float(_lvl(T[rm], o[rm], w[rm], tc, Smax) - _lvl(T[lm], o[lm], w[lm], tc, Smax))

# ---------- кривая варпа (сегментные прямые + резы) ----------
def warp_curve(o, w, pen=PEN, qpow=QPOW, S=SMAX, msize_s=MSIZE_S, T=_DEFT):
    """-> seglines [(t_lo,t_hi,a,b)], cuts [(tc, величина_локальная)]."""
    wq = w**qpow; msize = int(round(msize_s/STEP)); segs = _wsegment(T, o, wq, pen, msize); P = _wtab(T, o, wq)
    seglines = []
    for k, (i, j) in enumerate(segs):
        a, b = _wline(i, j, P); a = float(np.clip(a, -S, S))
        Wm = wq[i:j].sum(); b = float(np.sum(wq[i:j]*(o[i:j]-a*T[i:j]))/Wm) if Wm > 1e-9 else b
        t_lo = T[i] if k == 0 else T[segs[k-1][1]-1]; seglines.append((t_lo, T[j-1], a, b))
    cuts = []
    for k in range(len(segs)-1):
        tc = T[segs[k][1]-1]; aL, bL = seglines[k][2], seglines[k][3]; aR, bR = seglines[k+1][2], seglines[k+1][3]
        if abs((aR*tc+bR)-(aL*tc+bL)) >= MIN_FR:
            v = local_value(o, wq, tc, T=T); cuts.append((float(tc), v if v is not None else float((aR*tc+bR)-(aL*tc+bL))))
    m = []
    for tc, v in sorted(cuts):
        if m and abs(tc-m[-1][0]) < 15:
            if abs(v) > abs(m[-1][1]): m[-1] = (tc, v)
        else:
            m.append((tc, v))
    return seglines, m

# ---------- ПОСТФИЛЬТР по смещению хвоста (зацепка пользователя) ----------
def _gmask(base, w, gate):
    if gate > 0 and base.sum() >= 6:
        mm = base & (w >= np.quantile(w[base], gate))
        if mm.sum() >= 4: return mm
    return base

def tail_value(o, wq, w, tc, prev, nxt, win=25.0, buf=6.0, Smax=SMAX, gate=0.0, T=_DEFT):
    lm = _gmask((T >= max(prev+buf, tc-buf-win)) & (T <= tc-buf), w, gate)
    rm = _gmask((T >= tc+buf) & (T <= min(nxt-buf, tc+buf+win)), w, gate)
    if lm.sum() < 4 or rm.sum() < 4: return None
    return float(_lvl(T[rm], o[rm], wq[rm], tc, Smax) - _lvl(T[lm], o[lm], wq[lm], tc, Smax))

def tail_filter(o, w, cuts, qpow=QPOW, thr=MIN_FR, gate=0.0, T=_DEFT):
    """Оставить только резы с устойчивым смещением хвоста ≥ thr; величина = это смещение."""
    wq = w**qpow; bnds = [c[0] for c in cuts]; out = []
    for k, (tc, v) in enumerate(cuts):
        prev = bnds[k-1] if k > 0 else float(T[0])
        nxt = bnds[k+1] if k < len(cuts)-1 else float(T[-1])
        tv = tail_value(o, wq, w, tc, prev, nxt, T=T)
        if tv is None: tv = v
        if abs(tv) >= thr: out.append((tc, tv))
    return out

# ---------- краевой рез (близко к началу/концу, короче msize) ----------
EDGE_S = 15.0           # зона поиска краевого реза от края сегмента
EDGE_MIN_S = 2.5        # мин длина плато до И после реза (= окно half; меньше — не якорится → исключение)

def _fit_seg(o, wq, lo, hi, T=_DEFT):
    m = (T >= lo) & (T <= hi)
    a, b, _ = wfit(T[m], o[m], wq[m].copy()); a = float(np.clip(a, -SMAX, SMAX))
    W = wq[m].sum(); b = float(np.sum(wq[m]*(o[m]-a*T[m]))/W) if W > 1e-9 else b
    return (float(lo), float(hi), a, b)

def _find_edge_cut(o, wq, t_lo, t_hi, side, T=_DEFT):
    """Ступень плато→плато в краевой зоне сегмента [t_lo,t_hi]. -> (tc, jump) или None."""
    if t_hi - t_lo < EDGE_S + EDGE_MIN_S: return None        # короткий сегмент — рез уже выделен
    if side == "start": c0, c1 = t_lo+EDGE_MIN_S, t_lo+EDGE_S
    else:               c0, c1 = t_hi-EDGE_S, t_hi-EDGE_MIN_S
    cand = T[(T >= c0) & (T <= c1)]; best = None
    for tc in cand:
        lm = (T >= t_lo) & (T < tc); rm = (T >= tc) & (T <= t_hi)
        if lm.sum() < 4 or rm.sum() < 4: continue
        levL = _lvl(T[lm], o[lm], wq[lm], tc, SMAX); levR = _lvl(T[rm], o[rm], wq[rm], tc, SMAX)
        jump = levR - levL
        if abs(jump) >= MIN_FR and (best is None or abs(jump) > abs(best[1])):
            best = (float(tc), float(jump))
    return best

def _edge_refine(o, wq, seglines, cuts, T=_DEFT):
    """Выделить рез в первом/последнем сегменте (краевая зона) и разбить сегмент."""
    if len(seglines) < 1: return seglines, cuts
    new = list(seglines); extra = []
    ec = _find_edge_cut(o, wq, new[0][0], new[0][1], "start", T=T)
    if ec:
        tc, j = ec; s = new[0]
        new = [_fit_seg(o, wq, s[0], tc, T=T), _fit_seg(o, wq, tc, s[1], T=T)] + new[1:]; extra.append((tc, j))
    ec = _find_edge_cut(o, wq, new[-1][0], new[-1][1], "end", T=T)
    if ec:
        tc, j = ec; s = new[-1]
        new = new[:-1] + [_fit_seg(o, wq, s[0], tc, T=T), _fit_seg(o, wq, tc, s[1], T=T)]; extra.append((tc, j))
    return new, sorted(cuts + extra)

# ---------- публичный детект ----------
def detect(o, w, T=_DEFT):
    """-> (seglines [для сборки], cuts [(t,величина), фильтр по хвосту → 0 ложных])."""
    wq = w**QPOW
    seglines, cuts0 = warp_curve(o, w, T=T)
    cuts = tail_filter(o, w, cuts0, T=T)
    return _edge_refine(o, wq, seglines, cuts, T=T)
