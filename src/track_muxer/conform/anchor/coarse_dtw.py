# -*- coding: utf-8 -*-
"""ГРУБЫЙ аудио-проход на DTW (заменяет _coarse_off0). Находит аудио-события (вставки/вырезы),
которых зрение не видит, на УЖЕ уложенном зрением потоке (post-vision). Порт валидированного стенда
(research/dropdtw_audio): 1 реал/0 ложных на 340, license-free (DSP-48 band), без muq/контента, память-bounded.

Конвейер (detect_one на масштабе HW): benv → оконный DSP-дескриптор (±HW, P точек) → whiten по реф-стате
→ banded_dtw (полоса вокруг диагонали + colmin + СИГНАЛ-АДАПТИВНЫЙ ДИАГ-ПРИОР: тянет путь к лагу 0
ТОЛЬКО где вокруг нет матча) → (o,w) на сетке → vision_detect.global_trend/build_curve → roundtrip_filter
(выпрямить round-trip крупнее досягаемости тонкого band) → classify. cross_scale: событие реально ⟺
HW6 ∩ HW18 (короткое окно-матчер + длинное окно-арбитр, |Δtc|≤20с).

Параметры обоснованы (REFERENCE_DTW_methods_math.md, HANDOFF_postvision_singularity.md §✅):
  AMERCE=0.04 (штраф варп-шага), DIAG_PRIOR=2.0/SIG_*(сигнал-гейт, окно широкое diag∈[2,8]×sig_hi∈[0.40,0.50]),
  MINFR=80к (выше band ±0.7), flat_fr=band ±0.7 (round-trip выпрямлять). Знак: правее=+; кадр=41.708мс."""
import numpy as np, torch
from .maps import band as _B
from .params import FRAME, STEP
from ..kernel.dropdtw import drop_dtw_affine_guard_amerce, backtrack_affine
from .. import vision_detect as _VD
from .. import cache as _cache
from ..vision_detect import global_trend as _global_trend, build_curve as _build_curve, _wmedian

DEV = _B.DEV
# --- константы детектора (стендовые DTW-константы; ОТЛИЧНЫ от band_align 0.30/0.30) ---
HW_SHORT, HW_LONG, P = 6.0, 18.0, 32          # окно дескриптора (матчер/арбитр), точек на окно
AMERCE, DIAG_PRIOR = 0.04, 2.0                # штраф варп-шага; сила диаг-приора (окно [2,8])
SIG_LO, SIG_HI, SIG_WIN = 0.20, 0.40, 40      # сигнал-гейт приора (окно sig_hi [0.40,0.50]); сглаж. ±10с
MINFR = 80.0                                  # порог величины события, кадры (> band ±0.7)
OPEN, EXT, DSYN, MATCH_THR = 0.20, 0.02, 0.20, 0.15   # аффинное ядро (стенд)
CHUNK, OVERLAP, MARG = 800, 200, 160          # banded_dtw: кусок/перекрытие/полоса (память const)
COARSE_FR = 2.5 * 1000.0 / FRAME              # ±2.5с в кадрах (ret_tol round-trip)
BAND_FINE_FR = 0.7 * 1000.0 / FRAME           # досягаемость тонкого band ±0.7 (flat_fr round-trip)
CROSS_TOL_S = 20.0                            # кросс-масштаб: |Δtc| ≤ этого = подтверждено
FPS = _B.NB48["sr"] / _B.NB48["hop"]          # env-fps 62.5 (для classify длины)
_nfft, _hop, _sr = _B.NB48["nfft"], _B.NB48["hop"], _B.NB48["sr"]
_win = torch.hann_window(_nfft).to(DEV); _BM = _B._bands(48, 50.0, 14000.0, _sr, _nfft)
_fps_env = _sr / _hop


def _benv(x):
    """mono16 float32 → [48, Nf] band-огибающая (как _om_core, на всю дорожку)."""
    xt = torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(DEV)
    return _B._benv(xt.unsqueeze(0), _BM, _nfft, _hop, _win)[0]


@torch.no_grad()
def _desc(E, hw):
    """[48,Nf] → (дескриптор [n, 48*P], ts[c]). Окно ±hw ресэмплится в P точек, flatten. GRID=STEP."""
    Nf = E.shape[1]; dur = Nf / _fps_env; hwf = hw * _fps_env
    ts = np.arange(0, max(0.0, dur - 2 * hw), STEP) + hw
    if len(ts) == 0:
        return np.zeros((0, 48 * P), np.float32), ts
    centers = torch.from_numpy(ts * _fps_env).to(DEV).float(); pp = torch.linspace(0, 1, P, device=DEV)
    wlo = (centers - hwf).floor(); whi = (centers + hwf).floor(); Wd = (whi - wlo).clamp(min=1)
    pos = (wlo[:, None] + pp[None, :] * (Wd[:, None] - 1)).clamp(0, Nf - 1)
    lo = pos.floor().long(); hi = (lo + 1).clamp(max=Nf - 1); fr = (pos - lo.float())
    Elo = E[:, lo.reshape(-1)].reshape(48, len(ts), P); Ehi = E[:, hi.reshape(-1)].reshape(48, len(ts), P)
    seg = Elo * (1 - fr)[None] + Ehi * fr[None]
    return seg.permute(1, 0, 2).reshape(len(ts), 48 * P).contiguous().cpu().numpy().astype(np.float32), ts


def _whiten(Rr, Dr):
    """Whiten дескрипторов по СТАТИСТИКЕ РЕФА + L2-норм (аудио cos≈0.5, нужна нормировка)."""
    mu = Rr.mean(0, keepdims=True); sd = Rr.std(0, keepdims=True) + 1e-6
    wh = lambda o: ((o - mu) / sd) / (np.linalg.norm((o - mu) / sd, axis=1, keepdims=True) + 1e-8)
    return wh(Rr.astype(np.float64)), wh(Dr.astype(np.float64))


def _banded_dtw(R, Dd, off, *, amerce=AMERCE, diag_prior=DIAG_PRIOR):
    """MEMORY-BOUNDED Drop-DTW (закон проекта: длительность не ограничена): полоса ±MARG вокруг ТРЕНДА
    off, кусками CHUNK с OVERLAP. colmin по куску (аудио cos≈0.5). СИГНАЛ-АДАПТИВНЫЙ ДИАГ-ПРИОР после
    colmin: штраф diag_prior·|откл.лага от тренда|·wsig, wsig=вес «нет матча ВОКРУГ» (сглаж. 1−colmin,
    гейт SIG_LO/SIG_HI) — давит блуждание в слепых зонах, реальное событие свободно. -> pred[Nd] (ref-idx/-1)."""
    N = len(Dd); Rn = len(R); pred_full = np.full(N, -2, np.int64); quality = np.full(N, -1, np.int64)
    for a in range(0, N, CHUNK - OVERLAP):
        b = min(a + CHUNK - 1, N - 1); ks = np.arange(a, b + 1); refk = ks + off[ks]
        r1 = max(0, int(refk.min()) - MARG); r2 = min(Rn - 1, int(refk.max()) + MARG)
        if r2 - r1 < 50:
            if b == N - 1: break
            continue
        C = (1.0 - R[r1:r2 + 1] @ Dd[a:b + 1].T).astype(np.float64)
        colmin = C.min(0); C -= colmin[None, :]                        # относит. лучшего матча кадра
        if diag_prior > 0:                                            # сигнал-адаптивный приор (локальный)
            best = 1.0 - colmin
            sm = np.convolve(best, np.ones(SIG_WIN) / SIG_WIN, "same") if len(best) > SIG_WIN else best
            wsig = np.clip((SIG_HI - sm) / (SIG_HI - SIG_LO + 1e-9), 0.0, 1.0)   # 1=нет матча вокруг→приор ON
            dev = np.arange(r1, r2 + 1)[:, None] - refk[None, :]
            C = C + diag_prior * np.abs(dev) * wsig[None, :]
        fs = (a == 0)
        M, Dr, BM, BDr = drop_dtw_affine_guard_amerce(C, OPEN, EXT, DSYN, MATCH_THR, amerce, fs)
        pred, _, _ = backtrack_affine(M, Dr, BM, BDr, r1)
        for idx, kk in enumerate(range(a, b + 1)):
            q = min(kk - a, b - kk)
            if q > quality[kk]: quality[kk] = q; pred_full[kk] = pred[idx]
        if b == N - 1: break
    pred_full[pred_full == -2] = -1
    return pred_full


def _aow(tr, sh, cs, T):
    """Якоря (ref-время tr, сдвиг sh, cos cs) → (o,w) на сетке T (окно ±VIS_WIN, взвеш.медиана, agree)."""
    order = np.argsort(tr); trs, shs, css = tr[order], sh[order], np.clip(cs[order], 0, None)
    o = np.full(len(T), np.nan); w = np.zeros(len(T))
    for i, t in enumerate(T):
        lo = int(np.searchsorted(trs, t - _VD.VIS_WIN)); hi = int(np.searchsorted(trs, t + _VD.VIS_WIN))
        if hi - lo < 1: continue
        ww = css[lo:hi]; ss = shs[lo:hi]; om = _wmedian(ss, ww)
        o[i] = om; w[i] = float(np.median(ww)) * float(np.mean(np.abs(ss - om) <= 2.0))
    g = ~np.isnan(o); o = np.interp(np.arange(len(T)), np.where(g)[0], o[g]) if g.any() else np.zeros(len(T))
    pos = w > 0
    if pos.any(): w = w / np.median(w[pos])
    return o, w


def _levels(cuts, curve, T, win=8.0):
    lv = []
    for tc, dv, te, tn in cuts:
        bef = curve[(T >= tc - win) & (T < tc)]; aft = curve[(T > tc) & (T <= tc + win)]
        lv.append((float(np.median(bef)) if len(bef) else np.nan, float(np.median(aft)) if len(aft) else np.nan))
    return lv


def _roundtrip_filter(cuts, curve, T, *, ret_tol_fr=COARSE_FR, flat_fr=BAND_FINE_FR, win=8.0):
    """Снять резы round-trip (лаг ушёл с базы и ВЕРНУЛСЯ) + выпрямить кривую. flat_fr=band ±0.7:
    round-trip крупнее досягаемости тонкого band → кривую в БАЗУ (band его не вытянет; иначе ложный
    ED-матч ~2с остаётся → регресс на титрах). Реальная правка = постоянная ступень (не возвращается) → выживает."""
    curve = np.asarray(curve, float).copy(); n = len(cuts)
    if n < 2: return list(cuts), curve
    lv = _levels(cuts, curve, T, win); keep = [True] * n
    for k in range(n):
        if not keep[k]: continue
        base = lv[k][0]
        if not np.isfinite(base): continue
        for j in range(k + 1, n):
            la = lv[j][1]
            if np.isfinite(la) and abs(la - base) <= ret_tol_fr:
                for i in range(k, j + 1): keep[i] = False
                sp = (T > cuts[k][0]) & (T < cuts[j][0])
                mag = float(np.max(np.abs(curve[sp] - base))) if sp.any() else 0.0
                if mag > flat_fr: curve[(T >= cuts[k][0]) & (T <= cuts[j][0])] = base
                break
    return [c for i, c in enumerate(cuts) if keep[i]], curve


def _classify(cuts, tR, pred, drs, *, minfr=MINFR):
    """резы build_curve → события: вырез (drop≥.15 и |dv|≥minfr) / вставка (hwarp≥2 и длина≥minfr)."""
    cutsL = []; insL = []
    for tc, dv, te, tn in cuts:
        s = int(np.searchsorted(tR, te)); e = max(int(np.searchsorted(tR, tn)), s + 1)
        drop = sum(1 for r in drs if s <= r < e) / (e - s)
        n_dub = int(((pred >= s) & (pred < e)).sum()); hw = n_dub / (e - s)
        if drop >= 0.15 and abs(dv) >= minfr:
            cutsL.append((float(tc), float(dv), float(te), float(tn)))
        elif hw >= 2.0:
            ln = max(STEP, (n_dub - (e - s)) * STEP)
            if ln >= minfr * FRAME / 1000.0: insL.append((float(tc), float(ln)))
    return cutsL, insL


def _detect_one(Rr, tR, Dr, tD, vspans):
    """Один масштаб: whiten → banded_dtw(diag-приор) → (o,w) → global_trend/build_curve → roundtrip → classify.
    post-vision: тренд off=0 (поток уже уложен зрением). -> (events_cuts, events_inserts, curve[на tR], o, w)."""
    R, Dd = _whiten(Rr, Dr)
    off = np.clip(np.searchsorted(tR, tD), 0, len(tR) - 1) - np.arange(len(tD))   # ≈0 (диагональ)
    pred = _banded_dtw(R, Dd, off.astype(np.int64))
    m = pred >= 0; di = np.where(m)[0]; ri = pred[m]; trf = tR[np.clip(ri, 0, len(tR) - 1)]
    drs = (set(range(int(ri.min()), int(ri.max()) + 1)) - set(int(x) for x in ri)) if m.any() else set()
    sh = (trf - tD[di]) * 1000.0 / FRAME
    cos = np.array([float(Dd[di[i]] @ R[np.clip(ri[i], 0, len(R) - 1)]) for i in range(len(di))])
    T = tR.copy(); o, w = _aow(trf, sh, cos, T)
    if vspans:
        for sa, sb in vspans: w[(T >= sa) & (T <= sb)] = 0.0       # тишина зрения → не строим якоря
    a, b = _global_trend(o, w, T, FPS)
    curve, cuts_all, _fill, _ores, _conf, _body = _build_curve(o, w, T, a, b)
    cuts_kept, curve = _roundtrip_filter(cuts_all, curve, T)
    cutsL, insL = _classify(cuts_kept, tR, pred, drs)
    return cutsL, insL, curve, o, w


def _ref_benv(ref_mono16, ref_cache):
    """benv РЕФА с диск-кэшем CK4 (дорогой STFT переиспользуется между дублями эпизода/запусками).
    f32 .npy, бит-в-бит (GPU→CPU→save→load→GPU точно). Падение кэша не роняет — пересчёт."""
    if ref_cache is not None:
        from pathlib import Path as _P
        p = _P(ref_cache)
        if p.exists():
            try:
                e = torch.from_numpy(np.load(str(p))).to(DEV)
                _cache._a("CK4 benv реф", True, p.name)
                return e
            except Exception:  # noqa: BLE001 — битый кэш → пересчёт
                pass
        _cache._a("CK4 benv реф", False, p.name)
        E = _benv(ref_mono16)
        try:
            p.parent.mkdir(parents=True, exist_ok=True); np.save(str(p), E.cpu().numpy().astype(np.float32))
        except Exception:  # noqa: BLE001
            pass
        return E
    return _benv(ref_mono16)


def detect(ref_mono16, dub_mono16, *, vspans=None, ref_cache=None):
    """ПОЛНЫЙ детектор: кросс-масштаб (HW6 матчер ∩ HW18 арбитр). Вход — mono @16к (post-vision дубль + реф).
    ref_cache (путь .npy) — CK4: кэш benv рефа (реф переиспользуется между дублями). -> dict: curve
    (кадры, на сетке ts короткого окна), ts, o, w, events_cuts, events_inserts (подтв. кросс-масштабом)."""
    Er = _ref_benv(ref_mono16, ref_cache); Ed = _benv(dub_mono16)
    Rr6, tR6 = _desc(Er, HW_SHORT); Dr6, tD6 = _desc(Ed, HW_SHORT)
    cutsL, insL, curve, o, w = _detect_one(Rr6, tR6, Dr6, tD6, vspans)
    cand = [(t, dv, te, tn) for t, dv, te, tn in cutsL] + [(t, None, None, None) for t, ln in insL]
    conf_c, conf_i = cutsL, insL
    if cand:                                                       # кросс-масштаб ЛЕНИВО: только при кандидатах
        Rr18, tR18 = _desc(Er, HW_LONG); Dr18, tD18 = _desc(Ed, HW_LONG)
        lc, li, _cv, _o, _w = _detect_one(Rr18, tR18, Dr18, tD18, vspans)
        long_t = [t for t, dv, te, tn in lc] + [t for t, ln in li]
        ok = lambda tc: any(abs(t - tc) <= CROSS_TOL_S for t in long_t)
        conf_c = [(t, dv, te, tn) for t, dv, te, tn in cutsL if ok(t)]
        conf_i = [(t, ln) for t, ln in insL if ok(t)]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return dict(curve=curve, ts=tR6, o=o, w=w, events_cuts=conf_c, events_inserts=conf_i)
