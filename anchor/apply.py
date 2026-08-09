# -*- coding: utf-8 -*-
"""Прод-точка входа anchor-пайплайна на IN-MEMORY массивах (этап 2).

Сигнатура повторяет шов align.py (как `_audio_multispectral(out, ref_buf, ...)`):
правит `out` НА МЕСТЕ по таймлайну рефа, возвращает остаток в мс.

  out      — (n,2) float32 @ sr_audio: озвучка, уже разложенная по сетке РЕФА (выход зрения);
  ref_buf  — (n,2) float32 @ sr_audio: аудио рефа на той же сетке.

method ∈ {band, muq}; apply_cuts:
  True  → полный детект (дрейф + дискретные правки резов через сегментный варп);
  False → ТОЛЬКО дрейф (≤SMAX≈2%), непрерывная кривая, без дискретных правок резов.

Freeze в варпе не применяется: заморозка чинится на уровне зрения (файлы без неё).
Знак: правее=+; кадр=41.708мс."""
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np, torch
from loguru import logger
from scipy.ndimage import gaussian_filter1d
from .params import T as _DEFT, FRAME, SMAX, STEP, make_T
from . import detect
from . import coarse_dtw                       # ГРУБЫЙ детектор СОБЫТИЙ на DTW (вставки/вырезы вне окна band)
from .assemble import apply_warp
from .maps import band, muq, multispec
from ..interp_backend import warp_interp        # GPU/CPU блочный linear-interp (варпы аудио)
from ..memlog import memlog as _memlog          # отметки расхода памяти (CONFORM_MEMLOG=1)

MAP_SR = {"band": 16000, "muq": 24000}
COARSE_LAG_S = 2.5      # ширина окна грубого прохода band (видит сдвиг опенинга, ±0.7с не достаёт)
# ГЕЙТ band-подтверждения ЛОЖНОГО ВЫРЕЗА: coarse_dtw ловит вырез на ПОЛНОМ миксе (cos по flatten-дескриптору),
# где вокальная асимметрия (реф с оригинальным вокалом, дубль без) даёт ЛОЖНЫЙ drop в муз-зоне (kodik AniBar
# JoJo tv5-ч3 ep30: фантомный вырез 100с в [109,210] → off0_ev −20с → band-плато → 40 резов). band per-band-
# РОБАСТЕН (wmedian по 48 полосам) и на pre-DTW дубле видит ИСТИННУЮ синхронность там, где DTW-cos слепнет.
# Вырез отвергается ТОЛЬКО если band УВЕРЕННО (w>GATE_W·медианы) видит малый остаток (|o|<GATE_O_FR) в теле →
# дубль на месте → выреза нет. Слепой band (w≈0) НЕ режет → реальные вырезы целы. Гейт — ТОЛЬКО для ВЫРЕЗОВ
# (drop-побег): band-синхронность отрицает ВЫРЕЗ, но не ВСТАВКУ (та вне band-окна; мелкий insert гейт бы ложно
# срезал). Валид.: AniBar ложн-cut |o|=0.4/w=1.95 → отброшен; AniMaunt/Amazing вставки → не гейтятся, целы.
GATE_O_FR, GATE_W = 5.0, 0.5


def _wmedian(x, wts):
    """Взвешенная медиана (робастная статистика, без порогов)."""
    if len(x) == 0:
        return 0.0
    o = np.argsort(x); x = x[o]; wts = wts[o]; cw = np.cumsum(wts)
    if cw[-1] <= 0:
        return float(np.median(x))
    return float(x[int(np.searchsorted(cw, 0.5 * cw[-1]))])


def _smooth_w(tt, yy, ww, win):
    """Робастное сглаживание: взвеш. медиана в окне ±win узлов (давит выбросы). Копия
    vision_detect._smooth_w на локальном _wmedian — пакет anchor самодостаточен."""
    n = len(yy); out = np.empty(n); half = max(1, win // 2)
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        out[i] = _wmedian(yy[a:b], np.maximum(ww[a:b], 1e-6))
    return out


def _zero_w_in_spans(w, T, spans):
    """Занулить вес якорей band/muq в зонах ТИШИНЫ зрения (fill_spans, с): там дубль молчит —
    сравнивать с рефом нечего, якорь похожести напрасен (шум/ложь). Зрение уже разметило поток;
    слух его НЕ переоткрывает, лишь не цепляется за выброшенные зоны. Возврат: копия w."""
    if not spans:
        return w
    w = np.asarray(w, float).copy(); T = np.asarray(T, float)
    for a, b in spans:
        w[(T >= a) & (T <= b)] = 0.0
    return w


def _band_confirms_sync(o_band, w_band, T, te, tn):
    """True ⇔ band на pre-DTW дубле УВЕРЕННО видит синхронность в теле [te,tn] → DTW-событие ЛОЖНОЕ.
    band per-band-робастен (не слепнет от вокальной асимметрии, на которой DTW-cos ошибается). КОНСЕРВАТИВНО:
    слепой band (медиана w ≤ GATE_W) → False (событие НЕ режем → реальные вырезы/вставки целы).

    ⚠ Судим ТОЛЬКО по узлам, чьи band-окна (±COARSE_LAG_S) лежат СТРОГО ВНУТРИ тела (2026-08-06):
    узел у края тела меряет окном, перекрытым СИНХРОННЫМ контентом снаружи → «синхронность»
    самоподтверждается краями. Кейс (аудио-only, вырез аудио 5с, тело 5.5с): все 2-3 узла цепляли
    края → медиана |o|=3.8<5, w=1.05 → гейт убивал НАСТОЯЩЕЕ событие (реф §6.7 отчёта входной
    стойкости — та же ловушка перекрытых окон, что «кучные короткие блоки» 2026-06-11). Короткое
    тело (< 2·COARSE_LAG_S) гейт теперь честно НЕ ВИДИТ → False → событие живёт. AniBar-фантом
    (тело ~100с), ради которого гейт ставился (930766b), не задет — внутренних узлов там десятки."""
    body = (T >= te + COARSE_LAG_S) & (T <= tn - COARSE_LAG_S)
    if not body.any():
        return False
    ref = T >= 30
    med = float(np.median(o_band[ref])) if ref.any() else 0.0
    return bool(np.median(np.abs(o_band[body] - med)) < GATE_O_FR and np.median(w_band[body]) > GATE_W)


def _coarse_off0(o_c, w_c, *, med_win_s=5.0, stab_tol=4.0, spans=None, T=_DEFT):
    """ГРУБАЯ непрерывная кривая off0 (кадры на сетке T) — «тропа» для следящего band.
    ЧИСТАЯ СТАТИСТИКА поверх ГОТОВОГО широкого измерения (o_c, w_c) = band.build_arr ±COARSE_LAG_S
    (48 полос + взвеш. медиана ПО ПОЛОСАМ = толерантность к голосу). Само измерение делает
    вызывающий ОДИН раз на дубль — оно же служит судьёй гейта DTW-вырезов (_band_confirms_sync).
    Робастность ТОЛЬКО статистикой: страж края (пик на границе = нет пика) + гейт согласия
    соседей + взвеш. медиана. spans (с) — зоны тишины зрения: там вес зануляется (off0
    интерполируется через них, не цепляясь).
    -> off0."""
    glf = np.asarray(o_c, float); w0 = np.maximum(np.asarray(w_c, float), 1e-4)
    edge_k = COARSE_LAG_S * 1000.0 / FRAME
    w0 = np.where(np.abs(glf) > edge_k - 2.0, 0.0, w0)        # страж края: argmax на границе ±2.5с = нет пика
    w0 = _zero_w_in_spans(w0, T, spans)                      # тишина зрения → off0 не цепляется
    half = max(1, int(round(med_win_s / STEP)))

    def _sm(weights):
        out = np.full(len(T), np.nan)
        for i in range(len(T)):
            lo, hi = max(0, i - half), min(len(T), i + half + 1)
            if weights[lo:hi].sum() > 0:
                out[i] = _wmedian(glf[lo:hi].copy(), weights[lo:hi].copy())
        return out

    m0 = _sm(w0)
    stable = (~np.isnan(m0)) & (np.abs(glf - m0) <= stab_tol)  # гейт согласия соседей (устойчивость)
    off0 = _sm(np.where(stable, w0, 0.0))
    ok = ~np.isnan(off0)
    off0 = np.interp(np.arange(len(T)), np.where(ok)[0], off0[ok]) if ok.any() else np.zeros(len(T))
    return off0.astype(np.float32)


def _warp_by_off0(dub_ch, sr, off0, T=_DEFT):
    """Варп канала по НЕПРЕРЫВНОЙ кривой off0 (кадры на сетке T) — грубая коррекция."""
    n = len(dub_ch); t = np.arange(n) / sr
    dlt = np.interp(t, T, off0) * FRAME / 1000.0          # на сетке T (дёшево) — CPU
    return warp_interp(dub_ch, (t + dlt) * sr)            # варп полн.длины — GPU/CPU блочно


def _events_step_curve(curve, ts, cuts, inserts, T=_DEFT):
    """СТУПЕНЧАТАЯ пред-коррекция СОБЫТИЙ DTW (снять вставки/сдвиги вне окна band ±2.5): off0_ev
    (кадры на сетке T). ПЛОСКАЯ между событиями — дрейф НЕ трогаем, его доберёт band-коарс ниже.
    Уровни плато = медиана кривой DTW в сегменте между событиями; первый сегмент = опора (0).
    Знак как off0 (правее=+): off0_ev = −(уровень кривой DTW). Зовётся ТОЛЬКО для event-треков
    (где DTW нашёл вставку/вырез); где событий нет — вызывающий не строит off0_ev (base=src)."""
    cv = np.interp(np.asarray(T, float), np.asarray(ts, float), np.asarray(curve, float))
    ev = sorted([float(tc) for tc, _ in inserts] + [float(tc) for tc, _, _, _ in cuts])
    if not ev:
        return np.zeros(len(T), np.float32)
    bnds = [-1e9] + ev + [1e9]; off = np.zeros(len(T))
    for k in range(len(bnds) - 1):
        m = (T > bnds[k]) & (T <= bnds[k + 1])
        if m.any():
            off[m] = float(np.median(cv[m]))
    off = off - off[0]                          # первый сегмент = опора (0); дальше кумулятивные ступени событий
    return (-off).astype(np.float32)


def _smooth_drift_curve(o_total, w, cut_times, qpow=3.0, sigma_s=3.0, max_pct_s=1.25,
                        anchor_lookback_s=30.0, T=_DEFT):
    """КРИВАЯ ДРЕЙФА (следит за ГОРКАМИ): уверенно-взвешенное сглаживание o_total ПО КУСКАМ между
    резами (Надарая–Уотсон, вес=уверенность^qpow) + потолок СКОРОСТИ изменения сдвига max_pct_s
    (% в секунду — допустимое ускорение при построении звука; 1.25 = SMAX). Поворачивает (плавный
    излом) там, где сплошняком уверенные якоря отходят от прямой; стоит ровно на шуме. Разрывы —
    только на резах (cut_times). cut_times=[] → один кусок: ступени станут пандусами ≤max_pct_s.

    ОПОРА ПОСЛЕ РЕЗА (не левый край): в куске, который НАЧИНАЕТСЯ С РЕЗА, потолок скорости
    расходится в ОБЕ стороны от самого уверенного якоря в первых anchor_lookback_s (= начало
    плато), а не от первой точки. Иначе band слепнет на монтажном стыке сразу за резом (тусклые
    выбросы), занижает старт куска, и потолок скорости ~35с медленно вытягивает линию к плато.
    Теперь слепой старт подтягивается К плато, а не плато к старту. Первый кусок (с начала файла,
    без реза слева) — по-старому (левый край = опора)."""
    cw = np.maximum(np.asarray(w, float), 1e-6) ** qpow
    o = np.asarray(o_total, float)
    sig = max(1.0, sigma_s / STEP)
    lim = (max_pct_s / 100.0) * STEP * 1000.0 / FRAME   # макс |Δсдвиг| между узлами (кадры) = max_pct_s %/с
    nlb = max(1, int(round(anchor_lookback_s / STEP)))
    cur = o.copy()
    bnds = [-1e9] + sorted(float(t) for t in cut_times) + [1e9]
    for k in range(len(bnds) - 1):
        after_cut = k > 0                               # кусок начинается с реза?
        m = (T <= bnds[1]) if k == 0 else (T > bnds[k]) & (T <= bnds[k + 1])
        idx = np.where(m)[0]
        if len(idx) < 2:
            continue
        oo = o[idx]; ww = cw[idx]
        num = gaussian_filter1d(oo * ww, sig, mode="nearest")
        den = gaussian_filter1d(ww, sig, mode="nearest")
        sm = num / (den + 1e-12)
        if after_cut:
            a = int(np.argmax(ww[:min(len(ww), nlb)]))  # опора = увереннейший якорь начала плато
            for i in range(a + 1, len(sm)):             # потолок скорости ВПРАВО от опоры
                sm[i] = min(max(sm[i], sm[i - 1] - lim), sm[i - 1] + lim)
            for i in range(a - 1, -1, -1):              # и ВЛЕВО к резу: слепой старт → к плато
                sm[i] = min(max(sm[i], sm[i + 1] - lim), sm[i + 1] + lim)
        else:
            for i in range(1, len(sm)):                 # первый кусок: левый край = опора (как было)
                sm[i] = min(max(sm[i], sm[i - 1] - lim), sm[i - 1] + lim)
            for i in range(len(sm) - 2, -1, -1):
                sm[i] = min(max(sm[i], sm[i + 1] - lim), sm[i + 1] + lim)
        cur[idx] = sm
    return cur


def _robust_drift_curve(o_total, w, cut_times, max_pct_s=1.25, T=_DEFT):
    """РОБАСТНАЯ УКЛАДКА зелёной линии (R2, замена Надарая–Уотсона): на шумных данных БЕЗ погони
    за выбросами эмбеддера. ПО СЕГМЕНТАМ между резами (cut_times = прод-`det_cuts`):
      • ГЕЙТ уверенности: уровень считается ТОЛЬКО на якорях w ≥ 0.5·медианы(w>0); зоны w≈0 (band
        слеп: тёмные/повторные сцены) НЕ следуются, а МОСТЯТСЯ интерполяцией уверенных;
      • взвешенная МЕДИАНА (вес w³, окно 48 узлов = 24с) — робастна к выбросам (vs среднее прода);
      • ГАУСС σ3 — скруглить ступени медианы в плавные пандусы (уши терпят);
      • ПОТОЛОК скорости max_pct_s %/с (физпредел дрейфа) forward+backward.
    Разрывы — только на резах (варп СТУПЕНИТ на них, тишину ставит вызывающий). Детекцию НЕ трогает.
    Валидировано на кэш-библиотеке (1062 дубля, wob 25.6→5.8) + глаза юзера на графиках."""
    o = np.asarray(o_total, float); w = np.asarray(w, float)
    sig = max(1.0, 3.0 / STEP)
    lim = (max_pct_s / 100.0) * STEP * 1000.0 / FRAME    # макс |Δсдвиг| между узлами (кадры) = max_pct_s %/с
    gate = 0.5 * (np.median(w[w > 0]) if np.any(w > 0) else 1.0)   # порог уверенности (как прод coverage)
    cur = np.zeros(len(T)); last = 0.0
    bnds = [-1e9] + sorted(float(t) for t in cut_times) + [1e9]
    for k in range(len(bnds) - 1):                       # по сегментам между прод-резами
        m = (T <= bnds[1]) if k == 0 else (T > bnds[k]) & (T <= bnds[k + 1])
        idx = np.where(m)[0]
        if len(idx) < 1:
            continue
        ci = idx[w[idx] >= gate]                         # ТОЛЬКО уверенные якоря (НЕ w≈0)
        if len(ci) >= 2:
            lvl = _smooth_w(T[ci], o[ci], w[ci] ** 3, 48)   # робаст-уровень на УВЕРЕННЫХ
            raw = np.interp(T[idx], T[ci], lvl)          # зоны w≈0 МОСТЯТСЯ (не следуем за нулями)
        elif len(ci) == 1:
            raw = np.full(len(idx), o[ci[0]])
        else:
            raw = np.full(len(idx), last)                # нет опор — держим уровень соседа
        sm = gaussian_filter1d(raw, sig, mode="nearest")
        for i in range(1, len(sm)):                      # потолок скорости вперёд
            sm[i] = min(max(sm[i], sm[i - 1] - lim), sm[i - 1] + lim)
        for i in range(len(sm) - 2, -1, -1):             # и назад
            sm[i] = min(max(sm[i], sm[i + 1] - lim), sm[i + 1] + lim)
        cur[idx] = sm; last = float(sm[-1])
    return cur


def _warp_piecewise(dub, wcurve, cut_times, sr, T=_DEFT, dst=None):
    """Варп ПО КУСКАМ между резами: внутри куска — гладкая wcurve (горки), на резе — резкий стык.
    cut_times=[] → один кусок (непрерывно, ступени пандусом). Тишину в резах ставит вызывающий.

    dst (если задан) — куда писать результат; иначе выделяется новый массив. Передача
    выходного буфера убирает ВТОРУЮ полную копию дорожки: на фильме 1.5 ч с раскладкой
    5.1 это 6 ГБ (см. `_source_copy`).

    Память не зависит от длительности: индексы строятся на кусок (`arange(s0, s1)`, а не
    на всю дорожку — прежний `arange(n)` стоил 2 ГБ индексов), а канал подаётся в
    интерполятор ОКНОМ, покрывающим запрошенные точки, вместо копии канала целиком.
    """
    n = len(dub)
    bnds = [0.0] + sorted(float(t) for t in cut_times) + [n / sr]
    out = np.empty_like(dub) if dst is None else dst
    for k in range(len(bnds) - 1):
        c0, c1 = bnds[k], bnds[k + 1]
        s0 = int(round(c0 * sr)); s1 = n if k == len(bnds) - 2 else int(round(c1 * sr))
        if s1 <= s0:
            continue
        m = (T >= c0 - 0.6) & (T <= c1 + 0.6)
        tt = np.arange(s0, s1) / sr
        if m.sum() >= 2:
            dlt = np.interp(tt, T[m], wcurve[m]) * FRAME / 1000.0
        else:
            dlt = np.full(s1 - s0, float(np.interp(c0, T, wcurve)) * FRAME / 1000.0)
        src = (tt + dlt) * sr
        # Окно канала под запрошенные точки. Границы с запасом и клампом к [0, n], как в
        # блочном ресэмпле conform: интерполятор клампит края так же, как по целому каналу,
        # поэтому значения совпадают бит-в-бит.
        a1 = min(max(0, int(np.floor(src.min())) - 1), n - 1)
        a2 = max(min(n, int(np.ceil(src.max())) + 2), a1 + 2)
        src_w = src - a1
        for ch in range(dub.shape[1]):
            out[s0:s1, ch] = warp_interp(dub[a1:a2, ch], src_w)
    return out


PCM_FS = 32768.0          # полная шкала int16: фиксированная конверсия PCM→[-1,1] (как ffmpeg s16→f32le)


def _mono_sr(stereo, sr_in, sr_out):
    """(n,2|n,) int16-размах @ sr_in → mono float32 [-1,1] @ sr_out (даунмикс + /32768 + ресемпл).

    КОНТРАКТ: вход — PCM в int16-размахе (±32768), как `out`/`ref_buf` из conform
    (`_extract_wav`: wavfile.read s16 → float). Перевод к [-1,1] — фиксированная шкала
    /PCM_FS (та же, что ffmpeg f32le), на которой откалиброваны карты. Это обязательно:
    band/multispec используют log1p(энергия) — НЕ масштаб-инвариантно; MuQ ждёт [-1,1]."""
    import torchaudio.functional as AF
    mono = np.ascontiguousarray(stereo.mean(axis=1) if stereo.ndim == 2 else stereo, dtype=np.float32)
    t = torch.from_numpy(mono / PCM_FS)
    if sr_in != sr_out:
        t = AF.resample(t, sr_in, sr_out)
    return t.numpy().astype(np.float32)


def _eval_seglines(seglines, T=_DEFT):
    """Денойзенный сдвиг (со ступенями) в каждой точке сетки T по сегментным прямым."""
    bnds = np.array([s[0] for s in seglines[1:]]) if len(seglines) > 1 else np.array([])
    idx = np.searchsorted(bnds, T, side="right")
    a = np.array([s[2] for s in seglines]); b = np.array([s[3] for s in seglines])
    return a[idx]*T + b[idx]


def drift_curve(seglines, smax=SMAX, step=STEP, T=_DEFT):
    """База ±2%: следящая кривая за денойзенным сдвигом с ограничением скорости ≤smax.
    Ступени (резы) превращаются в пологие пандусы → аудио всё равно сходится, без резких
    правок. Центрированный пандус = среднее прямого и обратного slope-clamp пасса."""
    target = _eval_seglines(seglines, T=T)
    lim = smax*step                                   # макс |Δсдвиг| между соседними якорями
    cf = target.copy()
    for i in range(1, len(cf)):                       # прямой: ограничивает рост вправо
        cf[i] = min(max(cf[i], cf[i-1]-lim), cf[i-1]+lim)
    cb = target.copy()
    for i in range(len(cb)-2, -1, -1):                # обратный: ограничивает рост влево
        cb[i] = min(max(cb[i], cb[i+1]-lim), cb[i+1]+lim)
    return 0.5*(cf+cb)                                # центрированный пандус, всё ещё ≤smax


def _warp_by_curve(dub_ch, sr, curve, n_out, T=_DEFT):
    """Варп канала по непрерывной кривой сдвига curve(на сетке T), без дискретных правок."""
    t_out = np.arange(int(n_out))/sr
    dlt = np.interp(t_out, T, curve)*FRAME/1000.0
    src = (t_out+dlt)*sr
    return np.interp(src, np.arange(len(dub_ch)), dub_ch).astype(np.float32)


def _gcc_curve(ref_buf, dub_pre, sr_audio, *, win_s=4.0, maxlag_s=2.5,
               f_lo=50.0, f_hi=13500.0, T=_DEFT):
    """НЕЗАВИСИМЫЙ свидетель: по-оконный лаг GCC-PHAT (фаза, не энергия band) на сетке T.
    Поиск ШИРЕ рабочего окна band (±maxlag_s ≫ MAXLAG=0.7) — видно, что истинный пик может
    лежать ВНЕ окна band. dub_pre — дубль ДО варпа (как o/w). Знак как у o (rfft(dub)·conj(rfft(ref)):
    правее=+). -> (lag_fr[len(T)], conf[len(T)] — доля фазово-согласной полосы 0..~1)."""
    a = _mono_sr(ref_buf, sr_audio, 16000); b = _mono_sr(dub_pre, sr_audio, 16000)
    sr = 16000; n = min(len(a), len(b))
    W = int(win_s * sr); M = int(maxlag_s * sr)
    nfft = 1 << int(np.ceil(np.log2(W + 2 * M)))
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    bandm = (f >= f_lo) & (f <= min(f_hi, 0.45 * sr))
    coh_max = 2.0 * int(bandm.sum()) / nfft
    lag_fr = np.full(len(T), np.nan, np.float32); conf = np.zeros(len(T), np.float32)
    centers = (np.asarray(T) * sr).astype(int)
    valid = [i for i, c in enumerate(centers) if c - W // 2 >= 0 and c - W // 2 + W <= n]
    for b0 in range(0, len(valid), 64):
        chunk = valid[b0:b0 + 64]
        A = np.stack([a[centers[i] - W // 2:centers[i] - W // 2 + W] for i in chunk])
        B = np.stack([b[centers[i] - W // 2:centers[i] - W // 2 + W] for i in chunk])
        R = np.fft.rfft(B, nfft, axis=1) * np.conj(np.fft.rfft(A, nfft, axis=1))   # dub·conj(ref)=знак o
        R[:, ~bandm] = 0.0
        R[:, bandm] /= np.abs(R[:, bandm]) + 1e-12
        cc = np.fft.irfft(R, nfft, axis=1)
        CC = np.concatenate((cc[:, -M:], cc[:, :M + 1]), axis=1)                     # лаги [-M..M]
        pk = CC.max(1); pi = CC.argmax(1)
        for j, i in enumerate(chunk):
            lag_fr[i] = (pi[j] - M) / sr * 1000.0 / FRAME
            conf[i] = max(float(pk[j]) / coh_max, 0.0) if coh_max > 0 else 0.0
    return lag_fr, conf


def dump_cube(ref_buf, dub_pre, sr_audio, t0_s, t1_s, path, *, maxlag_s=2.5, T=_DEFT):
    """3D-куб band (окна×48 полос×лаги) для УЗКОЙ зоны [t0_s,t1_s] → npz (расследование).
    dub_pre — дубль ДО варпа. НЕ авто-дамп (на всю длину ~ГБ): зовётся вручную для места.
    npz: T(зона,с), cube[len(Tz),48,Ln], lags_fr(кадры), band_edges(Гц)."""
    sr = MAP_SR["band"]
    ref_m = _mono_sr(ref_buf, sr_audio, sr); dub_m = _mono_sr(dub_pre, sr_audio, sr)
    Tz = T[(T >= t0_s) & (T <= t1_s)]
    cube, lags_fr, edges = band.build_cube(ref_m, dub_m, Tz, maxlag_s=maxlag_s)
    np.savez_compressed(path, T=Tz.astype(np.float32), cube=cube,
                        lags_fr=lags_fr, band_edges=edges)
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# ГЛОБАЛЬНАЯ ПРЕД-СИНХРОНИЗАЦИЯ: constant A/V-десинк озвучки. Весь звук дубля равномерно
# съехал относительно СВОЕГО видео (студия так смуксила) — зрение слепо (смотрит картинку),
# а band ±0.7/_coarse_off0 ±2.5/coarse_dtw (ищет ступени) до него не достают → фантомные резы.
# Метод (доказан на кэше, БЕЗ декода): по-оконная корреляция 48-полос benv (тот же базис, что
# CK4 __dsp1), ШИРОКИЙ поиск лага ±ML. head = медиана лага первых K окон (ОПОРА на начало рефа,
# НЕ мода: мода на кусочном offset берёт хвост и ломает голову). flat = доля трека на лаге head.
# ГЕЙТ: |head|>MIN_S (вне band) И flat>FLAT_MIN (чистый constant, НЕ событие/краевой вырез) →
# снять head (сдвиг −head). Иначе no-op → src не трогается → ВЕСЬ путь ниже БИТ-В-БИТ.
# ─────────────────────────────────────────────────────────────────────────────
_PRE_DEV = "cuda" if torch.cuda.is_available() else "cpu"     # GPU-first, CPU-fallback
_PRE_SR, _PRE_HOP, _PRE_NFFT = 16000, 256, 2048
_PRE_FPS = _PRE_SR / _PRE_HOP                                 # 62.5 env-fps (базис CK4 __dsp1)
_PRE_HW_S, _PRE_ML_S, _PRE_STEP_S = 20.0, 70.0, 15.0          # окно-шаблон / диапазон лага / шаг центров
_PRE_HEAD_K = 6                                               # голова = первые K окон
_PRE_MIN_S, _PRE_FLAT_MIN, _PRE_TOL_S = 2.5, 0.9, 1.5         # гейт |head|>2.5с (вне band) И flat>0.9


def _pre_bands():
    from .maps.band import NB48 as _NB
    fb = torch.linspace(0, _PRE_SR / 2, _PRE_NFFT // 2 + 1); hi = min(_NB["fmax"], _PRE_SR / 2 - 1)
    edg = torch.logspace(np.log10(_NB["fmin"]), np.log10(hi), 49)
    BM = torch.zeros(48, _PRE_NFFT // 2 + 1)
    for b in range(48):
        BM[b, (fb >= edg[b]) & (fb < edg[b + 1])] = 1.0
    return BM.to(_PRE_DEV)


def _pre_benv(x16):
    """mono @16к float32 → [48,Nf] z-norm огибающая (band._benv, базис CK4 __dsp1)."""
    from .maps.band import _benv as _bnv
    BM = _pre_bands(); win = torch.hann_window(_PRE_NFFT).to(_PRE_DEV)
    xt = torch.from_numpy(np.ascontiguousarray(x16, np.float32)).to(_PRE_DEV)
    return _bnv(xt.unsqueeze(0), BM, _PRE_NFFT, _PRE_HOP, win)[0]


@torch.no_grad()
def _pre_window_lags(er, ed):
    """benv рефа/дубля [48,Nf] → (lags_s[K], conf[K]) по окнам: узкий шаблон дубля ±HW ищется в
    широком реф ±(HW+ML), per-band argmax лага → взвеш.медиана (вес=prominence)."""
    Nf = min(er.shape[1], ed.shape[1]); er = er[:, :Nf]; ed = ed[:, :Nf]
    hw = int(_PRE_HW_S * _PRE_FPS); ml = int(_PRE_ML_S * _PRE_FPS); step = int(_PRE_STEP_S * _PRE_FPS)
    pad = hw + ml; centers = np.arange(pad, Nf - pad, step)
    if len(centers) < _PRE_HEAD_K + 1:
        return np.zeros(0), np.zeros(0)
    nf = 1 << int(np.ceil(np.log2(4 * hw + 4 * ml)))
    lags = np.full(len(centers), np.nan); conf = np.zeros(len(centers))
    for b0 in range(0, len(centers), 24):
        ci = centers[b0:b0 + 24]
        D = torch.stack([ed[:, c - hw:c + hw] for c in ci])
        R = torch.stack([er[:, c - hw - ml:c + hw + ml] for c in ci])
        FD = torch.fft.rfft(D, nf, dim=2); FR = torch.fft.rfft(R, nf, dim=2)
        cc = torch.fft.irfft(FR * torch.conj(FD), nf, dim=2)[:, :, :2 * ml + 1]
        band_lag = torch.argmax(cc, 2).float() - ml
        prom = (cc.amax(2) - cc.median(2).values).clamp(min=0)
        order = torch.argsort(band_lag, 1)
        bl = torch.gather(band_lag, 1, order); pw = torch.gather(prom, 1, order)
        cw = torch.cumsum(pw, 1); mi = (cw < cw[:, -1:] * 0.5).sum(1).clamp(0, 47)
        med = bl.gather(1, mi[:, None]).squeeze(1)
        lags[b0:b0 + len(ci)] = (med / _PRE_FPS).cpu().numpy()
        conf[b0:b0 + len(ci)] = prom.median(1).values.cpu().numpy()
    return lags, conf


def _global_prealign(ref16, dub16, ref_dsp=None):
    """Глобальный constant A/V-десинк озвучки → (head_s, flat). ref_dsp (путь CK4 __dsp1) —
    переиспользование benv рефа (без пересчёта); None → считает из ref16."""
    er = None
    if ref_dsp:
        from pathlib import Path as _P
        if _P(ref_dsp).exists():
            try:
                er = torch.from_numpy(np.load(str(ref_dsp))).to(_PRE_DEV)
            except Exception:  # noqa: BLE001 — битый кэш → пересчёт из ref16
                er = None
    if er is None:
        er = _pre_benv(ref16)
    ed = _pre_benv(dub16)
    lags, conf = _pre_window_lags(er, ed)
    m = np.isfinite(lags)
    if m.sum() < _PRE_HEAD_K + 1:
        return 0.0, 0.0
    lg = lags[m]; w = np.maximum(conf[m], 1e-9)
    K = min(_PRE_HEAD_K, len(lg))
    head = _wmedian(lg[:K], w[:K])                               # опора на начало рефа
    flat = float(np.sum(w[np.abs(lg - head) <= _PRE_TOL_S]) / np.sum(w))
    return float(head), flat


def _shift_channel(x, sr, head_s):
    """Снять глобальный head: контент дубля отодвигается на head_s (тишина в начало, хвост за
    краем теряется). Знак −head доказан на кэше (band 38→1 рез, |o|→0). Тишину начала потом
    заливает _fill_silence_from_ref (реф звучит → оригинал)."""
    n = len(x); idx = np.arange(n, dtype=np.float64) - head_s * sr
    return np.interp(idx, np.arange(n), x, left=0.0, right=0.0).astype(np.float32)


_COPY_BLK = 1 << 22          # 4М кадров за раз: копия идёт кусками, пик не зависит от длительности


def _source_copy(out):
    """Снимок звука ДО варпа. Рядом с `out`, а НЕ в оперативной памяти.

    Прежний код делал `np.stack([out[:, c].copy() …])`: полная копия всей дорожки в
    памяти, причём вдвое — сначала список поканальных копий, потом `stack`. На фильме
    1.5 ч с раскладкой 5.1 это 6 ГБ и пик 12 ГБ (замер 2026-08-07: собственная память
    процесса скакала до 38 ГБ). Закон проекта требует, чтобы расход не рос с
    длительностью, поэтому снимок кладётся на диск рядом с `out` (он и сам файл),
    и копируется кусками.

    Если `out` — обычный массив в памяти (путь без выноса на диск), поведение прежнее.
    Значения идентичны прежним: тот же порядок, тот же тип, копия побайтная.
    """
    fn = getattr(out, "filename", None)
    if fn is None:                            # не файл — старый путь (короткие дорожки, тесты)
        return np.stack([out[:, c].copy() for c in range(out.shape[1])], axis=1)
    d = Path(tempfile.mkdtemp(prefix="asrc_", dir=str(Path(fn).parent)))
    dst = np.memmap(d / "src.f32", dtype=np.float32, mode="w+", shape=out.shape)
    for s in range(0, out.shape[0], _COPY_BLK):
        dst[s:s + _COPY_BLK] = out[s:s + _COPY_BLK]
    dst._conform_tmpdir = str(d)              # каталог удалит вызывающий (audio_anchor)
    return dst


def _map_channels(src_arr, fn):
    """Применить поканальное преобразование `fn(канал) -> канал` ко всей дорожке.

    Замена связки `np.stack([fn(a[:, c]) for c …], axis=1)`, которая держала в памяти и
    список поканальных результатов, и итоговый массив — то есть две полных копии дорожки.
    Здесь результат пишется поканально: рядом с источником, если тот на диске.
    """
    fn_path = getattr(src_arr, "filename", None)
    if fn_path is None:
        return np.stack([fn(src_arr[:, c]) for c in range(src_arr.shape[1])], axis=1)
    d = Path(tempfile.mkdtemp(prefix="amap_", dir=str(Path(fn_path).parent)))
    dst = np.memmap(d / "map.f32", dtype=np.float32, mode="w+", shape=src_arr.shape)
    for c in range(src_arr.shape[1]):
        dst[:, c] = fn(src_arr[:, c])
    dst._conform_tmpdir = str(d)
    return dst


def _drop_tmp(*arrs):
    """Убрать временные файлы, созданные `_source_copy`/`_map_channels`."""
    for a in arrs:
        d = getattr(a, "_conform_tmpdir", None)
        if d:
            shutil.rmtree(d, ignore_errors=True)


def audio_anchor(out, ref_buf, fps_ref=None, *, method="band", apply_cuts=True,
                 sr_audio=44100, drift_speed_pct=1.25, info=None, progress=None,
                 plot_dir=None, plot_stem=None, render_own=True, vision_spans=None, dsp_cache=None):
    """Карта→детект→варп `out` на месте. -> остаток |медиана| в мс.

    info (если задан) наполняется метриками укладки (audio_cuts/max_step/coverage/span/drift…)
    и слоями для единого графика (info["band_layers"]: o/w/off0/wcurve/det_cuts/gcc).
    plot_dir+plot_stem (если заданы) → сырьё (npz+json). render_own=True → ещё и СВОЙ PNG/HTML
    графика band; render_own=False → conform рисует ЕДИНЫЙ график (зрение+аудио), band свой не рисует."""
    if progress is not None:
        progress(0.0, "широкое измерение сдвига")
    n_out = out.shape[0]
    T = make_T(n_out / sr_audio)              # сетка от РЕАЛЬНОЙ длины пары (любая длительность)
    _memlog('вход аудио-слоя')
    src = _source_copy(out)                   # дубль ДО варпа (источник), НЕ в оперативной памяти
    _memlog('снимок дубля')
    # ═══ ШАГ 1 — СОБЫТИЯ (всё, что ВНЕ окна band ±2.5с): prealign → DTW-события → гейт → ступень ═══
    ref16 = _mono_sr(ref_buf, sr_audio, 16000)       # реф@16к ОДИН раз на весь band-путь (дедуп: DTW
    dub16d = _mono_sr(src, sr_audio, 16000)          # + широкий/тонкий проходы + resid); дубль — свой
    _memlog('моно 16к рефа и дубля')
    # Глобальная пред-синхронизация constant A/V-десинка озвучки (весь звук равномерно съехал
    # относит. своего видео; вне окна ±2.5с — band/dtw не достают). ГЕЙТ строгий → на здоровых no-op
    # (src не трогается) → весь путь ниже БИТ-В-БИТ. ref_dsp=CK4 → benv рефа из кэша (без пересчёта).
    g_head, g_flat = _global_prealign(ref16, dub16d, ref_dsp=dsp_cache)
    if abs(g_head) > _PRE_MIN_S and g_flat > _PRE_FLAT_MIN:
        prev = src
        src = _map_channels(src, lambda ch: _shift_channel(ch, sr_audio, g_head))
        _drop_tmp(prev)
        dub16d = _mono_sr(src, sr_audio, 16000)      # выровненный дубль → coarse_dtw/широкий проход на нём
        if info is not None:
            info["audio_global_offset_ms"] = round(g_head * 1000.0, 1)
    # ГРУБЫЙ детектор СОБЫТИЙ на DTW (вставки/вырезы вне окна band ±2.5, post-vision). Sparse +
    # валидирован (1 реал/0 ложных на 340). НЕТ события (почти вся выборка) → base=src → ВСЯ доводка НИЖЕ
    # идёт БИТ-В-БИТ со старым прод. Событие → пред-коррекция дубля (снять вставку), дальше та же доводка.
    _memlog('перед детектором событий')
    dres = coarse_dtw.detect(ref16, dub16d, vspans=vision_spans, ref_cache=dsp_cache)
    _memlog('после детектора событий')
    dtw_ins = dres["events_inserts"]; dtw_cuts = dres["events_cuts"]
    oc = wc = None                                   # широкое измерение band ±2.5 (единожды на дубль)
    if dtw_cuts:                                     # ГЕЙТ band-подтверждения: отсеять ЛОЖНЫЕ ВЫРЕЗЫ (drop-побег)
        oc, wc = band.build_arr(ref16, dub16d, T, maxlag=COARSE_LAG_S)   # band на pre-DTW дубле (сетка T)
        dtw_cuts = [c for c in dtw_cuts if not _band_confirms_sync(oc, wc, T, c[2], c[3])]
    if dtw_ins or dtw_cuts:
        off0_ev = _events_step_curve(dres["curve"], dres["ts"], dtw_cuts, dtw_ins, T=T)
        base = _map_channels(src, lambda ch: _warp_by_off0(ch, sr_audio, off0_ev, T=T))
        dub16b = _mono_sr(base, sr_audio, 16000)     # события сдвинули дубль → mono16 и измерение заново
        oc = wc = None
    else:
        off0_ev = None; base = src; dub16b = dub16d      # БИТ-В-БИТ старый путь (нет событий)
    # ═══ ШАГ 2 — ЛИНИЯ (всё В ПРЕДЕЛАХ ±2.5с): широкий проход → тонкий на mono-варпе → детект → R2 ═══
    # Широкое измерение band ±2.5с (толерантно к голосу) — ЕДИНСТВЕННОЕ на дубль: гейт выше
    # переиспользует его же (дубль без событий → вход тот же → бит-в-бит).
    if oc is None:
        oc, wc = band.build_arr(ref16, dub16b, T, maxlag=COARSE_LAG_S)
    # off0 = «тропа» для следящего тонкого прохода: чистая статистика поверх (oc, wc).
    # vision_spans (тишина зрения) → веса якорей там зануляются: слух не цепляется за выброшенные зоны.
    _memlog('после широкого измерения')
    off0 = _coarse_off0(oc, wc, spans=vision_spans, T=T)
    if progress is not None:
        progress(0.25, "измерение сдвига по частотным полосам" if method == "band"
                       else "измерение сдвига моделью MuQ")
    # Тонкий проход (±0.7с) СЛЕДИТ за off0 — ПЕРЕИЗМЕРЯЕТ остаток на пред-варпленном дубле. Детектор
    # резов со-адаптирован с этим переизмерением (o И w) — статистикой поверх широкого прохода оно НЕ
    # заменяется (доказано на case/, см. ROADMAP recreate-sluh). Варпится ТОЛЬКО mono@MAP_SR: измеритель
    # другого не слушает, а варп полного стерео 44.1к (буфер pre) — лишний. Эквивалентность A/B-доказана
    # на 180 дублях case/ (резы бит-в-бит 160/163, 3 пограничных флипа у порога MIN_FR).
    sr_map = MAP_SR[method]
    ref_map = ref16 if sr_map == 16000 else _mono_sr(ref_buf, sr_audio, sr_map)
    dub_map = dub16b if sr_map == 16000 else _mono_sr(base, sr_audio, sr_map)
    dub_map = _warp_by_off0(dub_map, sr_map, off0, T=T)
    _memlog('перед тонким проходом')
    want_diag = plot_dir is not None and bool(plot_stem)
    _bm = (band.build_arr if method == "band" else muq.build_arr)(ref_map, dub_map, T, diag=want_diag)
    if want_diag:
        o_res, w, diag = _bm
    else:
        o_res, w, diag = _bm[0], _bm[1], None
    _memlog('после тонкого прохода')
    w = _zero_w_in_spans(w, T, vision_spans)             # тишина зрения → тонкие якоря не строятся
    o = off0 + o_res                                      # полный сдвиг = грубый off0 + тонкий остаток
    seglines, cuts = detect.detect(o, w, T=T)            # cuts = большие СТУПЕНИ (сдвиг опенинга)
    det_cuts = list(cuts)
    if progress is not None:
        progress(0.65, "построение кривой сдвига")
    # Кривая дрейфа (ГОРКИ) + варп. apply_cuts=True → разрыв+тишина в резах;
    # =False → ТОТ ЖЕ алгоритм, но резы НЕ передаём → ступень станет пандусом ≤SMAX (без тишины).
    cut_times = [float(tc) for tc, _ in det_cuts] if apply_cuts else []
    wcurve = _robust_drift_curve(o, w, cut_times, max_pct_s=drift_speed_pct, T=T)   # R2 робаст-укладка (замена Надарая-Уотсона)
    # Пишем СРАЗУ в `out` (источник — отдельный буфер `base`), поэтому промежуточного
    # массива на всю дорожку больше нет: он стоил ещё одну полную копию звука.
    warped = _warp_piecewise(base, wcurve, cut_times, sr_audio, T=T, dst=out)
    if apply_cuts:                                        # ТИШИНА в резах: continuous-варп иначе переигрывает звук
        for tc, v in det_cuts:
            jms = v * FRAME
            if jms < 0:                                  # нехватка дубля → тишина |Δ| по центру
                h = abs(jms) / 1000.0 / 2.0
                a, b = int(max(0.0, tc - h) * sr_audio), int((tc + h) * sr_audio)
            else:                                        # лишнее вырезано → узкий шов
                a, b = int(max(0.0, tc - 0.15) * sr_audio), int((tc + 0.15) * sr_audio)
            warped[a:min(n_out, b)] = 0.0
        for _tc, _dv, te, tn in dtw_cuts:                # вырез DTW (нехватка дубля вне ±2.5) → тишина зоны [te,tn]
            a2, b2 = int(float(te) * sr_audio), int(float(tn) * sr_audio)
            if b2 > a2:
                warped[max(0, a2):min(n_out, b2)] = 0.0
    _memlog('после варпа')
    _drop_tmp(src, base)                       # снимок и пред-коррекция больше не нужны
    cuts = det_cuts if apply_cuts else []
    # ЛИНИЯ графика = band-доводка дрейфа (СЛЕДИТ за якорями o). Ступени событий DTW (вставки/вырезы)
    # сняты пред-коррекцией аудио и показываются ОТДЕЛЬНЫМИ МАРКЕРАМИ (dtw_inserts/dtw_cut_zones), НЕ
    # запекаются в линию — иначе линия уходит от якорей на величину события и распирает шкалу.
    wcurve_disp = wcurve
    # остаток независимым мультиспектром (16к) — реф@16к переиспользуем (дедуп), corr@16к свой
    _memlog('перед замером остатка')
    corr16 = _mono_sr(out, sr_audio, 16000)
    resid = multispec.drift(torch.from_numpy(ref16).to(multispec.DEV),
                            torch.from_numpy(corr16).to(multispec.DEV), T)
    m = (T >= 30); resid_fr = float(np.median(np.abs(resid[m])))   # без верхнего хардкода — вся длина

    # --- метрики укладки (из найденных резов и ломаной) ---
    sm = _eval_seglines(seglines, T=T)                   # денойзенная кривая сдвига на сетке T
    span_ms = float((sm.max() - sm.min()) * FRAME) if len(sm) else 0.0
    max_step_ms = float(max((abs(v) for _, v in det_cuts), default=0.0) * FRAME)
    sum_ms = float(sum(abs(v) for _, v in det_cuts) * FRAME)
    drift_ms = float((sm[-1] - sm[0]) * FRAME - sum(v for _, v in det_cuts) * FRAME) if len(sm) else 0.0
    wmed = float(np.median(w[w > 0])) if np.any(w > 0) else 1.0
    coverage = float(np.mean((w / (wmed if wmed > 1e-9 else 1.0)) >= 0.5))

    # --- ГРАФИКИ: трек-PNG (миниатюра) + интерактивный HTML (основная методика) ---
    # GCC-свидетель считаем ОДИН раз — для графиков и для сырья.
    # GCC-PHAT свидетель ОТКЛЮЧЁН (2026-06-26): строил лишь серую россыпь точек на аудио-панели
    # графика ценой ~34с/дубль чистого numpy-FFT на CPU (GCC-PHAT по окнам ±2.5с). На укладку звука
    # НЕ влияет (read-only), а линия признана слепым шумом и не-судьёй (ось Y он и так не задавал).
    # gl=gc=None → npz-поля gcc_lag_fr/gcc_conf пустые, band_layers["gcc"]=None, ветки графика gcc
    # пропускаются. Звук бит-в-бит (приёмка: PCM финального FLAC md5 не изменился). Функция
    # _gcc_curve оставлена в модуле для ручного расследования (dump_cube / сырьё band).
    gl = gc = None
    plots = []
    if render_own and plot_dir is not None and plot_stem:
        try:
            from pathlib import Path as _P
            from . import plots as _plt
            pd = _P(plot_dir); pd.mkdir(parents=True, exist_ok=True)
            ttl = (f"{plot_stem} — {method} · резов {len(det_cuts)} · "
                   f"остаток {resid_fr * FRAME:+.0f}мс · дрейф {drift_ms:+.0f}мс · "
                   f"скорость ≤{drift_speed_pct:.2f}%/с")
            tp = pd / f"{plot_stem}__track.png"
            _plt.render_track(tp, o, w, off0, wcurve_disp, det_cuts, title=ttl, T=T)   # полная кривая варпа (band+события DTW); T=реальная сетка (нестандартные длительности)
            plots.append({"kind": "track", "name": tp.name, "t": None, "v_ms": None})
            try:
                from . import plots_html as _ph
                hp = pd / f"{plot_stem}__track.html"
                _ph.render_track_html(hp, o, w, off0, gl, wcurve_disp, det_cuts, title=ttl, T=T)
                plots.append({"kind": "html", "name": hp.name, "t": None, "v_ms": None})
            except Exception:  # noqa: BLE001 — HTML опционален (plotly), PNG уже есть
                pass
        except Exception:  # noqa: BLE001 — графики не должны ронять conform
            plots = []

    # --- СЫРЫЕ ДАННЫЕ рядом с графиком (npz+json) для расследования/фильтрации/изоляции ---
    # Слои ДО схлопывания: грубая off0, кривая варпа, поверхность время×лаг, по-полосный расклад
    # (band), независимый свидетель GCC-PHAT (±2.5с). На звук НЕ влияет (read-only).
    if want_diag:
        try:
            from pathlib import Path as _P2
            pd2 = _P2(plot_dir); pd2.mkdir(parents=True, exist_ok=True)
            npz = {"T": np.asarray(T, np.float32), "o": o.astype(np.float32),
                   "w": w.astype(np.float32), "off0": off0.astype(np.float32),
                   "wcurve": np.asarray(wcurve_disp, np.float32),
                   "dtw_inserts": (np.array(dtw_ins, np.float64) if dtw_ins else np.zeros((0, 2))),
                   "dtw_cut_zones": (np.array([(te, tn) for _t, _d, te, tn in dtw_cuts], np.float64)
                                     if dtw_cuts else np.zeros((0, 2))),
                   "o_coarse": oc.astype(np.float32), "w_coarse": wc.astype(np.float32),
                   "seglines": np.array(seglines, np.float64) if seglines else np.zeros((0, 4)),
                   "det_cuts": (np.array([(t, v) for t, v in det_cuts], np.float64)
                                if det_cuts else np.zeros((0, 2))),
                   "gcc_lag_fr": gl if gl is not None else np.zeros(0, np.float32),
                   "gcc_conf": gc if gc is not None else np.zeros(0, np.float32)}
            if diag:
                npz.update(diag)                               # surf/lags_fr/band_shift/band_prom/band_edges
            np.savez_compressed(pd2 / f"{plot_stem}__raw.npz", **npz)
            meta = {
                "method": method, "stem": plot_stem, "apply_cuts": bool(apply_cuts),
                "frame_ms": FRAME, "grid_step_s": STEP, "t0_s": float(T[0]),
                "n_grid": int(len(T)),
                "maxlag_band_s": float(band.MAXLAG if method == "band" else muq.MAXLAG),
                "gcc_maxlag_s": 2.5,
                "params": {"QPOW": detect.QPOW, "PEN": detect.PEN, "SMAX": SMAX,
                           "MIN_FR": detect.MIN_FR, "MSIZE_S": detect.MSIZE_S},
                "metrics": {"audio_cuts": len(det_cuts), "max_step_ms": max_step_ms,
                            "sum_ms": sum_ms, "drift_ms": drift_ms, "span_ms": span_ms,
                            "coverage": coverage, "n_segments": len(seglines),
                            "resid_med_frames": resid_fr},
                "seglines": [[float(x) for x in s] for s in seglines],
                "det_cuts": [[round(float(t), 2), round(float(v), 3),
                              round(float(v * FRAME), 1)] for t, v in det_cuts],
                "arrays": {
                    "npz": f"{plot_stem}__raw.npz",
                    "T": "сетка времени, с",
                    "o": "сдвиг (argmax), кадры; правее=+",
                    "w": "уверенность band/muq, норм. по медиане",
                    "surf": "поверхность время×лаг (до argmax)",
                    "lags_fr": "ось лагов поверхности, кадры",
                    "band_shift": "[T,48] по-полосный сдвиг, кадры (только band)",
                    "band_prom": "[T,48] выраженность пика полосы (только band)",
                    "band_edges": "границы 48 полос, Гц (только band)",
                    "gcc_lag_fr": "лаг GCC-PHAT по окнам, кадры (свидетель, ±2.5с)",
                    "gcc_conf": "уверенность GCC (доля фазово-согласной полосы)",
                    "seglines": "[K,4] (t_lo,t_hi,a,b) ломаные варпа",
                    "det_cuts": "[M,2] (t_с, величина_кадры) найденные резы",
                },
            }
            (pd2 / f"{plot_stem}__raw.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            if info is not None:
                info["raw"] = f"{plot_stem}__raw.npz"
        except Exception:  # noqa: BLE001 — сырьё не должно ронять conform
            pass

    if info is not None:
        # слои для ЕДИНОГО графика (conform строит зрение+аудио вместе) — на сетке T, кадры
        info["band_layers"] = {
            "T": np.asarray(T, np.float32), "o": o.astype(np.float32), "w": w.astype(np.float32),
            "off0": off0.astype(np.float32), "wcurve": np.asarray(wcurve_disp, np.float32),
            "det_cuts": [(float(t), float(v)) for t, v in det_cuts],
            # D6: ВСТАВКА озвучки (зелёная вертикаль+длит.) / НЕДОСТАЧА дубля (оранжевая зона) — маркеры графика
            "dtw_inserts": [(float(tc), float(ln)) for tc, ln in dtw_ins],
            "dtw_cut_zones": [(float(te), float(tn)) for _tc, _dv, te, tn in dtw_cuts],
            "gcc": (gl if gl is not None else None)}
        info["anchor_method"] = method; info["apply_cuts"] = apply_cuts
        info["cuts"] = [(round(t, 1), round(v, 1)) for t, v in cuts]
        info["resid_med_frames"] = resid_fr
        info["audio_cuts"] = len(det_cuts)
        info["audio_max_step_ms"] = max_step_ms
        info["audio_sum_ms"] = sum_ms
        info["audio_drift_ms"] = drift_ms
        info["audio_span_ms"] = span_ms
        info["audio_coverage"] = coverage
        info["n_segments"] = len(seglines)
        info["plots"] = plots
    if progress is not None:
        progress(1.0, "готово")
    return resid_fr * FRAME


def detect_av_desync(out, ref_buf, sr_audio=44100, *, win_s=20.0, step_s=30.0,
                     maxlag_s=2.5, t0_s=60.0, big_frames=7.0, mad_max_frames=6.0):
    """Кросс-модальный детектор СТУДИЙНОГО A/V-десинка (read-only, на wav НЕ влияет).

    Идея: после видео-укладки `out` уже на сетке РЕФА. Меряем остаточный сдвиг аудио M&E к
    реф-аудио ШИРОКИМ окном (PHAT, ±maxlag_s — шире рабочего окна band 0.7с). У здоровой дорожки
    он ≈0; если он БОЛЬШОЙ И УСТОЙЧИВЫЙ (мал разброс по окнам) — аудио источника смещено
    относительно его же видео (студия сдвинула фон/M&E в миксе). Это дефект исходника, надёжно
    НЕ исправляется (разделение стемов цепляет исходную речь) → помечаем КРАСНЫМ.

    Паттерн валидирован на ep01 kamennyj-okean: AniLibria медиана −40к / MAD 2.2к (⚠), все 10
    здоровых дорожек ~0к / MAD ~0. PHAT устойчив к периодике (где band-NCC даёт ложные пики).

    -> dict(lag_frames, lag_ms, mad_frames, mad_ms, n_windows, danger) | None (мало окон)."""
    import numpy as np
    SR = 16000
    ref16 = _mono_sr(ref_buf, sr_audio, SR); out16 = _mono_sr(out, sr_audio, SR)
    n = min(len(ref16), len(out16)); dur = n / SR
    centers = np.arange(t0_s, dur - win_s, step_s)
    if len(centers) < 5:
        return None
    M = int(maxlag_s * SR); w = int(win_s * SR); nf = 1 << int(np.ceil(np.log2(2 * w)))
    lags = []
    for c in centers:
        s = int(c * SR - w // 2)
        a = ref16[s:s + w]; b = out16[s:s + w]
        if len(a) < w or len(b) < w:
            continue
        A = np.fft.rfft(a, nf); B = np.fft.rfft(b, nf)
        R = A * np.conj(B); R /= np.abs(R) + 1e-9
        cc = np.fft.irfft(R, nf)
        cc = np.concatenate([cc[-M:], cc[:M + 1]])      # лаги [-M..M]
        k = int(np.argmax(cc))
        lags.append((k - M) / SR * 1000.0 / FRAME)      # сдвиг в кадрах рефа
    if len(lags) < 5:
        return None
    lags = np.array(lags, float)
    med = float(np.median(lags)); mad = float(np.median(np.abs(lags - med)))
    danger = (abs(med) > big_frames) and (mad < mad_max_frames)
    return {"lag_frames": med, "lag_ms": med * FRAME, "mad_frames": mad,
            "mad_ms": mad * FRAME, "n_windows": len(lags), "danger": danger}
