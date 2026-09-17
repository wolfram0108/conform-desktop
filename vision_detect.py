# -*- coding: utf-8 -*-
"""Анализатор зрения: ЧИСТЫЙ построитель видео-карты (укладка на якорях + резы). Заменил
прежний «гербарий» detect+piecewise (~15 порогов: MSIZE/PEN/VC_*/EDGE_*), который СХЛОПЫВАЛ
короткие уверенные плато — линия уходила «в космос» мимо якорей на 40-90 кадров (узким местом
был video_cut_filter: буфер ±6с отбрасывал рез у короткого плато).

ЕДИНСТВЕННЫЙ построитель видео-карты в conform (вытеснил и «скат», и гербарий). Зрение видит
структуру сдвига (плато/ступени) точнее аудио; задача — лечь на УВЕРЕННЫЕ якоря и поставить
резы там, где сдвиг РЕАЛЬНО изменился.

Конвейер: видео-якоря (pred/cos) → (o,w) на сетке T (вес=cos·agree) → СНЯТИЕ реального масштаба
(`global_trend`, PAL/ускорение без потолка) → `clean_cuts` (РЕШАЕТ ДЕЛЬТА: руны по скачку>дрейфа,
острова-транзиенты в мост, фильтр краёв) → `build_curve` (робастная прямая между резами) →
tg_s + МОНОТОНИЗАЦИЯ (дубль нельзя играть назад). Резы/голова/хвост применяются ТИШИНОЙ
(разнести контент / нет дубля) вызывающим; финальная заливка рефом перекроет синхронно.

Принцип чистого алгоритма (зафиксирован пользователем 2026-06-22):
  - УВЕРЕННЫЕ якоря (w≥W_CONF) = истина, карта обязана лежать на них;
  - РЕШАЕТ ДЕЛЬТА (уровень сдвига), не длина: рез — только при ПОСТОЯННОМ изменении дельты;
    дельта ВЕРНУЛАСЬ (экскурсия вниз-вверх) = слепой выброс → мост, НЕ рез (сколько бы якорей);
  - край без опоры (голова/хвост) → экстраполяция плато + ТИШИНА (заливка рефом).
5 честных параметров (W_CONF/SMAX/TOL/ISLAND_MAX/EDGE_MAX) вместо ~15. Знак: правее=+;
кадр=41.708мс.
"""
from __future__ import annotations

import numpy as np

from track_muxer.conform.anchor.params import T as GT, SMAX, make_T  # noqa: F401

# Сетка/вес якорей (центры плато нечувствительности, sweep на 36 дорожках 2026-06-18)
VIS_WIN = 0.3             # окно агрегации якорей, с (плато 0.2-0.5)
VIS_SMAX = 0.45          # потолок наклона прямых = предел СКОРОСТИ ДРЕЙФА, к/с (физика проигрывания)
VIS_MAX_SCALE_PCT = 15.0  # санити-гард доверия глобальному наклону, %/с (реальные PAL/NTSC ≤±5%;
                          # >15% = якоря мусорные → масштаб не снимаем)
VIS_SCALE_BIN_S = 25.0    # бин оценки масштаба, с (плато 15-40): давит шум якорей перед медианой

# Clean builder: 5 honest parameters
VIS_W_CONF = 0.5         # порог уверенности якоря (надёжность измерения; w≈0 шум ↔ w≈1 уверен)
VIS_TOL = 6.0           # джиттер якоря, к (внутри плато якоря дрожат ~1.5к)
VIS_EDGE_MAX = 12       # короткий первый/последний рун без опоры с краю (якорей) = краевой скачок → фильтр

# Адаптивная прокладка кривой МЕЖДУ резами: прод-прямая (Тейл-Сен) если ложится, иначе RDP-ломаная.
VIS_RDP_EPS = 3.0       # RDP: макс отклонение ломаной от сглаженных якорей, к (минимум вершин)
VIS_SMOOTH_WIN = 8      # окно робастного сглаживания якорей перед RDP, узлов (давит выбросы)
VIS_ADAPT_THR = 8.0     # адаптив: 90%-невязка прод-прямой ≤ порога → прямая (ступени бит-в-бит), иначе RDP

# Vision noise on ambiguous content (dark or repeated scenes, NTSC sources): the layout uses
# TV denoising (1D total variation / fused lasso). A short excursion, even a tall one, does not pay
# for a base jump and is absorbed; a persistent step pays and the base moves. The optimisation is
# global over the track, so neighbouring excursions cannot contaminate each other locally.
# Validated on 690 cached dubs: 0 regressions against the greedy scan, 20 fewer false cuts;
# the lambda sweep shows a wide plateau [1200,3000].
VIS_TV_LAM = 1800.0     # порог окупаемости скачка TV, к·сэмпл (плато 1200-3000)
VIS_TV_STEP_MIN = 8.0   # |Δ уровня| рез, к (>8 убирает мелкую TV-лестницу на гладком гулянии)
VIS_OUT_THR = 12.0      # |o_res − B(TV)| = выброс → исключить из подгонки кривой, к


def _wmedian(x, wts):
    """Взвешенная медиана (робастная статистика, без порогов)."""
    if len(x) == 0:
        return 0.0
    o = np.argsort(x); x = x[o]; wts = wts[o]; cw = np.cumsum(wts)
    if cw[-1] <= 0:
        return float(np.median(x))
    return float(x[int(np.searchsorted(cw, 0.5 * cw[-1]))])


# ── Baked 3:2 telecine: NTSC rips carry 23.976 film as 29.97 without inverse telecine, so every
#    ~5th frame is a temporal hybrid, SRM gets noisy and anchors jitter. Detection is the peak of
#    the autocorrelation of neighbour-frame dissimilarity at PERIOD 5, computed over the ready
#    128x72 SRM and therefore resolution-invariant. The reaction is a soft IVTC: the redundant
#    frame of each five is dropped, giving ~23.976 and clean matching. Non-NTSC and non-telecine
#    input is a bit-exact no-op. Proven on 132 synthetic clips and both seasons of an NTSC-sourced series.
VIS_TC_NTSC = (29.0, 30.5)      # fps NTSC → кандидат на 3:2-телесин (PAL/film период-5 не дают)
VIS_TC_FLOOR = 0.05             # мин выступ периода-5 (защита от случайного argmax=5 на шуме)
_TC_SAMPLE = 6000               # кадров фикс-сэмпла детекта (память O(const), не растёт с длиной)


def _tele_signature(srm):
    """(выступ периода-5 автокорр несхожести соседей над фоном, argmax 2..12) на фикс-сэмпле."""
    n = len(srm)
    off = min(_TC_SAMPLE, n // 4)
    a = np.asarray(srm[off:off + _TC_SAMPLE], np.float32)
    if len(a) < 600:
        a = np.asarray(srm[:_TC_SAMPLE], np.float32)
    if len(a) < 600:
        return 0.0, 0
    c = np.sum(a[:-1] * a[1:], axis=1)
    x = (1.0 - c).astype(np.float64); x = x - x.mean()
    if x.std() < 1e-9:
        return 0.0, 0
    ac = np.correlate(x, x, "full")[len(x) - 1:]; ac = ac / (ac[0] + 1e-12)
    g = lambda p: float(ac[p]) if p < len(ac) else 0.0       # noqa: E731
    return float(max(g(5), g(10)) - float(np.mean([g(3), g(4), g(6), g(7)]))), int(np.argmax(ac[2:13]) + 2)


def is_baked_telecine(srm, fps):
    """Дёшево (фикс-сэмпл, БЕЗ прореживания): (telecine, tele_score, argmax). Гейт = NTSC fps И
    argmax период-5 И выступ ≥ порога. Используется для гейта CK3/паспорта ДО матчинга."""
    n = 0 if srm is None else len(srm)
    if n < 1000 or not (VIS_TC_NTSC[0] <= float(fps) <= VIS_TC_NTSC[1]):
        return False, 0.0, 0
    ts, am = _tele_signature(srm)
    return (am == 5 and ts >= VIS_TC_FLOOR), round(ts, 4), am


def _cadence_keep(srm, blk=4096):
    """Маска кадров на оставление: в каждом окне из 5 дроп самого избыточного (макс cos к
    предыдущему = pulldown-повтор). Сходство соседей ПОБЛОЧНО (память O(блока))."""
    n = len(srm)
    d = np.empty(n, np.float32); d[0] = -1.0
    for s in range(0, n, blk):
        lo = max(0, s - 1)
        a = np.asarray(srm[lo:s + blk], np.float32)
        if len(a) < 2:
            continue
        dd = np.sum(a[1:] * a[:-1], axis=1)
        d[lo + 1:lo + 1 + len(dd)] = dd
    keep = np.ones(n, bool)
    for w in range(0, n, 5):
        idx = [i for i in range(w, min(w + 5, n)) if i >= 1]
        if len(idx) >= 5:
            keep[max(idx, key=lambda i: d[i])] = False
    return keep


def detelecine(srm, fps, *, tmp_dir=None):
    """Запечённый 3:2-телесин (NTSC) → прорежает каденс на уровне SRM-фич (мягкий IVTC → ~23.976)
    для ЧИСТОГО матчинга. Иначе НО-ОП (тот же srm/fps бит-в-бит). tmp_dir задан → прореженный SRM
    в temp-memmap (память O(блока), low_mem); иначе RAM. -> (srm, fps, info)."""
    is_tc, ts, am = is_baked_telecine(srm, fps)
    info = {"telecine": is_tc, "tele_score": ts, "argmax": am, "dropped": 0}
    if not is_tc:
        return srm, fps, info
    n = len(srm); keep = _cadence_keep(srm); n2 = int(keep.sum()); D = int(srm.shape[1])
    if tmp_dir is not None:
        import tempfile
        from pathlib import Path as _P
        mp = _P(tempfile.mkdtemp(prefix="detc_", dir=str(tmp_dir))) / "f.f16"
        out = np.memmap(mp, dtype=np.float16, mode="w+", shape=(n2, D)); j = 0
        for s in range(0, n, 8192):
            sel = np.asarray(srm[s:s + 8192])[keep[s:s + 8192]]
            out[j:j + len(sel)] = sel; j += len(sel)
        out.flush(); del out
        srm2 = np.memmap(mp, dtype=np.float16, mode="r", shape=(n2, D))
    else:
        srm2 = np.asarray(srm)[keep]
    info["dropped"] = int(n - n2)
    return srm2, float(fps) * n2 / n, info


def vision_ow(pred, asg, cos, fps_ref, fps_dub, *, win_s=VIS_WIN, tol_fr=2.0, T=GT,
              ax_ref=None, ax_dub=None):
    """Видео-якоря → (o,w) на сетке params.T. o = взвеш.медиана сдвига (вес=cos) в окне
    ±win_s; w = медиана(cos)·доля согласных (|сдвиг−o|≤tol_fr — аналог band-«agree»).
    Пустые узлы: o интерполируется, w=0. Нормировка w по медиане положительных.

    ax_ref/ax_dub — РЕАЛЬНАЯ ось времени кадров (SrmFeatures.pts, только VFR): время кадра
    тогда = ax[индекс], а не индекс/fps (на VFR ложь до сотен секунд — замер s01e06 encoded:
    уход 713с). None → старый путь бит-в-бит. ЕДИНСТВЕННОЕ место, где индексы кадров
    превращаются во время для укладки; fps_ref дальше — лишь ЕДИНИЦА «кадры рефа»
    (сокращается в build_map: ×fps_ref здесь, /fps_ref в tg_s)."""
    t_ref = (ax_ref[pred[asg]] if ax_ref is not None
             else pred[asg].astype(np.float64) / fps_ref)
    t_dub = (ax_dub[asg] if ax_dub is not None
             else asg.astype(np.float64) / fps_dub)
    sh = (t_dub - t_ref) * fps_ref                       # сдвиг в кадрах рефа (правее=+)
    cs = np.asarray(cos, np.float64)
    order = np.argsort(t_ref)
    trs, shs, css = t_ref[order], sh[order], cs[order]
    o = np.full(len(T), np.nan); w = np.zeros(len(T))
    for i, t in enumerate(T):
        lo = int(np.searchsorted(trs, t - win_s)); hi = int(np.searchsorted(trs, t + win_s))
        if hi - lo < 1:
            continue
        ww = np.maximum(css[lo:hi], 0.0); ss = shs[lo:hi]
        om = _wmedian(ss, ww)
        agree = float(np.mean(np.abs(ss - om) <= tol_fr))
        o[i] = om; w[i] = float(np.median(ww)) * agree
    g = ~np.isnan(o)
    if g.any():
        o = np.interp(np.arange(len(T)), np.where(g)[0], o[g])
    else:
        o[:] = 0.0
    pos = w > 0
    if pos.any():
        w = w / np.median(w[pos])
    return o.astype(np.float64), w.astype(np.float64)


def global_trend(o, w, T, fps_ref, *, bin_s=VIS_SCALE_BIN_S, max_pct=VIS_MAX_SCALE_PCT):
    """ГЛОБАЛЬНЫЙ наклон укладки = РЕАЛЬНЫЙ масштаб дубля (PAL/NTSC/любое ускорение), ИЗ ДАННЫХ,
    БЕЗ потолка и БЕЗ всякого fps. Робастен И к резам, И к шуму якорей:
      1) грубые БИНЫ ~bin_s, в каждом взвеш. МЕДИАНА o → ДАВИТ ШУМ (у шумных дублей типа kodik
         per-step дрейф ~0.3 кадра тонет в джиттере соседних узлов сетки);
      2) МЕДИАНА наклонов СОСЕДНИХ бинов → рез = спайк в одной паре бинов, медиана его отсекает.
    Почему так: чистая медиана локальных наклонов хрупка к шуму (Amediateka давала 0%); МНК/`wfit`
    и широкие пары — резы УТЯГИВАЮТ (Reanimedia ep04 −9.4%). Бины+соседи = робастно к обоим.
    Санити-гард: |масштаб| > max_pct %/с = якоря мусорные → (0,0). Возврат: (a,b) прямой a·T+b."""
    o = np.asarray(o, float); w = np.asarray(w, float); T = np.asarray(T, float)
    if len(T) < 3:
        return 0.0, 0.0
    dur = float(T[-1] - T[0])
    nb = max(2, int(dur / bin_s)) if dur > 0 else 0
    edges = np.linspace(T[0], T[-1], nb + 1) if nb >= 2 else None
    tc: list[float] = []; oc: list[float] = []; wc: list[float] = []
    if edges is not None:
        for i in range(nb):
            m = (T >= edges[i]) & (T < edges[i + 1]) & (w > 0)
            if int(m.sum()) >= 3:
                tc.append(0.5 * (edges[i] + edges[i + 1]))
                oc.append(_wmedian(o[m], w[m]))          # уровень бина (взвеш. медиана — шум подавлен)
                wc.append(float(w[m].sum()))
    if len(tc) >= 3:                                      # наклон соседних бинов → медиана (рез отсеян)
        tca = np.asarray(tc); oca = np.asarray(oc); wca = np.asarray(wc)
        a = _wmedian(np.diff(oca) / np.diff(tca), np.minimum(wca[:-1], wca[1:]))
    else:                                                # мало бинов → откат на локальные наклоны узлов
        sl = np.diff(o) / np.maximum(np.diff(T), 1e-9); we = np.minimum(w[:-1], w[1:]); g = we > 0
        a = _wmedian(sl[g], we[g]) if g.any() else 0.0
    if (not np.isfinite(a)) or abs(a) / max(float(fps_ref), 1e-6) * 100.0 > max_pct:
        return 0.0, 0.0
    b = _wmedian(o - a * T, np.maximum(w, 1e-6))          # опорный уровень (центр лестницы)
    return float(a), float(b)


def tv1d(y, lam):
    """1D total variation denoising (Condat 2013, прямой O(n)). min 0.5·Σ(y−x)² + lam·Σ|Δx|.
    Кусочно-постоянная РОБАСТНАЯ база: короткий выброс (даже высокий) не окупает 2 скачка по lam
    → поглощается; устойчивая ступень окупает → скачок остаётся. Глобально по всей дорожке."""
    y = np.asarray(y, float); N = len(y)
    x = np.empty(N)
    if N == 0:
        return x
    if N == 1:
        x[0] = y[0]; return x
    k = k0 = kminus = kplus = 0
    vmin = y[0] - lam; vmax = y[0] + lam
    umin = lam; umax = -lam
    while True:
        if k == N - 1:
            if umin < 0.0:
                x[k0:kminus + 1] = vmin
                k = k0 = kminus = kminus + 1
                if k >= N:
                    break
                kplus = k; vmin = y[k]; vmax = y[k] + 2 * lam; umin = lam; umax = -lam
            elif umax > 0.0:
                x[k0:kplus + 1] = vmax
                k = k0 = kplus = kplus + 1
                if k >= N:
                    break
                kminus = k; vmin = y[k] - 2 * lam; vmax = y[k]; umin = lam; umax = -lam
            else:
                x[k0:N] = vmin + umin / (k - k0 + 1)
                break
            continue
        umin += y[k + 1] - vmin
        umax += y[k + 1] - vmax
        if umin < -lam:                                          # отрицательный скачок
            x[k0:kminus + 1] = vmin
            k = k0 = kminus = kminus + 1
            kplus = k; vmin = y[k]; vmax = y[k] + 2 * lam; umin = lam; umax = -lam
        elif umax > lam:                                         # положительный скачок
            x[k0:kplus + 1] = vmax
            k = k0 = kplus = kplus + 1
            kminus = k; vmin = y[k] - 2 * lam; vmax = y[k]; umin = lam; umax = -lam
        else:
            k += 1
            if umin >= lam:
                vmin += (umin - lam) / (k - k0 + 1); umin = lam; kminus = k
            if umax <= -lam:
                vmax += (umax + lam) / (k - k0 + 1); umax = -lam; kplus = k
    return x


def clean_cuts(o, w, T, a_g, b_g, *, w_conf=VIS_W_CONF, smax=VIS_SMAX, tol=VIS_TOL,
               lam=VIS_TV_LAM, step_min=VIS_TV_STEP_MIN, exc_thr=VIS_OUT_THR):
    """Границы резов видео-карты через TV-DENOISING (глобально, без жадной локальной контаминации
    соседними экскурсиями). o_res → TV-база B (кусочно-постоянная): короткая экскурсия не окупает
    скачок → поглощается; устойчивая ступень → B шагает. РЕЗЫ = границы сегментов B, но УРОВНИ/Δ
    берём из РЕАЛЬНЫХ якорей (медиана сегмента без выбросов — TV занижает Δ усадкой λ/m) + фильтр
    «резкий локальный скачок» (отсекает мелкую TV-лестницу на гладком гулянии). Выбросы =
    |o_res − уровень_сегмента| > exc_thr (от ИСТИННОГО уровня сегмента, не усаженной TV-базы B —
    иначе короткий сегмент перед большой ступенью теряется целиком) → exc_mask: ИСКЛЮЧИТЬ из подгонки.
    Возврат: (cuts[(tc,Δ,t_end,t_nxt)], o_res, conf, body, exc_mask)."""
    o = np.asarray(o, float); w = np.asarray(w, float); T = np.asarray(T, float)
    o_res = o - (a_g * T + b_g)                                   # детренд масштабом: рябь+ступени без наклона
    conf = w >= w_conf
    ti = T[conf]; ri = o_res[conf]
    n = len(ti)
    if n < 2:
        return [], o_res, conf, (float(T[0]), float(T[-1])), np.zeros(len(T), bool)
    B = tv1d(ri, lam)                                            # TV — ТОЛЬКО для СЕГМЕНТАЦИИ (границы резов)
    conf_idx = np.where(conf)[0]
    body = (float(ti[0]), float(ti[-1]))
    bj = list(np.where(np.abs(np.diff(B)) > 0.5)[0])            # скачок TV между якорями i и i+1
    seg_b = [0] + [i + 1 for i in bj] + [n]
    segs = [(seg_b[k], seg_b[k + 1]) for k in range(len(seg_b) - 1)]
    # ВЫБРОС считаем от ИСТИННОГО уровня сегмента (медиана реальных якорей), а НЕ от усаженной TV-базы B.
    # TV занижает ступень на ~λ/m → КОРОТКИЙ сегмент перед БОЛЬШОЙ ступенью (синхрон-голова перед
    # рекламной вставкой) иначе целиком уходит в exc → build_curve теряет его → укладка на ГЛОБАЛЬНОМ
    # уровне (первые секунды дубля сдвинуты на величину вставки). Уровни-из-якорей — заявленная
    # философия этого детектора; доводим до неё и выброс. (kolpakov ep08: голова 0-58с возвращается.)
    Blvl = np.empty(n)
    for a, b in segs:
        Blvl[a:b] = np.median(ri[a:b])
    exc = np.abs(ri - Blvl) > exc_thr                          # выбросы: далеко от уровня СВОЕГО сегмента

    def _lvl(a, b):                                             # уровень сегмента из РЕАЛЬНЫХ якорей (без выбросов)
        sl = ri[a:b][~exc[a:b]]
        return float(np.median(sl)) if len(sl) else float(np.median(ri[a:b]))
    seg_lvl = [_lvl(a, b) for a, b in segs]
    cuts = []
    for k in range(len(segs) - 1):
        i = segs[k][1] - 1                                      # последний якорь сегмента k
        dv = seg_lvl[k + 1] - seg_lvl[k]                        # ИСТИННЫЙ Δ уровня (без TV-усадки)
        sharp = any(abs(ri[j + 1] - ri[j]) > smax * (ti[j + 1] - ti[j]) + tol   # резкий локальный скачок
                    for j in range(max(0, i - 2), min(n - 1, i + 3)))           # (не гладкая лестница гуляния)
        if abs(dv) > step_min and sharp:
            cuts.append((0.5 * (ti[i] + ti[i + 1]), dv, float(ti[i]), float(ti[i + 1])))
    exc_mask = np.zeros(len(T), bool)
    exc_mask[conf_idx[exc]] = True
    return cuts, o_res, conf, body, exc_mask


def _theilsen(tt, yy, smax):
    """Прод-прямая: разреженный Тейл-Сен, наклон clamp ±smax."""
    n = len(tt)
    if n < 2:
        return 0.0, float(yy[0]) if n else 0.0
    step = max(1, n // 20)
    sl = [(yy[j] - yy[i]) / (tt[j] - tt[i])
          for i in range(0, n, step) for j in range(i + 1, n, step) if tt[j] > tt[i] + 1]
    s = float(np.clip(np.median(sl or [0.0]), -smax, smax))
    return s, float(np.median(yy - s * tt))


def _smooth_w(tt, yy, ww, win):
    """Робастное сглаживание: взвеш. медиана в окне ±win узлов (давит выбросы перед RDP)."""
    n = len(yy); out = np.empty(n); half = max(1, win // 2)
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        out[i] = _wmedian(yy[a:b], np.maximum(ww[a:b], 1e-6))
    return out


def _rdp(t, y, eps):
    """Ramer-Douglas-Peucker: ломаная с гарантией макс отклонения ≤ eps, минимум вершин."""
    keep = np.zeros(len(t), bool); keep[0] = keep[-1] = True
    stack = [(0, len(t) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        x0, y0, x1, y1 = t[a], y[a], t[b], y[b]
        sl = 0.0 if x1 == x0 else (y1 - y0) / (x1 - x0)
        d = np.abs(y[a:b + 1] - (y0 + sl * (t[a:b + 1] - x0)))
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[a + k] = True; stack.append((a, a + k)); stack.append((a + k, b))
    idx = np.where(keep)[0]
    return t[idx], y[idx]


def build_curve(o, w, T, a_g, b_g, *, smax=VIS_SMAX):
    """Чистая карта: на каждый отрезок между резами — АДАПТИВНО: прод-прямая (Тейл-Сен ≤smax),
    если ложится на якоря (90%-невязка ≤ VIS_ADAPT_THR) → ступенчатые/линейные дубли бит-в-бит;
    не ложится (ГУЛЯЮЩИЙ горками, анти-бан) → RDP-ломаная по сглаженным якорям. Резы +
    exc_mask = clean_cuts (TV-сегментация, финальные). Подгонка — по базовым якорям (не выбросам).
    Вне body (голова/хвост) — экстраполяция плато.
    Возврат: (curve[кадры на T], cuts, fill[(t_end,t_nxt) пролёты тишины], o_res, conf, body)."""
    o = np.asarray(o, float); w = np.asarray(w, float); T = np.asarray(T, float)
    cuts, o_res, conf, body, exc_mask = clean_cuts(o, w, T, a_g, b_g, smax=smax)  # TV: резы финальные
    base = a_g * T + b_g
    inbody = (T >= body[0]) & (T <= body[1])                      # вне тела (голова/хвост) — не тянем линию за краем
    fit_ok = conf & ~exc_mask                                     # подгонка ТОЛЬКО по базовым якорям (не выбросам)
    bnds = [float(T[0]) - 1] + [c[0] for c in cuts] + [float(T[-1]) + 1]
    curve = np.zeros(len(T))
    for k in range(len(bnds) - 1):
        m = (T > bnds[k]) & (T <= bnds[k + 1])
        idx = np.where(m)[0]
        if not len(idx):
            continue
        cm = fit_ok & m & inbody
        if int(cm.sum()) >= 2:
            tt = T[cm]; yy = o_res[cm]; ww = w[cm]
            sl, inter = _theilsen(tt, yy, smax)                  # прод-прямая (тот же Тейл-Сен, что был inline)
            line_resid = float(np.percentile(np.abs(yy - (sl * tt + inter)), 90))
            if line_resid <= VIS_ADAPT_THR:                      # ПРЯМАЯ ложится → ступени/линия БИТ-В-БИТ как было
                curve[idx] = sl * T[idx] + inter + base[idx]
            else:                                                # ГУЛЯЮЩИЙ горками → RDP-ломаная по сглаженным
                ys = _smooth_w(tt, yy, ww, VIS_SMOOTH_WIN)
                nt, ny = _rdp(tt, ys, VIS_RDP_EPS)
                curve[idx] = np.interp(T[idx], nt, ny) + base[idx]
        else:
            curve[idx] = base[idx]
    fill = [(c[2], c[3]) for c in cuts]                           # пролёты тишины (резы)
    return curve, cuts, fill, o_res, conf, body


def build_map(pred, asg, cos, fps_ref, fps_dub, dur_ref, dt, ax_ref=None, ax_dub=None):
    """ВИДЕО-КАРТА ЧИСТЫМ построителем — ЕДИНСТВЕННЫЙ источник истины conform: СНЯТИЕ реального
    масштаба + clean_cuts (РЕШАЕТ ДЕЛЬТА) + робастная укладка. Карта `tg_s` = укладка НАПРЯМУЮ,
    БЕЗ монотонизации: между резами она и так монотонна (наклон ≤VIS_SMAX ≪ fps → дубль назад не
    играется). Вырезы (где дубля нет) вызывающий гасит ТИШИНОЙ по резам `cuts` (−Δ → вырез ширины
    |Δ|/fps; на резе tg_s локально падает, но та зона занулена — после неё дубль непрерывен).
    Возврат: (grid, tg_s, cuts, o, w, curve).
      cuts [(tc, Δкадры, t_end, t_nxt)] — РЕЗЫ укладки = ЕДИНСТВЕННЫЙ детектор; вырез =
      промежуток МЕЖДУ якорями [t_end, t_nxt] (там нет дубля); o/w/curve — для графиков.

    РЕАЛЬНЫЙ МАСШТАБ (вариант B): глобальный наклон дубля (PAL/ускорение/любой fps) снимается
    БЕЗ потолка (`global_trend`), а детект+укладка идут на ОСТАТКЕ — где потолок VIS_SMAX
    ограничивает уже ОТКЛОНЕНИЕ от реального масштаба, а не от нуля. Иначе ускоренный дубль
    (напр. PAL −4.1%) упирался в потолок ~1.9% и уезжал «в космос». Здоровым (наклон~0) —
    без изменений; вырезы (мгновенные ступени) остаются в остатке для детекта резов."""
    T = make_T(dur_ref)                                   # сетка от РЕАЛЬНОЙ длины рефа (любая длительность)
    o, w = vision_ow(pred, asg, cos, fps_ref, fps_dub, T=T, ax_ref=ax_ref, ax_dub=ax_dub)
    a_g, b_g = global_trend(o, w, T, fps_ref)             # реальный масштаб дубля (без потолка)
    curve, cuts4, _fill, _o_res, _conf, _body = build_curve(o, w, T, a_g, b_g)
    # [(tc, Δкадры, t_end, t_nxt)] — рез + ГРАНИЦЫ выреза = промежуток МЕЖДУ якорями плато
    # (t_end=последний якорь плато до, t_nxt=первый якорь плато после). Это и есть «нет дубля».
    cuts = [(float(tc), float(dv), float(te), float(tn)) for tc, dv, te, tn in cuts4]
    grid = np.arange(0, dur_ref, dt)
    shift = np.interp(grid, T, curve)                     # сдвиг (кадры рефа) на grid рефа = укладка
    tg_s = grid + shift / fps_ref                         # время дубля, с: кадры→сек по РЕАЛЬНОМУ fps_ref
    #   (НЕ FRAME=23.976 — на NTSC/PAL давало масштаб 1.25×; сдвиг измерен в кадрах рефа vision_ow)
    return grid, tg_s.astype(np.float64), cuts, o, w, curve


def render_plots(plot_dir, stem, o, w, curve, cuts):
    """Графики укладки зрения (png + html) рядом с аудио — как band/muq. Read-only,
    падение не должно ронять conform (ловит вызывающий тоже). Переиспользует anchor/plots
    (точки o по уверенности + кривая укладки + резы); off0 нет → нули, GCC нет → None."""
    from pathlib import Path
    pd = Path(plot_dir); pd.mkdir(parents=True, exist_ok=True)
    off0 = np.zeros(len(o), np.float64)
    ttl = f"{stem} — зрение · резов {len(cuts)}"
    from track_muxer.conform.anchor import plots as _p
    _p.render_track(pd / f"{stem}__vision.png", o, w, off0, curve, cuts, title=ttl)
    try:
        from track_muxer.conform.anchor import plots_html as _ph
        _ph.render_track_html(pd / f"{stem}__vision.html", o, w, off0, None, curve, cuts, title=ttl)
    except Exception:  # noqa: BLE001 — html опционален (plotly)
        pass
