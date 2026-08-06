# -*- coding: utf-8 -*-
"""ЕДИНЫЙ график укладки: ЗРЕНИЕ (всегда) + АУДИО band/muq (если включено) в ОДНОМ.

Обе панели — в ДРЕЙФ-форме (откл. сдвига от робастной линейной базы): наклон/дрейф и
ступени-резы видны вокруг 0. Зрение всегда есть → минимум 1 панель; аудио добавляет вторую
(общая ось времени, синхронный зум/ховер).

⭐ ГРАФИК ЗРЕНИЯ = РОВНО ТО, ЧТО В ЗВУКЕ (один источник, 100% по построению):
  - оранжевая = `vision["curve"]` = применённая укладка (сдвиг из tg_s, по которой ресэмплится out),
    РВЁТСЯ в вырезах (там дубля нет — показывать нечего);
  - красные зоны = `vision["fill_spans"]` = РОВНО зоны, занулённые в out (вырезы −Δ + швы +Δ).
  Никакого «рефа» в этих зонах график не утверждает: реф туда может прийти лишь финальным
  _fill_silence_from_ref по факту тишины — это отдельный этап, не часть карты.

API:
  render_unified(plot_dir, stem, *, vision, audio=None, title="") -> list[plots]
    vision = dict(o, w, curve, cuts, fill_spans, t_ref, shift_fr, cos, scale_a, scale_b, head_s, dur)
    audio  = dict(T, o, w, wcurve, det_cuts, gcc) | None
  Пишет: <stem>__track.png (превью) + <stem>__track.html (plotly, для модалки).

Read-only диагностика: падение рендера не должно ронять conform. Знак: правее=+; кадр=FRAME мс."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .params import T as GT, FRAME


# ───────────────────────── общие помощники ─────────────────────────
def _baseline(t_ref, shift_fr, head_s):
    """Робастная линейная база сдвига зрения по ТЕЛУ (Тейл-Сен) — для дрейф-формы (старые данные)."""
    tr = np.asarray(t_ref, float); sh = np.asarray(shift_fr, float)
    body = tr > (head_s + 5.0)
    if body.sum() < 10:
        body = np.ones(len(tr), bool)
    x = tr[body]; y = sh[body]
    n = len(x); idx = np.linspace(0, n - 1, min(n, 400)).astype(int)
    xs, ys = x[idx], y[idx]; sl = []
    for i in range(0, len(xs), 2):
        for j in range(i + 1, len(xs), 7):
            dx = xs[j] - xs[i]
            if abs(dx) > 1.0:
                sl.append((ys[j] - ys[i]) / dx)
    a = float(np.median(sl)) if sl else 0.0
    b = float(np.median(y - a * x))
    return a, b


def _breaks(curve, cut_t, Tarr):
    """NaN на точках-резах — линия не соединяет куски через разрыв (для аудио-панели)."""
    c = np.asarray(curve, float).copy()
    for tc in cut_t:
        j = int(np.searchsorted(Tarr, tc))
        if 0 < j < len(c):
            c[j] = np.nan
    return c


def _mask_spans(Tarr, curve, spans):
    """РАЗРЫВ линии в зонах spans=[(a,b)] (там дубля нет → показывать нечего). Возврат: копия с NaN."""
    T = np.asarray(Tarr, float); c = np.asarray(curve, float).copy()
    for a, b in spans:
        c[(T >= a) & (T <= b)] = np.nan
    return c


def _vis_dev(vis):
    """Зрение в дрейф-форме: (база a,b; девиация якорей; девиация ПРИМЕНЁННОЙ карты (raw, без
    разрывов); предел оси). Разрывы по fill_spans накладывает вызывающий — это ровно зоны звука."""
    Tv = np.asarray(vis.get("T", GT), float)
    if "scale_a" in vis:                                 # тот же РОБАСТНЫЙ масштаб, что в карте
        a, b = float(vis["scale_a"]), float(vis["scale_b"])
    else:                                                # старые данные без масштаба → Тейл-Сен база
        a, b = _baseline(vis["t_ref"], vis["shift_fr"], vis["head_s"])
    base_T = a * Tv + b
    dev_anchor = np.asarray(vis["shift_fr"], float) - (a * np.asarray(vis["t_ref"], float) + b)
    dev_curve = np.asarray(vis["curve"], float) - base_T
    md = np.isfinite(dev_anchor)
    cf = dev_curve[np.isfinite(dev_curve)]               # кривую укладки обрезать НЕЛЬЗЯ — по ней ресэмпл
    peak = float(np.percentile(np.abs(dev_anchor[md]), 99)) if md.any() else 0.0  # якоря: p99 (выбросы cos не раздувают)
    if cf.size:
        peak = max(peak, float(np.abs(cf).max()))        # кривая: целиком, иначе скачок укладки срежет потолок
    lim = max(12.0, peak * 1.06 + 3) if peak > 0 else 15.0   # запас ПРОПОРЦИОНАЛЬНЫЙ + мини-поле
    return a, b, dev_anchor, dev_curve, lim


# ───────────────────────── PNG (превью, 1/2 панели) ─────────────────────────
def _png(plot_dir, stem, vision, audio, title, xlim=None, suffix="track"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    two = audio is not None
    a, b, dev_anchor, dev_curve, vlim = _vis_dev(vision)
    med_abs = float(np.median(vision["shift_fr"])) if len(vision["shift_fr"]) else 0.0
    if two:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7.2), dpi=100, sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(15, 4.6), dpi=100); ax2 = None
    dur = float(vision["dur"]); xl = xlim or (0, dur)
    Tc = np.asarray(vision.get("T", GT), float)
    spans = vision.get("fill_spans") or []
    # --- зрение ---
    ax1.axhline(0, color="#bbb", lw=0.7)
    st = max(1, len(vision["t_ref"]) // 8000)
    sc1 = ax1.scatter(vision["t_ref"][::st], dev_anchor[::st], c=vision["cos"][::st],
                      cmap="viridis", s=4, vmin=0, vmax=1, linewidths=0, zorder=3)
    for a_s, b_s in spans:                               # вырезы (нет дубля) = ровно занулённое в out
        ax1.axvspan(a_s, b_s, color="crimson", alpha=0.16, lw=0, zorder=0)
    for tc in (vision.get("cut_marks") or []):           # границы разрывов — штриховые вертикали (опц.)
        ax1.axvline(float(tc), color="#444", ls="--", lw=1.0, alpha=0.85, zorder=6)
    ax1.plot(Tc, _mask_spans(Tc, dev_curve, spans), color="#ff7f0e", lw=2.6,
             label="укладка (где лежит дубль)", zorder=5)        # рвётся в вырезах
    ax1.set_ylim(-vlim, vlim); ax1.set_xlim(*xl); ax1.set_ylabel("Δ дрейф зрения, кадры")
    ax1.set_title(f"ЗРЕНИЕ — где лежит дубль (откл. от масштаба; абс. медиана {med_abs:+.0f}к)", fontsize=10)
    ax1.grid(True, alpha=0.15)
    h, lab = ax1.get_legend_handles_labels()
    h.append(Patch(facecolor="crimson", alpha=0.16)); lab.append("вырез (нет дубля)")
    ax1.legend(h, lab, loc="upper right", fontsize=8)
    fig.colorbar(sc1, ax=ax1, pad=0.01, fraction=0.02).set_label("cos")
    # --- аудио ---
    if two:
        ao = np.asarray(audio["o"], float); Tb = np.asarray(audio["T"], float)
        ax2.axhline(0, color="#bbb", lw=0.7)
        if audio.get("gcc") is not None:                 # gcc-свидетель тоже не рисуем в вырезах (тишина)
            ax2.scatter(Tb, _mask_spans(Tb, np.asarray(audio["gcc"], float), spans), s=2, color="lightgray", alpha=0.4)
        for a_s, b_s in spans:                           # вырезы зрения — band там не работает
            ax2.axvspan(a_s, b_s, color="crimson", alpha=0.10, lw=0, zorder=0)
        wA = np.asarray(audio["w"], float); am = wA > 1e-3   # вес≈0 = тишина зрения → НЕ якорь, не рисуем
        Tm, aom, wm = Tb[am], ao[am], wA[am]
        bst = max(1, len(Tm) // 8000)
        sc2 = ax2.scatter(Tm[::bst], aom[::bst], c=np.clip(wm, 0, 1.2)[::bst],
                          cmap="cividis", s=4, vmin=0, vmax=1.2, linewidths=0)
        bc = [float(t) for t, _ in audio["det_cuts"]]
        ax2.plot(Tb, _mask_spans(Tb, _breaks(audio["wcurve"], bc, Tb), spans), color="#2ca02c", lw=2.2,
                 label="кривая band (дрейф)", zorder=5)   # рвётся в вырезах: band там не работает
        for t, v in audio["det_cuts"]:                              # резы band (off0+o) — фиолетовый пунктир (как старый прод)
            ax2.axvline(float(t), color="purple", ls=":", lw=1.0)
        _amv = np.abs(ao[wA > 1e-3])                     # масштаб по ВИДИМЫМ якорям band (реальные замеры, вес>0)
        _peak = float(np.percentile(_amv, 99)) if _amv.size else 0.0  # wcurve экстраполирует без опор на краях, gcc — шум: ось НЕ задают
        bl = max(8.0, _peak * 1.08 + 2)                  # проп. запас + мини-поле; пол 8 как раньше
        for te, tn in (audio.get("dtw_cut_zones") or []):           # НЕДОСТАЧА дубля (вырез DTW) — оранжевая зона
            ax2.axvspan(float(te), float(tn), color="darkorange", alpha=0.22, lw=0, zorder=1)
        for tc, ln in (audio.get("dtw_inserts") or []):             # ВСТАВКА озвучки (лишнее) — зелёная вертикаль+длит.
            ax2.axvline(float(tc), color="#17a020", lw=1.4, zorder=6)
            ax2.annotate(f"вставка +{float(ln):.0f}с", (float(tc), bl * 0.8), color="#17a020",
                         fontsize=7, ha="center", rotation=90, va="top")
        ax2.set_ylim(-bl, bl); ax2.set_xlim(*xl); ax2.set_ylabel("Δ дрейф аудио, кадры")
        ax2.set_title("АУДИО — дрейф остатка (доводка поверх зрения)", fontsize=10)
        ax2.grid(True, alpha=0.15); ax2.legend(loc="upper right", fontsize=8)
        ax2.set_xlabel("время рефа, с")
        fig.colorbar(sc2, ax=ax2, pad=0.01, fraction=0.02).set_label("w")
    else:
        ax1.set_xlabel("время рефа, с")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    p = Path(plot_dir) / f"{stem}__{suffix}.png"
    fig.savefig(p); plt.close(fig)
    return p.name


# ───────────────────────── HTML (plotly, для модалки) ─────────────────────────
def _html(plot_dir, stem, vision, audio, title):
    from plotly.subplots import make_subplots
    two = audio is not None
    a, b, dev_anchor, dev_curve, vlim = _vis_dev(vision)
    med_abs = float(np.median(vision["shift_fr"])) if len(vision["shift_fr"]) else 0.0
    rows = 2 if two else 1
    titles = [f"ЗРЕНИЕ — где лежит дубль (откл. от масштаба; абс. медиана {med_abs:+.0f}к)"]
    if two:
        titles.append("АУДИО — дрейф остатка (доводка поверх зрения)")
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=titles)
    Tc = np.asarray(vision.get("T", GT), float)
    spans = vision.get("fill_spans") or []
    # зрение
    fig.add_hline(y=0, line=dict(color="#bbb", width=0.7), row=1, col=1)
    st = max(1, len(vision["t_ref"]) // 9000)
    fig.add_scattergl(x=np.asarray(vision["t_ref"])[::st], y=dev_anchor[::st], mode="markers",
                      marker=dict(size=3, color=np.asarray(vision["cos"])[::st], colorscale="Viridis",
                                  cmin=0, cmax=1, colorbar=dict(title="cos", x=1.02, len=0.5, y=0.78 if two else 0.5)),
                      name="якоря зрения (цвет=cos)",
                      hovertemplate="t=%{x:.1f}с  Δ=%{y:+.1f}к<extra>зрение</extra>", row=1, col=1)
    for a_s, b_s in spans:                               # вырезы (нет дубля) = ровно занулённое в out
        fig.add_vrect(x0=float(a_s), x1=float(b_s), fillcolor="crimson", opacity=0.16, line_width=0, row=1, col=1)
    for tc in (vision.get("cut_marks") or []):           # границы разрывов — штриховые вертикали (опц.)
        fig.add_vline(x=float(tc), line=dict(color="#444", dash="dash", width=1), row=1, col=1)
    # scattergl ПОСЛЕ точек → линия сверху; рвётся в вырезах (connectgaps=False)
    fig.add_scattergl(x=Tc, y=_mask_spans(Tc, dev_curve, spans), mode="lines",
                      line=dict(color="#ff7f0e", width=3.0), connectgaps=False,
                      name="укладка (где лежит дубль)",
                      hovertemplate="t=%{x:.1f}с  дубль на рефе %{y:+.1f}к<extra>зрение</extra>", row=1, col=1)
    fig.add_scattergl(x=[None], y=[None], mode="markers",
                      marker=dict(size=10, color="crimson", opacity=0.4, symbol="square"),
                      name="вырез (нет дубля)", row=1, col=1)
    fig.update_yaxes(range=[-vlim, vlim], title_text="Δ дрейф зрения, кадры", row=1, col=1)
    # аудио
    if two:
        Tb = np.asarray(audio["T"], float); ao = np.asarray(audio["o"], float)
        fig.add_hline(y=0, line=dict(color="#bbb", width=0.7), row=2, col=1)
        if audio.get("gcc") is not None and len(np.asarray(audio["gcc"])):  # gcc тоже не в вырезах
            fig.add_scattergl(x=Tb, y=_mask_spans(Tb, np.asarray(audio["gcc"], float), spans), mode="markers",
                              marker=dict(size=2, color="lightgray"), opacity=0.4, name="gcc-свидетель",
                              hovertemplate="t=%{x:.1f}с  gcc=%{y:+.1f}к<extra>аудио</extra>", row=2, col=1)
        for a_s, b_s in spans:                           # вырезы зрения — band там не работает
            fig.add_vrect(x0=float(a_s), x1=float(b_s), fillcolor="crimson", opacity=0.10, line_width=0, row=2, col=1)
        wA = np.asarray(audio["w"], float); am = wA > 1e-3   # вес≈0 = тишина зрения → НЕ якорь, не рисуем
        Tm, aom, wm = Tb[am], ao[am], wA[am]
        bst = max(1, len(Tm) // 9000)
        fig.add_scattergl(x=Tm[::bst], y=aom[::bst], mode="markers",
                          marker=dict(size=3, color=np.clip(wm, 0, 1.2)[::bst],
                                      colorscale="Cividis", cmin=0, cmax=1.2,
                                      colorbar=dict(title="w", x=1.02, len=0.5, y=0.22)),
                          name="якоря аудио",
                          hovertemplate="t=%{x:.1f}с  остаток=%{y:+.1f}к<extra>аудио</extra>", row=2, col=1)
        bc = [float(t) for t, _ in audio["det_cuts"]]
        fig.add_scattergl(x=Tb, y=_mask_spans(Tb, _breaks(audio["wcurve"], bc, Tb), spans), mode="lines",
                          connectgaps=False,                # рвётся в вырезах: band там не работает
                          line=dict(color="#2ca02c", width=2.4), name="кривая band (дрейф)",
                          hovertemplate="t=%{x:.1f}с  band=%{y:+.1f}к<extra>аудио</extra>", row=2, col=1)
        for t, v in audio["det_cuts"]:                              # резы band (off0+o) — фиолетовый пунктир (как старый прод)
            fig.add_vline(x=float(t), line=dict(color="purple", dash="dot", width=1.0), row=2, col=1)
        for te, tn in (audio.get("dtw_cut_zones") or []):           # НЕДОСТАЧА дубля (вырез DTW) — оранжевая зона
            fig.add_vrect(x0=float(te), x1=float(tn), fillcolor="darkorange", opacity=0.22, line_width=0, row=2, col=1)
        for tc, ln in (audio.get("dtw_inserts") or []):             # ВСТАВКА озвучки (лишнее) — зелёная вертикаль+аннотация
            fig.add_vline(x=float(tc), line=dict(color="#17a020", width=1.5),
                          annotation_text=f"вставка +{float(ln):.0f}с", annotation_position="top",
                          annotation=dict(font_size=9, font_color="#17a020"), row=2, col=1)
        _amv = np.abs(ao[wA > 1e-3])                     # масштаб по ВИДИМЫМ якорям band (реальные замеры, вес>0)
        _peak = float(np.percentile(_amv, 99)) if _amv.size else 0.0  # wcurve экстраполирует без опор на краях, gcc — шум: ось НЕ задают
        bl = max(8.0, _peak * 1.08 + 2)                  # проп. запас + мини-поле; пол 8 как раньше
        fig.update_yaxes(range=[-bl, bl], title_text="Δ дрейф аудио, кадры", row=2, col=1)
    fig.update_xaxes(title_text="время рефа, с", row=rows, col=1)
    fig.update_layout(title=title, template="plotly_white", height=760 if two else 460,
                      hovermode="x unified", legend=dict(orientation="h", y=-0.08))
    p = Path(plot_dir) / f"{stem}__track.html"
    fig.write_html(str(p), include_plotlyjs="inline")
    return p.name


# ───────────────────────── публичный рендер ─────────────────────────
def render_unified(plot_dir, stem, *, vision, audio=None, title=""):
    """Единый график (1 панель зрение / 2 панели зрение+аудио). -> список PlotRef."""
    pd = Path(plot_dir); pd.mkdir(parents=True, exist_ok=True)
    plots = []
    name = _png(pd, stem, vision, audio, title)
    plots.append({"kind": "track", "name": name, "t": None, "v_ms": None})
    try:
        hname = _html(pd, stem, vision, audio, title)
        plots.append({"kind": "html", "name": hname, "t": None, "v_ms": None})
    except Exception:  # noqa: BLE001 — HTML опционален (plotly)
        pass
    return plots
