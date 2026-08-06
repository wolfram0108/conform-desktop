# -*- coding: utf-8 -*-
"""PNG-миниатюра укладки band/muq (read-only диагностика, matplotlib Agg).
Кривая варпа (дрейф+горки, оранжевая) + грубая off0 (синяя) + точки o по уверенности +
вертикали резов. Знак: правее=+; кадр=41.708мс. Интерактивный график — в plots_html.py."""
import numpy as np

from .params import T as _DEFT, FRAME


def _setup():
    import matplotlib
    matplotlib.use("Agg")            # без дисплея (сервер)
    import matplotlib.pyplot as plt
    return plt


def _wn(w):
    med = np.median(w[w > 0]) if np.any(w > 0) else 1.0
    return w / (med if med > 1e-9 else 1.0)


def _wc_breaks(wcurve, cuts, T):
    """Кривая с NaN на резах — чтобы линия не соединяла куски через разрыв."""
    wc = np.asarray(wcurve, float).copy()
    for tc, _ in cuts:
        j = int(np.searchsorted(T, tc))
        if 0 < j < len(wc):
            wc[j] = np.nan
    return wc


def render_track(path, o, w, off0, wcurve, cuts, *, title="", T=None):
    """Весь трек: точки o (яркость=уверенность) + грубая off0 + кривая варпа + резы.

    T — сетка времени тех же длин, что o/w (make_T реальной длительности пары). None →
    дефолтная сетка params.T (как было; годится ТОЛЬКО когда длительность пары совпадает
    со стандартной — на нестандартной падало «x and y must be the same size», 2026-08-06)."""
    T = _DEFT if T is None else np.asarray(T, float)
    plt = _setup()
    fig, ax = plt.subplots(figsize=(15, 4.6), dpi=110)
    wn = np.clip(_wn(w), 0.0, 1.5)
    sc = ax.scatter(T, o, c=wn, cmap="viridis", s=7, vmin=0.0, vmax=1.2, linewidths=0)
    ax.plot(T, off0, color="royalblue", lw=1.0, alpha=0.6, label="off0 (грубая)")
    ax.plot(T, _wc_breaks(wcurve, cuts, T), color="#ff7f0e", lw=2.0, zorder=4,
            label="кривая варпа (дрейф+горки)")
    for tc, v in cuts:
        ax.axvline(tc, color="#d62728", lw=1.0, ls="--", alpha=0.8, zorder=2)
        ax.annotate(f"{v * FRAME:+.0f}мс", xy=(tc, 0), xytext=(2, 4), textcoords="offset points",
                    fontsize=8, color="#d62728", rotation=90, va="bottom", ha="left")
    ax.axhline(0, color="#888", lw=0.7)
    ax.set_xlim(float(T[0]), float(T[-1]))
    ax.set_xlabel("время, с"); ax.set_ylabel("сдвиг, кадры")
    secy = ax.secondary_yaxis("right", functions=(lambda f: f * FRAME, lambda m: m / FRAME))
    secy.set_ylabel("сдвиг, мс")
    cb = fig.colorbar(sc, ax=ax, pad=0.06, fraction=0.025); cb.set_label("уверенность (ярче)")
    ax.legend(loc="upper right", fontsize=8); ax.set_title(title); ax.grid(True, alpha=0.15)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
