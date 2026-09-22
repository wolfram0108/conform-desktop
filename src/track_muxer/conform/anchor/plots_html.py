# -*- coding: utf-8 -*-
"""Interactive HTML alignment plot for band/muq (plotly, self-contained — opens offline). Read-only.
GCC witness + coarse off0 + o points colored by confidence + the warp curve (drift and bumps) +
cuts. Sign convention: rightward = +; frame = 41.708 ms."""
import numpy as np

from .params import T as _DEFT, FRAME


def _wc_breaks(wcurve, cuts, T):
    wc = np.asarray(wcurve, float).copy()
    for tc, _ in cuts:
        j = int(np.searchsorted(T, tc))
        if 0 < j < len(wc):
            wc[j] = np.nan
    return wc


def render_track_html(path, o, w, off0, gl, wcurve, cuts, *, title="", T=None):
    """Self-contained HTML (plotly inline). A failed plotly import does not bring down conform — the
    caller catches it. T is the time grid matching the length of o/w (make_T at the real duration);
    None uses the default params.T."""
    import plotly.graph_objects as go
    T = _DEFT if T is None else np.asarray(T, float)
    o = np.asarray(o, float); off0 = np.asarray(off0, float); wcurve = np.asarray(wcurve, float)
    fig = go.Figure()
    if gl is not None and len(np.asarray(gl)):
        fig.add_scatter(x=T, y=np.asarray(gl, float), mode="markers",
                        marker=dict(size=3, color="lightgray"), opacity=0.4,
                        name="GCC-свидетель ±2.5с",
                        hovertemplate="t=%{x:.1f}с  gcc=%{y:+.1f}к<extra></extra>")
    fig.add_scatter(x=T, y=off0, mode="lines", line=dict(color="royalblue", width=1.5), opacity=0.6,
                    name="off0 (грубая)", hovertemplate="t=%{x:.1f}с  off0=%{y:+.1f}к<extra></extra>")
    fig.add_scatter(x=T, y=o, mode="markers",
                    marker=dict(size=4, color=np.clip(w, 0, 1.2), colorscale="Viridis", cmin=0, cmax=1.2,
                                colorbar=dict(title="увер. w", x=1.07)),
                    name="o (итог)", hovertemplate="t=%{x:.1f}с  o=%{y:+.1f}к<extra></extra>")
    fig.add_scatter(x=T, y=_wc_breaks(wcurve, cuts, T), mode="lines", line=dict(color="orange", width=2.5),
                    name="кривая варпа (дрейф+горки)",
                    hovertemplate="t=%{x:.1f}с  варп=%{y:+.1f}к<extra></extra>")
    for tc, v in cuts:
        fig.add_vline(x=float(tc), line=dict(color="red", dash="dash", width=1),
                      annotation_text=f"{v * FRAME:+.0f}мс", annotation_position="top")
    ymin = float(min(o.min(), off0.min(), -2)); ymax = float(max(o.max(), off0.max(), 2))
    fig.update_layout(title=title, xaxis_title="время, с", yaxis_title="сдвиг, кадры",
                      template="plotly_white", hovermode="x unified", height=620,
                      yaxis=dict(range=[ymin - 2, ymax + 2]),
                      yaxis2=dict(title="сдвиг, мс", overlaying="y", side="right",
                                  range=[(ymin - 2) * FRAME, (ymax + 2) * FRAME], showgrid=False))
    fig.add_scatter(x=[float(T[0])], y=[(ymin - 2) * FRAME], mode="markers", marker=dict(opacity=0),
                    yaxis="y2", showlegend=False, hoverinfo="skip")
    fig.write_html(str(path), include_plotlyjs="inline")
