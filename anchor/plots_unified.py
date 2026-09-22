# -*- coding: utf-8 -*-
"""UNIFIED layout plot: VISION (always) + AUDIO band/muq (when enabled) in ONE chart.

Both panels are in DRIFT form (the shift's deviation from a robust linear baseline): the
slope/drift and the cut steps are visible around 0. Vision is always present, so there is
at least 1 panel; audio adds a second one (a shared time axis, synchronized zoom/hover).

THE VISION PLOT IS EXACTLY WHAT ENDS UP IN THE AUDIO (one source, 100% by construction):
  - the orange line is `vision["curve"]`, the applied layout (the shift from tg_s that out
    is resampled by), which BREAKS across excisions (there is no dub there to show);
  - the red zones are `vision["fill_spans"]`, EXACTLY the zones zeroed out in out (excisions
    -delta plus seams +delta).

The pair's passport (`passport`) is the single source of zones, events, speed segments and
header metrics for both panels and both renders; the plot computes nothing itself, it only
lays the passport out across the panels. All labels are keys from `plot_terms` (language is
a parameter).

API:
  render_unified(plot_dir, stem, *, vision, audio=None, title="", passport=None, lang="ru") -> list[plots]
    vision   = dict(o, w, curve, cuts, fill_spans, t_ref, shift_fr, cos, scale_a, scale_b, head_s, dur)
    audio    = dict(T, o, w, wcurve, det_cuts, gcc) | None
    passport = dict(zones=[{kind,a,b}], events=[{kind,t,value}], segments=[{a,b,speed_pct}],
                    header=[[{key,value[,tpl]}, ...], [...]], verdict, verdict_text)
  Writes: <stem>__track.png (preview) + <stem>__track.html (plotly, for the modal).

Read-only diagnostics: a render failure must not bring conform down. Sign convention:
right = positive; a frame is FRAME ms."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from .params import T as GT
from .plot_terms import term, DEFAULT_LANG

# Zone kinds: colour, alpha on (vision panel, audio panel), hatch. Event kinds: colour, dash, panel.
ZONE_STYLE = {
    "vision_cut": dict(color="crimson", alpha=(0.16, 0.10), hatch=None),
    "video_freeze": dict(color="#606060", alpha=(0.22, 0.12), hatch="//"),
    "audio_gap": dict(color="purple", alpha=(0.08, 0.14), hatch=None),
    "dtw_cut": dict(color="darkorange", alpha=(0.10, 0.22), hatch=None),
}
EVENT_STYLE = {
    "vision_step": dict(color="#444444", ls="--", dash="dash", panel=1),
    "audio_cut": dict(color="purple", ls=":", dash="dot", panel=2),
    "audio_jump": dict(color="purple", ls="-", dash="solid", panel=2),
    "dtw_insert": dict(color="#17a020", ls="-", dash="solid", panel=2),
}
SPEED_LABEL_MIN_PCT = 0.1        # layout segments slower/faster than the scale by less are not labelled
LABEL_ROWS = 5                   # rows of event labels under the top edge of a panel
LABEL_ROW_FRAC = 0.045           # row height as a share of the panel height
LABEL_CHAR_PX = 5.6              # average glyph width of the label font
PNG_PLOT_PX = 1300               # width of the plot area in the fixed-size PNG
HTML_MIN_PLOT_PX = 600           # the HTML chart is responsive: slots are laid out for the narrowest sane width
PNG_HEADER_CHARS = 250           # header line capacity of the fixed-width PNG at its header font
ITEM_SEP = " · "


def _wrap_items(lines, limit):
    """Lines no longer than `limit`, broken only between items, so a metric is never split in two."""
    out = []
    for line in lines:
        current = ""
        for item in line.split(ITEM_SEP):
            candidate = item if not current else current + ITEM_SEP + item
            if current and len(candidate) > limit:
                out.append(current)
                current = item
            else:
                current = candidate
        out.append(current)
    return out


def _label_slots(items, x0, x1, plot_px):
    """(row, side) for each (t, text) so that no label covers another one or leaves the plot area;
    None when every row is taken near that line — the line stays, the text lives in the event list.

    A label is a box beside its line: to the right of it, or to the left when the right side is taken
    or runs past the axis. Rows go down from the top edge of the panel, inside the plot area, so a
    label never reaches the panel title above it."""
    span = max(float(x1) - float(x0), 1e-9)
    taken = [[] for _ in range(LABEL_ROWS)]
    slots = [None] * len(items)
    for i in sorted(range(len(items)), key=lambda k: items[k][0]):
        t, text = float(items[i][0]), items[i][1]
        width = (len(text) * LABEL_CHAR_PX + 8) / plot_px * span
        for row in range(LABEL_ROWS):
            for side, box in (("right", (t, t + width)), ("left", (t - width, t))):
                inside = box[0] >= x0 and box[1] <= x1
                if inside and all(box[1] <= a or box[0] >= b for a, b in taken[row]):
                    taken[row].append(box)
                    slots[i] = (row, side)
                    break
            if slots[i] is not None:
                break
    return slots


# ───────────────────────── shared helpers ─────────────────────────
def _baseline(t_ref, shift_fr, head_s):
    """Robust linear baseline of the vision shift over the BODY (Theil-Sen) -- used for the
    drift form when the data has no stored scale."""
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
    """NaN at the cut points, so the line does not connect pieces across a break (for the audio panel)."""
    c = np.asarray(curve, float).copy()
    for tc in cut_t:
        j = int(np.searchsorted(Tarr, tc))
        if 0 < j < len(c):
            c[j] = np.nan
    return c


def _mask_spans(Tarr, curve, spans):
    """Breaks the line across the zones spans=[(a,b)] (there is no dub there to show). Returns a copy with NaN in those zones."""
    T = np.asarray(Tarr, float); c = np.asarray(curve, float).copy()
    for a, b in spans:
        c[(T >= a) & (T <= b)] = np.nan
    return c


def _vis_dev(vis):
    """Vision in drift form: (baseline a,b; anchor deviation; deviation of the APPLIED map;
    axis limit; count of anchors beyond the limit). The LAYOUT sets the axis, since it is
    the actual result; anchors that stray further (frozen frames, false matches) are
    clipped and counted rather than stretching the axis."""
    Tv = np.asarray(vis.get("T", GT), float)
    if "scale_a" in vis:                                 # the same robust scale the time map itself uses
        a, b = float(vis["scale_a"]), float(vis["scale_b"])
    else:                                                # older data with no stored scale falls back to a Theil-Sen baseline
        a, b = _baseline(vis["t_ref"], vis["shift_fr"], vis["head_s"])
    base_T = a * Tv + b
    dev_anchor = np.asarray(vis["shift_fr"], float) - (a * np.asarray(vis["t_ref"], float) + b)
    dev_curve = np.asarray(vis["curve"], float) - base_T
    md = np.isfinite(dev_anchor)
    cf = dev_curve[np.isfinite(dev_curve)]
    peak = float(np.abs(cf).max()) if cf.size else 0.0     # the layout alone sets the axis
    lim = max(12.0, peak * 1.06 + 3) if peak > 0 else 15.0
    clipped = int(np.sum(np.abs(dev_anchor[md]) > lim)) if md.any() else 0
    return a, b, dev_anchor, dev_curve, lim, clipped


def _mmss(s: float) -> str:
    s = max(0, int(round(float(s))))
    return f"{s // 60}:{s % 60:02d}"


def _zones(passport, kind=None):
    for z in ((passport or {}).get("zones") or []):
        if kind is None or z["kind"] == kind:
            yield z


def _events(passport, panel):
    for e in ((passport or {}).get("events") or []):
        st = EVENT_STYLE.get(e["kind"])
        if st and st["panel"] == panel:
            yield e, st


def _header_lines(passport, lang):
    """Header lines from passport['header']: each item {key, value} or {key, tpl, value: dict}
    for composite values; None → the catalog's dash; bool → yes/no. Units live in the catalog."""
    out = []
    for line_key, items in zip(("header.vision", "header.audio"), (passport or {}).get("header") or []):
        if not items:
            continue
        parts = []
        for it in items:
            v = it.get("value")
            if v is None:
                v = term("value.none", lang)
            elif isinstance(v, dict):
                v = term(it["tpl"], lang, **v)
            elif isinstance(v, bool):
                v = term("value.yes" if v else "value.no", lang)
            parts.append(term("metric." + it["key"], lang, value=v))
        out.append(f"{term(line_key, lang)}: " + ITEM_SEP.join(parts))
    vd = (passport or {}).get("verdict")
    if vd:
        txt = term("metric.verdict", lang, value=term("verdict." + vd, lang))
        vt = (passport or {}).get("verdict_text")
        if vt:
            txt += f" — {vt}"
        out.append(txt)
    return out


def _legend(passport, panel, present_kinds, lang):
    """(handles, labels) proxies for the zones and events shown on a panel — matplotlib only."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles, labels = [], []
    for kind in ZONE_STYLE:
        if kind in present_kinds:
            zs = ZONE_STYLE[kind]
            handles.append(Patch(facecolor=zs["color"], alpha=max(zs["alpha"]), hatch=zs["hatch"]))
            labels.append(term("zone." + kind, lang))
    seen = set()
    for e, est in _events(passport, panel):
        if e["kind"] in seen:
            continue
        seen.add(e["kind"])
        handles.append(Line2D([], [], color=est["color"], ls=est["ls"])); labels.append(term("legend." + e["kind"], lang))
    return handles, labels


# ───────────────────────── PNG (preview, 1 or 2 panels) ─────────────────────────
def _png(plot_dir, stem, vision, audio, title, passport=None, lang=DEFAULT_LANG, xlim=None, suffix="track"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    two = audio is not None
    a, b, dev_anchor, dev_curve, vlim, clipped = _vis_dev(vision)
    header = _header_lines(passport, lang)
    if two:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8.2), dpi=100, sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(15, 5.4), dpi=100); ax2 = None
    dur = float(vision["dur"]); xl = xlim or (0, dur)
    Tc = np.asarray(vision.get("T", GT), float)
    spans = vision.get("fill_spans") or []
    present = {z["kind"] for z in _zones(passport)}
    if passport is None and spans:
        present.add("vision_cut")

    def draw_zones(ax, pi):
        for z in _zones(passport):
            zs = ZONE_STYLE[z["kind"]]
            ax.axvspan(float(z["a"]), float(z["b"]), color=zs["color"], alpha=zs["alpha"][pi],
                       hatch=zs["hatch"], lw=0, zorder=0)
        if passport is None:                              # old callers without a passport: vision cuts only
            for a_s, b_s in spans:
                ax.axvspan(a_s, b_s, color="crimson", alpha=(0.16, 0.10)[pi], lw=0, zorder=0)

    def draw_events(ax, panel, ylim):
        found = list(_events(passport, panel))
        texts = [term("event." + e["kind"], lang, value=float(e["value"])) for e, _ in found]
        slots = _label_slots([(e["t"], tx) for (e, _), tx in zip(found, texts)], xl[0], xl[1], PNG_PLOT_PX)
        for (e, est), text, slot in zip(found, texts, slots):
            ax.axvline(float(e["t"]), color=est["color"], ls=est["ls"], lw=1.1, alpha=0.9, zorder=6)
            if slot is None:
                continue
            row, side = slot
            ax.annotate(text, (float(e["t"]), 1.0 - 0.012 - row * LABEL_ROW_FRAC), xycoords=("data", "axes fraction"),
                        xytext=(3 if side == "right" else -3, 0), textcoords="offset points",
                        color=est["color"], fontsize=7, ha="left" if side == "right" else "right", va="top",
                        zorder=7)

    # --- vision ---
    ax1.axhline(0, color="#bbb", lw=0.7)
    st = max(1, len(vision["t_ref"]) // 8000)
    sc1 = ax1.scatter(vision["t_ref"][::st], np.clip(dev_anchor, -vlim, vlim)[::st], c=vision["cos"][::st],
                      cmap="viridis", s=4, vmin=0, vmax=1, linewidths=0, zorder=3)
    draw_zones(ax1, 0)
    if passport is None:
        for tc in (vision.get("cut_marks") or []):
            ax1.axvline(float(tc), color="#444", ls="--", lw=1.0, alpha=0.85, zorder=6)
    ax1.plot(Tc, _mask_spans(Tc, dev_curve, spans), color="#ff7f0e", lw=2.6, zorder=5)
    for seg in ((passport or {}).get("segments") or []):
        if abs(float(seg["speed_pct"])) >= SPEED_LABEL_MIN_PCT:
            tm = 0.5 * (float(seg["a"]) + float(seg["b"]))
            ym = float(np.interp(tm, Tc, np.nan_to_num(dev_curve)))
            ax1.annotate(term("segment.speed", lang, value=float(seg["speed_pct"])), (tm, ym),
                         xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7, color="#b35c00")
    draw_events(ax1, 1, vlim)
    if clipped:
        ax1.text(0.005, 0.97, term("anchors_clipped", lang, value=clipped), transform=ax1.transAxes,
                 fontsize=7, va="top", color="#333")
    ax1.set_ylim(-vlim, vlim); ax1.set_xlim(*xl); ax1.set_ylabel(term("axis.vision_y", lang))
    ax1.set_title(term("panel.vision", lang), fontsize=10)
    ax1.grid(True, alpha=0.15)
    h1 = [Line2D([], [], marker="o", ls="", color="#3b528b", ms=4), Line2D([], [], color="#ff7f0e", lw=2.6)]
    l1 = [term("legend.anchor_vision", lang), term("legend.layout", lang)]
    hz, lz = _legend(passport, 1, present, lang)
    ax1.legend(h1 + hz, l1 + lz, loc="lower right", fontsize=7)   # the top rows belong to event labels
    fig.colorbar(sc1, ax=ax1, pad=0.01, fraction=0.02).set_label(term("colorbar.cos", lang))
    # --- audio ---
    if two:
        ao = np.asarray(audio["o"], float); Tb = np.asarray(audio["T"], float)
        ax2.axhline(0, color="#bbb", lw=0.7)
        draw_zones(ax2, 1)
        wA = np.asarray(audio["w"], float); am = wA > 1e-3   # weight near 0 means vision silence, not an anchor: skip drawing it
        Tm, aom, wm = Tb[am], ao[am], wA[am]
        bst = max(1, len(Tm) // 8000)
        sc2 = ax2.scatter(Tm[::bst], aom[::bst], c=np.clip(wm, 0, 1.2)[::bst],
                          cmap="cividis", s=4, vmin=0, vmax=1.2, linewidths=0)
        bc = [float(t) for t, _ in audio["det_cuts"]]
        ax2.plot(Tb, _mask_spans(Tb, _breaks(audio["wcurve"], bc, Tb), spans), color="#2ca02c", lw=2.2, zorder=5)
        _amv = np.abs(ao[wA > 1e-3])
        _peak = float(np.percentile(_amv, 99)) if _amv.size else 0.0
        bl = max(8.0, _peak * 1.08 + 2)
        if passport is None:
            for t, v in audio["det_cuts"]:
                ax2.axvline(float(t), color="purple", ls=":", lw=1.0)
        draw_events(ax2, 2, bl)
        ax2.set_ylim(-bl, bl); ax2.set_xlim(*xl); ax2.set_ylabel(term("axis.audio_y", lang))
        ax2.set_title(term("panel.audio", lang), fontsize=10)
        ax2.grid(True, alpha=0.15)
        h2 = [Line2D([], [], marker="o", ls="", color="#7f7f7f", ms=4), Line2D([], [], color="#2ca02c", lw=2.2)]
        l2 = [term("legend.anchor_audio", lang), term("legend.audio_curve", lang)]
        hz2, lz2 = _legend(passport, 2, present, lang)
        ax2.legend(h2 + hz2, l2 + lz2, loc="lower right", fontsize=7)
        fig.colorbar(sc2, ax=ax2, pad=0.01, fraction=0.02).set_label(term("colorbar.w", lang))
    axb = ax2 if two else ax1
    axb.set_xlabel(f"{term('axis.time', lang)} · {term('axis.time.mmss', lang)}")
    axb.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:.0f}\n{_mmss(x)}"))
    # The title is placed by hand above the header; left in the layout it reserves its height twice.
    fig.suptitle(title, fontsize=11, y=0.995).set_in_layout(False)
    header = _wrap_items(header, PNG_HEADER_CHARS)       # the PNG has a fixed width: long lines wrap at items
    for i, line in enumerate(header):
        fig.text(0.01, 0.972 - 0.021 * i, line, fontsize=7.5, ha="left", va="top", color="#222")
    fig.tight_layout(rect=(0, 0, 1, 0.975 - 0.021 * len(header)))
    p = Path(plot_dir) / f"{stem}__{suffix}.png"
    fig.savefig(p); plt.close(fig)
    return p.name


# ───────────────────────── HTML (plotly, for the modal) ─────────────────────────
def _html(plot_dir, stem, vision, audio, title, passport=None, lang=DEFAULT_LANG):
    from plotly.subplots import make_subplots
    two = audio is not None
    a, b, dev_anchor, dev_curve, vlim, clipped = _vis_dev(vision)
    rows = 2 if two else 1
    titles = [term("panel.vision", lang)] + ([term("panel.audio", lang)] if two else [])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=titles)
    Tc = np.asarray(vision.get("T", GT), float)
    spans = vision.get("fill_spans") or []
    hover = "%{x:.1f} s (%{customdata})  %{y:+.1f}<extra></extra>"

    def zones(row, pi):
        for z in _zones(passport):
            zs = ZONE_STYLE[z["kind"]]
            fig.add_vrect(x0=float(z["a"]), x1=float(z["b"]), fillcolor=zs["color"], opacity=zs["alpha"][pi],
                          line_width=0, row=row, col=1)
        if passport is None:
            for a_s, b_s in spans:
                fig.add_vrect(x0=float(a_s), x1=float(b_s), fillcolor="crimson", opacity=(0.16, 0.10)[pi],
                              line_width=0, row=row, col=1)
        if row == 1:                                     # legend proxies once, for the zones present
            for kind in ZONE_STYLE:
                if any(True for _ in _zones(passport, kind)):
                    zs = ZONE_STYLE[kind]
                    fig.add_scattergl(x=[None], y=[None], mode="markers",
                                      marker=dict(size=10, color=zs["color"], opacity=0.5, symbol="square"),
                                      name=term("zone." + kind, lang), row=1, col=1)

    def events(row, panel):
        found = list(_events(passport, panel))
        texts = [term("event." + e["kind"], lang, value=float(e["value"])) for e, _ in found]
        slots = _label_slots([(e["t"], tx) for (e, _), tx in zip(found, texts)],
                             0.0, float(vision["dur"]), HTML_MIN_PLOT_PX)
        for (e, est), text, slot in zip(found, texts, slots):
            fig.add_vline(x=float(e["t"]), line=dict(color=est["color"], dash=est["dash"], width=1.2),
                          row=row, col=1)
            if slot is None:
                continue
            level, side = slot
            fig.add_annotation(x=float(e["t"]), y=1.0 - 0.012 - level * LABEL_ROW_FRAC, yref="y domain",
                               text=text, showarrow=False, font=dict(size=9, color=est["color"]),
                               xanchor="left" if side == "right" else "right", yanchor="top",
                               xshift=3 if side == "right" else -3, row=row, col=1)

    # vision
    fig.add_hline(y=0, line=dict(color="#bbb", width=0.7), row=1, col=1)
    st = max(1, len(vision["t_ref"]) // 9000)
    tr = np.asarray(vision["t_ref"])[::st]
    fig.add_scattergl(x=tr, y=np.clip(dev_anchor, -vlim, vlim)[::st], mode="markers",
                      marker=dict(size=3, color=np.asarray(vision["cos"])[::st], colorscale="Viridis",
                                  cmin=0, cmax=1, colorbar=dict(title=term("colorbar.cos", lang), x=1.02, len=0.5,
                                                                y=0.78 if two else 0.5)),
                      name=term("legend.anchor_vision", lang), customdata=[_mmss(x) for x in tr],
                      hovertemplate=hover, row=1, col=1)
    zones(1, 0)
    if passport is None:
        for tc in (vision.get("cut_marks") or []):
            fig.add_vline(x=float(tc), line=dict(color="#444", dash="dash", width=1), row=1, col=1)
    fig.add_scattergl(x=Tc, y=_mask_spans(Tc, dev_curve, spans), mode="lines",
                      line=dict(color="#ff7f0e", width=3.0), connectgaps=False, customdata=[_mmss(x) for x in Tc],
                      name=term("legend.layout", lang), hovertemplate=hover, row=1, col=1)
    for seg in ((passport or {}).get("segments") or []):
        if abs(float(seg["speed_pct"])) >= SPEED_LABEL_MIN_PCT:
            tm = 0.5 * (float(seg["a"]) + float(seg["b"]))
            fig.add_annotation(x=tm, y=float(np.interp(tm, Tc, np.nan_to_num(dev_curve))),
                               text=term("segment.speed", lang, value=float(seg["speed_pct"])),
                               showarrow=False, yshift=12, font=dict(size=9, color="#b35c00"), row=1, col=1)
    events(1, 1)
    if clipped:
        fig.add_annotation(xref="x domain", yref="y domain", x=0.005, y=0.97, showarrow=False,
                           text=term("anchors_clipped", lang, value=clipped), font=dict(size=9), row=1, col=1)
    fig.update_yaxes(range=[-vlim, vlim], title_text=term("axis.vision_y", lang), row=1, col=1)
    # audio
    if two:
        Tb = np.asarray(audio["T"], float); ao = np.asarray(audio["o"], float)
        fig.add_hline(y=0, line=dict(color="#bbb", width=0.7), row=2, col=1)
        zones(2, 1)
        wA = np.asarray(audio["w"], float); am = wA > 1e-3
        Tm, aom, wm = Tb[am], ao[am], wA[am]
        bst = max(1, len(Tm) // 9000)
        fig.add_scattergl(x=Tm[::bst], y=aom[::bst], mode="markers",
                          marker=dict(size=3, color=np.clip(wm, 0, 1.2)[::bst], colorscale="Cividis", cmin=0, cmax=1.2,
                                      colorbar=dict(title=term("colorbar.w", lang), x=1.02, len=0.5, y=0.22)),
                          name=term("legend.anchor_audio", lang), customdata=[_mmss(x) for x in Tm[::bst]],
                          hovertemplate=hover, row=2, col=1)
        bc = [float(t) for t, _ in audio["det_cuts"]]
        fig.add_scattergl(x=Tb, y=_mask_spans(Tb, _breaks(audio["wcurve"], bc, Tb), spans), mode="lines",
                          connectgaps=False, line=dict(color="#2ca02c", width=2.4), customdata=[_mmss(x) for x in Tb],
                          name=term("legend.audio_curve", lang), hovertemplate=hover, row=2, col=1)
        if passport is None:
            for t, v in audio["det_cuts"]:
                fig.add_vline(x=float(t), line=dict(color="purple", dash="dot", width=1.0), row=2, col=1)
        events(2, 2)
        _amv = np.abs(ao[wA > 1e-3])
        _peak = float(np.percentile(_amv, 99)) if _amv.size else 0.0
        bl = max(8.0, _peak * 1.08 + 2)
        fig.update_yaxes(range=[-bl, bl], title_text=term("axis.audio_y", lang), row=2, col=1)
    fig.update_xaxes(title_text=term("axis.time", lang), row=rows, col=1)
    fig.update_layout(template="plotly_white", height=760 if two else 460, margin=dict(t=40),
                      hovermode="x unified", legend=dict(orientation="h", y=-0.08))
    p = Path(plot_dir) / f"{stem}__track.html"
    p.write_text(_page(title, _header_lines(passport, lang), _event_list(passport, lang),
                       fig.to_html(full_html=False, include_plotlyjs="inline"), lang), encoding="utf-8")
    return p.name


PAGE_STYLE = """
body { margin: 0; font: 13px/1.45 system-ui, sans-serif; color: #2a3f5f; background: #fff; }
header { padding: 12px 20px 0; }
h1 { margin: 0 0 4px; font-size: 17px; font-weight: 500; overflow-wrap: anywhere; }
header p { margin: 0; font-size: 12px; overflow-wrap: anywhere; }
details { margin-top: 4px; font-size: 12px; }
details ol { margin: 4px 0 0; padding-left: 22px; columns: 22em; }
"""


def _event_list(passport, lang):
    """Every event as text, in time order: a label that found no free row on the chart is still readable here."""
    found = sorted(((passport or {}).get("events") or []), key=lambda e: float(e["t"]))
    return [f"{_mmss(float(e['t']))} — {term('event.' + e['kind'], lang, value=float(e['value']))}"
            for e in found if e["kind"] in EVENT_STYLE]


def _page(title, header, event_list, chart_div, lang):
    """The passport header is ordinary HTML above the chart, not text inside the SVG: the browser wraps it
    to any window width, and its height can never collide with the panels."""
    esc = html.escape
    parts = [f"<!doctype html><html lang='{esc(lang)}'><head><meta charset='utf-8'>",
             f"<title>{esc(title)}</title><style>{PAGE_STYLE}</style></head><body><header>",
             f"<h1>{esc(title)}</h1>"]
    parts += [f"<p>{esc(line)}</p>" for line in header]
    if event_list:
        parts.append(f"<details><summary>{esc(term('list.events', lang, value=len(event_list)))}</summary><ol>")
        parts += [f"<li>{esc(item)}</li>" for item in event_list]
        parts.append("</ol></details>")
    parts += ["</header>", chart_div, "</body></html>"]
    return "".join(parts)


# ───────────────────────── public render entry point ─────────────────────────
def render_unified(plot_dir, stem, *, vision, audio=None, title="", passport=None, lang=DEFAULT_LANG):
    """The unified plot (1 panel for vision / 2 panels for vision+audio). Returns a list of PlotRef."""
    pd = Path(plot_dir); pd.mkdir(parents=True, exist_ok=True)
    plots = []
    name = _png(pd, stem, vision, audio, title, passport=passport, lang=lang)
    plots.append({"kind": "track", "name": name, "t": None, "v_ms": None})
    try:
        hname = _html(pd, stem, vision, audio, title, passport=passport, lang=lang)
        plots.append({"kind": "html", "name": hname, "t": None, "v_ms": None})
    except Exception:  # noqa: BLE001 — HTML is optional (depends on plotly)
        pass
    return plots
