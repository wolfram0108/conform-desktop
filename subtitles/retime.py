"""Transfer subtitle cues from a source timeline onto the reference timeline.

The time map is the one conform lays the audio by: between cuts the source plays
continuously, at a cut the reference either has content the source lacks (a hole)
or the source has content the reference lacks (dropped). Cues follow the same law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from track_muxer.conform.subtitles.vtt import Cue


@dataclass(frozen=True)
class TimeMap:
    """Linear pieces of the map: reference span [t0, t1] shows source span [x0, x1], x1 > x0."""

    t0: np.ndarray
    t1: np.ndarray
    x0: np.ndarray
    x1: np.ndarray
    # One reference frame, seconds: a cue piece shorter than this can never be displayed.
    frame_s: float

    def __len__(self) -> int:
        return len(self.t0)


def _shift_at(t: float, tn: np.ndarray, sn: np.ndarray, hold_head: bool, hold_tail: bool) -> float:
    # Towards a cut the segment's fitted line continues; clamping would bend it at the cut.
    # At the ends of the timeline the audio is laid with the shift held, and cues follow the audio.
    if len(tn) > 1 and t < tn[0] and not hold_head:
        return float(sn[0] + (sn[1] - sn[0]) / (tn[1] - tn[0]) * (t - tn[0]))
    if len(tn) > 1 and t > tn[-1] and not hold_tail:
        return float(sn[-1] + (sn[-1] - sn[-2]) / (tn[-1] - tn[-2]) * (t - tn[-1]))
    return float(np.interp(t, tn, sn))


def build_time_map(T, curve, cuts, fps_ref: float, dur_ref: float) -> TimeMap:
    """Vision layout (shift in reference frames on grid T, cuts [(tc, dv, t_end, t_nxt)]) -> TimeMap.

    Segments are split exactly where the layout splits them; a negative cut leaves the hole
    [t_end, t_nxt] the audio is silenced in, a positive cut joins the two sides at tc.
    """
    T = np.asarray(T, float)
    curve = np.asarray(curve, float)
    cuts = sorted(cuts, key=lambda c: c[0])
    bnds = [-np.inf] + [float(c[0]) for c in cuts] + [np.inf]
    t0: list[float] = []
    t1: list[float] = []
    x0: list[float] = []
    x1: list[float] = []
    for k in range(len(bnds) - 1):
        m = (T > bnds[k]) & (T <= bnds[k + 1])
        if not m.any():
            continue
        tn, sn = T[m], curve[m]
        a = 0.0 if k == 0 else float(cuts[k - 1][3] if cuts[k - 1][1] < 0 else cuts[k - 1][0])
        b = float(dur_ref) if k == len(bnds) - 2 else \
            float(cuts[k][2] if cuts[k][1] < 0 else cuts[k][0])
        if b <= a:
            continue
        knots = np.concatenate(([a], tn[(tn > a) & (tn < b)], [b]))
        head, tail = k == 0, k == len(bnds) - 2
        xs = np.array([t + _shift_at(t, tn, sn, head, tail) / fps_ref for t in knots])
        for i in range(len(knots) - 1):
            # A span the source does not advance through shows nothing new; it carries no cues.
            if xs[i + 1] > xs[i]:
                t0.append(knots[i]); t1.append(knots[i + 1])
                x0.append(xs[i]); x1.append(xs[i + 1])
    return TimeMap(np.array(t0), np.array(t1), np.array(x0), np.array(x1), 1.0 / fps_ref)


def retime_cues(cues, tmap: TimeMap, dur_src: float) -> list[Cue]:
    """Cues on the source timeline -> cues on the reference timeline, ordered by start."""
    out: list[tuple[float, int, Cue]] = []
    for n, cue in enumerate(cues):
        s, e = max(0.0, cue.start), min(float(dur_src), cue.end)
        if e <= s or not len(tmap):
            continue
        lo = np.maximum(tmap.x0, s)
        hi = np.minimum(tmap.x1, e)
        idx = np.where(hi > lo)[0]
        if not len(idx):
            continue
        k = (tmap.t1[idx] - tmap.t0[idx]) / (tmap.x1[idx] - tmap.x0[idx])
        a = tmap.t0[idx] + (lo[idx] - tmap.x0[idx]) * k
        b = tmap.t0[idx] + (hi[idx] - tmap.x0[idx]) * k
        order = np.argsort(a, kind="stable")
        spans: list[list[float]] = []
        for i in order:
            # Pieces of one cue that touch are the same appearance on screen.
            if spans and a[i] - spans[-1][1] <= 1e-6:
                spans[-1][1] = max(spans[-1][1], float(b[i]))
            else:
                spans.append([float(a[i]), float(b[i])])
        for a_s, b_s in spans:
            if b_s - a_s >= tmap.frame_s:
                out.append((a_s, n, cue.at(a_s, b_s)))
    out.sort(key=lambda r: (r[0], r[1]))
    return [c for _, _, c in out]
