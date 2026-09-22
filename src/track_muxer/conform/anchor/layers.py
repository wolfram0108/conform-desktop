"""The audio layout as one map: the resamplings the audio layer applies one after another, composed."""

from __future__ import annotations

import numpy as np


def layered_layout(T, layers, unit_s: float, guard_s: float):
    """Several resamplings applied one after another -> one layout for build_time_map.

    A layer is (curve, breaks, holes): its output at time t plays its input at t + curve(t) * unit_s;
    `breaks` are the times where the curve steps, `holes` the spans it silences, both on the layer's
    own output axis. Layers come outermost first: the first one produces the final timeline, the last
    one reads the source. -> (T, shift in seconds on T, cuts [(tc, dv, t_end, t_nxt)]), all on the final
    timeline. Grid nodes closer than guard_s to a break or inside a hole are left out: the grid
    cannot place a step finer than its cell, and the segment's line is continued to the break instead.
    """
    T = np.asarray(T, float)
    pos = T.copy()                      # where the current layer's output axis is read, per final node
    breaks: list[float] = []
    holes: list[tuple[float, float]] = []

    def to_final(u: float, pos_now: np.ndarray) -> float:
        # Invert pos(t) = u near u: the outer layers move slowly between their own breaks.
        t = float(u)
        for _ in range(3):
            t = float(u) - (float(np.interp(t, T, pos_now)) - t)
        return t

    for curve, layer_breaks, layer_holes in layers:
        breaks += [to_final(b, pos) for b in layer_breaks]
        holes += [(to_final(a, pos), to_final(b, pos)) for a, b in layer_holes]
        pos = pos + np.interp(pos, T, np.asarray(curve, float)) * unit_s
    holes = [(a, b) for a, b in holes if b > a]
    keep = np.ones(len(T), bool)
    for b in breaks:
        keep &= np.abs(T - b) >= guard_s
    for a, b in holes:
        keep &= (T < a - guard_s) | (T > b + guard_s)
    cuts = [(0.5 * (a + b), -1.0, a, b) for a, b in holes]
    cuts += [(b, 1.0, b, b) for b in breaks if not any(a <= b <= e for a, e in holes)]
    return T[keep], (pos - T)[keep], sorted(cuts)
