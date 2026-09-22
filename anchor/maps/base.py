# -*- coding: utf-8 -*-
"""Shared by the maps: a correlation surface -> (o, w). o[t] is the shift in frames, w[t] the quality
of the anchor, normalised by its median."""
import numpy as np


def offset_quality(S, lagf):
    """Surface S[t,lag] -> o[t] (argmax plus parabolic interpolation), w[t] (peak height, normalized)."""
    n = len(S); o = np.full(n, np.nan); w = np.zeros(n)
    for i in range(n):
        row = S[i]
        if np.any(np.isnan(row)): continue
        k = int(np.argmax(row)); pk = row[k]
        if 0 < k < len(row)-1:
            y0, y1, y2 = row[k-1], row[k], row[k+1]; den = y0-2*y1+y2
            off = 0.5*(y0-y2)/den if abs(den) > 1e-9 else 0.0
        else:
            off = 0.0
        kk = np.clip(k+off, 0, len(lagf)-1)
        o[i] = np.interp(kk, np.arange(len(lagf)), lagf); w[i] = max(pk, 0.0)
    g = ~np.isnan(o)
    if g.any(): o = np.interp(np.arange(n), np.where(g)[0], o[g])
    if w.max() > 0: w = w/np.median(w[w > 0])
    return o, w

