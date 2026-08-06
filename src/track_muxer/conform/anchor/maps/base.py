# -*- coding: utf-8 -*-
"""Общее для карт: поверхность корреляции → (o,w) и опциональный файловый кеш.
o[t] — сдвиг (кадры), w[t] — качество якоря (норм. по медиане). Кеш по методу+хэшу пути."""
import os, hashlib, numpy as np
from ..params import T as _T


def offset_quality(S, lagf):
    """Поверхность S[t,lag] → o[t] (argmax+парабола), w[t] (высота пика, норм.)."""
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


def _key(method, dub):
    h = hashlib.md5(os.path.abspath(str(dub)).encode()).hexdigest()[:6]
    return f"{method}_{os.path.splitext(os.path.basename(str(dub)))[0]}_{h}_n{len(_T)}"


def cached(method, dub, build_fn, cache_dir=None):
    """(o,w) из кеша или построить build_fn()->(o,w). cache_dir=None → без файлового кеша."""
    if cache_dir is None:
        return build_fn()
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f"om_{_key(method, dub)}.npz")
    if os.path.exists(p):
        d = np.load(p); return d["o"], d["w"]
    o, w = build_fn(); np.savez(p, o=o, w=w); return o, w
