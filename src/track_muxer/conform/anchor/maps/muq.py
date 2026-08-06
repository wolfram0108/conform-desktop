# -*- coding: utf-8 -*-
"""Карта MuQ — музыкальный SSL-эмбеддер (OpenMuQ/MuQ-large-msd-iter, ЛИЦЕНЗИРОВАН).
Эмбеддинг трека (bf16+батч) → оконная кросс-корреляция эмбеддингов → поверхность →
base.offset_quality → (o,w). build(ref,dub,T) -> (o,w). Знак: правее=+; кадр=41.708мс.
Пакет `muq`+`transformers` импортируются ЛЕНИВО внутри _model() — прод без них собирается."""
import time, numpy as np, torch
from ..params import FRAME
from ..audioio import load_window
from . import base

DEV = "cuda"; SR = 24000; DUR = 1422.0; WIN = 8.0; MAXLAG = 0.7
CH = 20.0; OV = 3.0                       # чанк/перекрытие (с): мелкие чанки + большой батч

_muq = None
def _model():
    global _muq
    if _muq is None:
        from muq import MuQ
        t = time.time(); _muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(DEV).eval()
        print(f"# MuQ загружен за {time.time()-t:.1f}c")
    return _muq

@torch.no_grad()
def _embed(path, B=16):
    return _embed_arr(load_window(path, 0, DUR, SR), B)

@torch.no_grad()
def _embed_arr(x, B=16):
    """Эмбеддинг in-memory mono @ SR(24000) (float32). -> (E[Nf,D] cuda, fps)."""
    x = np.ascontiguousarray(x, dtype=np.float32); dur = len(x)/SR; N = len(x)
    CHs = int(CH*SR); stepn = int((CH-OV)*SR)
    starts = list(range(0, max(1, N-int(2*SR)), stepn)); chunks = []; meta = []
    for s in starts:
        seg = x[s:s+CHs]; v = len(seg)
        if v < int(2*SR): continue
        if v < CHs: seg = np.pad(seg, (0, CHs-v))
        chunks.append(seg); meta.append((s, v))
    m = _model(); arr = np.stack(chunks); feats = []
    for i in range(0, len(arr), B):
        wav = torch.from_numpy(arr[i:i+B]).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):       # bf16: без NaN на паддинге
            out = m(wav); h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        feats.append(torch.nan_to_num(h.float()))
    H = torch.cat(feats, 0); Ffull = H.shape[1]; fps = Ffull/CH; embs = []
    for idx, (s, v) in enumerate(meta):
        pos = s/SR; vf = int(round(v/CHs*Ffull)); h = H[idx, :vf]; ft = pos + np.arange(vf)/fps
        lo = pos + (0 if s == 0 else OV/2); hi = pos + v/SR - (0 if (s+v) >= N-1 else OV/2)
        keep = (ft >= lo) & (ft < hi); embs.append(h[keep])
    E = torch.cat(embs, 0); fps = E.shape[0]/dur
    E = E - E.mean(0, keepdim=True); E = E / (E.norm(dim=1, keepdim=True) + 1e-8)
    return E, fps

_REF = {}
def _ref_emb(path):
    if path not in _REF: _REF[path] = _embed(path)
    return _REF[path]

@torch.no_grad()
def _surface(er, ed, fps, T, B=48):
    Wf = int(WIN*fps); half = Wf//2; Mf = int(round(MAXLAG*fps)); Nf = min(er.shape[0], ed.shape[0])
    er = er[:Nf]; ed = ed[:Nf]; nf = 1 << int(np.ceil(np.log2(2*Wf)))
    lags = torch.arange(-(Wf-1), Wf, device=DEV); sel = (lags >= -Mf) & (lags <= Mf); lg = lags[sel].float()
    nl = int(sel.sum().item()); S = np.full((len(T), nl), np.nan)
    for i in range(0, len(T), B):
        idx = (T[i:i+B]*fps).round().astype(int); ok = (idx >= half) & (idx < Nf-half-1)
        if not ok.any(): continue
        ci = idx[ok]
        A = torch.stack([er[c-half:c-half+Wf] for c in ci]); D = torch.stack([ed[c-half:c-half+Wf] for c in ci])
        FA = torch.fft.rfft(A, nf, dim=1); FD = torch.fft.rfft(D, nf, dim=1)
        cc = torch.fft.irfft(FD*torch.conj(FA), nf, dim=1); cc = torch.roll(cc, Wf-1, dims=1)[:, :2*Wf-1, :]
        s = cc.sum(2)[:, sel].cpu().numpy(); oi = np.where(ok)[0]; S[i+oi] = s
    lagf = (lg/fps*1000.0/FRAME).cpu().numpy()
    return S, lagf

def build(ref, dub, T, diag=False):
    er, fps = _ref_emb(ref); ed, _ = _embed(dub)
    S, lagf = _surface(er, ed, fps, T)
    o, w = base.offset_quality(S, lagf)
    if diag:
        return o, w, {"surf": S.astype(np.float32), "lags_fr": lagf.astype(np.float32)}
    return o, w


def build_arr(ref_mono, dub_mono, T, diag=False):
    """По in-memory mono @24к (float32). Реф эмбеддится каждый вызов (кэш серии — на этапе 3).
    diag=True → (o, w, D) с поверхностью время×лаг (по-полосного у muq нет)."""
    er, fps = _embed_arr(ref_mono); ed, _ = _embed_arr(dub_mono)
    S, lagf = _surface(er, ed, fps, T)
    o, w = base.offset_quality(S, lagf)
    if diag:
        return o, w, {"surf": S.astype(np.float32), "lags_fr": lagf.astype(np.float32)}
    return o, w
