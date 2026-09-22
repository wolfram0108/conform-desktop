# -*- coding: utf-8 -*-
"""The MuQ map — a musical SSL embedder (OpenMuQ/MuQ-large-msd-iter, LICENSED).
Track embedding (bf16+batch) → windowed cross-correlation of embeddings → surface →
base.offset_quality → (o,w). build_arr(ref_mono, dub_mono, T) -> (o, w). Sign: rightward=+; frame=41.708ms.
The `muq`+`transformers` packages are imported LAZILY inside _model() — production builds without them.

DURATION LAW: nothing on the GPU grows with the track's length. The embedding of a chunk batch is
trimmed and copied to host memory immediately, and the surface lifts only the window batch it is
correlating; the full-track embedding lives in host memory, like the mono input it comes from.
"""
import time, numpy as np, torch
from ..params import FRAME
from . import base

DEV = "cuda"; SR = 24000; DUR = 1422.0; WIN = 8.0; MAXLAG = 0.7
CH = 20.0; OV = 3.0                       # chunk/overlap (s): small chunks + a large batch
CHS = int(CH*SR); STEPN = int((CH-OV)*SR)  # chunk / hop in samples
# Chunks per forward. The GPU peak is linear in this (measured on MuQ-large: 778 MB at 1, 1181 at 4,
# 2274 at 8, 4476 at 16), while the time is already saturated at 2 (0.50 s per 10 min at 2, 4 and 8,
# against 1.01 s at 1) — a larger batch buys nothing but memory, and the VRAM is shared with the
# other pairs of the queue and with the NVDEC sessions of the decoder.
EMB_BATCH = 4
# Windows per surface batch. Same shape of measurement (62 MB at 4, 125 at 8, 251 at 16, 749 at 48;
# 0.64 s at 4, then 0.49/0.47/0.48 s), and here the surface is bit-identical at any batch, so the
# smallest saturated batch wins. Each window carries WIN seconds of embedding, so the peak is
# bounded by this number, never by the track.
SURF_BATCH = 8

_muq = None
def _model():
    global _muq
    if _muq is None:
        from muq import MuQ
        t = time.time(); _muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(DEV).eval()
        print(f"# MuQ загружен за {time.time()-t:.1f}c")
    return _muq


def _chunks(N):
    """(start, valid_samples) of every chunk of a track of N samples. A chunk shorter than 2 s
    carries no embedding and is skipped."""
    return [(s, min(CHS, N - s)) for s in range(0, max(1, N - int(2*SR)), STEPN)
            if min(CHS, N - s) >= int(2*SR)]


def _spans(meta, Ffull, fps, N):
    """[k0, k1) — the frames each chunk contributes: half of the overlap belongs to the neighbour,
    so every frame of the track is taken exactly once, from the chunk that holds it away from a seam."""
    out = []
    for s, v in meta:
        pos = s/SR; vf = int(round(v/CHS*Ffull)); ft = pos + np.arange(vf)/fps
        lo = pos + (0 if s == 0 else OV/2); hi = pos + v/SR - (0 if (s+v) >= N-1 else OV/2)
        k = np.flatnonzero((ft >= lo) & (ft < hi))       # a contiguous span: ft is increasing
        out.append((int(k[0]), int(k[-1]) + 1) if len(k) else (0, 0))
    return out


@torch.no_grad()
def _embed_arr(x, B=EMB_BATCH):
    """Embed in-memory mono @ SR(24000) (float32). -> (E[Nf,D] float32 in HOST memory, fps)."""
    x = np.ascontiguousarray(x, dtype=np.float32); dur = len(x)/SR; N = len(x)
    meta = _chunks(N)
    m = _model(); E = None; spans = None; fps = None; w = 0
    for i in range(0, len(meta), B):
        blk = meta[i:i+B]
        wav = np.zeros((len(blk), CHS), np.float32)      # a short tail chunk is zero-padded
        for j, (s, v) in enumerate(blk):
            wav[j, :v] = x[s:s+v]
        with torch.autocast("cuda", dtype=torch.bfloat16):       # bf16: no NaN on padding
            out = m(torch.from_numpy(wav).to(DEV))
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        h = torch.nan_to_num(h.float())
        if E is None:                                    # the frame rate is known from the first forward
            Ffull = h.shape[1]; fps = Ffull/CH
            spans = _spans(meta, Ffull, fps, N)
            E = np.empty((sum(b - a for a, b in spans), h.shape[2]), np.float32)
        hc = h.cpu().numpy()                             # one transfer per batch: the GPU holds no history
        for j in range(len(blk)):
            a, b = spans[i+j]
            if b > a:
                E[w:w+b-a] = hc[j, a:b]
                w += b - a
        del h, out
    if E is None:                                        # under one chunk minimum (2 s) there is nothing to embed
        raise ValueError(f"трек короче 2 с ({dur:.2f} с): нечего вкладывать")
    fps = E.shape[0]/dur
    E -= E.mean(0, dtype=np.float64).astype(np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    return E, fps


@torch.no_grad()
def _surface(er, ed, fps, T, B=SURF_BATCH):
    """Windowed cross-correlation of two host-memory embeddings. -> (S[len(T), nlags], lags in frames)."""
    Wf = int(WIN*fps); half = Wf//2; Mf = int(round(MAXLAG*fps)); Nf = min(er.shape[0], ed.shape[0])
    er = er[:Nf]; ed = ed[:Nf]; nf = 1 << int(np.ceil(np.log2(2*Wf)))
    lags = torch.arange(-(Wf-1), Wf, device=DEV); sel = (lags >= -Mf) & (lags <= Mf); lg = lags[sel].float()
    nl = int(sel.sum().item()); S = np.full((len(T), nl), np.nan)
    for i in range(0, len(T), B):
        idx = (T[i:i+B]*fps).round().astype(int); ok = (idx >= half) & (idx < Nf-half-1)
        if not ok.any(): continue
        ci = idx[ok]
        # The windows of one batch overlap heavily (WIN wide, STEP apart), so the batch's whole
        # span is lifted ONCE and cut on the device: the transfer is the span, not the windows.
        lo = int(ci[0]) - half; hi = int(ci[-1]) - half + Wf
        eb = torch.from_numpy(er[lo:hi]).to(DEV); db = torch.from_numpy(ed[lo:hi]).to(DEV)
        off = [int(c) - half - lo for c in ci]
        A = torch.stack([eb[o:o+Wf] for o in off]); D = torch.stack([db[o:o+Wf] for o in off])
        FA = torch.fft.rfft(A, nf, dim=1); FD = torch.fft.rfft(D, nf, dim=1)
        cc = torch.fft.irfft(FD*torch.conj(FA), nf, dim=1); cc = torch.roll(cc, Wf-1, dims=1)[:, :2*Wf-1, :]
        s = cc.sum(2)[:, sel].cpu().numpy(); oi = np.where(ok)[0]; S[i+oi] = s
        del eb, db, A, D, FA, FD, cc
    lagf = (lg/fps*1000.0/FRAME).cpu().numpy()
    return S, lagf


def build_arr(ref_mono, dub_mono, T, diag=False):
    """Over in-memory mono @24 kHz (float32). The reference is embedded on every call: its embedding
    is not cached anywhere.
    diag=True → (o, w, D) with the time×lag surface (muq has no per-band breakdown)."""
    er, fps = _embed_arr(ref_mono); ed, _ = _embed_arr(dub_mono)
    S, lagf = _surface(er, ed, fps, T)
    o, w = base.offset_quality(S, lagf)
    if diag:
        return o, w, {"surf": S.astype(np.float32), "lags_fr": lagf.astype(np.float32)}
    return o, w
