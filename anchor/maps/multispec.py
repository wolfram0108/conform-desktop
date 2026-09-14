# -*- coding: utf-8 -*-
"""Мультиспектр 16 полос (чистый DSP, GPU) — ИЗМЕРИТЕЛЬ ОСТАТКА после сборки.
drift(refz,dubz,T) -> Δ(t) в кадрах (по согласию полос). Знак: правее=+; кадр=41.708мс.
Используется core/assemble для замера остаточного дрейфа corrected↔ref."""
import numpy as np, torch

FRAME = 41.708; SR = 16000
DEV = "cuda" if torch.cuda.is_available() else "cpu"     # GPU-first, CPU-fallback (закон проекта)
NB = 16; NFFT = 1024; HOP = 256; WIN = 5.0; MAXLAG = 0.7
_fb = torch.linspace(0, SR/2, NFFT//2+1)
_edg = torch.logspace(np.log10(50), np.log10(min(14000, SR/2-1)), NB+1)
_BM = torch.zeros(NB, NFFT//2+1)
for _b in range(NB): _BM[_b, (_fb >= _edg[_b]) & (_fb < _edg[_b+1])] = 1.0
_BM = _BM.to(DEV); _WINDOW = torch.hann_window(NFFT).to(DEV)

def _benv(x):
    Z = torch.stft(x, NFFT, HOP, window=_WINDOW, return_complex=True); P = Z.abs()**2
    E = torch.einsum("nf,bft->bnt", _BM, P); E = torch.log1p(E); E = E - E.mean(2, keepdim=True)
    return E / (E.std(2, keepdim=True) + 1e-6)

@torch.no_grad()
def drift(refz, dubz, T, win=WIN, on_prog=None):
    w = int(win*SR); half = w//2; out = np.empty(len(T))
    n = min(int(refz.shape[0]), int(dubz.shape[0]))      # окна не за конец дорожки (короткие)
    for i in range(0, len(T), 256):
        idx = (T[i:i+256]*SR).astype(int)
        c0 = np.clip(idx - half, 0, max(0, n - w))
        A = torch.stack([refz[s:s+w] for s in c0]); B = torch.stack([dubz[s:s+w] for s in c0])
        er = _benv(A); ed = _benv(B); Tf = er.shape[2]; nf = 1 << int(np.ceil(np.log2(2*Tf)))
        Er = torch.fft.rfft(er, nf, dim=2); Ed = torch.fft.rfft(ed, nf, dim=2)
        cc = torch.fft.irfft(Ed*torch.conj(Er), nf, dim=2); cc = torch.roll(cc, Tf-1, dims=2)[:, :, :2*Tf-1]
        s = cc.sum(1); fps = SR/HOP; lags = torch.arange(-(Tf-1), Tf, device=DEV)
        Mf = int(round(MAXLAG*fps)); sel = (lags >= -Mf) & (lags <= Mf)
        ss = s[:, sel]; k = torch.argmax(ss, 1); kk = k.clamp(1, ss.shape[1]-2)
        y0 = ss.gather(1, (kk-1)[:, None]).squeeze(1); y1 = ss.gather(1, kk[:, None]).squeeze(1)
        y2 = ss.gather(1, (kk+1)[:, None]).squeeze(1)
        den = y0 - 2*y1 + y2
        off = torch.where(den.abs() > 1e-9, 0.5*(y0-y2)/den, torch.zeros_like(den)).clamp(-1, 1)
        lg = lags[sel].float(); out[i:i+256] = ((lg[k]+off)/fps*1000.0).cpu().numpy()
        if on_prog is not None: on_prog((i + 256) / len(T))
    return out/FRAME
