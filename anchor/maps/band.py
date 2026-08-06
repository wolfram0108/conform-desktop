# -*- coding: utf-8 -*-
"""Карта BAND — многополосный DSP (без модели). Рецепт: 16к, 48 полос (nfft 2048),
агрегация wmedian (взвеш. медиана по-полосных сдвигов вместо суммы cc), качество =
согласованность полос (agree).

API:
  build(ref, dub, T)               — по путям-файлам (этап-1 стенд; ffmpeg-загрузка)
  build_arr(ref_mono, dub_mono, T) — по in-memory mono @16к (этап-2 прод)
Знак: правее=+; кадр=41.708мс."""
import numpy as np, torch
from ..params import FRAME
from ..audioio import load_window

DEV = "cuda" if torch.cuda.is_available() else "cpu"     # GPU-first, CPU-fallback (закон проекта)
DUR = 1422.0; MAXLAG = 0.7
NB48 = dict(sr=16000, NB=48, fmin=50.0, fmax=14000.0, nfft=2048, hop=256,
            win=5.0, agg="wmedian", quality="agree", promp=1.0, tol=2.0)

_AUD = {}
def _load(path, sr):
    k = (path, sr)
    if k not in _AUD:
        _AUD[k] = torch.from_numpy(load_window(path, 0, DUR, sr)).to(DEV)
    return _AUD[k]

def _bands(NB, fmin, fmax, sr, nfft):
    fb = torch.linspace(0, sr/2, nfft//2+1); hi = min(fmax, sr/2 - 1)
    edg = torch.logspace(np.log10(fmin), np.log10(hi), NB+1)
    BM = torch.zeros(NB, nfft//2+1)
    for b in range(NB): BM[b, (fb >= edg[b]) & (fb < edg[b+1])] = 1.0
    return BM.to(DEV)

def _benv(x, BM, nfft, hop, window):
    Z = torch.stft(x, nfft, hop, window=window, return_complex=True); P = Z.abs()**2
    E = torch.einsum("nf,bft->bnt", BM, P); E = torch.log1p(E)
    E = E - E.mean(2, keepdim=True)
    return E / (E.std(2, keepdim=True) + 1e-6)

@torch.no_grad()
def _om_core(refz, dubz, T, *, sr, NB, fmin, fmax, nfft, hop, win, agg, quality, promp, tol,
             diag=False, maxlag=None):
    """Ядро: refz/dubz — тензоры mono @ sr на DEV. -> (o кадры, q качество норм.).
    diag=True → доп. третий возврат: сырьё под графики/расследование (поверхность время×лаг,
    по-полосный сдвиг/выраженность). Знак surf/lags как у o (Ed·conj(Er): правее=+).
    maxlag (с) — ОКНО поиска лага; None → штатное MAXLAG (0.7). Грубый проход зовёт с ±2.5с
    (НЕ мутируя глобал MAXLAG — потокобезопасно для очереди)."""
    window = torch.hann_window(nfft).to(DEV); BM = _bands(NB, fmin, fmax, sr, nfft)
    ml = MAXLAG if maxlag is None else maxlag
    w = int(win*sr); half = w//2; fps = sr/hop; Mf = int(round(ml*fps))
    n = min(int(refz.shape[0]), int(dubz.shape[0]))      # окна не должны выходить за конец дорожки
    o = np.full(len(T), np.nan); q = np.zeros(len(T))
    _surf, _bsh, _prom_l, _lags_fr = [], [], [], None
    for i in range(0, len(T), 128):
        idx = (T[i:i+128]*sr).astype(int)
        # короткая дорожка: центр у конца давал срез < w → ragged torch.stack (краш). Старт окна
        # клампим в [0, n−w] (для дорожек ≥ сетки — тождество, бит-в-бит), а узлы, реально
        # вышедшие за край, ниже гасим (o=nan, q=0 → не влияют на укладку).
        c0 = np.clip(idx - half, 0, max(0, n - w))
        inb = (idx - half >= 0) & (idx - half <= n - w)
        A = torch.stack([refz[s:s+w] for s in c0])
        B = torch.stack([dubz[s:s+w] for s in c0])
        er = _benv(A, BM, nfft, hop, window); ed = _benv(B, BM, nfft, hop, window)
        Tf = er.shape[2]; nf = 1 << int(np.ceil(np.log2(2*Tf)))
        Er = torch.fft.rfft(er, nf, dim=2); Ed = torch.fft.rfft(ed, nf, dim=2)
        cc = torch.fft.irfft(Ed*torch.conj(Er), nf, dim=2)
        cc = torch.roll(cc, Tf-1, dims=2)[:, :, :2*Tf-1]
        lags = torch.arange(-(Tf-1), Tf, device=DEV); sel = (lags >= -Mf) & (lags <= Mf)
        cc = cc[:, :, sel]; lg = lags[sel].float(); Ln = cc.shape[2]
        prom = (cc.amax(2) - cc.median(2).values).clamp(min=0)
        if agg == "sum":
            s = cc.sum(1)
        elif agg == "wsum":
            s = (cc * (prom**promp).unsqueeze(2)).sum(1)
        elif agg == "wmedian":
            bk = torch.argmax(cc, 2); bsh = lg[bk]; wgt = prom**promp
            order = torch.argsort(bsh, dim=1)
            bs_s = torch.gather(bsh, 1, order); wg_s = torch.gather(wgt, 1, order)
            cw = torch.cumsum(wg_s, 1); half_w = cw[:, -1:]*0.5
            mi = (cw < half_w).sum(1).clamp(0, NB-1)
            med = bs_s.gather(1, mi[:, None]).squeeze(1)
            s = torch.zeros(cc.shape[0], Ln, device=DEV)
            kk = torch.clamp(torch.searchsorted(lg, med), 0, Ln-1)
            s.scatter_(1, kk[:, None], 1.0)
        else:
            raise ValueError(agg)
        k = torch.argmax(s, 1); kk = k.clamp(1, Ln-2)
        y0 = s.gather(1, (kk-1)[:, None]).squeeze(1); y1 = s.gather(1, kk[:, None]).squeeze(1)
        y2 = s.gather(1, (kk+1)[:, None]).squeeze(1)
        den = y0 - 2*y1 + y2
        off = torch.where(den.abs() > 1e-9, 0.5*(y0-y2)/den, torch.zeros_like(den)).clamp(-1, 1)
        shift = (lg[k] + off) / fps * 1000.0 / FRAME
        if quality == "peak":
            qual = y1.clamp(min=0)
        elif quality == "agree":
            bk = torch.argmax(cc, 2); bshift = lg[bk] / fps * 1000.0 / FRAME
            near = (bshift - shift.unsqueeze(1)).abs() <= tol
            qual = (prom * near).sum(1) / (prom.sum(1) + 1e-9)
        else:
            raise ValueError(quality)
        sh = shift.cpu().numpy(); ql = qual.cpu().numpy()
        sh[~inb] = np.nan; ql[~inb] = 0.0                # узлы за концом дорожки — без влияния
        o[i:i+len(idx)] = sh; q[i:i+len(idx)] = ql
        if diag:
            bkd = torch.argmax(cc, 2)                          # по-полосный аргмакс-лаг
            _surf.append(cc.sum(1).cpu().numpy().astype(np.float32))            # поверхность (сумма полос)
            _bsh.append((lg[bkd]/fps*1000.0/FRAME).cpu().numpy().astype(np.float32))
            _prom_l.append(prom.cpu().numpy().astype(np.float32))
            if _lags_fr is None:
                _lags_fr = (lg/fps*1000.0/FRAME).cpu().numpy().astype(np.float32)
    g = ~np.isnan(o)
    if g.any(): o = np.interp(np.arange(len(o)), np.where(g)[0], o[g])
    if q.max() > 0: q = q/np.median(q[q > 0])
    if diag:
        D = {"surf": np.concatenate(_surf) if _surf else np.zeros((0, 0), np.float32),
             "lags_fr": _lags_fr if _lags_fr is not None else np.zeros(0, np.float32),
             "band_shift": np.concatenate(_bsh) if _bsh else np.zeros((0, 0), np.float32),
             "band_prom": np.concatenate(_prom_l) if _prom_l else np.zeros((0, 0), np.float32),
             "band_edges": band_edges()}
        return o, q, D
    return o, q


def build(ref, dub, T, diag=False, maxlag=None):
    """По путям-файлам (как стенд): ffmpeg-загрузка mono @16к → _om_core."""
    refz = _load(ref, NB48["sr"]); dubz = _load(dub, NB48["sr"])
    return _om_core(refz, dubz, T, **NB48, diag=diag, maxlag=maxlag)


def build_arr(ref_mono, dub_mono, T, diag=False, maxlag=None):
    """По in-memory mono @16к (float32). Для прод-врезки из conform (без перечтения файлов).
    diag=True → (o, w, D) с поверхностью/по-полосным сырьём (см. _om_core).
    maxlag (с) — ширина окна лага; None → 0.7. Грубый проход зовёт с 2.5с."""
    refz = torch.from_numpy(np.ascontiguousarray(ref_mono, dtype=np.float32)).to(DEV)
    dubz = torch.from_numpy(np.ascontiguousarray(dub_mono, dtype=np.float32)).to(DEV)
    return _om_core(refz, dubz, T, **NB48, diag=diag, maxlag=maxlag)


def band_edges():
    """Границы 48 лог-полос (Гц), NB+1 значений — подписи для по-полосного сырья/куба."""
    hi = min(NB48["fmax"], NB48["sr"]/2 - 1)
    return np.logspace(np.log10(NB48["fmin"]), np.log10(hi), NB48["NB"]+1).astype(np.float32)


@torch.no_grad()
def build_cube(ref_mono, dub_mono, T, maxlag_s=2.5):
    """ПОЛНЫЙ 3D-куб band для зоны: окна T × 48 полос × лаги ±maxlag_s (по запросу, НЕ
    авто-дамп — на всю длину ~ГБ). Лаг шире штатного MAXLAG=0.7 (видно, что истинный пик
    вне рабочего окна band). Знак как у o (Ed·conj(Er): правее=+).
    -> (cube[len(T),48,Ln] f32, lags_fr[Ln], edges[49])."""
    sr = NB48["sr"]; NB = NB48["NB"]; nfft = NB48["nfft"]; hop = NB48["hop"]; win = NB48["win"]
    refz = torch.from_numpy(np.ascontiguousarray(ref_mono, dtype=np.float32)).to(DEV)
    dubz = torch.from_numpy(np.ascontiguousarray(dub_mono, dtype=np.float32)).to(DEV)
    window = torch.hann_window(nfft).to(DEV); BM = _bands(NB, NB48["fmin"], NB48["fmax"], sr, nfft)
    w = int(win*sr); half = w//2; fps = sr/hop; Mf = int(round(maxlag_s*fps))
    n = min(int(refz.shape[0]), int(dubz.shape[0]))      # окна не за конец дорожки (короткие)
    cubes = []; lags_fr = None
    Tn = np.asarray(T, dtype=np.float64)
    for i in range(0, len(Tn), 64):
        idx = (Tn[i:i+64]*sr).astype(int)
        c0 = np.clip(idx - half, 0, max(0, n - w))
        A = torch.stack([refz[s:s+w] for s in c0])
        B = torch.stack([dubz[s:s+w] for s in c0])
        er = _benv(A, BM, nfft, hop, window); ed = _benv(B, BM, nfft, hop, window)
        Tf = er.shape[2]; nf = 1 << int(np.ceil(np.log2(2*Tf)))
        Er = torch.fft.rfft(er, nf, dim=2); Ed = torch.fft.rfft(ed, nf, dim=2)
        cc = torch.fft.irfft(Ed*torch.conj(Er), nf, dim=2)
        cc = torch.roll(cc, Tf-1, dims=2)[:, :, :2*Tf-1]
        lags = torch.arange(-(Tf-1), Tf, device=DEV); sel = (lags >= -Mf) & (lags <= Mf)
        cc = cc[:, :, sel]
        if lags_fr is None:
            lags_fr = (lags[sel].float()/fps*1000.0/FRAME).cpu().numpy().astype(np.float32)
        cubes.append(cc.cpu().numpy().astype(np.float32))
    cube = np.concatenate(cubes) if cubes else np.zeros((0, NB, 0), np.float32)
    return cube, (lags_fr if lags_fr is not None else np.zeros(0, np.float32)), band_edges()
