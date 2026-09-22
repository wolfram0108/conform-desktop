# -*- coding: utf-8 -*-
"""The BAND map — multi-band DSP (no model). Recipe: 16 kHz, 48 bands (nfft 2048),
wmedian aggregation (a weighted median of the per-band shifts instead of a cc sum), quality =
agreement across bands (agree).

API: build_arr(ref_mono, dub_mono, T) -> (o, w), over mono arrays at 16 kHz held in memory.
Sign: rightward=+; frame=41.708ms."""
import numpy as np, torch
from ..params import FRAME

DEV = "cuda" if torch.cuda.is_available() else "cpu"     # GPU-first with a CPU fallback
DUR = 1422.0; MAXLAG = 0.7
NB48 = dict(sr=16000, NB=48, fmin=50.0, fmax=14000.0, nfft=2048, hop=256,
            win=5.0, agg="wmedian", quality="agree", promp=1.0, tol=2.0)


def _bands(NB, fmin, fmax, sr, nfft):
    fb = torch.linspace(0, sr/2, nfft//2+1); hi = min(fmax, sr/2 - 1)
    edg = torch.logspace(np.log10(fmin), np.log10(hi), NB+1)
    BM = torch.zeros(NB, nfft//2+1)
    for b in range(NB): BM[b, (fb >= edg[b]) & (fb < edg[b+1])] = 1.0
    return BM.to(DEV)

_ENV_BLK = 1 << 21          # ~2M samples (~131s at 16k) per chunk: the peak does NOT depend on duration


def _benv(x, BM, nfft, hop, window):
    """A per-band envelope [B, NB, frames], z-normalized over time.

    ⚠ Computed in CHUNKS. A direct `torch.stft` over the whole track builds the full complex
    spectrogram at once: for 30 minutes that is 2.8 GB of GPU memory, for a 90-minute movie
    ~8.8 GB — usage scales linearly with duration, which is forbidden (measured: this is exactly
    where the process's own memory jumped by 2.8 GB). The band convolution compresses 1025
    frequencies down to NB bands right away, so what gets accumulated is already compressed: the
    result takes a few megabytes regardless of length.

    Values match a single-pass computation: chunks are taken with a MARGIN at the edges and then
    trimmed down to frames that rest only on real samples, and normalization (the mean and spread
    over time) is applied at the end — that is, globally, just as a single-pass computation would.
    """
    n = int(x.shape[-1])
    guard = nfft                                   # margin covering the whole frame window
    if n <= _ENV_BLK + 2 * guard:                  # short track: fits in a single chunk
        Z = torch.stft(x, nfft, hop, window=window, return_complex=True); P = Z.abs()**2
        E = torch.einsum("nf,bft->bnt", BM, P)
    else:
        n_frames = n // hop + 1                    # exactly what stft gives with centering
        parts, done = [], 0
        while done < n_frames:
            f0 = done
            f1 = min(n_frames, f0 + _ENV_BLK // hop)
            s0 = max(0, f0 * hop - guard)          # margin on both sides: chunk edges are not used
            s1 = min(n, (f1 - 1) * hop + guard + 1)
            seg = x[..., s0:s1]
            Zs = torch.stft(seg, nfft, hop, window=window, return_complex=True)
            Ps = Zs.abs()**2
            Es = torch.einsum("nf,bft->bnt", BM, Ps)
            del Zs, Ps
            lo = (f0 * hop - s0) // hop            # frame f0 within the chunk
            parts.append(Es[..., lo:lo + (f1 - f0)].clone())
            del Es
            done = f1
        E = torch.cat(parts, dim=2)
        del parts
    E = torch.log1p(E)
    E = E - E.mean(2, keepdim=True)
    return E / (E.std(2, keepdim=True) + 1e-6)

@torch.no_grad()
def track_envelope(x, BM, nfft, hop, window, blk=_ENV_BLK):
    """A whole track's per-band envelope [NB, Nf], z-normalised over time, returned in HOST memory.

    Same recipe as `_benv`, but for a track of any length: the spectrogram is built block by block
    and each block leaves the device at once, so the GPU peak is the block and the result — a few
    tens of megabytes even for a film — lives in host memory, where its callers slice it. The
    statistics of the normalisation are taken over the whole track (float64 on the host), exactly
    as a single-pass computation takes them.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = int(x.shape[-1]); guard = nfft
    n_frames = n // hop + 1                        # exactly what stft gives with centering
    NB = int(BM.shape[0])
    E = np.empty((NB, n_frames), np.float32)
    done = 0
    while done < n_frames:
        f0 = done
        f1 = min(n_frames, f0 + blk // hop)
        s0 = max(0, f0 * hop - guard)              # margin on both sides: chunk edges are not used
        s1 = min(n, (f1 - 1) * hop + guard + 1)
        seg = torch.from_numpy(x[s0:s1]).to(BM.device).unsqueeze(0)
        Z = torch.stft(seg, nfft, hop, window=window, return_complex=True)
        Es = torch.einsum("nf,bft->bnt", BM, Z.abs()**2)[0]
        lo = (f0 * hop - s0) // hop                # frame f0 within the chunk
        E[:, f0:f1] = torch.log1p(Es[:, lo:lo + (f1 - f0)]).cpu().numpy()
        del seg, Z, Es
        done = f1
    mu = E.mean(1, dtype=np.float64, keepdims=True)
    sd = E.std(1, ddof=1, dtype=np.float64, keepdims=True)      # ddof=1, as torch.std defaults to
    return ((E - mu) / (sd + 1e-6)).astype(np.float32)


@torch.no_grad()
def _om_core(refz, dubz, T, *, sr, NB, fmin, fmax, nfft, hop, win, agg, quality, promp, tol,
             diag=False, maxlag=None, on_prog=None):
    """The core: refz/dubz are mono arrays @ sr in HOST memory. -> (o frames, q normalized quality).
    diag=True → an extra third return: raw data for plots/investigation (a time×lag surface,
    the per-band shift/prominence). surf/lags share o's sign convention (Ed·conj(Er): rightward=+).
    maxlag (s) — the lag search WINDOW; None → the standard MAXLAG (0.7). The coarse pass calls it
    with ±2.5s (without mutating the global MAXLAG — thread-safe for the queue)."""
    window = torch.hann_window(nfft).to(DEV); BM = _bands(NB, fmin, fmax, sr, nfft)
    ml = MAXLAG if maxlag is None else maxlag
    w = int(win*sr); half = w//2; fps = sr/hop; Mf = int(round(ml*fps))
    n = min(int(refz.shape[0]), int(dubz.shape[0]))      # windows must not run past the end of the track
    o = np.full(len(T), np.nan); q = np.zeros(len(T))
    _surf, _bsh, _prom_l, _lags_fr = [], [], [], None
    for i in range(0, len(T), 128):
        idx = (T[i:i+128]*sr).astype(int)
        # short track: a center near the end gave a slice < w -> ragged torch.stack (crash). Clamp
        # the window start to [0, n-w] (identity, bit-for-bit, for tracks >= the grid), and nodes that
        # actually run past the edge are zeroed out below (o=nan, q=0: they don't affect the fit).
        c0 = np.clip(idx - half, 0, max(0, n - w))
        inb = (idx - half >= 0) & (idx - half <= n - w)
        # DURATION LAW: the tracks stay in host memory; only the span this batch of windows covers
        # goes to the device (the windows overlap heavily, so the span is far smaller than their sum).
        s0 = int(c0[0]); s1 = int(c0[-1]) + w
        rb = torch.from_numpy(refz[s0:s1]).to(DEV); db = torch.from_numpy(dubz[s0:s1]).to(DEV)
        A = torch.stack([rb[s-s0:s-s0+w] for s in c0])
        B = torch.stack([db[s-s0:s-s0+w] for s in c0])
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
        sh[~inb] = np.nan; ql[~inb] = 0.0                # nodes past the end of the track: no effect
        o[i:i+len(idx)] = sh; q[i:i+len(idx)] = ql
        if on_prog is not None:
            on_prog((i + 128) / len(T))
        if diag:
            bkd = torch.argmax(cc, 2)                          # per-band argmax lag
            _surf.append(cc.sum(1).cpu().numpy().astype(np.float32))            # surface (sum across bands)
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


def build_arr(ref_mono, dub_mono, T, diag=False, maxlag=None, on_prog=None):
    """Over in-memory mono @16 kHz (float32). For production use from conform (no re-reading files).
    diag=True → (o, w, D) with the surface/per-band raw data (see _om_core).
    maxlag (s) — the lag window width; None → 0.7. The coarse pass calls it with 2.5s."""
    refz = np.ascontiguousarray(ref_mono, dtype=np.float32)
    dubz = np.ascontiguousarray(dub_mono, dtype=np.float32)
    return _om_core(refz, dubz, T, **NB48, diag=diag, maxlag=maxlag, on_prog=on_prog)


def band_edges():
    """Edges of the 48 log bands (Hz), NB+1 values — labels for the per-band raw data/cube."""
    hi = min(NB48["fmax"], NB48["sr"]/2 - 1)
    return np.logspace(np.log10(NB48["fmin"]), np.log10(hi), NB48["NB"]+1).astype(np.float32)

