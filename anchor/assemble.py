# -*- coding: utf-8 -*-
"""ОБЩАЯ сборка (одна на оба метода): freeze-схлоп + варп ОБОИХ каналов на ТОЧНУЮ
длину рефа по гладкой сегментной кривой (наклон ≤SMAX ≈ ≤2%). Остаток мерится
независимым мультиспектром. Знак: правее=+; кадр=41.708мс.

Порт-этап 1: тракт по файлам (как в стенде). `write_output=False` пропускает запись
большого 44.1к аудиофайла (только метрики/остаток) — для быстрой регрессии без I/O на NAS."""
import os, json, subprocess, wave, time, numpy as np, torch
from .params import T as _DEFT, FRAME, make_T
from .audioio import load_window
from .maps import multispec

# ---------- зависшие семплы (freeze) ----------
def detect_freezes(dub, vthr=1e-4, min_ms=80.0):
    """Стук-семплы (нативно 44.1к): прогоны равных НЕнулевых сэмплов -> [(t_сек, df_кадры)]."""
    x = load_window(dub, 0, _dur(dub), 44100); sr = 44100
    d = np.diff(x); z = np.r_[False, d == 0]; minn = int(min_ms/1000*sr); i = 0; n = len(x); out = []
    while i < n:
        if z[i]:
            j = i
            while j < n and z[j]: j += 1
            if (j-i+1) >= minn and abs(x[i]) >= vthr: out.append((i/sr, (j-i+1)/sr*1000/FRAME))
            i = j
        else:
            i += 1
    return out

# ---------- варп / сборка ----------
def _warp_at(t, seglines, freezes):
    bnds = np.array([s[0] for s in seglines[1:]]); idx = np.searchsorted(bnds, t, side="right")
    a = np.array([s[2] for s in seglines]); b = np.array([s[3] for s in seglines])
    off = a[idx]*t + b[idx]
    for tf, df in freezes: off = off - df*(t < tf)   # держит синхрон СЕРЕДИНЫ (остаток 0.05); артефакт начала — отд. задача
    return off

def apply_warp(dub_ch, sr, seglines, n_out, freezes=()):
    t_out = np.arange(int(n_out))/sr; dlt = _warp_at(t_out, seglines, freezes)*FRAME/1000.0
    src = (t_out+dlt)*sr
    return np.interp(src, np.arange(len(dub_ch)), dub_ch).astype(np.float32)

# ---------- утилиты ----------
def _dur(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(p)],
                       stdout=subprocess.PIPE)
    return float(json.loads(r.stdout)["format"]["duration"])

def _load_stereo(path, sr):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ar", str(sr), "-f", "f32le", "-"]
    a = np.frombuffer(subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout, np.float32)
    ch = 2 if a.size % 2 == 0 else 1; return a.reshape(-1, ch)

def _wwrite(path, x, sr):
    """Устойчивая запись: временный файл → atomic replace с ретраями."""
    if x.ndim == 1: x = x[:, None]
    data = (np.clip(x, -1, 1)*32767).astype("<i2").tobytes(); tmp = str(path)+".tmp"
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(x.shape[1]); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(data)
    for k in range(8):
        try:
            os.replace(tmp, path); return
        except PermissionError:
            if k == 7: raise
            time.sleep(0.5)

def measure_resid(ref16, corr16, T=_DEFT):
    rz = torch.from_numpy(ref16).to(multispec.DEV); dz = torch.from_numpy(corr16).to(multispec.DEV)
    return multispec.drift(rz, dz, T)

# ---------- ПУБЛИЧНАЯ сборка ----------
def build(ref, dub, out_path, seglines, sr=44100, write_output=True):
    """Пишет стерео corrected (длина=рефу) по seglines + freeze. -> метрики + остаток.
    write_output=False → пропустить запись 44.1к аудио (остаток считается всё равно)."""
    fz = detect_freezes(dub)
    DUR = _dur(ref); n_out = int(round(DUR*sr))
    ch = 2; len_s = n_out/sr
    if write_output:
        st = _load_stereo(dub, sr)
        corr = np.stack([apply_warp(st[:, c], sr, seglines, n_out, freezes=fz)
                         for c in range(st.shape[1])], axis=1)
        _wwrite(out_path, corr, sr); ch = corr.shape[1]; len_s = corr.shape[0]/sr
    ref16 = load_window(ref, 0, DUR, 16000); dub16 = load_window(dub, 0, DUR, 16000)
    corr16 = apply_warp(dub16, 16000, seglines, len(ref16), freezes=fz)
    T = make_T(DUR)
    aft = measure_resid(ref16, corr16, T=T); m = (T >= 30)
    return dict(freezes=fz, resid_med=float(np.median(np.abs(aft[m]))),
                resid_max=float(np.max(np.abs(aft[m]))), ch=ch,
                len_s=len_s, ref_len_s=DUR)
