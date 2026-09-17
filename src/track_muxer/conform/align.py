"""Одна ПАРА ref↔dub → выровненный FLAC (conform).

Конвейер (единый путь, без legacy-тумблеров):
  SRM-матчинг (coarse_robust → band_align, free_start) → pred;
  _level_decide: разбор правок по СДВИГУ УРОВНЯ offset (восстановление слепых drop-syn, вырезы);
  ВИДЕО-карта зрения (vision_detect.build_curve → tg_s, БЕЗ монотонизации) — ЕДИНСТВЕННЫЙ
  источник структуры; ресэмпл аудио на сетку REF; ТИШИНА в вырезах [t_end,t_nxt];
  аудио-слой band/muq (anchor) ПОВЕРХ (не строит якоря в тишине зрения); ПОСЛЕ —
  fill_silence заполняет тишину озвучки синхронным рефом; запись FLAC l12.
Дополнительно (НЕ влияет на wav) — паспорт качества (PairResult).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, medfilt, resample_poly, sosfiltfilt

from track_muxer.conform import cache as cache_mod
from track_muxer.conform.config import FFMPEG, FFPROBE
from track_muxer.conform import procreg
from track_muxer.conform import tmpfiles
from track_muxer.conform.memlog import memlog
from track_muxer.conform.decode_backend import decode_backend
from track_muxer.conform.interp_backend import warp_interp        # GPU/CPU audio warp (resample onto the reference grid)
from track_muxer.conform.features import (
    build_srm,
    probe_audio_channels,
    probe_av_delay,
    av_delay_filters,
    probe_duration,
    probe_has_video,
    probe_resolution,
)
from track_muxer.conform.kernel.band_align import band_align
from track_muxer.conform.kernel.coarse import coarse_robust, coarse_windowed
from track_muxer.conform.kernel.orient import mirror_srm, orientation_probe
from track_muxer.conform.models import PairResult, SrmFeatures
from track_muxer.conform.progress import Reporter, part
from track_muxer.conform.trace import Trace
from track_muxer.conform.vision_detect import build_map as _vision_build_map, global_trend as _vision_global_trend, vision_ow as _vision_ow, detelecine as _detelecine, is_baked_telecine as _is_telecine
from track_muxer.conform.anchor.params import FRAME as VFRAME, T as VGT, make_T as _make_T  # ms per frame + anchor grid T (cut silence, unified plot)

# conform v8 constants: algorithm calibration, do not change
SR = 44100
DT = 0.005
ABORT_ASSIGNED_PCT = 60.0  # assigned% below this means a foreign video (dub of another episode):
                           # fail the pair right after matching, before resampling and GPU audio
AUDIO_RESID_MAX_MS = 80.0       # project criterion: a place is out of sync beyond ±80 ms (2 frames)
AUDIO_COVERAGE_BLIND = 0.15     # below: the files share almost no sound, the layer measured nothing
AUDIO_COVERAGE_LOW = 0.5        # below: the layer had support on less than half of the track
AUDIO_EXCESS_WARN_MS = 2000.0   # cut movement that cancelled out (Σ|steps| − |net|): the layer went back and forth
AUDIO_EXCESS_CRIT_MS = 10000.0  # population: healthy ≤ 0.4 s (95th pct), blind chase ≥ 12 s; red above this
FREEZE_MIN_S = 10.0             # identical frames this long are a frozen picture (broken encode), not a scene
FREEZE_COS = 0.999              # SRM cosine of frames ~0.5 s apart that only identical decoded frames reach
FREEZE_GAP_S = 1.0              # a frozen run survives cadence dips shorter than this
MIRROR_RATIO = 2.0         # the mirrored frame sample must beat the plain one by this factor to switch orientation
GEOM_GATE = 50             # coarse pass found < N anchors: vision is blind (crop/zoom/anamorph/bars),
                           # so run the geometry pass before the foreign-video cutoff
COARSE_SAME_MIN_FRAC = 0.10  # share of thinned (K=8) frames in the monotone coarse chain that still
                           # means the same episode despite low assigned%
CUT_MIN_S = 0.3          # a cut is longer than 0.3 s
FADE = int(0.010 * SR)
FILL_MIN_S = 1.0         # cuts longer than 1 s are filled with the reference, shorter ones get silence
XFADE = int(0.030 * SR)  # crossfade dub<->original at fill seams (30 ms)
# Final fill of dub silence with the reference (after band/muq, once the track is in sync):
SIL_FILL_DB = -90.0      # dub silence threshold, dBFS: below -80 is the plateau of real gaps, above is
                         # quiet content; -90 sits mid-plateau, far from the -40..-60 edge
SIL_FILL_WIN_S = 0.02    # RMS window (20 ms)
SIL_FILL_MIN_S = 0.15    # shortest silence zone; shorter dips are left alone
EDGE_MIN_FR = 24         # edge re-pass only if more than ~1 s is dropped or uncovered
EDGE_COS_MIN = 0.5       # a recovered edge match is kept only above this cosine
DSYN = 0.30
MATCH_THR = 0.30
# level_edits layer: edits are decided by the offset LEVEL shift instead of restore_blind+R
LEVEL_EDIT_S = 1.0       # offset level shift above this is a real edit, otherwise a blind zone
LEVEL_MIN_INS_S = 2.0    # an insert is a solid dub block longer than this, shorter is jitter
LEVEL_WIN_S = 4.0        # median window of the offset level before/after a dropped run
# Level-aware cut detection: a transient offset outlier on static content is not a cut
LEVEL_CUT_COALESCE_S = 3.0  # reference gaps closer than this merge into one cluster
LEVEL_CUT_RECOVER_S = 8.0   # в пределах этого ищем ВОЗВРАТ уровня (выброс-пила) vs устойчивый сдвиг (реальный вырез)
# Creep detector: a zone of ASSIGNED frames whose content is foreign to the reference.
# Thresholds sit in the gap measured on 8 pairs: healthy zones have cos>=0.22 and last <=1.0 s,
# the defect has cos<=0.02 and lasts 11.3 s; both keep a >x1.5 margin to either side.
CREEP_COS = 0.15         # медиана cos присвоенной зоны ниже → кадры чужие
CREEP_MIN_S = 3.0        # зона длиннее этого → выбросить (короче не трогаем; арбитр — _level_decide)
CREEP_BRIDGE_S = 1.0     # мостик через уже выброшенные кадры внутри зоны


def _extract_base(video: Path, ffmpeg: str, audio_fix: bool = True, channels: int = 2,
                  atrack: int = 0, delay_s: float = 0.0) -> list[str]:
    """Общие аргументы декода аудио → PCM s16le @ SR, `channels` каналов. Любой кодек входа
    (AAC/FLAC/AC3/PCM) ffmpeg декодирует; `-ar SR` ресэмплит ТОЛЬКО при несовпадении частот
    (44.1 на входе → no-op). `-ac channels`: для рефа/анализа = 2 (даунмикс), для ВЫХОДНОГО
    дубля = его родное число каналов (2.0/5.1/7.1 → no-op remix, раскладка сохраняется).

    ⭐ `aresample=async=1:first_pts=0` применяется ВСЕГДА (2026-08-06). Раньше это был тумблер
    `audio_fix` (выкл. по умолчанию) — тумблеров в проде быть не должно, решение принимает
    алгоритм по данным. Обоснование замерами:
      • ЛЕЧИТ дыры PTS в аудио. Контейнерный гэп — это разрыв в метках, а НЕ тишина
        (`silencedetect` его не видит): при дефолтном декоде PCM склеивается подряд и ВЕСЬ
        последующий звук съезжает. Стенд `dub_51_audio_gap_pts` (дыра 3.02с): без фильтра
        6 окон из 11 вне ±80мс (макс 3310мс) → с фильтром **0 из 13**.
      • БЕЗОПАСЕН: на чистом входе бит-в-бит no-op (md5 совпал на синтетике И на реальной паре
        из библиотеки случаев — выход идентичен, resid 1.1489566 в обоих прогонах); на стенде из
        10 дублей не изменил НИ ОДНОГО остального кейса.
      • `async=1` = только filling/trimming, БЕЗ растяжения (замер: тоны 440/880 Гц сохранены
        точно; растяжение начинается при async>1).
    Параметр `audio_fix` сохранён в сигнатуре для совместимости вызовов (CLI/API/queue), но на
    команду больше НЕ влияет — подлежит удалению из API и UI отдельным шагом.

    atrack — индекс аудиодорожки файла (многодорожечность, этап 5.1 standalone); 0 = как было.

    delay_s — container delay video_start − audio_start (features.probe_av_delay): the audio is
    laid on the VIDEO axis because SRM takes frame time from index 0. 0 → command unchanged."""
    # Delay filters go before aresample: first_pts=0 would otherwise refill the trimmed head.
    af = ",".join([*av_delay_filters(delay_s), "aresample=async=1:first_pts=0"])
    return [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-map", f"0:a:{int(atrack)}", "-af", af,
            "-ac", str(channels), "-ar", str(SR), "-f", "s16le"]


def _pump_progress(stream, dur, reporter) -> None:
    """Feed ffmpeg `-progress` lines (out_time_us) into the stage reporter; with no reporter
    just drain the pipe so ffmpeg never blocks on a full buffer. Accepts bytes and str."""
    for raw in stream:
        line = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw
        if reporter is not None and dur > 0 and line.startswith("out_time_us="):
            try:
                us = int(line.split("=", 1)[1])
            except ValueError:
                continue
            reporter(min(us / 1e6 / dur, 0.999))


def _decode_audio(video: Path, ffmpeg: str, audio_fix: bool, *, channels: int = 2,
                 atrack: int = 0, reporter: Reporter | None = None,
                 delay_s: float | None = None) -> np.ndarray:
    """Декод аудио (`channels` каналов) в float32 (ЦЕЛИКОМ в RAM) ПРЯМО из пайпа ffmpeg — без
    временного файла на диске. PCM s16le → stdout; прогресс — со stderr (`-progress pipe:2`),
    читается в отдельном потоке (иначе блокировка при заполнении любого из пайпов).
    Возврат (N, channels) float32 (int16-размах).
    delay_s=None → the container delay is probed here; callers that trace it pass it in."""
    if delay_s is None:
        delay_s = probe_av_delay(video, FFPROBE, atrack=atrack)
    dur = (probe_duration(video, FFPROBE) or 0.0) if reporter is not None else 0.0
    cmd = [*_extract_base(video, ffmpeg, audio_fix, channels, atrack, delay_s),
           "-progress", "pipe:2", "-nostats", "pipe:1"]
    proc = procreg.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    th = threading.Thread(target=_pump_progress, args=(proc.stderr, dur, reporter), daemon=True)
    th.start()
    buf = proc.stdout.read()
    proc.wait(); procreg.done(proc); th.join(timeout=2)
    if proc.returncode not in (0, None):
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    a = np.frombuffer(buf, np.int16)
    return a[: (a.size // channels) * channels].reshape(-1, channels).astype(np.float32)


def _decode_audio_mmap(video: Path, ffmpeg: str, audio_fix: bool, *, channels: int = 2,
                      atrack: int = 0, reporter: Reporter | None = None,
                      dest: Path | None = None, delay_s: float | None = None):
    """low_mem: ffmpeg декодирует PCM s16le (`channels` каналов) ПРЯМО в рабочий файл `a.raw`
    (без WAV-обёртки и без перечитывания), затем `np.memmap` поверх — RAM не растёт на длинных
    дорожках. Файл — рабочий буфер (НЕ throwaway), удаляется вызывающим (rmtree каталога). PCM
    в файл → stdout свободен для `-progress pipe:1`. Значения бит-в-бит как у _decode_audio.
    Возврат (int16-memmap [N,channels], каталог_на_удаление|None).

    dest задан (CK2) → писать raw ПРЯМО в него (кэш аудио), tmp-каталог не создаём, второй
    элемент = None (удалять нечего, raw переживает запуск)."""
    if dest is not None:
        dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
        raw = dest; d = None
    else:
        tmpdir = Path(video).parent / "_tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        d = Path(tempfile.mkdtemp(prefix="extract_", dir=str(tmpdir)))
        raw = d / "a.raw"
    if delay_s is None:
        delay_s = probe_av_delay(video, FFPROBE, atrack=atrack)
    dur = (probe_duration(video, FFPROBE) or 0.0) if reporter is not None else 0.0
    cmd = [*_extract_base(video, ffmpeg, audio_fix, channels, atrack, delay_s),
           "-progress", "pipe:1", "-nostats", str(raw)]
    proc = procreg.popen(cmd, stdout=subprocess.PIPE, text=True)
    _pump_progress(proc.stdout, dur, reporter)   # дренит stdout до конца
    proc.wait(); procreg.done(proc)
    if proc.returncode not in (0, None):
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    a = np.memmap(raw, dtype=np.int16, mode="r")
    return a[: (a.size // channels) * channels].reshape(-1, channels), d


def _write_audio_streamed(path: Path, out, ffmpeg: str, *, layout: str | None = None, on_prog=None) -> None:
    """Многоканальный int16 → FLAC (lossless, level 8) ПОТОКОМ через ffmpeg, БЕЗ промежуточного
    WAV: int16 кусками в stdin (`-f s16le`) → `-c:a flac -compression_level 8`.
    out — (N,C) float32 (memmap или ndarray), весь массив в RAM не держим; число каналов C
    берётся из out. layout (если задан) наследует раскладку дубля (5.1/7.1) — корректный тег
    каналов на выходе. Сэмплы бит-в-бит к прежнему WAV (та же int16-конверсия), FLAC
    декодируется в них без потерь.

    Уровень 8 (не 12): кодирование ×6 быстрее при +~1.6% размера и ТОЙ ЖЕ декодированной дорожке
    (FLAC lossless: PCM бит-в-бит независимо от уровня — уровень влияет только на скорость/размер)."""
    ch = out.shape[1] if out.ndim == 2 else 1
    lay = ["-channel_layout", layout] if layout else []
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "s16le", "-ar", str(SR), "-ac", str(ch), *lay, "-i", "pipe:0",
           "-c:a", "flac", "-compression_level", "8", str(path)]
    # stderr перехватываем: если приёмник умрёт посреди записи, запись в его канал даёт
    # «Broken pipe» — сообщение БЕЗ причины. Настоящая причина лежит в stderr ffmpeg,
    # и без перехвата она терялась насовсем (случай 2026-08-07: полтора часа работы → пусто).
    n = len(out)
    logger.info("запись {}: {} сэмплов × {} кан. ({:.1f} мин, ~{:.1f} ГБ int16), раскладка {}",
                path.name, n, ch, n / SR / 60, n * ch * 2 / 2**30, layout or "по числу каналов")
    proc = procreg.popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    BLK = 1 << 20
    broken = None
    try:
        for s in range(0, n, BLK):
            proc.stdin.write(np.ascontiguousarray(out[s:s + BLK]).astype(np.int16).tobytes())
            if on_prog is not None:
                on_prog(min(1.0, (s + BLK) / max(1, n)))
    except (BrokenPipeError, OSError) as e:
        broken = e                      # приёмник закрыл канал — дочитаем ЕГО жалобу ниже
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
    err = (proc.stderr.read() or b"").decode("utf-8", "replace").strip() if proc.stderr else ""
    rc = proc.wait(); procreg.done(proc)
    if broken is not None or rc != 0:
        written = path.stat().st_size if path.exists() else 0
        raise RuntimeError(
            f"запись {path.name} не удалась: код {rc}"
            + (f", обрыв канала ({broken})" if broken is not None else "")
            + f", записано {written / 2**20:.1f} МБ, ожидалось сэмплов {n}"
            + (f"; ffmpeg: {err[-2000:]}" if err else "; ffmpeg промолчал"))


def _fill_silence_from_ref(out, ref_buf, *, sr=SR, sil_db=SIL_FILL_DB,
                           win_s=SIL_FILL_WIN_S, min_dur_s=SIL_FILL_MIN_S, xfade=XFADE):
    """ФИНАЛЬНЫЙ проход: ТИШИНУ озвучки заполнить звуком РЕФА. Зовётся ТОЛЬКО ПОСЛЕ полного
    аудио-выравнивания (band/muq) — когда озвучка уже синхронна рефу (на несинхронной дорожке
    вставка рефа создаёт мнимое смещение и плодит вырезы для аудиоанализа — потому fill_cuts
    «по видео до анализа» запрещён). Здесь анализ завершён, реф 1-в-1 на сетке REF и синхронен.

    Зоны, где озвучка молчит (RMS < sil_db) И реф звучит (RMS >= sil_db), заменяем рефом с
    кроссфейдом XFADE на стыках. Где у ОБОИХ тишина — остаётся тишина. Озвучка не теряется
    (трогаем только пустоты). Мутирует out на месте. -> секунд заполнено."""
    n = out.shape[0]; win = int(win_s * sr)
    if win < 1 or n < 2 * win:
        return 0.0
    om = out.mean(1) if out.ndim == 2 else out
    rm = ref_buf.mean(1) if ref_buf.ndim == 2 else ref_buf
    k = min(len(om), len(rm)) // win
    if k < 1:
        return 0.0

    def rms_db(sig):
        s = np.asarray(sig[:k * win], np.float64).reshape(k, win)
        return 20.0 * np.log10(np.sqrt((s * s).mean(1)) / 32768.0 + 1e-12)   # dBFS (int16 fullscale)

    do = rms_db(om); dr = rms_db(rm)
    fill = (do < sil_db) & (dr >= sil_db)             # озвучка молчит, реф звучит
    minw = max(1, int(round(min_dur_s / win_s)))
    filled = 0; i = 0
    while i < k:
        if not fill[i]:
            i += 1; continue
        j = i
        while j < k and fill[j]:
            j += 1
        if j - i >= minw:
            s0 = i * win; s1 = min(j * win, n, ref_buf.shape[0])
            if s1 > s0:
                dub = out[s0:s1].copy(); rb = ref_buf[s0:s1]
                if rb.ndim == 2 and out.ndim == 2 and rb.shape[1] != out.shape[1]:
                    # дубль многоканальный (5.1/7.1), реф стерео → дыру заливаем МОНО-рефом,
                    # размноженным на все каналы дубля (звук слышен во всех каналах, без потери)
                    rb = np.broadcast_to(rb.mean(1, keepdims=True), (rb.shape[0], out.shape[1]))
                out[s0:s1] = rb
                X = min(xfade, (s1 - s0) // 2)
                if X >= 1:
                    w = np.linspace(0.0, 1.0, X)[:, None]
                    out[s0:s0 + X] = dub[:X] * (1 - w) + rb[:X] * w          # озвучка → реф
                    out[s1 - X:s1] = rb[-X:] * (1 - w) + dub[-X:] * w        # реф → озвучка
                filled += s1 - s0
        i = j
    return filled / sr


def _recover_side(sub_syn, sub_ref, pred, syn_off, ref_off):
    """Один доп-проход: под-кусок дубляжа × под-кусок рефа → band_align → вписать
    вернувшиеся матчи (cos > EDGE_COS_MIN) в pred. Возврат: число вписанных кадров."""
    if len(sub_syn) < 5 or len(sub_ref) < 5:
        return 0
    off_sub, nrel, _, _ = coarse_robust(sub_syn, sub_ref)
    if nrel < 5:
        return 0
    pred_sub = band_align(sub_syn, sub_ref, off_sub, affine=True, DSYN=DSYN,
                          MATCH_THR=MATCH_THR, free_start=True)
    rec = np.where(pred_sub >= 0)[0]
    if not len(rec):
        return 0
    cosr = np.einsum("ij,ij->i", sub_syn[rec].astype(np.float32),
                     sub_ref[pred_sub[rec]].astype(np.float32))
    good = rec[cosr > EDGE_COS_MIN]
    for k in good:
        pred[syn_off + int(k)] = ref_off + int(pred_sub[int(k)])
    return int(len(good))


def _recover_edges(syn, refv, pred):
    """Доп-проходы на краях (пост-обработка, КАК R-фильтр v7 — ЯДРО НЕ ТРОГАЕМ).
    free_start мог выкинуть СОВПАДАЮЩЕЕ начало/конец дубляжа (напр. общий опенинг при
    разном вступлении BD↔WEB). Берём выкинутый кусок дубляжа против непокрытого куска
    рефа, гоняем band_align на под-задаче и вписываем вернувшиеся матчи. Мутирует pred.
    Возврат: число возвращённых кадров. Запускается ТОЛЬКО при заметном выбросе И
    непокрытом куске рефа (в обычных треках с рекламой в начале — не триггерится:
    после выброса рекламы реф покрыт с ~0)."""
    asg = np.where(pred >= 0)[0]
    if len(asg) < 2:
        return 0
    n = len(syn); rn = len(refv)
    rec = 0
    a0 = int(asg[0]); r0 = int(pred[a0])                  # ЛЕВЫЙ край
    if a0 > EDGE_MIN_FR and r0 > EDGE_MIN_FR:
        rec += _recover_side(syn[:a0], refv[:r0], pred, 0, 0)
    aN = int(asg[-1]); rN = int(pred[aN])                 # ПРАВЫЙ край
    if (n - 1 - aN) > EDGE_MIN_FR and (rn - 1 - rN) > EDGE_MIN_FR:
        rec += _recover_side(syn[aN + 1:], refv[rN + 1:], pred, aN + 1, rN + 1)
    return rec


def _creep_drop(syn, refv, pred, fps_dub):
    """Детектор «налипания» (пост-проход — ядро band_align НЕ трогаем).

    Дефект: вставка в дубле (повтор СВОЕГО контента с наложенным текстом) присваивается
    Drop-DTW «ползучим ходом» вместо drop — близнецы кадров лежат в рефе ВПЕРЕДИ, и
    монотонность запрещает присвоить их туда, а знакомый контент мешает чистому выбросу.
    Сигнал кричащий: присвоенные кадры по содержимому ЧУЖИЕ своим реф-партнёрам
    (cos≈0.00–0.02 на 11.5с подряд), у честных матчей даже в тёмных сценах cos≥0.22
    дольше 1с не проседает.

    Механика: скользящая медиана (окно ~1с) cos присвоенных кадров; связные зоны ниже
    CREEP_COS (мостик через уже выброшенные кадры ≤ CREEP_BRIDGE_S) длиной ≥ CREEP_MIN_S
    → pred=-1. Судьбу решает ДАЛЬШЕ _level_decide: сдвиг уровня через зону → реальная
    вставка (остаётся выброшенной, пробел рефа зальётся оригиналом); сдвига нет →
    восстановление интерполяцией (карта не меняется) — ложное срабатывание безвредно.

    Мутирует pred. Возврат: [(j0, j1, cos_сред, наклон)…] выброшенных зон (телеметрия)."""
    asgm = pred >= 0
    asg = np.where(asgm)[0]
    if len(asg) < 2:
        return []
    cosv = np.full(len(pred), np.nan, np.float32)
    CH = 4096
    for i in range(0, len(asg), CH):
        sl = asg[i:i + CH]
        a = syn[sl].astype(np.float32); b = refv[pred[sl]].astype(np.float32)
        cosv[sl] = np.einsum("ij,ij->i", a, b) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    W = max(3, int(fps_dub) | 1)
    half = W // 2
    sw = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([np.full(half, np.nan, np.float32), cosv,
                        np.full(half, np.nan, np.float32)]), W)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN окна на drop-прогонах
        med = np.nanmedian(sw[:len(pred)], axis=1)
    low = asgm & (med < CREEP_COS)
    BR = int(CREEP_BRIDGE_S * fps_dub)
    zones: list[list[int]] = []
    for s, e in _runs1d(np.where(low)[0]):
        # мостик: склеиваем с предыдущей зоной, если между ними нет присвоенных кадров
        if zones and s - zones[-1][1] <= BR and not asgm[zones[-1][1] + 1:s].any():
            zones[-1][1] = e
        else:
            zones.append([s, e])
    out = []
    MIN = CREEP_MIN_S * fps_dub
    for s, e in zones:
        if e - s + 1 < MIN:
            continue
        slope = (int(pred[e]) - int(pred[s])) / max(1, e - s)
        out.append((int(s), int(e), float(np.nanmean(cosv[s:e + 1])), float(slope)))
        pred[s:e + 1] = -1
    return out


def _runs1d(idx):
    """Связные прогоны подряд идущих индексов → [(s,e)…]."""
    out = []
    if not len(idx):
        return out
    s = p = int(idx[0])
    for j in idx[1:]:
        if j == p + 1:
            p = int(j)
        else:
            out.append((s, p)); s = p = int(j)
    out.append((s, p))
    return out


def _level_decide(pred, fps_ref, fps_dub, n_ref, n_syn, *,
                  edit_s=LEVEL_EDIT_S, min_ins_s=LEVEL_MIN_INS_S, win_s=LEVEL_WIN_S,
                  cut_min_s=CUT_MIN_S):
    """НОВЫЙ слой разбора правок ЕДИНЫМ устойчивым правилом (замена restore_blind + R-фильтра).

    Идея: реальная правка = СТОЙКИЙ сдвиг УРОВНЯ offset; слепая зона = уровень держится.
    Для каждого выброшенного блока дубляжа (drop-syn) сравниваем медиану offset ДО и ПОСЛЕ
    (окно win_s). Если |Δ| ≤ edit_s ИЛИ блок короче min_ins_s (дрожь) → СЛЕПАЯ зона →
    восстановить интерполяцией (это заодно перекрывает парный пробел рефа). Иначе → реальная
    вставка → оставить выброшенной. После: любой оставшийся пробел рефа > cut_min_s = реальный
    ВЫРЕЗ (доказано: пробел рефа при непрерывном дубляже всегда сдвигает уровень ⇒ слепых не
    остаётся). Краевые непокрытые зоны рефа (> FILL_MIN_S) — тоже вырезы (анти-заморозка).

    Один физический порог (сдвиг ~1с, блок ~2с, в АБСОЛЮТНЫХ кадрах — устойчивее отношения R_ins).
    Возврат: (pred, cut_intervals[(R1,R2)…], n_restored)."""
    pred = pred.copy()
    asg = np.where(pred >= 0)[0]
    if len(asg) < 2:
        return pred, [], 0
    off_asg = pred[asg].astype(np.float64) - asg
    W = max(1, int(win_s * fps_dub))
    EDIT = edit_s * fps_dub
    MIN_INS = min_ins_s * fps_dub

    def lvl(frame, before):
        m = ((asg < frame) & (asg >= frame - W)) if before else ((asg > frame) & (asg <= frame + W))
        return float(np.median(off_asg[m])) if m.any() else None

    n_restored = 0
    for a, b in _runs1d(np.where(pred == -1)[0]):
        prev = asg[asg < a]; nxt = asg[asg > b]
        if not len(prev) or not len(nxt):
            continue
        lb = lvl(a, True); la = lvl(b, False)
        if lb is None or la is None:
            continue
        if not (abs(la - lb) > EDIT and (b - a + 1) > MIN_INS):    # слепая/дрожь → восстановить
            p0, p1 = int(prev[-1]), int(nxt[0])
            pred[a:b + 1] = np.round(np.interp(
                np.arange(a, b + 1), [p0, p1], [pred[p0], pred[p1]])).astype(np.int64)
            n_restored += b - a + 1
        # иначе — реальная вставка → оставить выброшенной (её аудио выпадет / зальётся вырез рефа)

    # --- ВЫРЕЗЫ: пробел рефа = вырез ТОЛЬКО при УСТОЙЧИВОМ сдвиге уровня (фикс «пилы») ---
    # Соседние пробелы склеиваем в кластер; если уровень offset ВОЗВРАЩАЕТСЯ (выброс на
    # статике) → восстановить интерполяцией (звук озвучки сохранён, тишины нет). Если уровень
    # держится → реальный вырез: кластер схлопываем в ОДИН интервал (>1с → заливка рефом,
    # а не пила тишин <1с).
    asg = np.where(pred >= 0)[0]
    ratio = fps_ref / fps_dub
    off = pred[asg].astype(np.float64) - asg * ratio
    Wc = max(1, int(win_s * fps_dub)); EDIT_R = edit_s * fps_ref
    cand = [i for i in range(len(asg) - 1)
            if int(pred[asg[i + 1]]) - int(pred[asg[i]]) > cut_min_s * fps_ref]
    COAL = LEVEL_CUT_COALESCE_S * fps_dub
    clusters: list[list[int]] = []
    for i in cand:
        if clusters and (asg[i] - asg[clusters[-1][-1] + 1]) <= COAL:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    cut_intervals: list[tuple[int, int]] = []
    for cl in clusters:
        i0, i1 = cl[0], cl[-1]
        bef = off[(asg < asg[i0]) & (asg >= asg[i0] - Wc)]
        lb = float(np.median(bef)) if len(bef) else float(off[i0])
        recover = None; lim = asg[i1] + LEVEL_CUT_RECOVER_S * fps_dub
        for k in range(i1 + 1, len(asg)):
            if asg[k] > lim:
                break
            if abs(off[k] - lb) <= EDIT_R:
                recover = k; break
        if recover is not None:                       # выброс с возвратом (пила) → восстановить
            d0, d1 = int(asg[i0]), int(asg[recover])
            pred[d0:d1 + 1] = np.round(np.interp(np.arange(d0, d1 + 1), [d0, d1],
                                                 [pred[d0], pred[d1]])).astype(np.int64)
            n_restored += d1 - d0 + 1
        else:                                          # устойчивый сдвиг → ОДИН вырез на кластер
            cut_intervals.append((int(pred[asg[i0]]) + 1, int(pred[asg[i1 + 1]])))
    asg = np.where(pred >= 0)[0]                       # пересчёт после восстановлений
    # краевые непокрытые зоны рефа — КАК БЫЛО (старт-заливка = техдолг, тут не трогаем)
    r_first = int(pred[int(asg[0])]); r_last = int(pred[int(asg[-1])])
    if r_first > FILL_MIN_S * fps_ref:
        cut_intervals.insert(0, (0, r_first))
    if (n_ref - r_last) > FILL_MIN_S * fps_ref:
        cut_intervals.append((r_last, n_ref))
    return pred, sorted(set(cut_intervals)), n_restored


def _cos_anchors(syn, refv, pred, asg):
    """cos каждого присвоенного кадра (asg) к его реф-партнёру (SRM L2-норм ~1) — вес
    анализатора зрения (видео-карта) и единого графика."""
    cos = np.empty(len(asg), np.float32)
    CH = 4096
    for i in range(0, len(asg), CH):
        sl = asg[i:i + CH]
        a = syn[sl].astype(np.float32); b = refv[pred[sl]].astype(np.float32)
        cos[i:i + len(sl)] = np.einsum("ij,ij->i", a, b) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    return cos


# ── tmp-mode checkpoints: stage versions + the align stamp (CK3 validity) ──
ALIGN_VER = 3   # matching + vision map (band_align/_level_decide/vision_detect); bump when they change


def _align_stamp(dub_audio, ref, fps_ref, fps_tst, *, free_start, recover_edges) -> str:
    """Штамп CK3 (align-результат): меняется при смене входа (файлы), версий (EMB/ALIGN) или
    опций, влияющих на матчинг+карту. Совпал → __vision.npz можно переиспользовать."""
    import json as _j
    return _j.dumps({
        "emb": cache_mod.EMB_VER, "align": ALIGN_VER,
        "dub": cache_mod.cache_key(Path(dub_audio)),
        "ref": cache_mod.cache_key(Path(ref.src)) if ref.src else "",
        "fr": round(float(fps_ref), 6), "ft": round(float(fps_tst), 6),
        "fs": bool(free_start), "re": bool(recover_edges),
    }, sort_keys=True)


def _load_align_ckpt(out_path: Path, n_syn: int, stamp: str):
    """CK3: вернуть (pred, asg, cut_intervals) из `<stem>__vision.npz`, если штамп совпал; иначе
    None. pred восстанавливается pred[asg]=pred_asg. Карта зрения (grid/tg_s/vision_cuts) НЕ
    кэшируется — дёшево пересчитывается build_map из pred/asg ниже (бит-в-бит)."""
    p = Path(out_path).parent / "_plots" / (Path(out_path).stem + "__vision.npz")
    if not p.exists():
        return None
    try:
        d = np.load(str(p), allow_pickle=True)
        if "stamp" not in d.files or str(d["stamp"]) != stamp or "cut_intervals" not in d.files:
            return None
        asg = np.asarray(d["asg"], np.int64)
        pred = np.full(int(n_syn), -1, np.int64)
        if len(asg):
            pred[asg] = np.asarray(d["pred_asg"], np.int64)
        ci = [(int(a), int(b)) for a, b in np.asarray(d["cut_intervals"], np.int64).reshape(-1, 2)]
        return pred, asg, ci
    except Exception:  # noqa: BLE001 — битый кэш → пересчёт
        return None


@dataclass
class _Match:
    """Features and everything measured on them. Replaced as one unit whenever the features
    change (detelecine, geom rebuild), so no decision can read a value from an older state."""

    refv: np.ndarray
    syn: np.ndarray
    fps_ref: float
    fps_dub: float
    ax_ref: np.ndarray | None
    ax_dub: np.ndarray | None
    tc_ref: dict
    tc_dub: dict
    state: str                                   # "plain" | "detelecine" | "mirror" | "geom"
    off0: np.ndarray | None = None
    nrel: int = 0
    n_keep: int = -1                             # -1: no coarse pass on this state (CK3 reuse)
    chain: np.ndarray | None = None
    mirror: bool = False                         # dub features are those of the horizontally mirrored frames

    @property
    def n(self) -> int:
        return len(self.syn)

    @property
    def strong_thr(self) -> float:
        return COARSE_SAME_MIN_FRAC * self.n / 8  # K=8 is the thinning step of coarse._anchors

    @property
    def coarse_strong(self) -> bool:
        return self.n_keep >= self.strong_thr


def _frozen_runs(syn, fps: float, ax=None) -> list[tuple[float, float]]:
    """Runs of identical consecutive frames longer than FREEZE_MIN_S, in seconds. Probes the rows the
    coarse pass thins to (every K-th frame, so they are already in the page cache after it) and
    compares each with the probe two steps later; a telecine cadence dents single probes, so a run
    survives dips up to FREEZE_GAP_S."""
    K = 8; n = len(syn)
    idx = np.arange(0, n - 2 * K, K)
    if len(idx) < 4:
        return []
    a = np.asarray(syn[idx]).astype(np.float32); b = np.asarray(syn[idx + 2 * K]).astype(np.float32)
    cos = np.einsum("ij,ij->i", a, b)
    t = (ax[idx] if ax is not None else idx / fps).astype(np.float64)
    runs: list[tuple[float, float]] = []; start = None; last_hi = None
    for ti, c in zip(t, cos):
        if c >= FREEZE_COS:
            start = ti if start is None else start; last_hi = ti
        elif start is not None and ti - last_hi > FREEZE_GAP_S:
            if last_hi - start >= FREEZE_MIN_S:
                runs.append((float(start), float(last_hi)))
            start = None
    if start is not None and last_hi - start >= FREEZE_MIN_S:
        runs.append((float(start), float(last_hi)))
    return runs


def _run_coarse(m: _Match, low_mem: bool, rep: Reporter | None, trace: Trace) -> _Match:
    """Coarse pass on the given state; the chain and n_keep stay bound to that state."""
    if rep is not None:
        rep.mark(0.0)
    m.off0, m.nrel, m.n_keep, m.chain = (coarse_windowed(m.syn, m.refv, on_prog=rep) if low_mem
                                         else coarse_robust(m.syn, m.refv))
    trace.event("coarse", state=m.state, n_dub=m.n, n_ref=len(m.refv), anchors=int(m.nrel),
                n_keep=int(m.n_keep), chain_len=int(len(m.chain)),
                off0_median_s=float(np.median(m.off0)) / m.fps_ref if m.fps_ref else 0.0)
    return m


def _mirrored_copy(v: np.ndarray, low_mem: bool, tmp_dir) -> np.ndarray:
    """SRM of the mirrored frames; on disk under tmp_dir in low_mem mode like the other rebuilt features."""
    if low_mem and tmp_dir is not None:
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        mp = Path(tempfile.mkdtemp(prefix="mirror_", dir=str(tmp_dir))) / "f.f16"
        out = np.memmap(mp, dtype=np.float16, mode="w+", shape=v.shape)
    else:
        out = np.empty(v.shape, np.float16)
    return mirror_srm(v, out=out)


def _try_mirror(m: _Match, low_mem: bool, tmp_dir, rep: Reporter | None, trace: Trace) -> _Match | None:
    """Orientation second chance, before geometry: a frame sample is matched in both orientations
    (seconds, no decode); when the mirrored sample clearly wins, the coarse pass is redone on the
    mirrored features. Returns the new state or None; the decision is traced either way."""
    pr = orientation_probe(m.syn, m.refv)
    need = COARSE_SAME_MIN_FRAC * pr["sample"]
    win = pr["mirror"] >= need and pr["mirror"] > MIRROR_RATIO * max(pr["plain"], 1)
    trace.decide("mirror", state=m.state, inputs=dict(pr, n_keep=int(m.n_keep)),
                 thresholds={"min_strong": need, "MIRROR_RATIO": MIRROR_RATIO},
                 verdict="applied" if win else "none")
    if not win:
        return None
    mm = _Match(m.refv, _mirrored_copy(m.syn, low_mem, tmp_dir), m.fps_ref, m.fps_dub,
                m.ax_ref, m.ax_dub, m.tc_ref, m.tc_dub, "mirror", mirror=True)
    return _run_coarse(mm, low_mem, rep, trace)


def _detelecine_state(m: _Match, tmp_dir, trace: Trace) -> _Match:
    """Thin the baked 3:2 cadence on both sides; a side with a live VFR axis is left alone
    because thinning would break its 1:1 index-to-time stitching."""
    if m.ax_ref is None:
        m.refv, m.fps_ref, m.tc_ref = _detelecine(m.refv, m.fps_ref, tmp_dir=tmp_dir)
    if m.ax_dub is None:
        m.syn, m.fps_dub, m.tc_dub = _detelecine(m.syn, m.fps_dub, tmp_dir=tmp_dir)
    m.state = "detelecine"
    m.off0 = None; m.nrel = 0; m.n_keep = -1; m.chain = None
    trace.event("detelecine", state=m.state, dropped_ref=int(m.tc_ref["dropped"]),
                dropped_dub=int(m.tc_dub["dropped"]), fps_ref=m.fps_ref, fps_dub=m.fps_dub, n_dub=m.n)
    return m


def _try_geom(ref, dub_audio, fps_ref, fps_dub, low_mem, cache_dir, keep_tmp,
              ffmpeg, rep: Reporter | None, trace: Trace, mirror: bool = False) -> tuple[_Match | None, dict | None]:
    """Geometry pass for a pair vision cannot match (crop/zoom/anamorph/bars): consensus_G finds
    the global transform, both SRM are rebuilt with `crop` (a second decode, only for such pairs)
    and the coarse pass runs on the rebuilt features. Returns the new state and G, or (None, None)
    with the reason recorded in the trace."""
    from track_muxer.conform import geom
    if ref.src is None:
        trace.decide("geom", state="geom", verdict="none", inputs={"reason": "reference has no source file"})
        return None, None
    # The LoFTR consensus is expensive and not deterministic, so a cached G is reused as is.
    G = (cache_mod.load_geom(cache_dir, Path(ref.src), Path(dub_audio))
         if (keep_tmp and cache_dir is not None) else None)
    if G is not None and bool(G.get("flip", False)) != mirror:   # cached for the other orientation
        G = None
    source = "cache"
    if G is None:
        if not geom.available():
            trace.decide("geom", state="geom", verdict="none", inputs={"reason": "geom backend unavailable"})
            return None, None
        if rep is not None:
            rep.mark(0.0, "оценка кадрирования и масштаба")
        G = geom.consensus_G(Path(ref.src), Path(dub_audio), on_prog=part(rep, 0.0, 0.40), flip=mirror)
        source = "computed"
        if G is None:
            trace.decide("geom", state="geom", verdict="none", inputs={"reason": "no consensus transform"})
            return None, None
        if keep_tmp and cache_dir is not None:
            cache_mod.save_geom(cache_dir, Path(ref.src), Path(dub_audio), G)
    trace.decide("geom", state="geom", verdict="applied", source=source,
                 inputs={"sx": G["sx"], "sy": G["sy"], "n_in": G["n_in"],
                         "crop_ref": G["crop_ref"], "crop_dub": G["crop_dub"], "mirror": mirror})
    if rep is not None:
        rep.mark(0.40, "пересчёт признаков кадров")

    def _srm(video: Path, fps: float, crop: str, sub: Reporter | None) -> SrmFeatures:
        # tmp-чекпоинт кропнутого SRM (CK1 geom): повтор слепой пары без передекода
        h = probe_resolution(video)
        if low_mem and keep_tmp and cache_dir is not None:
            cached = cache_mod.load_srm(cache_dir, video, crop=crop)
            if cached is not None:
                return cached
            mp = cache_mod.srm_file(cache_dir, video, crop=crop)
            with decode_backend(h, ffmpeg) as _be:
                f = build_srm(video, fps, ffmpeg=ffmpeg, crop=crop, mmap_path=mp, backend=_be, reporter=sub)
            cache_mod.save_meta(cache_dir, video, len(f.srm), f.fps, crop=crop)
            return f
        mp = (Path(tempfile.mkdtemp(prefix="geomsrm_")) / "f.f16") if low_mem else None
        with decode_backend(h, ffmpeg) as _be:
            return build_srm(video, fps, ffmpeg=ffmpeg, crop=crop, mmap_path=mp, backend=_be, reporter=sub)

    ref2 = _srm(Path(ref.src), fps_ref, G["crop_ref"], part(rep, 0.40, 0.70))
    dub2 = _srm(Path(dub_audio), fps_dub, G["crop_dub"], part(rep, 0.70, 0.96))
    # Cropping keeps the frame set, so the VFR axes of the rebuilt SRM are the source's own.
    ax_ref2 = ref2.pts if (ref2.pts is not None and len(ref2.pts) == len(ref2.srm)) else None
    ax_dub2 = dub2.pts if (dub2.pts is not None and len(dub2.pts) == len(dub2.srm)) else None
    _tctmp = (Path(dub_audio).parent / "_tmp") if low_mem else None
    if _tctmp is not None:
        _tctmp.mkdir(parents=True, exist_ok=True)
    _no_tc = {"telecine": False, "tele_score": 0.0, "argmax": 0, "dropped": 0}
    syn2 = _mirrored_copy(dub2.srm, low_mem, _tctmp) if mirror else dub2.srm
    m = _Match(ref2.srm, syn2, fps_ref, fps_dub, ax_ref2, ax_dub2, dict(_no_tc), dict(_no_tc), "geom",
               mirror=mirror)
    if m.ax_ref is None:
        m.refv, m.fps_ref, m.tc_ref = _detelecine(m.refv, m.fps_ref, tmp_dir=_tctmp)
    if m.ax_dub is None:
        m.syn, m.fps_dub, m.tc_dub = _detelecine(m.syn, m.fps_dub, tmp_dir=_tctmp)
    if rep is not None:
        rep.mark(0.96, "повторное грубое соответствие")
    return _run_coarse(m, low_mem, part(rep, 0.96, 1.0), trace), G


def _edge_silence(out, grid, tg_s, n_aud, sr, fill_spans, fade):
    """КРАЕВОЙ вырез = дубля физически НЕТ на краю таймлайна рефа: ГОЛОВА (дубль начинается позже
    рефа → tg_s<0) / ХВОСТ (дубль короче рефа → tg_s за концом дубля). build_curve вне тела
    экстраполирует ЛИНИЮ (curve=base) → warp клампит первый/последний сэмпл дубля в КОНСТАНТУ, а не
    тишину. Эта константа-аномалия ломает band по ВСЕМУ треку (нормировка/сегментация глобальны).
    Зануляем ТИШИНОЙ (+fade) и вносим в fill_spans — как внутренние вырезы: band игнорит зону
    (vision_spans), _fill_silence_from_ref заливает синхронным оригиналом рефа. Только НЕПРЕРЫВНЫЕ
    края (внутренние вырезы гасит вызывающий по vision_cuts). tg_s на крае монотонна (base-линия)."""
    tg = np.asarray(tg_s, float); n_out = out.shape[0]; dur_dub = n_aud / sr
    EDGE_MIN_S = 0.3                                              # <0.3с края = округление/суб-кадр → не трогаем

    def _zero(a_s, b_s):
        if b_s - a_s < EDGE_MIN_S:                               # микро-край (дубль ~ровно длины рефа) — no-op
            return
        s1 = max(0, int(a_s * sr)); s2 = min(n_out, int(b_s * sr))
        if s2 - s1 <= 0:
            return
        out[s1:s2] = 0.0
        if s1 - fade >= 0:
            out[s1 - fade:s1] *= np.linspace(1, 0, fade)[:, None]
        if s2 + fade <= n_out:
            out[s2:s2 + fade] *= np.linspace(0, 1, fade)[:, None]
        fill_spans.append((float(a_s), float(b_s)))

    if len(tg) >= 2 and tg[0] < 0 and (tg >= 0.0).any():          # ГОЛОВА: дубля нет до grid[k]
        _zero(0.0, float(grid[int(np.argmax(tg >= 0.0))]))
    if len(tg) >= 2 and tg[-1] > dur_dub and (tg <= dur_dub).any():  # ХВОСТ: дубля нет после grid[k]
        k = len(tg) - 1 - int(np.argmax((tg <= dur_dub)[::-1]))
        _zero(float(grid[k]), float(grid[-1]))


def conform_features(
    ref: SrmFeatures,
    dub: SrmFeatures,
    dub_audio: Path,
    out_path: Path,
    *,
    fps_ref: float | None = None,
    fps_dub: float | None = None,
    free_start: bool = True,
    recover_edges: bool = True,
    fill_silence: bool = True,      # «вырезы оригиналом»: тишину озвучки заполнить рефом ПОСЛЕ band/muq
                                    # (_fill_silence_from_ref) — только на уже синхронной дорожке
    audio_band: bool = False,       # аудио-метод: anchor-пайплайн, карта DSP 48 полос (без модели)
    audio_muq: bool = False,        # аудио-метод: anchor-пайплайн, карта MuQ (опц., GPU+transformers)
    apply_cuts: bool = True,        # band/muq: ВКЛ=резкая правка резов поверх дрейфа; ВЫКЛ=только дрейф ±2% (пандусы)
    drift_speed_pct: float = 1.25,  # band/muq: потолок СКОРОСТИ изменения сдвига кривой дрейфа, %/с (1.25 = SMAX)
    audio_fix: bool = False,
    ref_audio: np.ndarray | None = None,
    ref_atrack: int = 0,            # ⭐ 5.1: индекс аудиодорожки РЕФА (звуковой эталон band/заливки)
    dub_atrack: int = 0,            # ⭐ 5.1: индекс аудиодорожки ДУБЛЯ (что выравниваем)
    ffmpeg: str = FFMPEG,
    progress=None,
    dub_name: str | None = None,
    progress_meta: tuple[int, int, str] = (0, 0, ""),
    low_mem: bool = False,
    cache_dir: Path | str | None = None,   # чекпоинты CK2/CK3/CK5 (режим tmp); None=выкл
    keep_tmp: bool = False,
    trace: Trace | None = None,     # decision trace of the pair; created here when the caller has none
) -> PairResult:
    """Выровнять озвучку (фичи dub + её аудио dub_audio) на таймлайн ref → out_path.

    keep_tmp+cache_dir → чекпоинты в `cache_dir` (аудио CK2 / align CK3 / выход CK5):
    при повторе пропуск декода аудио и матчинга. (CK1 SRM
    дубля — в conform_pair выше.)"""
    t0 = time.perf_counter()

    def _rep(stage: str, detail: str = "") -> Reporter | None:
        return Reporter.of(progress, stage, progress_meta, detail)

    name = dub_name or (dub.src.name if dub.src else dub_audio.name)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fps_ref = fps_ref if fps_ref is not None else ref.fps
    fps_tst = fps_dub if fps_dub is not None else dub.fps
    fps0_ref, fps0_tst = fps_ref, fps_tst   # исходные (до детелесина) — для geom-второго-шанса ниже
    # SRM держим в f16 (как в кэше); до f32 апкастим ТОЧЕЧНО на срезах перед
    # перемножением (band_align/_recover/_metrics) — полную f32-копию не материализуем.
    syn = dub.srm
    refv = ref.srm
    if len(refv) == 0 or not (fps_ref and fps_ref > 0):   # пустой/битый SRM рефа (сборка упала) → понятная
        raise ValueError(f"SRM рефа пуст/битый (кадров={len(refv)}, fps={fps_ref}) — пересоберите кэш рефа")  # ошибка, не div0
    if len(syn) == 0 or not (fps_tst and fps_tst > 0):
        raise ValueError(f"SRM дубля пуст/битый (кадров={len(syn)}, fps={fps_tst})")
    N = len(syn)

    # ── Реальная ось времени кадров (SrmFeatures.pts) — только VFR, иначе None → путь
    #    индекс/fps бит-в-бит. Длина оси ОБЯЗАНА совпадать с SRM (сшивка по индексу 1:1). ──
    ax_ref = ref.pts if (ref.pts is not None and len(ref.pts) == len(refv)) else None
    ax_dub = dub.pts if (dub.pts is not None and len(dub.pts) == len(syn)) else None

    # ── Запечённый 3:2-телесин (NTSC риперы пекут фильм 23.976 в 29.97 без IVTC → SRM шумит →
    #    шумная укладка/джиттер якорей). Детект ДЁШЕВО (фикс-сэмпл) для гейта CK3 и паспорта;
    #    само прореживание каденса — в матчинге ниже (на не-телесине НО-ОП → бит-в-бит).
    #    Поток с живой VFR-осью телесин НЕ детектим: автокорреляция периода-5 предполагает
    #    равномерную сетку по построению, а прореживание сломало бы сшивку оси 1:1. ──
    _tcr0 = _is_telecine(refv, fps_ref) if ax_ref is None else (False, 0.0, 0)
    _tcd0 = _is_telecine(syn, fps_tst) if ax_dub is None else (False, 0.0, 0)
    tc_ref = {"telecine": _tcr0[0], "tele_score": _tcr0[1], "argmax": _tcr0[2], "dropped": 0}
    tc_dub = {"telecine": _tcd0[0], "tele_score": _tcd0[1], "argmax": _tcd0[2], "dropped": 0}
    _is_tc = _tcr0[0] or _tcd0[0]
    _tctmp = (Path(dub_audio).parent / "_tmp") if low_mem else None
    if _is_tc and _tctmp is not None:
        _tctmp.mkdir(parents=True, exist_ok=True)

    # CK3 (tmp mode): the matching result (pred/asg/cut_intervals) comes from __vision.npz when
    # the stamp matches, skipping the costly coarse/band_align/_level_decide. The vision map
    # (build_map below) is cheap to rebuild from pred/asg, bit-exact.
    # The telecine CK3 track is not reused: decimation shifts reference indices, cached pred is stale.
    trace = trace if trace is not None else Trace(name)
    trace.event("telecine", state="plain", ref_fps=fps_ref, dub_fps=fps_tst, n_ref=len(refv), n_dub=N,
                ref=tc_ref, dub=tc_dub, vfr_ref=ax_ref is not None, vfr_dub=ax_dub is not None)
    n_restore = 0; n_recovered = 0; n_blind = 0
    geom_info = None                                  # G of the geometry pass when it was applied
    creep_zones: list[tuple[int, int, float, float]] = []
    cut_intervals: list[tuple[int, int]] = []
    m = _Match(refv, syn, fps_ref, fps_tst, ax_ref, ax_dub, tc_ref, tc_dub, "plain")
    freeze_runs: list[tuple[float, float]] = []
    _astamp = (_align_stamp(dub_audio, ref, fps_ref, fps_tst, free_start=free_start,
                            recover_edges=recover_edges)
               if keep_tmp else "")
    _ck3 = (_load_align_ckpt(out_path, N, _astamp)
            if (keep_tmp and cache_dir is not None and not _is_tc) else None)
    cache_mod._a("CK3 зрение", _ck3 is not None, out_path.stem)        # реюз матчинга (пропуск GPU-матчинга)
    trace.decide("ck3_reuse", state="cache", source="cache",
                 inputs={"checkpoints": bool(keep_tmp and cache_dir is not None), "telecine": _is_tc},
                 verdict="hit" if _ck3 is not None else "miss")
    _band = None

    def _match(m: _Match, detail: str):
        """Band + edge/creep/level passes on one state; the chain is the one coarse found on it."""
        nonlocal _band
        _band = _rep("band", detail)
        if _band is not None:
            _band.mark(0.0)
        pred = band_align(m.syn, m.refv, m.off0, affine=True, DSYN=DSYN, MATCH_THR=MATCH_THR,
                          free_start=free_start, on_prog=part(_band, 0.0, 0.85), chain_aj=m.chain,
                          chain_off=(m.off0[m.chain] if m.chain is not None else None))
        if _band is not None:
            _band.mark(0.85, "разбор уровней и краёв")
        raw = int((pred >= 0).sum())
        n_rec = _recover_edges(m.syn, m.refv, pred) if recover_edges else 0
        creep = _creep_drop(m.syn, m.refv, pred, m.fps_dub)
        pred, cuts, n_res = _level_decide(pred, m.fps_ref, m.fps_dub, len(m.refv), m.n)
        asg = np.where(pred >= 0)[0]
        trace.event("band", state=m.state, chain_len=int(len(m.chain)) if m.chain is not None else 0,
                    assigned_raw_pct=100.0 * raw / max(1, m.n), edge_recovered=int(n_rec),
                    creep_zones=len(creep), cuts=len(cuts), restored=int(n_res),
                    assigned_pct=100.0 * len(asg) / max(1, m.n))
        return pred, asg, n_rec, creep, cuts, n_res

    if _ck3 is not None:
        pred, asg, cut_intervals = _ck3
    else:
        m = _run_coarse(m, low_mem, _rep("coarse", "оценка общего смещения"), trace)
        freeze_runs = _frozen_runs(m.syn, m.fps_dub, m.ax_dub)      # after coarse: its thinned rows are cached
        trace.event("video_freeze", state=m.state, runs=[(round(a, 2), round(b, 2)) for a, b in freeze_runs],
                    total_s=round(sum(b - a for a, b in freeze_runs), 2))
        # Too few anchors means vision is blind (crop/zoom/anamorph/bars): geometry before anything else.
        blind = m.n_keep < GEOM_GATE
        trace.decide("geom_gate", state=m.state, inputs={"n_keep": m.n_keep},
                     thresholds={"GEOM_GATE": GEOM_GATE},
                     verdict="geom" if blind else ("detelecine" if _is_tc else "band"))
        if blind:
            # Orientation is checked before geometry: the probe costs seconds and needs no decode.
            mm = _try_mirror(m, low_mem, _tctmp, _rep("coarse", "проверка ориентации кадра"), trace)
            if mm is not None:
                m = mm
                if _is_tc:
                    m = _detelecine_state(m, _tctmp, trace)
                    m = _run_coarse(m, low_mem, _rep("coarse", "повторная оценка после прореживания каденса"), trace)
                blind = m.n_keep < GEOM_GATE
        if blind:
            gm, G = _try_geom(ref, dub_audio, fps_ref, fps_tst, low_mem,
                              cache_dir, keep_tmp, ffmpeg, _rep("geom"), trace, mirror=m.mirror)
            if G is not None:
                m, geom_info = gm, G
        elif _is_tc and m.state == "plain":
            m = _detelecine_state(m, _tctmp, trace)
            m = _run_coarse(m, low_mem, _rep("coarse", "повторная оценка после прореживания каденса"), trace)
        pred, asg, n_recovered, creep_zones, cut_intervals, n_restore = _match(
            m, "сопоставление кадров в полосе поиска")

    assigned_pct = 100.0 * len(asg) / max(1, m.n)
    # A low assigned share is only a proxy for a foreign video: a soft zoom leaves enough coarse
    # anchors to pass GEOM_GATE yet starves the band, so geometry gets a second chance first.
    second = assigned_pct < ABORT_ASSIGNED_PCT and not m.coarse_strong and geom_info is None
    trace.decide("geom_second_chance", state=m.state,
                 inputs={"assigned_pct": assigned_pct, "n_keep": m.n_keep, "geom_used": geom_info is not None},
                 thresholds={"ABORT_ASSIGNED_PCT": ABORT_ASSIGNED_PCT, "strong_thr": m.strong_thr},
                 verdict="try" if second else "skip")
    if second and not m.mirror:
        mm = _try_mirror(m, low_mem, _tctmp, _rep("coarse", "проверка ориентации кадра"), trace)
        if mm is not None:
            m = mm
            if _is_tc:
                m = _detelecine_state(m, _tctmp, trace)
                m = _run_coarse(m, low_mem, _rep("coarse", "повторная оценка после прореживания каденса"), trace)
            pred, asg, n_recovered, creep_zones, cut_intervals, n_restore = _match(
                m, "сопоставление кадров после отражения")
            assigned_pct = 100.0 * len(asg) / max(1, m.n)
            second = assigned_pct < ABORT_ASSIGNED_PCT and not m.coarse_strong
    if second:
        # fps0: the current ones may already be thinned by detelecine, and the geometry pass
        # rebuilds the SRM from the files and thins on its own.
        gm, G = _try_geom(ref, dub_audio, fps0_ref, fps0_tst, low_mem,
                          cache_dir, keep_tmp, ffmpeg, _rep("geom"), trace, mirror=m.mirror)
        if G is not None:
            m, geom_info = gm, G
            pred, asg, n_recovered, creep_zones, cut_intervals, n_restore = _match(
                m, "сопоставление кадров после коррекции")
            assigned_pct = 100.0 * len(asg) / max(1, m.n)
    refv, syn, fps_ref, fps_tst, N = m.refv, m.syn, m.fps_ref, m.fps_dub, m.n
    ax_ref, ax_dub, tc_ref, tc_dub, n_keep = m.ax_ref, m.ax_dub, m.tc_ref, m.tc_dub, m.n_keep
    _geom_kw = dict(mirror_used=bool(m.mirror), geom_used=geom_info is not None,
                    geom_n_in=int(geom_info["n_in"]) if geom_info else 0,
                    geom_sx=float(geom_info["sx"]) if geom_info else 0.0,
                    geom_sy=float(geom_info["sy"]) if geom_info else 0.0,
                    telecine_ref=bool(tc_ref["telecine"]), telecine_dub=bool(tc_dub["telecine"]),
                    tele_score=round(max(tc_ref["tele_score"], tc_dub["tele_score"]), 3),
                    tc_dropped=int(tc_ref["dropped"]) + int(tc_dub["dropped"]))
    # Foreign video only when both signals agree; a strong coarse chain with a low assigned share
    # is a low-cosine encode of the same episode and proceeds with a warning.
    low_cos_proceed = assigned_pct < ABORT_ASSIGNED_PCT and m.coarse_strong
    foreign = assigned_pct < ABORT_ASSIGNED_PCT and not m.coarse_strong
    trace.decide("foreign_gate", state=m.state,
                 inputs={"assigned_pct": assigned_pct, "n_keep": m.n_keep, "n_dub": m.n},
                 thresholds={"ABORT_ASSIGNED_PCT": ABORT_ASSIGNED_PCT, "strong_thr": m.strong_thr},
                 verdict="abort" if foreign else ("proceed_low_cos" if low_cos_proceed else "proceed"))
    if foreign:
        return PairResult(
            dub=name, out_path=None, ok=False,
            error=f"чужое видео: сопоставлено {assigned_pct:.0f}% (< {ABORT_ASSIGNED_PCT:.0f}%) — выравнивание прервано",
            fps_ref=fps_ref, fps_dub=fps_tst, n_frames=N,
            duration_s=len(refv) / fps_ref, assigned_pct=assigned_pct,
            elapsed_s=time.perf_counter() - t0, trace=trace.to_list(), **_geom_kw)

    # --- ВИДЕО-КАРТА анализатором зрения (ЕДИНСТВЕННЫЙ путь): детект ступеней + ломаная вместо
    #     ската. tg_s со ступенями на резах (без maximum.accumulate) — резы перекроет тишина ниже. ---
    if _band is not None:
        _band.mark(0.88, "построение карты соответствия")
    # Длина рефа: по VFR-оси = время последнего кадра + средний кадр (индекс/fps на VFR врёт).
    dur_ref = (float(ax_ref[-1]) + 1.0 / fps_ref) if ax_ref is not None else len(refv) / fps_ref
    cos_asg = _cos_anchors(syn, refv, pred, asg)   # cos якорей: вес анализатора зрения И единого графика
    grid, tg_s, vision_cuts, _vo, _vw, _vcurve = _vision_build_map(
        pred, asg, cos_asg, fps_ref, fps_tst, dur_ref, DT, ax_ref=ax_ref, ax_dub=ax_dub)

    # --- ресэмпл аудио на сетку REF (low_mem: дубль-аудио и out через memmap, ресэмпл КУСКАМИ) ---
    # Раскладку дубля (2.0/5.1/7.1) НАСЛЕДУЕМ на выход: декодируем в РОДНОЕ число каналов,
    # буфер out на C каналов, варп применяется к каждому каналу. Анализ (band/muq) идёт по моно
    # (среднее всех каналов) — варп один на все каналы, фаза между каналами сохраняется.
    n_out = int(dur_ref * SR)
    _extract = _rep("extract", "декодирование звука озвучки")
    if _extract is not None:
        _extract.mark(0.0)
    dub_ch, dub_layout = probe_audio_channels(dub_audio, FFPROBE, atrack=dub_atrack)
    dub_delay = probe_av_delay(dub_audio, FFPROBE, atrack=dub_atrack)
    aud_cleanup = out_tmp = ref_tmp = None
    memlog('перед декодом аудио озвучки')
    if low_mem:
        tmp = Path(dub_audio).parent / "_tmp"; tmp.mkdir(parents=True, exist_ok=True)
        # CK2: аудио дубля из кэша (пропуск декода) или декод ПРЯМО в кэш (режим tmp).
        aud = (cache_mod.load_audio_mmap(cache_dir, dub_audio, dub_ch, atrack=dub_atrack)
               if (keep_tmp and cache_dir is not None) else None)
        _aud_src = "cache" if aud is not None else "decoded"
        if aud is None:
            dest = (cache_mod.audio_raw(cache_dir, dub_audio, dub_atrack)
                    if (keep_tmp and cache_dir is not None) else None)
            aud, aud_cleanup = _decode_audio_mmap(dub_audio, ffmpeg, audio_fix, channels=dub_ch,
                                                 atrack=dub_atrack, reporter=_extract,
                                                 dest=dest, delay_s=dub_delay)  # int16-memmap [N,C]
            if dest is not None:
                cache_mod.save_audio_meta(cache_dir, dub_audio, dub_ch, atrack=dub_atrack)   # CK2 мета (+EXT_VER)
        n_aud = len(aud)
        trace.event("dub_audio", state="audio", source=_aud_src, channels=dub_ch, layout=dub_layout,
                    atrack=dub_atrack, seconds=n_aud / SR, container_delay_s=dub_delay)
        out_tmp = Path(tempfile.mkdtemp(prefix="out_", dir=str(tmp))) / "out.f32"
        out = np.memmap(out_tmp, dtype=np.float32, mode="w+", shape=(n_out, dub_ch))
        BLK = 30 * SR                                    # 30с кусок: индексы и чтение аудио — по куску
        _resample = _rep("resample", "перекладка звука на таймлайн референса")
        for s1 in range(0, n_out, BLK):
            s2 = min(s1 + BLK, n_out)
            if _resample is not None:
                _resample(s2 / max(1, n_out))
            src = np.interp(np.arange(s1, s2) / SR, grid, tg_s) * SR
            # Полоса [a1:a2] покрывает src. clamp в [0,n_aud] обязателен: карта (vision/любая)
            # может указывать ЗА пределы аудио дубля (дубль КОРОЧЕ рефа → хвостовые блоки src за
            # концом дубля) → иначе a1≥a2 → пустой xp → ValueError. np.interp сам клампит src к
            # краям полосы → за концом дубля звучит край (бит-в-бит с не-low_mem веткой).
            a1 = min(max(0, int(np.floor(src.min())) - 1), n_aud - 1)
            a2 = max(min(n_aud, int(np.ceil(src.max())) + 2), a1 + 1)
            for ch in range(dub_ch):
                out[s1:s2, ch] = warp_interp(aud[a1:a2, ch], src - a1)   # xp=arange(a1,a2) → сдвиг -a1; GPU/CPU
        del aud
        memlog('после ресэмпла звука по зрению')
    else:
        aud = _decode_audio(dub_audio, ffmpeg, audio_fix, channels=dub_ch, atrack=dub_atrack,
                           reporter=_extract, delay_s=dub_delay)
        n_aud = len(aud)
        trace.event("dub_audio", state="audio", source="decoded", channels=dub_ch, layout=dub_layout,
                    atrack=dub_atrack, seconds=n_aud / SR, container_delay_s=dub_delay)
        t_out = np.arange(n_out) / SR
        t_syn_at = np.interp(t_out, grid, tg_s)
        src = t_syn_at * SR; sg = np.arange(len(aud))
        out = np.empty((n_out, dub_ch), np.float32)
        _resample = _rep("resample", "перекладка звука на таймлайн референса")
        if _resample is not None:
            _resample.mark(0.5)
        for ch in range(dub_ch):
            out[:, ch] = warp_interp(aud[:, ch], src)        # sg=arange(len(aud)) → grid; GPU/CPU
        del aud, t_out, t_syn_at, src, sg   # 1.2: освобождаем крупные индекс-массивы сразу после ресэмпла

    # --- ТИШИНА В ВЫРЕЗАХ = ЕДИНСТВЕННЫЙ источник: резы укладки зрения (build_curve). Ни
    #     cut_intervals (_level_decide), ни монотонизация в этот путь НЕ участвуют (один выверенный
    #     детектор). -Д-рез (дельта падает: в рефе есть кусок, которого НЕТ в дубле) -> ВЫРЕЗ ровно
    #     ширины |Д|/fps; после него дубль непрерывен (tg_s БЕЗ монотонизации). +Д-рез (лишнее у
    #     дубля) -> узкий шов склейки. band/muq ПОВЕРХ (видит тишину); _fill_silence_from_ref зальёт рефом.
    fill_spans: list[tuple[float, float]] = []           # вырезы (нет дубля) для графика и телеметрии
    for tc, v, te, tn in vision_cuts:                   # ВЫРЕЗ = промежуток МЕЖДУ якорями [te, tn]
        if v >= 0:
            continue
        a_s = float(te); b_s = float(tn)
        s1 = max(0, int(a_s * SR)); s2 = min(n_out, int(b_s * SR))
        if s2 - s1 <= 0:
            continue
        out[s1:s2] = 0.0
        if s1 - FADE >= 0:
            out[s1 - FADE:s1] *= np.linspace(1, 0, FADE)[:, None]
        if s2 + FADE <= n_out:
            out[s2:s2 + FADE] *= np.linspace(0, 1, FADE)[:, None]
        fill_spans.append((a_s, b_s))
    for tc, v, te, tn in vision_cuts:                   # ВСТАВКА: +Д-рез (лишнее у дубля) -> узкий шов
        if v > 0:
            a_s, b_s = float(max(0.0, tc - 0.15)), float(tc + 0.15)
            out[int(a_s * SR):min(n_out, int(b_s * SR))] = 0.0
            fill_spans.append((a_s, b_s))             # в fill_spans → график = РОВНО занулённое
    # КРАЕВОЙ вырез (голова/хвост, где дубля физически нет): тишина + fill_span вместо клампнутой
    # константы (build_curve экстраполирует линию за телом → warp клампит → аномалия рвёт band).
    _edge_silence(out, grid, tg_s, n_aud, SR, fill_spans, FADE)
    fill_spans.sort()

    # ЕДИНЫЙ график (зрение + аудио) строится НИЖЕ, ПОСЛЕ аудио-слоя (один на оба слоя).

    # --- реф-аудио на сетке REF (нужно аудио-слою band/muq/легаси И финальному заполнению тишины) ---
    # ref_audio передан (извлечён ОДИН раз на серию) → используем его, иначе извлекаем сами.
    ref_buf = None
    if audio_band or audio_muq:
        try:
            ra = ref_audio
            if ra is None and ref.src is not None:
                ra = _decode_audio(Path(ref.src), ffmpeg, False, atrack=ref_atrack,
                                  reporter=_rep("extract", "аудио рефа"))
            if ra is not None:
                mr = min(len(ra), n_out)
                if low_mem:
                    ref_tmp = Path(tempfile.mkdtemp(prefix="refbuf_", dir=str(Path(dub_audio).parent / "_tmp"))) / "ref.f32"
                    ref_buf = np.memmap(ref_tmp, dtype=np.float32, mode="w+", shape=(n_out, 2))
                    ref_buf[:mr] = ra[:mr]
                    if mr < n_out:
                        ref_buf[mr:] = 0.0                # хвост — тишина (как np.zeros)
                else:
                    ref_buf = np.zeros((n_out, 2), np.float32)
                    ref_buf[:mr] = ra[:mr]
        except Exception:  # noqa: BLE001 — нет аудио у рефа → откат на тишину
            ref_buf = None

    # --- аудио-слой band/muq (anchor): доводка по аудио ПОСЛЕ ресэмпла зрения. Видит реф/тишину в
    #     вырезе (как боевой off, на чём валидирован), а не сырой дубль → без ложного краевого реза. ---
    audio_resid_ms = 0.0
    audio_info: dict = {}
    band_on = audio_band and ref_buf is not None
    muq_on = (not band_on) and audio_muq and ref_buf is not None
    anchor_on = band_on or muq_on

    # --- настоящие вырезы: ТИШИНА (синхронное заполнение рефом — финальным проходом ниже) ---
    real_cuts_s = sum(b - a for a, b in fill_spans)      # вырезы УЖЕ занулены выше (резы build_curve)
    n_filled = 0                                 # переопределится ниже = секунд тишины залито рефом

    # --- ДЕТЕКТОР студийного A/V-десинка (read-only, на wav НЕ влияет) — НА ВИДЕО-УЛОЖЕННОМ out
    #     ДО band-доводки. ⚠ Замер ОБЯЗАН быть до band: band своими резами «размазывает» десинк →
    #     на финале детектор слепнет (MAD растёт, danger гаснет). Здесь out = чистая видео-укладка,
    #     сигнал устойчив (= gcc-свидетель band, MAD≈0). Большой И устойчивый PHAT-остаток M&E =
    #     аудио источника смещено относительно его видео (студия: оригинал на 1-2с, закадр записан
    #     по картинке) → дефект исходника, дорожка НЕ годится для дальнейшего. Поведение укладки и
    #     band НЕ трогаем — только сигнализируем КРАСНЫМ.
    memlog('перед буфером рефа')
    trace.event("vision_map", state=m.state, cuts=len(vision_cuts), silenced_spans=len(fill_spans),
                silenced_s=sum(b - a for a, b in fill_spans), dur_ref_s=dur_ref, n_anchors=int(len(asg)))
    dv = None
    if ref_buf is not None:
        try:
            from .anchor import apply as _anchor_det
            dv = _anchor_det.detect_av_desync(out, ref_buf, sr_audio=SR)
            # Diagnostics only: a constant offset is what the audio layer removes; the track verdict
            # is taken from what remains after it (_audio_verdict), never from this pre-layer value.
            trace.decide("av_desync", state="audio", verdict="offset" if (dv and dv["danger"]) else "ok",
                         inputs={k: dv[k] for k in ("lag_ms", "mad_ms", "n_windows") if dv and k in dv})
        except Exception as e:  # noqa: BLE001 — детектор не должен ронять conform
            trace.decide("av_desync", state="audio", verdict="error", inputs={"error": str(e)[:200]})
    else:
        trace.decide("av_desync", state="audio", verdict="skipped", inputs={"reason": "no reference audio"})

    # --- НОВЫЙ аудио-слой (Band/MuQ) — ПОСЛЕ заливки вырезов: видит реф/тишину в вырезе (как
    #     боевой off, на чём валидирован), а не сырой дубль → без ложного краевого реза. Варпит out.
    memlog('перед аудио-слоем')
    _audio = _rep("audio", "звуковой анализ") if anchor_on else None
    if anchor_on:
        from .anchor import apply as _anchor          # ленивый импорт: GPU+опц. transformers только при выборе
        if _audio is not None:
            _audio.mark(0.0)
        # CK4: кэш benv рефа (coarse_dtw) — реф переиспользуется между дублями эпизода/запусками (keep_tmp).
        _dsp_cache = (cache_mod.dsp_ref_path(cache_dir, ref.src, ref_atrack)
                      if (keep_tmp and cache_dir is not None and ref.src is not None) else None)
        audio_resid_ms = _anchor.audio_anchor(
            out, ref_buf, fps_ref, method=("muq" if muq_on else "band"),
            apply_cuts=apply_cuts, drift_speed_pct=drift_speed_pct, info=audio_info, progress=part(_audio, 0.0, 0.70),
            plot_dir=out_path.parent / "_plots", plot_stem=out_path.stem, render_own=False,
            vision_spans=fill_spans, dsp_cache=_dsp_cache)   # зоны тишины зрения + CK4-кэш benv рефа
            # render_own=False: band свой график НЕ рисует — conform строит ЕДИНЫЙ (зрение+аудио) ниже
        trace.event("audio_layer", state="audio", method=("muq" if muq_on else "band"),
                    resid_ms=audio_resid_ms, cuts=int(audio_info.get("audio_cuts", 0)),
                    max_step_ms=float(audio_info.get("audio_max_step_ms", 0.0)),
                    coverage=float(audio_info.get("audio_coverage", 0.0)),
                    span_ms=float(audio_info.get("audio_span_ms", 0.0)),
                    drift_ms=float(audio_info.get("audio_drift_ms", 0.0)),
                    excess_ms=_audio_excess_ms(audio_info),
                    global_offset_ms=float(audio_info.get("audio_global_offset_ms", 0.0)),
                    structure=audio_info.get("audio_structure"))
    else:
        trace.event("audio_layer", state="audio", method="none",
                    reason=("no reference audio" if (audio_band or audio_muq) else "audio layers disabled"))

    # --- ФИНАЛЬНОЕ заполнение ТИШИНЫ озвучки рефом: ТОЛЬКО ПОСЛЕ band/muq (дорожка синхронна
    #     рефу). В дырах озвучки (резы, вырезы, края), где реф звучит, подставляем синхронный
    #     реф. На несинхронной дорожке (аудио off) НЕ делаем — там вставка плодит рассинхрон. ---
    memlog('после аудио-слоя')
    if _audio is not None:
        _audio.mark(0.70, "заполнение тишины рефом")
    ref_filled_s = 0.0
    if fill_silence and anchor_on and ref_buf is not None:
        ref_filled_s = _fill_silence_from_ref(out, ref_buf)
        audio_info["ref_filled_s"] = round(ref_filled_s, 1)
    n_filled = int(round(ref_filled_s))          # PairResult.filled_cuts = секунд тишины залито рефом
    trace.event("fill_silence", state="audio", enabled=bool(fill_silence and anchor_on and ref_buf is not None),
                filled_s=ref_filled_s)

    # --- паспорт качества + предупреждения (read-only, на wav не влияет); до графика: шапка берёт вердикт ---
    metrics = _metrics(syn, refv, pred, asg, fps_ref, fps_tst, free_start)
    warns = _warnings(asg=asg, n_syn=N, pred=pred, fps_ref=fps_ref, fps_dub=fps_tst,
                      dur_ref=dur_ref, real_cuts_s=real_cuts_s, metrics=metrics,
                      creep_zones=creep_zones, freeze_runs=freeze_runs)
    critical: list[str] = []
    if anchor_on and audio_info:
        critical, audio_warns = _audio_verdict(audio_info, audio_resid_ms)
        warns += audio_warns
    verdict = "critical" if critical else ("warn" if warns else "ok")
    trace.decide("verdict", state="audio", verdict=verdict, inputs={"critical": critical, "warnings": warns})

    # --- ЕДИНЫЙ график укладки: ЗРЕНИЕ (всегда, видео-укладка) + АУДИО (если был anchor-слой).
    #     Read-only: падение не роняет conform. 1 панель (только зрение) / 2 панели (зрение+аудио). ---
    if _audio is not None:
        _audio.mark(0.78, "графики укладки")
    unified_plots: list[dict] = []
    try:
        from .anchor import plots_unified as _pu
        T_v = _make_T(dur_ref)                                        # сетка графика = от реальной длины
        o_v, w_v = _vision_ow(pred, asg, cos_asg, fps_ref, fps_tst, T=T_v,
                              ax_ref=ax_ref, ax_dub=ax_dub)
        t_ref_v = (ax_ref[pred[asg]] if ax_ref is not None
                   else pred[asg].astype(np.float64) / fps_ref)
        t_dub_v = (ax_dub[asg] if ax_dub is not None
                   else asg.astype(np.float64) / fps_tst)
        shift_v = (t_dub_v - t_ref_v) * fps_ref
        shift_T = np.interp(T_v, grid, (tg_s - grid) * fps_ref)       # боевая карта → сдвиг(кадры) на T
        cuts_g = [(float(tc), float(v)) for tc, v, te, tn in vision_cuts]   # резы детекта зрения (tc,Δ)
        # ГРАФИК: линия укладки РВЁТСЯ на резах (разрыв = разрыв, не вертикаль через скачок) +
        # пунктирные вертикали границ [te,tn]. shift_T_plot — копия ТОЛЬКО для графика (NaN на резах);
        # curve_fr в npz и tg_s остаются чистыми (NaN их бы сломал).
        shift_T_plot = shift_T.copy(); cut_marks: list[float] = []
        for tc, v, te, tn in vision_cuts:
            m = (T_v >= te) & (T_v <= tn)
            if m.any():
                shift_T_plot[m] = np.nan
            else:                                                     # узкая вставка между узлами → ближайший узел
                shift_T_plot[int(np.argmin(np.abs(T_v - tc)))] = np.nan
            cut_marks += [float(te), float(tn)]
        sa, sb = _vision_global_trend(o_v, w_v, T_v, fps_ref)          # ТОТ ЖЕ масштаб, что в карте —
        vision_d = dict(o=o_v, w=w_v, curve=shift_T_plot, cuts=cuts_g, cut_marks=cut_marks,
                        T=np.asarray(T_v, np.float32),
                        t_ref=t_ref_v.astype(np.float32), shift_fr=shift_v.astype(np.float32),
                        cos=np.asarray(cos_asg, np.float32),
                        scale_a=float(sa), scale_b=float(sb),         # база графика (дрейф-форма) без «пилы»
                        fill_spans=fill_spans,                        # вырезы (нет дубля) = РОВНО занулённое в out
                        head_s=float(t_ref_v.min()) if len(t_ref_v) else 0.0, dur=dur_ref)
        # RAW-дамп ЗРЕНИЯ (.npz + .json) рядом с графиком — ВСЕ данные верхней панели + БОЕВАЯ
        # карта варпа (grid→tg_s) + сырые видео-якоря, чтобы диагностировать укладку зрения
        # БЕЗ перегенерации (паритет с band-__raw.npz). Дамп не должен ронять conform.
        try:
            import json as _json
            _pd = out_path.parent / "_plots"; _pd.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                _pd / f"{out_path.stem}__vision.npz",
                T=np.asarray(T_v, np.float32), o=o_v.astype(np.float32), w=w_v.astype(np.float32),
                curve_fr=shift_T.astype(np.float32),               # боевая карта зрения, сдвиг (кадры) на T
                cuts=np.asarray(cuts_g, np.float64).reshape(-1, 2),  # (tc_сек, v_кадры)
                t_ref=t_ref_v.astype(np.float32),                   # время рефа сырых якорей, с
                shift_fr=shift_v.astype(np.float32),                # сдвиг сырых якорей, кадры рефа
                cos=np.asarray(cos_asg, np.float32),                # уверенность якоря (cos)
                asg=np.asarray(asg, np.int32),                      # кадры дубля с привязкой
                pred_asg=np.asarray(pred[asg], np.int32),           # их кадры рефа (pred)
                grid=np.asarray(grid, np.float32),                  # сетка ресэмпла, с (реф)
                tg_s=np.asarray(tg_s, np.float64),                  # ПРИМЕНЁННАЯ карта: время дубля, с
                fill_spans=(np.asarray(fill_spans, np.float64).reshape(-1, 2)
                            if fill_spans else np.zeros((0, 2), np.float64)),  # вырезы (нет дубля) = занулённое
                cut_intervals=(np.asarray(cut_intervals, np.int64).reshape(-1, 2)
                               if cut_intervals else np.zeros((0, 2), np.int64)),  # legacy _level_decide (CK3/legacy auto)
                stamp=np.asarray(_astamp),                          # штамп CK3 (валидность переиспользования)
                fps_ref=np.float64(fps_ref), fps_dub=np.float64(fps_tst),
                scale_a=np.float64(sa), scale_b=np.float64(sb),     # реальный масштаб (медиана наклонов)
                dur_ref=np.float64(dur_ref))
            (_pd / f"{out_path.stem}__vision.json").write_text(_json.dumps({
                "fps_ref": float(fps_ref), "fps_dub": float(fps_tst), "dur_ref": float(dur_ref),
                "scale_pct": float(sa) / max(float(fps_ref), 1e-6) * 100.0,  # масштаб дубля, %/с
                "n_anchors": int(len(asg)),
                "median_shift_fr": float(np.median(shift_v)) if len(shift_v) else 0.0,
                "cuts": [[float(tc), float(v)] for tc, v in cuts_g],
                "npz": f"{out_path.stem}__vision.npz",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001 — дамп зрения не должен ронять conform
            pass
        audio_d = audio_info.get("band_layers")                       # None → только зрение (1 панель)
        passport = _pair_passport(
            vision_cuts=vision_cuts, fill_spans=fill_spans, freeze_runs=freeze_runs, audio_info=audio_info,
            shift_T=shift_T, T_v=T_v, scale_a=sa, fps_ref=fps_ref, assigned_pct=assigned_pct,
            metrics=metrics, tc_ref=tc_ref, tc_dub=tc_dub, geom_info=geom_info, mirror=bool(m.mirror),
            container_delay=dub_delay, anchor_on=anchor_on, audio_resid_ms=audio_resid_ms,
            av_desync=dv, verdict=verdict, verdict_text=(critical or warns or [""])[0])
        unified_plots = _pu.render_unified(out_path.parent / "_plots", out_path.stem,
                                           vision=vision_d, audio=audio_d, title=out_path.stem,
                                           passport=passport)
    except Exception:  # noqa: BLE001 — графики не должны ронять conform
        unified_plots = []

    _write = _rep("write", out_path.name)
    if _write is not None:
        _write.mark(0.0)
    memlog('перед записью файла')
    try:
        _write_audio_streamed(out_path, out, ffmpeg, layout=dub_layout, on_prog=_write)  # FLAC, раскладка дубля
        trace.event("write", state="output", path=out_path, layout=dub_layout, seconds=n_out / SR,
                    bytes=out_path.stat().st_size if out_path.exists() else 0)
    finally:
        # ⚠ Уборка ОБЯЗАНА идти при любом исходе. Раньше она стояла просто после записи:
        # запись падала — и десятки гигабайт промежуточных файлов оставались лежать
        # (случай 2026-08-07: 34.6 ГБ на сетевом диске от двух оборванных прогонов).
        if low_mem:                                      # очистка memmap-временных (out/ref_buf/дубль-аудио)
            del out
            if ref_buf is not None:
                del ref_buf
            for d in (out_tmp.parent if out_tmp else None,
                      ref_tmp.parent if ref_tmp else None, aud_cleanup):
                tmpfiles.drop_dir(d)          # подключения закрыты выше (del), проверка внутри

    # The vision layout is the pair's time map; stored so subtitles can follow the audio.
    try:
        from .subs_transfer import save_time_map
        save_time_map(out_path, _make_T(dur_ref), _vcurve, vision_cuts, fps_ref, dur_ref)
        trace.event("time_map", state="output", cuts=len(vision_cuts))
    except Exception as e:  # noqa: BLE001
        trace.event("time_map", state="output", error=str(e)[:200])

    if low_cos_proceed:                               # назн% < порога, но грубый проход подтвердил серию
        warns.insert(0, f"низкий косинус энкода (назн {assigned_pct:.0f}% < {ABORT_ASSIGNED_PCT:.0f}%): "
                        f"отсечка «чужое видео» НЕ применена — грубый проход подтвердил серию "
                        f"(n_keep={n_keep}); ПРОВЕРИТЬ вручную")
    return PairResult(
        dub=name, out_path=out_path, ok=True,
        fps_ref=fps_ref, fps_dub=fps_tst, n_frames=N, duration_s=dur_ref,
        assigned_pct=100.0 * len(asg) / max(1, N),
        slope=metrics["slope"], cos_median=metrics["cos_median"],
        monotonic_violations=metrics["mono"],
        real_cuts=fill_spans, real_cuts_s=real_cuts_s,
        filled_cuts=n_filled,
        blind_zones=n_blind, blind_restored=n_restore, edge_recovered=n_recovered,
        dropped_intro_s=metrics["intro_s"], audio_resid_ms=audio_resid_ms,
        audio_cuts=int(audio_info.get("audio_cuts", 0)),
        audio_max_step_ms=float(audio_info.get("audio_max_step_ms", 0.0)),
        audio_coverage=float(audio_info.get("audio_coverage", 0.0)),
        audio_span_ms=float(audio_info.get("audio_span_ms", 0.0)),
        plots=unified_plots,
        warnings=warns,
        critical=critical,
        elapsed_s=time.perf_counter() - t0, trace=trace.to_list(), **_geom_kw,
    )


def _metrics(syn, refv, pred, asg, fps_ref, fps_tst, free_start) -> dict:
    if len(asg) == 0:
        return {"slope": 0.0, "cos_median": 0.0, "mono": 0, "intro_s": 0.0}
    cosp = np.empty(len(asg), np.float32)
    CH = 4096
    pr = pred[asg]
    for i in range(0, len(asg), CH):
        sl = asg[i:i + CH]
        cosp[i:i + len(sl)] = np.einsum("ij,ij->i", syn[sl].astype(np.float32),
                                        refv[pred[sl]].astype(np.float32))
    A = np.vstack([asg, np.ones(len(asg))]).T
    slope = float(np.linalg.lstsq(A, pr, rcond=None)[0][0])
    mono = int((np.diff(pr) < 0).sum())
    intro = 0
    if free_start:
        k0 = int(asg[0])
        intro = k0 if (pred[:k0] == -1).all() else 0
    return {"slope": slope, "cos_median": float(np.median(cosp)),
            "mono": mono, "intro_s": intro / fps_tst}


def _mmss(t: float) -> str:
    t = max(0, int(t)); return f"{t // 60}:{t % 60:02d}"


def _pair_passport(*, vision_cuts, fill_spans, freeze_runs, audio_info, shift_T, T_v, scale_a, fps_ref,
                   assigned_pct, metrics, tc_ref, tc_dub, geom_info, mirror, container_delay, anchor_on,
                   audio_resid_ms, av_desync, verdict, verdict_text) -> dict:
    """One passport of the pair for both chart renders: zones and events with their kind and size,
    layout segments whose speed differs from the global scale, header metrics of vision and hearing,
    and the verdict. Numbers only; labels and units come from the chart's term catalog."""
    st = audio_info.get("audio_structure") or {}
    zones = ([{"kind": "vision_cut", "a": float(a), "b": float(b)} for a, b in fill_spans]
             + [{"kind": "video_freeze", "a": float(a), "b": float(b)} for a, b in freeze_runs]
             + [{"kind": "audio_gap", "a": float(a), "b": float(b)} for a, b in st.get("gaps", [])]
             + [{"kind": "dtw_cut", "a": float(a), "b": float(b)}
                for a, b in (audio_info.get("band_layers") or {}).get("dtw_cut_zones", [])])
    events = ([{"kind": "vision_step", "t": float(tc), "value": float(v) / fps_ref} for tc, v, _te, _tn in vision_cuts]
              + [{"kind": "audio_cut", "t": float(t), "value": float(v) * VFRAME} for t, v in audio_info.get("cuts", [])]
              + [{"kind": "audio_jump", "t": float(t), "value": float(d)} for t, d in st.get("jumps", [])]
              + [{"kind": "dtw_insert", "t": float(tc), "value": float(ln)}
                 for tc, ln in (audio_info.get("band_layers") or {}).get("dtw_inserts", [])])
    # Layout speed per segment between vision steps: slope of the applied map against the global scale.
    segments = []
    bounds = [0.0] + sorted(float(tc) for tc, *_ in vision_cuts) + [float(T_v[-1])]
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = (T_v >= a) & (T_v <= b) & np.isfinite(shift_T)
        if m.sum() >= 10 and b - a >= 20.0:
            slope = float(np.polyfit(T_v[m], shift_T[m], 1)[0])          # frames of layout per second
            segments.append({"a": a, "b": b, "speed_pct": (slope - scale_a) / fps_ref * 100.0})
    vision_line = [
        {"key": "assigned", "value": float(assigned_pct)},
        {"key": "cos", "value": float(metrics["cos_median"])},
        {"key": "scale", "value": float(scale_a) / max(float(fps_ref), 1e-6) * 100.0},
        {"key": "telecine", "tpl": "value.telecine",
         "value": {"ref": float(tc_ref["tele_score"]), "dub": float(tc_dub["tele_score"])}},
        {"key": "tc_dropped", "value": int(tc_ref["dropped"]) + int(tc_dub["dropped"])},
        {"key": "geom", "tpl": "value.geom",
         "value": ({"sx": float(geom_info["sx"]), "sy": float(geom_info["sy"]), "n": int(geom_info["n_in"])}
                   if geom_info else None)},
        {"key": "mirror", "value": bool(mirror)},
        {"key": "container_delay", "value": float(container_delay)},
        {"key": "vision_cuts", "tpl": "value.count_dur",
         "value": {"n": len(fill_spans), "dur": float(sum(b - a for a, b in fill_spans))}},
        {"key": "freezes", "tpl": "value.count_dur",
         "value": {"n": len(freeze_runs), "dur": float(sum(b - a for a, b in freeze_runs))}},
    ]
    audio_line = []
    if anchor_on and audio_info:
        audio_line = [
            {"key": "method", "value": str(audio_info.get("anchor_method", ""))},
            {"key": "structure", "tpl": "value.structure",
             "value": ({"plateaus": len(st["plateaus"]), "jumps": len(st["jumps"])} if st else None)},
            {"key": "resid", "value": float(abs(audio_resid_ms))},
            {"key": "cuts", "value": int(audio_info.get("audio_cuts", 0))},
            {"key": "max_step", "value": float(audio_info.get("audio_max_step_ms", 0.0))},
            {"key": "coverage", "value": float(audio_info.get("audio_coverage", 0.0)) * 100.0},
            {"key": "excess", "value": _audio_excess_ms(audio_info) / 1000.0},
            {"key": "filled", "value": float(audio_info.get("ref_filled_s", 0.0))},
            {"key": "av_offset", "tpl": "value.av_offset",
             "value": ({"lag": float(av_desync["lag_ms"]), "mad": float(av_desync["mad_ms"])} if av_desync else None)},
        ]
    return {"zones": zones, "events": events, "segments": segments,
            "header": [vision_line, audio_line], "verdict": verdict, "verdict_text": verdict_text}


def _audio_excess_ms(audio_info: dict) -> float:
    """Cut movement that cancelled itself out: Σ|steps| − |Σsteps|. Real audio edits accumulate in one
    direction; a layer chasing a blind measurement goes back and forth and this grows to seconds."""
    return max(0.0, float(audio_info.get("audio_sum_ms", 0.0)) - abs(float(audio_info.get("audio_net_ms", 0.0))))


def _audio_verdict(audio_info: dict, resid_ms: float) -> tuple[list[str], list[str]]:
    """Track verdict from what remains AFTER the audio layer -> (critical, warnings). One chokepoint
    for both modes: constant offsets the layer removed never count, only its result does."""
    crit: list[str] = []; warn: list[str] = []
    resid = abs(float(resid_ms))
    cov = float(audio_info.get("audio_coverage", 0.0))
    cuts = int(audio_info.get("audio_cuts", 0)); excess = _audio_excess_ms(audio_info)
    drift = float(audio_info.get("audio_drift_ms", 0.0))
    if cov < AUDIO_COVERAGE_BLIND:
        crit.append(f"у файлов почти нет общего звука (опора {cov * 100:.0f}% длительности) — "
                    f"синхронность выхода не измерена")
    elif cov < AUDIO_COVERAGE_LOW:
        warn.append(f"звуковое измерение имело опору лишь на {cov * 100:.0f}% длительности")
    if resid > AUDIO_RESID_MAX_MS:
        crit.append(f"остаток после доводки {resid:.0f} мс — больше допустимых ±{AUDIO_RESID_MAX_MS:.0f} мс")
    if excess > AUDIO_EXCESS_CRIT_MS:
        crit.append(f"слух метался: {cuts} резов, взаимно погашено {excess / 1000:.1f} с хода — "
                    f"синхронность выхода не гарантирована")
    elif excess > AUDIO_EXCESS_WARN_MS:
        warn.append(f"резы туда-обратно: {cuts} резов, взаимно погашено {excess / 1000:.1f} с хода — проверить")
    if abs(drift) > 1000.0:
        warn.append(f"сдвиг звука уходит на {drift / 1000:+.1f} с — вероятно, дефект исходного файла")
    st = audio_info.get("audio_structure")
    if st:
        gap_s = sum(b - a for a, b in st["gaps"])
        jumps = ", ".join(f"{j[1]:+.1f} с на {j[0]:.0f} с" for j in st["jumps"])
        warn.append(f"структура звука: {len(st['plateaus'])} плато" + (f", скачки {jumps}" if jumps else "")
                    + (f"; звука озвучки нет {gap_s:.0f} с — залито рефом" if gap_s else ""))
    return crit, warn


def _warnings(*, asg, n_syn, pred, fps_ref, fps_dub, dur_ref, real_cuts_s,
              metrics, creep_zones=(), freeze_runs=()) -> list[str]:
    """Эвристические предупреждения «обрати внимание» (НА wav НЕ влияют). Терпимы к ложным:
    лучше пере-предупредить. Каждое — с причиной-ярлыком, чтобы ложное было легко отмести."""
    w: list[str] = []
    if len(asg) < 2:
        return ["почти нет сопоставленных кадров видео"]
    for j0, j1, cz, sl in creep_zones:
        w.append(f"налипшая вставка выброшена: {_mmss(j0 / fps_dub)}–{_mmss(j1 / fps_dub)} "
                 f"дубля (~{(j1 - j0 + 1) / fps_dub:.0f}с, cos {cz:.2f}, ход {sl:.2f}×)")
    for a, b in freeze_runs:
        w.append(f"кадр озвучки не меняется {_mmss(a)}–{_mmss(b)} ({b - a:.0f}с) — замершая картинка или статичная заставка")
    exp = (fps_ref / fps_dub) if fps_dub else 1.0
    # — геометрия —
    if metrics["slope"] and abs(metrics["slope"] - exp) > 0.05:
        w.append(f"ход времени {metrics['slope']:.3f} вместо ожидаемого {exp:.3f}")
    if metrics["mono"] > max(20, int(0.01 * len(asg))):
        w.append(f"нарушений монотонности {metrics['mono']}")
    # — покрытие видео —
    ap = 100.0 * len(asg) / max(1, n_syn)
    if ap < 85.0:
        w.append(f"сопоставлено лишь {ap:.0f}% кадров")
    d = np.diff(asg)
    if len(d):
        i = int(np.argmax(d)); gap_s = (int(d[i]) - 1) / fps_dub
        if gap_s > max(15.0, 0.06 * dur_ref):
            w.append(f"крупный неразмеченный участок ~{gap_s:.0f}с (≈{_mmss(pred[asg[i]] / fps_ref)}) — вставка или сбой")
    # — заливка / начало —
    ff = real_cuts_s / max(1e-6, dur_ref)
    if ff > 0.15:
        w.append(f"звуком референса заполнено {ff * 100:.0f}% длительности")
    if metrics["intro_s"] > 5.0:
        w.append(f"выпало/залито начало ~{metrics['intro_s']:.0f}с")
    return w


def _align_audio_only(
    ref: SrmFeatures,
    dub_audio: Path,
    out_path: Path,
    *,
    ffmpeg: str = FFMPEG,
    progress=None,
    progress_meta: tuple[int, int, str] = (0, 0, ""),
    ref_audio=None,
    low_mem: bool = False,
    cache_dir: Path | str | None = None,
    keep_tmp: bool = False,
    dub_name: str | None = None,
    fill_silence: bool = False,
    audio_band: bool = False,
    audio_muq: bool = False,
    apply_cuts: bool = True,
    drift_speed_pct: float = 1.25,   # как в _align: потолок скорости кривой дрейфа band/muq, %/с
    ref_atrack: int = 0,             # ⭐ 5.1: аудиодорожка рефа (звуковой эталон)
    dub_atrack: int = 0,             # ⭐ 5.1: аудиодорожка дубля (голое аудио обычно одна, но mka бывают многодорожечными)
    should_stop=None,
    trace: Trace | None = None,
    **_video_only_opts,          # free_start/recover_edges/fps_dub/audio_fix — видео-понятия, в аудио-only не участвуют
) -> PairResult:
    """⭐ АУДИО-ONLY conform: озвучка
    БЕЗ видеопотока (голый flac/mka/mp3/aac…). Зрения нет по построению — дубль кладётся на
    таймлайн рефа КАК ЕСТЬ (identity, от нуля), после чего ВЕСЬ рассинхрон снимает штатный
    аудио-слой ровно тем же конвейером, что и после зрения:
      `_global_prealign` (константный сдвиг, охват ±70с) → `coarse_dtw` (вставки/вырезы вне
      окна полосы ±2.5с) → band-доводка (дрейф + резы ≤2.5с) → заливка тишины рефом.
    Графики: band рисует СВОЙ трек (`render_own=True`, `<stem>__track.png/html` в `_plots`) —
    панель зрения не строится (нечего показывать).

    Границы метода (по построению, предупреждаем — не чиним): нелинейный видео-масштаб
    (PAL-ускорение и т.п. зрение снимало из данных, слуху здесь взять неоткуда — прямая
    скорость аудио) и структурные расхождения за пределами охвата prealign/DTW.
    PairResult.mode="audio"; поля зрения (assigned/cos/slope) не имеют смысла и остаются 0."""
    t0 = time.perf_counter()

    def _rep(stage: str, detail: str = "") -> Reporter | None:
        return Reporter.of(progress, stage, progress_meta, detail)

    dub_audio = Path(dub_audio)
    name = dub_name or dub_audio.name
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace = trace if trace is not None else Trace(name)

    fps_ref = ref.fps
    dur_ref = ref.duration_s
    if not (dur_ref and dur_ref > 0 and fps_ref and fps_ref > 0):
        raise ValueError(f"реф пуст/битый (dur={dur_ref}, fps={fps_ref}) — аудио-only без длины рефа невозможен")
    n_out = int(dur_ref * SR)
    dub_ch, dub_layout = probe_audio_channels(dub_audio, FFPROBE, atrack=dub_atrack)
    dub_delay = probe_av_delay(dub_audio, FFPROBE, atrack=dub_atrack)
    trace.event("dub_audio", state="audio-only", channels=dub_ch, layout=dub_layout, atrack=dub_atrack,
                dur_ref_s=dur_ref, container_delay_s=dub_delay)

    # --- декод дубля + identity-укладка на таймлайн рефа (хвост за длиной рефа — отрез,
    #     нехватка — тишина; сдвиги/вырезы дальше разберёт аудио-слой) ---
    aud_cleanup = out_tmp = ref_tmp = None
    if low_mem:
        tmp = dub_audio.parent / "_tmp"; tmp.mkdir(parents=True, exist_ok=True)
        aud = (cache_mod.load_audio_mmap(cache_dir, dub_audio, dub_ch, atrack=dub_atrack)
               if (keep_tmp and cache_dir is not None) else None)
        if aud is None:
            dest = (cache_mod.audio_raw(cache_dir, dub_audio, dub_atrack)
                    if (keep_tmp and cache_dir is not None) else None)
            aud, aud_cleanup = _decode_audio_mmap(dub_audio, ffmpeg, True, channels=dub_ch,
                                                  atrack=dub_atrack,
                                                  reporter=_rep("extract", "декодирование звука озвучки"),
                                                  dest=dest, delay_s=dub_delay)
            if dest is not None:
                cache_mod.save_audio_meta(cache_dir, dub_audio, dub_ch, atrack=dub_atrack)   # CK2 мета (+EXT_VER)
        out_tmp = Path(tempfile.mkdtemp(prefix="out_", dir=str(tmp))) / "out.f32"
        out = np.memmap(out_tmp, dtype=np.float32, mode="w+", shape=(n_out, dub_ch))
        m = min(len(aud), n_out)
        BLK = 60 * SR                                    # поблочно: RAM O(блока), как весь low_mem-путь
        for s1 in range(0, m, BLK):
            out[s1:min(s1 + BLK, m)] = aud[s1:min(s1 + BLK, m)]
        if m < n_out:
            out[m:] = 0.0
        del aud
    else:
        aud = _decode_audio(dub_audio, ffmpeg, True, channels=dub_ch, atrack=dub_atrack,
                            reporter=_rep("extract", "декодирование звука озвучки"), delay_s=dub_delay)
        out = np.zeros((n_out, dub_ch), np.float32)
        m = min(len(aud), n_out)
        out[:m] = aud[:m]
        del aud

    # --- реф-аудио (как в _align: переданное с серии ИЛИ извлечь из ref.src) ---
    ref_buf = None
    if audio_band or audio_muq:
        try:
            ra = ref_audio
            if ra is None and ref.src is not None:
                ra = _decode_audio(Path(ref.src), ffmpeg, True, atrack=ref_atrack,
                                  reporter=_rep("extract", "аудио рефа"))
            if ra is not None:
                mr = min(len(ra), n_out)
                if low_mem:
                    ref_tmp = Path(tempfile.mkdtemp(prefix="refbuf_", dir=str(dub_audio.parent / "_tmp"))) / "ref.f32"
                    ref_buf = np.memmap(ref_tmp, dtype=np.float32, mode="w+", shape=(n_out, 2))
                    ref_buf[:mr] = ra[:mr]
                    if mr < n_out:
                        ref_buf[mr:] = 0.0
                else:
                    ref_buf = np.zeros((n_out, 2), np.float32)
                    ref_buf[:mr] = ra[:mr]
        except Exception:  # noqa: BLE001 — нет аудио у рефа → слой невозможен
            ref_buf = None

    band_on = audio_band and ref_buf is not None
    muq_on = (not band_on) and audio_muq and ref_buf is not None
    anchor_on = band_on or muq_on
    audio_resid_ms = 0.0
    audio_info: dict = {}
    _audio = _rep("audio", "звуковой анализ (файл без видеоряда)") if anchor_on else None
    if anchor_on:
        from .anchor import apply as _anchor
        if _audio is not None:
            _audio.mark(0.0)
        _dsp_cache = (cache_mod.dsp_ref_path(cache_dir, ref.src, ref_atrack)
                      if (keep_tmp and cache_dir is not None and ref.src is not None) else None)
        audio_resid_ms = _anchor.audio_anchor(
            out, ref_buf, fps_ref, method=("muq" if muq_on else "band"),
            apply_cuts=apply_cuts, drift_speed_pct=drift_speed_pct, info=audio_info, progress=part(_audio, 0.0, 0.70),
            plot_dir=out_path.parent / "_plots", plot_stem=out_path.stem,
            render_own=True,                       # свой график band: панели зрения в этом режиме нет
            vision_spans=None, dsp_cache=_dsp_cache)
        trace.event("audio_layer", state="audio-only", method=("muq" if muq_on else "band"),
                    resid_ms=audio_resid_ms, cuts=int(audio_info.get("audio_cuts", 0)),
                    max_step_ms=float(audio_info.get("audio_max_step_ms", 0.0)),
                    coverage=float(audio_info.get("audio_coverage", 0.0)),
                    span_ms=float(audio_info.get("audio_span_ms", 0.0)),
                    drift_ms=float(audio_info.get("audio_drift_ms", 0.0)))
    else:
        trace.event("audio_layer", state="audio-only", method="none",
                    reason=("no reference audio" if (audio_band or audio_muq) else "audio layers disabled"))

    if _audio is not None:
        _audio.mark(0.70, "заполнение тишины рефом")
    ref_filled_s = 0.0
    if fill_silence and anchor_on and ref_buf is not None:
        ref_filled_s = _fill_silence_from_ref(out, ref_buf)
        audio_info["ref_filled_s"] = round(ref_filled_s, 1)
    trace.event("fill_silence", state="audio-only", enabled=bool(fill_silence and anchor_on and ref_buf is not None),
                filled_s=ref_filled_s)

    # --- предупреждения (read-only): режим без зрения обязан честно сигналить о слепоте ---
    warns: list[str] = []; critical: list[str] = []
    if not anchor_on:
        warns.append("аудио-only БЕЗ аудио-слоя (band/muq выключены или у рефа нет аудио): "
                     "дорожка уложена КАК ЕСТЬ, выравнивание не выполнялось")
    else:
        critical, warns = _audio_verdict(audio_info, audio_resid_ms)
    trace.decide("verdict", state="audio-only", verdict="critical" if critical else ("warn" if warns else "ok"),
                 inputs={"critical": critical, "warnings": warns})

    _write = _rep("write", out_path.name)
    if _write is not None:
        _write.mark(0.0)
    try:
        _write_audio_streamed(out_path, out, ffmpeg, layout=dub_layout, on_prog=_write)
        trace.event("write", state="output", path=out_path, layout=dub_layout, seconds=n_out / SR,
                    bytes=out_path.stat().st_size if out_path.exists() else 0)
    finally:
        if low_mem:                                      # уборка при любом исходе (см. видео-путь)
            del out
            if ref_buf is not None:
                del ref_buf
            for d in (out_tmp.parent if out_tmp else None,
                      ref_tmp.parent if ref_tmp else None, aud_cleanup):
                tmpfiles.drop_dir(d)          # подключения закрыты выше (del), проверка внутри

    return PairResult(
        dub=name, out_path=out_path, ok=True, mode="audio",
        fps_ref=fps_ref, duration_s=dur_ref,
        filled_cuts=int(round(ref_filled_s)),
        audio_resid_ms=audio_resid_ms,
        audio_cuts=int(audio_info.get("audio_cuts", 0)),
        audio_max_step_ms=float(audio_info.get("audio_max_step_ms", 0.0)),
        audio_coverage=float(audio_info.get("audio_coverage", 0.0)),
        audio_span_ms=float(audio_info.get("audio_span_ms", 0.0)),
        plots=list(audio_info.get("plots", [])),
        warnings=warns, critical=critical,
        elapsed_s=time.perf_counter() - t0, trace=trace.to_list(),
    )


def conform_pair(
    ref: SrmFeatures,
    dub_video: Path | str,
    out_path: Path | str,
    *,
    fps_dub: float | None = None,
    ffmpeg: str = FFMPEG,
    progress=None,
    should_stop=None,
    progress_meta: tuple[int, int, str] = (0, 0, ""),
    low_mem: bool = False,
    cache_dir: Path | str | None = None,
    keep_tmp: bool = False,
    **opts,
) -> PairResult:
    """Удобная обёртка: построить фичи озвучки и выровнять её на ref → out_path.

    low_mem=True → SRM дубля строится ПОТОКОМ на диск (`_tmp/.../dub.f16`, memmap),
    в RAM не копится; файл удаляется после пары. Результат бит-в-бит как in-RAM.

    keep_tmp=True + cache_dir → ЧЕКПОИНТ CK1: SRM дубля кэшируется в `cache_dir`
    (`epXX/_conform_cache`, как реф) и при повторном запуске берётся оттуда — пропуск
    тяжёлого GPU-декода. Ключ=файл+EMB_VER (cache.load_srm)."""
    dub_video = Path(dub_video)
    out_path = Path(out_path)
    trace = Trace(dub_video.name)
    # One chokepoint for every outcome: the trace reaches the result and the disk whether the
    # pair succeeded, was rejected by a gate or crashed; a crash alone must not lose the record.
    try:
        res = _conform_pair(ref, dub_video, out_path, trace, fps_dub=fps_dub, ffmpeg=ffmpeg,
                            progress=progress, should_stop=should_stop, progress_meta=progress_meta,
                            low_mem=low_mem, cache_dir=cache_dir, keep_tmp=keep_tmp, **opts)
    except Exception as e:  # noqa: BLE001 — one pair must not take the episode down
        logger.exception("conform: озвучка {} упала на серии {}", dub_video.name, ref.src)
        trace.event("exception", state="error", type=type(e).__name__, error=str(e)[:500])
        res = PairResult(dub=dub_video.name, out_path=None, ok=False, error=str(e), trace=trace.to_list())
    trace.save(out_path)
    return res


def _conform_pair(ref, dub_video: Path, out_path: Path, trace: Trace, *, fps_dub, ffmpeg, progress,
                  should_stop, progress_meta, low_mem, cache_dir, keep_tmp, **opts) -> PairResult:
    # A dub without a video stream cannot be matched by vision; the audio-only path lays it down
    # as is and lets the audio layer do all the alignment. Decided from the file, never a switch.
    if not probe_has_video(dub_video):
        trace.decide("mode", state="probe", verdict="audio", inputs={"has_video": False})
        return _align_audio_only(ref, dub_video, out_path, ffmpeg=ffmpeg,
                                 progress=progress, progress_meta=progress_meta,
                                 low_mem=low_mem, cache_dir=cache_dir, keep_tmp=keep_tmp,
                                 dub_name=dub_video.name, should_stop=should_stop, trace=trace, **opts)
    trace.decide("mode", state="probe", verdict="av", inputs={"has_video": True})
    _decode = Reporter.of(progress, "decode", progress_meta, "разбор файла")
    if _decode is not None:
        _decode.mark(0.0)
    # CK1: попытка взять SRM дубля из кэша серии (пропуск GPU-декода).
    dub = cache_mod.load_srm(cache_dir, dub_video) if (keep_tmp and cache_dir is not None) else None
    trace.decide("ck1_srm", state="cache", source="cache", verdict="hit" if dub is not None else "miss",
                 inputs={"checkpoints": bool(keep_tmp and cache_dir is not None)})
    if dub is not None:
        if fps_dub is not None:
            dub.fps = float(fps_dub)
        return conform_features(ref, dub, dub_video, out_path, ffmpeg=ffmpeg,
                                progress=progress, dub_name=dub_video.name,
                                progress_meta=progress_meta, low_mem=low_mem,
                                cache_dir=cache_dir, keep_tmp=keep_tmp, trace=trace, **opts)
    # Нет валидного CK1 → строим SRM. В tmp — ПРЯМО в кэш (не удаляем); иначе в _tmp (удаляем).
    mp = None; into_cache = False
    if keep_tmp and cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        mp = cache_mod.srm_file(cache_dir, dub_video); into_cache = True
    elif low_mem:
        tmpdir = dub_video.parent / "_tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        mp = Path(tempfile.mkdtemp(prefix="srm_", dir=str(tmpdir))) / "dub.f16"
    with decode_backend(probe_resolution(dub_video), ffmpeg) as _be:   # 1080+ → GPU (потолок NVDEC), иначе CPU
        dub = build_srm(dub_video, fps_dub, ffmpeg=ffmpeg, reporter=_decode,
                        should_stop=should_stop, mmap_path=mp, backend=_be)
    trace.event("srm_dub", state="plain", source="decoded", frames=len(dub.srm), fps=dub.fps,
                vfr=dub.pts is not None, cached=into_cache)
    if into_cache:
        cache_mod.save_meta(cache_dir, dub_video, len(dub.srm), dub.fps)   # CK1 мета (+EMB_VER)
    try:
        return conform_features(ref, dub, dub_video, out_path, ffmpeg=ffmpeg,
                                progress=progress, dub_name=dub_video.name,
                                progress_meta=progress_meta, low_mem=low_mem,
                                cache_dir=cache_dir, keep_tmp=keep_tmp, trace=trace, **opts)
    finally:
        if mp is not None and not into_cache:   # _tmp-копию удаляем; кэш-копию (CK1) оставляем
            del dub                          # освободить memmap перед удалением файла
            try:
                mp.unlink()
                mp.parent.rmdir()
            except OSError:
                pass
