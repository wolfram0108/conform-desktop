"""SRM-фичи: видео → вектора кадров (декод ffmpeg + свёртка KB/D1).

Канон ИДЕНТИЧЕН research-сборке (_real_test.build_srm / _build_srm_cache):
gray 128×72, ядра KB и D1, clip ±3, L2-норма по каналу, concat ×1/√2, float16,
ffmpeg `-vsync 0` (passthrough — индекс кадра = позиция). Любое отклонение здесь
ломает бит-в-бит совпадение с эталонным кэшем/wav.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from track_muxer.conform.config import FFMPEG, FFPROBE
from track_muxer.conform import procreg
from track_muxer.conform.models import Progress, SrmFeatures

GW, GH = 128, 72
KB = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], np.float32)
D1 = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float32)
T = 3.0
_RBLOCK = 1024


def probe_duration(video: Path, ffprobe: str = FFPROBE) -> float | None:
    r = procreg.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def _parse_hhmmss(s: str | None) -> float | None:
    """'00:25:11.410000000' → секунды (float). Невалидно → None."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        parts = s.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(s)
    except ValueError:
        return None


def probe_video_duration(video: Path, ffprobe: str = FFPROBE) -> float | None:
    """Реальная длительность ВИДЕОПОТОКА (НЕ контейнерная format=duration). Контейнерная бывает
    раздута битым Segment Duration в mkv или длинным хвостом аудио/субтитров — тогда
    fps=кадры/длительность врёт (Призрак-2: format 1748с против видео 1511с → fps 25.9 вместо
    29.97 → растяжка выхода и развал аудио-доводки). Приоритет: длительность видеопотока →
    его тег DURATION → format.duration (фолбэк)."""
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    try:
        v = float(r.stdout.strip())
        if v > 0:
            return v
    except ValueError:
        pass
    r = procreg.run(                                   # тег DURATION видеопотока ('00:25:11.41' — надёжен в mkv)
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream_tags=DURATION", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    v = _parse_hhmmss(r.stdout)
    if v and v > 0:
        return v
    return probe_duration(video, ffprobe)                 # фолбэк: контейнер (когда видеопоток молчит)


def probe_audio_channels(video: Path, ffprobe: str = FFPROBE,
                         atrack: int = 0) -> tuple[int, str | None]:
    """Число каналов и раскладка аудиодорожки `atrack` (дефолт 0 — как было) — чтобы
    наследовать 2.0/5.1/7.1 на выход. -> (channels, channel_layout|None). При любой
    неудаче — безопасный fallback (2, None).
    csv-строка вида '6,5.1(side)' / '2,stereo' / '2,' (пустая раскладка → None)."""
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", f"a:{int(atrack)}",
         "-show_entries", "stream=channels,channel_layout", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    line = next((s for s in r.stdout.splitlines() if s.strip()), "")
    parts = line.split(",", 1)                    # раскладки запятых не содержат → split(1) безопасен
    try:
        ch = int(parts[0])
    except (ValueError, IndexError):
        return (2, None)
    layout = parts[1].strip() if len(parts) > 1 else ""
    return (max(1, ch), layout if layout and layout != "unknown" else None)


def probe_audio_tracks(video: Path, ffprobe: str = FFPROBE) -> list[dict]:
    """Список ВСЕХ аудиодорожек файла — для выбора дорожки в UI (реф-дорожка /
    дорожки озвучек, требование 10 CHARTER standalone). Каждая запись:
    {index (0-based среди аудио), codec, channels, layout, lang, title, default}.
    Ошибка/нет аудио → []."""
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", "a",
         "-show_entries",
         "stream=codec_name,channels,channel_layout:stream_disposition=default"
         ":stream_tags=language,title",
         "-of", "json", str(video)],
        capture_output=True, text=True,
    )
    try:
        streams = json.loads(r.stdout or "{}").get("streams") or []
    except json.JSONDecodeError:
        return []
    out: list[dict] = []
    for i, s in enumerate(streams):
        tags = s.get("tags") or {}
        layout = (s.get("channel_layout") or "").strip()
        out.append({
            "index": i,
            "codec": s.get("codec_name") or "",
            "channels": int(s.get("channels") or 0),
            "layout": layout if layout and layout != "unknown" else None,
            "lang": (tags.get("language") or "").strip() or None,
            "title": (tags.get("title") or "").strip() or None,
            "default": bool((s.get("disposition") or {}).get("default")),
        })
    return out


def probe_resolution(video: Path, ffprobe: str = FFPROBE) -> int:
    """Высота кадра ПЕРВОГО видеопотока — для выбора бэкенда декода (CPU/GPU). Ошибка → 0 (→ CPU)."""
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=height", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    try:
        return int(r.stdout.strip())
    except (ValueError, AttributeError):
        return 0


def _parse_fps(s: str | None) -> float | None:
    """'24000/1001' → 23.976. Пусто/0/мусор → None."""
    s = (s or "").strip()
    try:
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b) if float(b) else None
        return float(s) if s else None
    except (ValueError, ZeroDivisionError):
        return None


def probe_fps(video: Path, ffprobe: str = FFPROBE) -> float | None:
    """fps видеопотока (r_frame_rate, напр. '24000/1001') — для ОЦЕНКИ числа кадров в прогрессе."""
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    return _parse_fps(r.stdout)


def probe_frame_count_hints(video: Path, ffprobe: str = FFPROBE) -> tuple[int | None, float | None]:
    """(nb_frames, avg_frame_rate) видеопотока — из МЕТАДАННЫХ, без прохода по файлу.

    Зачем в обход `r_frame_rate`: он бывает МУСОРНЫМ. Замер на 92 реальных файлах веб-плееров
    (cvh/Persona99): r_frame_rate=48.0 и 90000.0 (последнее — timebase MPEG-TS 90кГц, просочившийся
    в поле) при реальных 23.976 → гейт полноты ниже считал ожидание вдвое/в 3750 раз завышенным и
    ронял ЦЕЛУЮ серию на честном файле. При этом `nb_frames` был заполнен и ТОЧНО равен числу
    видеопакетов у всех 71 файла, где он есть (0 расхождений).
    ⚠ `avg_frame_rate` НЕ является честным источником на VFR: у Matroska с переменной частотой
    он остаётся номинальным (замер: 24000/1001 при реальных 3237 кадрах за 180с = 17.98) —
    поэтому при отсутствии `nb_frames` считаем пакеты честно (`probe_packet_count`).
    Отчёт: doc/reports/conform_input_robustness/."""
    # JSON, не csv: ffprobe выводит поля во ВНУТРЕННЕМ порядке, а не в порядке запроса
    # (проверено: `stream=nb_frames,avg_frame_rate` → csv «avg,nb»), позиционный разбор хрупок.
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,avg_frame_rate", "-of", "json", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        st = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    except (json.JSONDecodeError, IndexError):
        return None, None
    avg = _parse_fps(st.get("avg_frame_rate"))
    try:
        nb = int(st.get("nb_frames"))
    except (TypeError, ValueError):
        nb = None
    return (nb if (nb and nb > 0) else None), avg


def probe_packet_count(video: Path, ffprobe: str = FFPROBE) -> int | None:
    """ЧЕСТНОЕ число видеопакетов — один проход по файлу БЕЗ декода пикселей.

    Единственный источник, не врущий на VFR (там и `nb_frames` отсутствует, и `avg_frame_rate`
    номинален). Цена замерена: 0.04с на клип 180с, 0.68с на MP4 182МБ, 4.3с на MKV 1.5ГБ —
    против 40-60с самого декода SRM, т.е. ≤10% накладных, и только когда `nb_frames` нет."""
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        n = int((r.stdout or "").strip().rstrip(",").splitlines()[0])
    except (ValueError, IndexError):
        return None
    return n if n > 0 else None


def probe_has_video(video: Path, ffprobe: str = FFPROBE) -> bool:
    """Есть ли у файла НАСТОЯЩИЙ видеопоток. Голое аудио (flac/mka/mp3/aac…) → False —
    признак ветки АУДИО-ONLY conform (требование 9 CHARTER standalone-миссии).
    ⚠ Обложка (attached_pic у mp3/flac) — формально видеопоток, но НЕ видео: исключаем по
    disposition. Ошибка пробы → True (консервативно: пусть падает видео-путь с понятной
    ошибкой, а не молча уходит в аудио-режим)."""
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", "v",
         "-show_entries", "stream=codec_type:stream_disposition=attached_pic",
         "-of", "json", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        streams = json.loads(r.stdout or "{}").get("streams") or []
    except json.JSONDecodeError:
        return True
    for st in streams:
        if int((st.get("disposition") or {}).get("attached_pic", 0)) == 0:
            return True
    return False


def probe_frame_pts(video: Path, ffprobe: str = FFPROBE) -> np.ndarray | None:
    """PTS всех ВИДЕОПАКЕТОВ (сек), отсортированные по возрастанию — это и есть времена кадров
    в порядке показа. Замер (VFR-клип стенда): sorted(packet pts_time) == frame pts_time
    БИТ-В-БИТ (0.000 мс расхождения) при цене 0.05с против 2.80с покадрового прохода с декодом;
    на реальном MKV 700МБ — 1.2с. Любой N/A / мусор / пусто → None (осторожный отказ)."""
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    vals: list[float] = []
    for x in (r.stdout or "").split():
        x = x.strip().rstrip(",")
        if not x:
            continue
        try:
            vals.append(float(x))
        except ValueError:                # N/A и прочий мусор → оси нет
            return None
    if not vals:
        return None
    return np.sort(np.asarray(vals, np.float64))


def vfr_time_axis(video: Path, n_frames: int, ffprobe: str = FFPROBE) -> np.ndarray | None:
    """Ось времени кадров (сек ОТ ПЕРВОГО КАДРА, t[0]=0) — ТОЛЬКО для реального VFR; иначе None.

    None ⟹ потребители считают время как индекс/fps — СТАРЫЙ путь бит-в-бит (CFR-регресс
    по построению). Критерий VFR — невязка к НАИЛУЧШЕЙ равномерной сетке
    (шаг = t[-1]/(n-1)), порог 1.5 кадра. Почему не медианный шаг: Matroska квантует метки
    в мс (42/41 вперемешку при истинных 41.708) — медиана 42.00 «уплывает» на 10с за серию
    и даёт ложный VFR на честном CFR. Замер на 5 файлах: CFR-mkv квантованный → невязка
    0.02 кадра; реальные VFR-энкоды → 539–1425 кадров. Зазор классов — 4 порядка.

    len(pts) != n_frames (битые пакеты, обрыв декода) → None: сшивка пакет↔кадр по индексу
    обязана быть 1:1, иначе оси не верим."""
    if n_frames < 2:
        return None
    pts = probe_frame_pts(video, ffprobe)
    if pts is None or len(pts) != n_frames:
        return None
    t = pts - pts[0]
    slope = float(t[-1]) / (n_frames - 1)
    if slope <= 0:
        return None
    dev = np.abs(t - np.arange(n_frames) * slope)
    return t if float(dev.max()) > 1.5 * slope else None


def build_srm(
    video: Path | str,
    fps: float | None = None,
    *,
    ffmpeg: str = FFMPEG,
    ffprobe: str = FFPROBE,
    progress=None,
    should_stop=None,
    progress_meta: tuple[int, int, str] = (0, 0, ""),
    mmap_path: Path | None = None,
    crop: str | None = None,
    backend: str = "cpu",
) -> SrmFeatures:
    """Декод видео + свёртка → SrmFeatures. fps=None → авто (кадры/длительность).

    progress(Progress) вызывается периодически на этапе "decode".
    should_stop() == True → прерывает декод (RuntimeError("stopped")).
    mmap_path задан → фичи ПОТОКОМ пишутся в raw-файл f16 (в RAM не копятся), srm
    возвращается как np.memmap (read-only). Значения идентичны in-RAM пути (бит-в-бит).
    crop='W:H:X:Y' (геом-коррекция, conform.geom): обрезать кадр ДО scale=128:72 —
    приводит дубль к кадрированию рефа, когда SRM слепнет от кропа/зума/анаморфа/полос.
    None (по умолчанию) → как было, бит-в-бит.
    backend='cuda' → декод на GPU (NVDEC, `-hwaccel cuda`); scale 128×72 ОСТАЁТСЯ на CPU
    (без output_format cuda) → кадры бит-в-бит идентичны 'cpu' (декод детерминирован), кэш не
    зависит от бэкенда. Выбор делает conform.decode_backend по разрешению/потолку NVDEC.
    """
    video = Path(video)
    dur = probe_video_duration(video, ffprobe)                # видеопоток, не контейнер
    # fps видеопотока — для оценки числа кадров: прогресс-бар И гейт полноты декода (ниже).
    fps_est = fps if fps else probe_fps(video, ffprobe)
    nb_meta, avg_fps = probe_frame_count_hints(video, ffprobe)   # честные источники для гейта
    di, dt, dname = progress_meta

    vf = (f"crop={crop},scale={GW}:{GH},format=gray" if crop
          else f"scale={GW}:{GH},format=gray")
    pre = ["-hwaccel", "cuda"] if backend == "cuda" else []   # GPU-декод; scale остаётся на CPU → бит-в-бит
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", *pre, "-i", str(video), "-an",
           "-vf", vf, "-vsync", "0", "-f", "rawvideo", "-"]
    fb = GW * GH
    D = GW * GH * 2                       # размерность вектора кадра (rk[9216]+rd[9216])
    p = procreg.popen(cmd, stdout=subprocess.PIPE, bufsize=fb * _RBLOCK)
    out_f = open(mmap_path, "wb") if mmap_path is not None else None   # поток на диск (низкая RAM)
    vecs: list[np.ndarray] = []           # используется только при mmap_path is None
    n_written = 0
    buf = b""
    t0 = time.perf_counter(); last = t0
    # Ожидаемое число кадров для гейта полноты (ниже) и прогресс-бара. Источники по убыванию
    # честности. `r_frame_rate` — ТОЛЬКО последний фолбэк, он бывает мусорным (48.0 и 90000.0
    # при реальных 23.976 у cvh/Persona99 → гейт ронял честную серию целиком); `avg_frame_rate`
    # тоже НЕ честен на VFR (номинальный 24000/1001 при реальных 17.98) → при отсутствии
    # nb_frames считаем пакеты (цена ≤10% от декода, см. probe_packet_count).
    if nb_meta:
        n_expect = nb_meta                                   # точное число кадров из контейнера
    else:
        n_pkt = probe_packet_count(video, ffprobe)            # честный счёт (VFR/Matroska)
        if n_pkt:
            n_expect = n_pkt
        elif dur and avg_fps:
            n_expect = int(round(dur * avg_fps))
        elif dur and fps_est:
            n_expect = int(round(dur * fps_est))             # как было (последний фолбэк)
        else:
            n_expect = 0
    try:
        while True:
            if should_stop is not None and should_stop():
                p.kill()
                raise RuntimeError("stopped")
            chunk = p.stdout.read(fb * _RBLOCK)
            if not chunk:
                break
            buf += chunk
            k = len(buf) // fb
            if k:
                block = np.frombuffer(buf[:k * fb], np.uint8).reshape(k, GH, GW).astype(np.float32)
                buf = buf[k * fb:]
                bvs = np.empty((k, D), np.float16)
                for j in range(k):
                    g = block[j]
                    rk = np.clip(ndimage.convolve(g, KB, mode="reflect"), -T, T).ravel()
                    rd = np.clip(ndimage.convolve(g, D1, mode="reflect"), -T, T).ravel()
                    rk /= np.linalg.norm(rk) + 1e-6
                    rd /= np.linalg.norm(rd) + 1e-6
                    bvs[j] = (np.concatenate([rk, rd]) * 0.7071068).astype(np.float16)
                if out_f is not None:
                    out_f.write(bvs.tobytes())     # на диск, в RAM держим только блок
                else:
                    vecs.append(bvs)
                n_written += k
                if progress is not None and time.perf_counter() - last >= 2.0:
                    done = n_written
                    frac = (done / n_expect) if n_expect else 0.0
                    kps = done / (time.perf_counter() - t0 + 1e-9)
                    progress(Progress("decode", min(frac, 0.999),
                                      f"{done} кадров, {kps:,.0f} к/с", di, dt, dname))
                    last = time.perf_counter()
    finally:
        p.stdout.close()
        p.wait(); procreg.done(p)
        if out_f is not None:
            out_f.close()

    # ── Гейт целостности декода: обрыв ffmpeg (NVDEC/OOM/битый поток) по пайпу неотличим от конца
    #    файла → без гейта обрубок молча становился «успешным» SRM (dr-stone ep09: 4357/34552
    #    кадров, fps=n/dur=3.02 заражал кэш и валил все озвучки серии). ──
    if p.returncode not in (0, None):
        raise RuntimeError(f"декод SRM упал (ffmpeg rc={p.returncode}) на кадре {n_written}: {video.name}")
    # 0.9: реальный обрыв = доли файла (12.6% в кейсе dr-stone); честные потери кадров — единицы
    # процентов. Ожидание считается от nb_frames/avg_fps (см. выше), а НЕ от r_frame_rate —
    # иначе мусорный r_frame_rate роняет честный файл (cvh/Persona99: 0.4995 и 0.0003 от «ожидания»).
    if n_expect and n_written < 0.9 * n_expect:
        raise RuntimeError(
            f"декод SRM неполон: {n_written} из ~{n_expect} кадров — обрыв декодера ({video.name})")

    if mmap_path is not None:
        arr = (np.memmap(mmap_path, dtype=np.float16, mode="r", shape=(n_written, D))
               if n_written else np.zeros((0, D), np.float16))
    else:
        arr = np.concatenate(vecs) if vecs else np.zeros((0, D), np.float16)
    if fps is None:
        if dur is None:
            dur = probe_video_duration(video, ffprobe)    # видеопоток, не раздутый контейнер
        fps = (len(arr) / dur) if dur else 0.0
    # Реальная ось времени кадров — только для VFR (иначе None → старый путь индекс/fps
    # бит-в-бит). На VFR попутно честнеет fps: средняя плотность кадров по оси, а не n/dur
    # от контейнерной длительности (для плотностных окон и ratio ref/dub в даунстриме).
    pts_ax = vfr_time_axis(video, len(arr), ffprobe)
    if pts_ax is not None and pts_ax[-1] > 0:
        fps = (len(arr) - 1) / float(pts_ax[-1])
    return SrmFeatures(srm=arr, fps=float(fps), src=video, pts=pts_ax)
