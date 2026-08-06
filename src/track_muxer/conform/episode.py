"""Одна СЕРИЯ: 1 референс + список озвучек. Реф декодируется ОДИН раз и
переиспользуется в RAM для всех озвучек; опционально кешируется на диск
(`cache_dir`) для переиспользования между ЗАПУСКАМИ (докональная синхра новой
озвучки без передекода рефа).

Выход: out_dir/<имя_озвучки>.<ext> на каждую (формат — деталь записи, сейчас FLAC l12).
skip_existing — готовые (файл уже на диске) пропускаются. keep_tmp=False → кеш-подкаталог
удаляется по завершении; True → остаётся
(всё промежуточное: реф SRM + дубль CK1/2/3).
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from track_muxer.conform import cache as cache_mod
from track_muxer.conform.align import conform_pair
from track_muxer.conform.config import FFMPEG
from track_muxer.conform.decode_backend import decode_backend
from track_muxer.conform.features import build_srm, probe_resolution
from track_muxer.conform.models import EpisodeResult, PairResult, Progress, SrmFeatures


def conform_episode(
    ref_video: Path | str,
    dub_videos: list[Path | str],
    out_dir: Path | str,
    *,
    fps_ref: float | None = None,
    ref_features: SrmFeatures | None = None,
    cache_dir: Path | str | None = None,
    skip_existing: bool = True,
    ffmpeg: str = FFMPEG,
    low_mem: bool = False,
    keep_tmp: bool = False,            # режим tmp-чекпоинтов (CK1 SRM дубля/CK2 аудио/…) в cache_dir
    ref_atrack: int = 0,               # ⭐ 5.1: индекс аудиодорожки РЕФА (звуковой эталон band/заливки)
    dub_atracks: list[int] | None = None,   # ⭐ 5.1: индекс дорожки КАЖДОЙ озвучки (параллельно
                                       # dub_videos; None/короче списка → дорожка 0). «Виртуальный
                                       # дубль» = тот же файл (хоть сам реф) с другой дорожкой.
    progress=None,
    should_stop=None,
    on_pair=None,
    **pair_opts,
) -> EpisodeResult:
    """Выровнять все озвучки серии на таймлайн ref_video. Возвращает EpisodeResult.

    pair_opts → conform_pair/conform_features (fps_dub, free_start, recover_edges,
    fill_silence, audio_band, audio_muq, apply_cuts, drift_speed_pct).
    on_pair(PairResult) — после КАЖДОЙ озвучки (живой прогресс в очереди).

    Многодорожечность (этап 5.1 standalone, 2026-08-06): дорожка ≠ 0 у озвучки → выход
    получает суффикс `__a<N>` (иначе виртуальные дубли одного файла затирали бы друг друга).
    CK3 (видео-матчинг) от аудиодорожки не зависит и реюзается между дорожками одного файла.
    """
    t0 = time.perf_counter()
    ref_video = Path(ref_video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dubs = [Path(d) for d in dub_videos]
    total = len(dubs)

    # Реф: ref_features → дисковый кеш → декод (+сохранить в кеш).
    if progress is not None:
        progress(Progress("decode", 0.0, f"реф {ref_video.name}", 0, total, "реф"))
    ref = ref_features
    if ref is None and cache_dir is not None:
        ref = cache_mod.load_srm(cache_dir, ref_video)        # memmap, если в кеше
    ref_tmp = None                                            # папка реф-memmap в _tmp (low_mem без кеша) — на удаление
    if ref is None:
        with decode_backend(probe_resolution(ref_video), ffmpeg) as _be:   # 1080+ → GPU (потолок NVDEC), иначе CPU
            if low_mem and cache_dir is not None:             # строим реф ПОТОКОМ прямо в кеш → memmap
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                ref = build_srm(ref_video, fps_ref, ffmpeg=ffmpeg, progress=progress,
                                should_stop=should_stop, progress_meta=(0, total, "реф"),
                                mmap_path=cache_mod.srm_file(cache_dir, ref_video), backend=_be)
                cache_mod.save_meta(cache_dir, ref_video, len(ref.srm), ref.fps)
            elif low_mem:                                     # без кеша — во временный memmap, удалим в конце
                td = ref_video.parent / "_tmp"
                td.mkdir(parents=True, exist_ok=True)
                ref_tmp = Path(tempfile.mkdtemp(prefix="srmref_", dir=str(td)))
                ref = build_srm(ref_video, fps_ref, ffmpeg=ffmpeg, progress=progress,
                                should_stop=should_stop, progress_meta=(0, total, "реф"),
                                mmap_path=ref_tmp / "ref.f16", backend=_be)
            else:
                ref = build_srm(ref_video, fps_ref, ffmpeg=ffmpeg, progress=progress,
                                should_stop=should_stop, progress_meta=(0, total, "реф"), backend=_be)
                if cache_dir is not None:
                    cache_mod.save_srm(cache_dir, ref_video, ref)
    if fps_ref is not None:                       # явный fps перекрывает сохранённый в кеше
        ref.fps = float(fps_ref)

    # Реф-АУДИО извлекаем ОДИН раз на серию и переиспользуем для всех озвучек (реф может
    # быть на гигабайты — повторное извлечение читало бы весь файл на каждую озвучку).
    # ЛЕНИВО (на первой реально обрабатываемой паре), в RAM; по выходу из функции освобождается.
    # реф-аудио нужен аудио-слою (band/muq/легаси) И финальному заполнению тишины рефом
    # (fill_silence работает поверх band/muq, реф-аудио затягивается ими же).
    want_ref_audio = pair_opts.get("audio_band", False) or pair_opts.get("audio_muq", False)
    ref_audio = None

    pairs: list[PairResult] = []
    stopped = False
    for i, dub_video in enumerate(dubs, 1):
        if should_stop is not None and should_stop():
            stopped = True
            break
        atrack = int(dub_atracks[i - 1]) if (dub_atracks and i - 1 < len(dub_atracks)) else 0
        stem = dub_video.stem + (f"__a{atrack}" if atrack else "")   # дорожка≠0 → суффикс (виртуальные дубли)
        out_path = out_dir / f"{stem}.flac"                     # выход conform — FLAC l12 16/44.1
        old = out_dir / f"{stem}.wav"                           # старый формат (миграция: тоже «готово»)
        done = next((p for p in (out_path, old) if p.exists() and p.stat().st_size > 0), None)
        if skip_existing and done is not None:
            res = PairResult(dub=dub_video.name, out_path=done, ok=True, skipped=True)
        else:
            if want_ref_audio and ref_audio is None:        # извлечь реф-аудио единожды (с прогрессом)
                try:
                    from track_muxer.conform.align import _decode_audio
                    ref_audio = _decode_audio(ref_video, ffmpeg, False, atrack=ref_atrack,
                                             progress=progress,
                                             progress_meta=(0, total, "реф"), label="аудио рефа (1 раз)")
                except Exception:  # noqa: BLE001 — нет аудио → пары извлекут сами/откатятся на тишину
                    ref_audio = None
            meta = (i, total, dub_video.name)
            try:
                res = conform_pair(ref, dub_video, out_path, ffmpeg=ffmpeg,
                                   progress=progress, should_stop=should_stop,
                                   progress_meta=meta, ref_audio=ref_audio,
                                   ref_atrack=ref_atrack, dub_atrack=atrack,
                                   low_mem=low_mem, cache_dir=cache_dir, keep_tmp=keep_tmp,
                                   **pair_opts)
            except Exception as e:  # noqa: BLE001 — одна озвучка не валит серию
                res = PairResult(dub=dub_video.name, out_path=None, ok=False, error=str(e))
        pairs.append(res)
        if on_pair is not None:
            try:
                on_pair(res)
            except Exception:  # noqa: BLE001
                pass

    ref_audio = None        # освободить реф-аудио из RAM по завершении серии (на диск не пишем)
    if ref_tmp is not None:            # реф-memmap из _tmp (low_mem без кеша) → освободить и удалить
        del ref
        shutil.rmtree(ref_tmp, ignore_errors=True)

    # Очистка кеша, если НЕ tmp-чекпоинты (и серия не прервана). keep_tmp ⊃ сохранение всего
    # промежуточного (реф SRM + дубль CK1/2/3): tmp ON → кеш остаётся; OFF → чистим после серии.
    if cache_dir is not None and not keep_tmp and not stopped:
        cache_mod.clear_dir(cache_dir)

    return EpisodeResult(ref=ref_video.name, out_dir=out_dir, pairs=pairs,
                         elapsed_s=time.perf_counter() - t0)
