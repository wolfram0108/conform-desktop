"""Дисковый SRM-кеш (для РЕФА). Кладётся в подкаталог каталога серии
(`<серия>/_conform_cache/`), ключ — по содержимому файла (имя+размер+mtime).

Зачем: реф переиспользуется между озвучками (в пределах серии) И между ЗАПУСКАМИ
(появилась новая озвучка — реф берётся из кеша без передекода). Озвучки не кешируем.

Формат: фичи лежат raw `.f16` (плоский float16, форма (N, D)), мета — `.json` (N, fps).
Загрузка отдаёт **np.memmap** (read-only) — реф НЕ грузится в RAM целиком (важно для
длинных файлов). Старые `.npz`-кеши не читаются — просто перестроятся.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
from loguru import logger

from track_muxer.conform.features import GH, GW, probe_video_duration, vfr_time_axis
from track_muxer.conform.models import SrmFeatures

_CACHE_AUDIT = bool(os.environ.get("TM_CACHE_AUDIT"))


def _a(stage: str, hit: bool, name: str = "") -> None:
    """Аудит реюза кэша (env TM_CACHE_AUDIT=1): код САМ пишет, взял кэш отсюда или строит на GPU.
    Включается только для тестовых прогонов; в проде по умолчанию молчит."""
    if _CACHE_AUDIT:
        print(f"[КЭШ] {stage:16s} {'РЕЮЗ ✓' if hit else 'промах → строю/декодю'}  {name}", flush=True)

D = GW * GH * 2          # размерность вектора кадра (rk+rd)

# Версии этапов для инвалидации чекпоинтов (режим tmp). Бампать при смене кода ЭТАПА —
# тогда стухший чекпоинт пересоберётся, а тяжёлое выше возьмётся из кэша. Пользователь не
# управляет. EMB_VER — SRM/эмбединг (features.build_srm), покрывает кэш SRM рефа И дубля.
EMB_VER = 1
EXT_VER = 3              # извлечение аудио (_extract_wav[_mmap]) — кэш аудио CK2
#   v3: audio is laid on the video axis by the container delay (features.probe_av_delay);
#   v2 caches of files with video_start != audio_start hold the audio shifted by that delay.
#   v2 (2026-08-06): декод аудио ВСЕГДА идёт через `aresample=async=1:first_pts=0`
#   (был тумблер audio_fix, выкл. по умолчанию — см. align._extract_base). На чистом входе
#   данные идентичны v1 (бит-в-бит), но кэш файла С ДЫРОЙ PTS, снятый по v1, рассинхронен.
DSP_VER = 3             # DSP-48 benv РЕФА (coarse_dtw) — кэш CK4. Бампать при смене benv/полос/ресэмпла.
#   v3: reference audio decode now applies the container delay (see EXT_VER v3).
#   v2 (2026-08-06): benv считается по аудио рефа, а его декод изменился (см. EXT_VER).


def cache_key(video: Path) -> str:
    st = Path(video).stat()
    return f"{Path(video).stem}__{st.st_size}__{int(st.st_mtime)}"


def _crop_tag(crop: str | None) -> str:
    """Суффикс ключа для КРОПНУТОГО SRM (геом-коррекция): None → '' (обычный кэш)."""
    return ("__c" + crop.replace(":", "_")) if crop else ""


def srm_file(cache_dir: Path | str, video: Path | str, crop: str | None = None) -> Path:
    """Путь к raw-файлу фич (.f16) — туда же build_srm пишет потоком при low_mem.
    crop задан → отдельный ключ кропнутого SRM (геом-режим), не пересекается с обычным."""
    return Path(cache_dir) / (cache_key(Path(video)) + _crop_tag(crop) + ".f16")


def _meta_file(cache_dir: Path | str, video: Path | str, crop: str | None = None) -> Path:
    return Path(cache_dir) / (cache_key(Path(video)) + _crop_tag(crop) + ".json")


def load_srm(cache_dir: Path | str, video: Path | str, crop: str | None = None) -> SrmFeatures | None:
    """Реф/дубль из кеша как memmap (read-only). None — если кеша нет/битый. Логирует реюз (аудит).
    crop задан → кропнутый SRM (геом-режим)."""
    r = _load_srm(cache_dir, video, crop)
    _a("CK1 SRM" + (" geom" if crop else ""), r is not None, Path(video).name)
    return r


def _load_srm(cache_dir: Path | str, video: Path | str, crop: str | None = None) -> SrmFeatures | None:
    try:
        sp = srm_file(cache_dir, video, crop); mp = _meta_file(cache_dir, video, crop)
    except OSError:
        return None
    if not (sp.exists() and mp.exists()):
        return None
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
        # Инвалидация по версии эмбединга: отсутствие поля = версия 1 (так строили существующие
        # кэши до ввода чекпоинтов) → не ломаем их сейчас, но бамп EMB_VER пересоберёт.
        if int(meta.get("emb_ver", 1)) != EMB_VER:
            return None
        n = int(meta["n"]); d = int(meta.get("d", D))
        if n <= 0:                       # ПУСТОЙ/битый SRM-кэш (сборка упала, оставила n=0/0-байт) →
            return None                  # инвалид → conform пересоберёт SRM (иначе len/fps=0 → div0)
        # fps ПЕРЕ-ВЫВОДИМ из файла, НЕ доверяем meta: старые кэши хранят fps от раздутой
        # контейнерной длительности (битый Segment Duration → fps врёт). Кадры от fps не зависят →
        # SRM не пересобираем, лишь освежаем fps. Сбой probe → фолбэк на сохранённое значение.
        fps = float(meta.get("fps", 0.0))
        try:
            vd = probe_video_duration(Path(video))
            if vd and n:
                fps = n / vd
        except Exception:  # noqa: BLE001 — probe недоступен → старое значение
            pass
        arr = np.memmap(sp, dtype=np.float16, mode="r", shape=(n, d)) if n else np.zeros((0, d), np.float16)
        # Ось VFR НЕ кэшируем — перепробиваем из исходника при каждой загрузке (0.05–1.2с,
        # пакетный проход без декода; сам SRM-декод, который кэш экономит, 40–60с). CFR → None
        # (без затрат на подавляющем большинстве файлов кроме той же пробы). Как в build_srm:
        # при живой оси fps = средняя плотность по оси.
        try:
            pts_ax = vfr_time_axis(Path(video), n)
        except Exception:  # noqa: BLE001 — проба недоступна → консервативно без оси
            pts_ax = None
        if pts_ax is not None and pts_ax[-1] > 0:
            fps = (n - 1) / float(pts_ax[-1])
        return SrmFeatures(srm=arr, fps=fps, src=Path(video), pts=pts_ax)
    except Exception as e:  # noqa: BLE001
        logger.warning("conform cache load failed {}: {}", sp, e)
        return None


def save_meta(cache_dir: Path | str, video: Path | str, n: int, fps: float, d: int = D,
              crop: str | None = None) -> None:
    """Записать мету (когда build_srm уже записал .f16 потоком напрямую в кеш)."""
    if int(n) <= 0:                      # НЕ кэшируем пустой SRM (сборка упала) — иначе poison → div0
        return
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        _meta_file(cache_dir, video, crop).write_text(
            json.dumps({"n": int(n), "d": int(d), "fps": float(fps), "emb_ver": EMB_VER}),
            encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.warning("conform cache meta save failed: {}", e)


def save_srm(cache_dir: Path | str, video: Path | str, feats: SrmFeatures) -> None:
    """Сохранить уже построенные (в RAM) фичи: raw .f16 + мета. Для in-RAM пути."""
    try:
        arr = np.asarray(feats.srm, np.float16)
        if arr.shape[0] <= 0:            # пустой SRM не кэшируем (сборка упала) — иначе poison → div0
            return
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        arr.tofile(srm_file(cache_dir, video))
        save_meta(cache_dir, video, arr.shape[0], feats.fps, arr.shape[1] if arr.ndim == 2 else D)
    except Exception as e:  # noqa: BLE001
        logger.warning("conform cache save failed: {}", e)


# ── CK2: кэш извлечённого аудио (raw int16 [N,C], как _extract_wav_mmap) ──

def _atr_tag(atrack: int) -> str:
    """Суффикс ключа для НЕ-нулевой аудиодорожки (многодорожечность, этап 5.1 standalone).
    atrack=0 → пустой (старые ключи бит-в-бит валидны)."""
    return f"__a{int(atrack)}" if atrack else ""


def audio_raw(cache_dir: Path | str, video: Path | str, atrack: int = 0) -> Path:
    """Путь к raw-аудио (.raw, int16 [N*C]) — туда _extract_wav_mmap(dest=) пишет напрямую."""
    return Path(cache_dir) / (cache_key(Path(video)) + _atr_tag(atrack) + "__audio.raw")


def _audio_meta(cache_dir: Path | str, video: Path | str, atrack: int = 0) -> Path:
    return Path(cache_dir) / (cache_key(Path(video)) + _atr_tag(atrack) + "__audio.json")


def load_audio_mmap(cache_dir: Path | str, video: Path | str, channels: int,
                    *, ext_ver: int = EXT_VER, atrack: int = 0):
    """CK2: int16-memmap [N,channels] из кэша (read-only) или None. Ключ=файл+EXT_VER+каналы+дорожка."""
    r = _load_audio_mmap(cache_dir, video, channels, ext_ver=ext_ver, atrack=atrack)
    _a("CK2 аудио дубль", r is not None, Path(video).name)
    return r


def _load_audio_mmap(cache_dir: Path | str, video: Path | str, channels: int,
                     *, ext_ver: int = EXT_VER, atrack: int = 0):
    try:
        raw = audio_raw(cache_dir, video, atrack); mp = _audio_meta(cache_dir, video, atrack)
    except OSError:
        return None
    if not (raw.exists() and mp.exists()):
        return None
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
        if int(m.get("ext_ver", 1)) != ext_ver or int(m.get("channels", 0)) != int(channels):
            return None
        a = np.memmap(raw, dtype=np.int16, mode="r")
        return a[: (a.size // channels) * channels].reshape(-1, channels)
    except Exception as e:  # noqa: BLE001
        logger.warning("conform audio cache load failed {}: {}", raw, e)
        return None


def save_audio_meta(cache_dir: Path | str, video: Path | str, channels: int,
                    *, ext_ver: int = EXT_VER, atrack: int = 0) -> None:
    """Записать мету CK2 (raw уже записан _extract_wav_mmap(dest=audio_raw(...)))."""
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        _audio_meta(cache_dir, video, atrack).write_text(
            json.dumps({"channels": int(channels), "ext_ver": int(ext_ver)}), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.warning("conform audio cache meta save failed: {}", e)


# ── CK4: кэш DSP-48 benv РЕФА (coarse_dtw; реф переиспользуется между дублями эпизода) ──

def dsp_ref_path(cache_dir: Path | str, video: Path | str, atrack: int = 0) -> Path:
    """Путь к кэшу benv рефа (.npy, f32 [48,Nf]) с версией DSP_VER в имени (бамп → пересборка).
    benv считается по АУДИО рефа ⟹ реф-дорожка входит в ключ (atrack=0 → старое имя)."""
    return Path(cache_dir) / (cache_key(Path(video)) + _atr_tag(atrack) + f"__dsp{DSP_VER}.npy")


# ── CK-geom: кэш РЕЗУЛЬТАТА consensus_G (crop/scale per-pair). LoFTR-geom ДОРОГОЙ (декод+нейроматчер)
#    И флаки (анкоры/GPU) → НЕЛЬЗЯ перезапускать каждый conform. Ключ = дубль (уникален на пару) +
#    проверка идентичности рефа. Хит → кропаем БЕЗ повторного geom (и даже без kornia — она нужна
#    лишь для ВЫЧИСЛЕНИЯ кропа, не для применения). Бамп GEOM_VER при смене geom-алгоритма. ──
GEOM_VER = 1


def _geom_meta(cache_dir: Path | str, dub_video: Path | str) -> Path:
    return Path(cache_dir) / (cache_key(Path(dub_video)) + "__geom.json")


def load_geom(cache_dir: Path | str, ref_video: Path | str, dub_video: Path | str):
    """Геом-результат (dict crop_ref/crop_dub/sx/sy/n_in) из кэша или None (нет/версия/реф сменился)."""
    try:
        p = _geom_meta(cache_dir, dub_video)
    except OSError:
        return None
    if not p.exists():
        _a("CK-geom", False, Path(dub_video).name)
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if int(d.get("ver", 0)) != GEOM_VER or d.get("ref_key") != cache_key(Path(ref_video)):
            return None
        _a("CK-geom", True, Path(dub_video).name)
        return d["G"]
    except Exception as e:  # noqa: BLE001
        logger.warning("geom cache load failed {}: {}", p, e)
        return None


def save_geom(cache_dir: Path | str, ref_video: Path | str, dub_video: Path | str, G: dict) -> None:
    """Сохранить геом-результат пары (после успешного consensus_G)."""
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        _geom_meta(cache_dir, dub_video).write_text(
            json.dumps({"ver": GEOM_VER, "ref_key": cache_key(Path(ref_video)), "G": G}),
            encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.warning("geom cache save failed: {}", e)


def clear_dir(cache_dir: Path | str) -> None:
    """Удалить кеш-подкаталог целиком (когда флаг «сохранять кеш рефа» выключен)."""
    try:
        d = Path(cache_dir)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("conform cache clear failed: {}", e)
