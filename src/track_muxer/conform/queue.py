"""Очередь conform — плоский реестр СЕРИЙ-задач + пул с порогом. По образцу
core/pipeline.py (но проще): единица = серия (1 реф + список озвучек), воркер =
conform_episode (реф декодится один раз). Состояние — источник правды, наружу
ТОЛЬКО через API. Персист `_conform.json`; restore: оборванные running→queued.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from track_muxer.conform import procreg
from track_muxer.conform.episode import conform_episode
from track_muxer.conform.models import PairResult

QUEUED = "queued"
PAUSED = "paused"                # ждёт РУЧНОГО пуска (кнопка ▶ у строки очереди)
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
_TERMINAL = {DONE, FAILED, CANCELLED}
_PERSIST = "_conform.json"
DEFAULT_LIMIT = 1                # серии по очереди (порог меняется на лету)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pfps(s: str | None) -> float | None:
    """fps из строки ('24000/1001', '24.0') → float; None/мусор → None (авто)."""
    if not s:
        return None
    try:
        return float(eval(str(s), {"__builtins__": {}}, {}))  # noqa: S307 — доверенный ввод
    except Exception:  # noqa: BLE001
        return None


class PlotRef(BaseModel):
    """Ссылка на PNG-график укладки (band/muq). Файл в <out_dir>/_plots/<name>."""

    kind: str                       # "track" (весь трек) | "cut" (зум на рез)
    name: str                       # имя файла png
    t: float | None = None          # время реза, с (для kind=cut)
    v_ms: float | None = None       # величина реза, мс (для kind=cut)


class DubResult(BaseModel):
    """Паспорт одной озвучки (снимок для API)."""

    dub: str
    ok: bool = False
    mode: str = "av"                # "av" = зрение+звук | "audio" = аудио-only (озвучка без видео)
    skipped: bool = False           # уже было готово (файл на диске)
    out_path: str | None = None     # путь к выходному аудиофайлу (формат — деталь записи)
    assigned_pct: float = 0.0
    slope: float = 0.0
    cos_median: float = 0.0
    real_cuts: int = 0
    filled_cuts: int = 0            # вырезов заполнено оригиналом рефа
    edge_recovered: int = 0        # кадров возвращено доп-проходом на краях (опенинг/концовка)
    audio_resid_ms: float = 0.0    # остаточный аудио-сдвиг ПОСЛЕ доводки (мс; меньше=лучше)
    blind_zones: int = 0
    dropped_intro_s: float = 0.0
    audio_cuts: int = 0            # band/muq: дискретных правок стыков (резов)
    audio_max_step_ms: float = 0.0 # band/muq: крупнейший рез, мс
    audio_coverage: float = 0.0    # band/muq: доля трека с надёжными якорями (0..1)
    audio_span_ms: float = 0.0     # band/muq: диапазон движения сдвига, мс
    geom_used: bool = False        # применена геом-коррекция (кроп/зум/анаморф/полосы) — зрение слепло без неё
    geom_n_in: int = 0             # геом: inlier-якорей консенсуса (надёжность регистрации)
    geom_sx: float = 0.0           # геом: масштаб по X (анаморф = sx≠sy)
    geom_sy: float = 0.0           # геом: масштаб по Y
    plots: list[PlotRef] = Field(default_factory=list)  # PNG-графики укладки
    suspect: bool = False          # авто-флаг брака (cos<0.3 / назнач<90 / ошибка)
    warnings: list[str] = Field(default_factory=list)  # предупреждения «обрати внимание» (с причинами)
    critical: list[str] = Field(default_factory=list)  # КРАСНОЕ: не исправляемый авто дефект (студийный A/V-десинк)
    error: str | None = None
    elapsed_s: float = 0.0


class ConformJob(BaseModel):
    """Одна серия-задача. Сериализуется в JSON (API + персист)."""

    id: str
    seq: int = 0
    label: str = ""
    ref: str
    dubs: list[str]
    out_dir: str
    cache_dir: str | None = None       # подкаталог кеша (<серия>/_conform_cache)
    keep_tmp: bool = False             # tmp-чекпоинты: сохранить ВСЁ промежуточное (реф SRM + дубль
                                       # CK1/2/3) → повтор без GPU-декода; OFF → кеш чистится после серии
                                       # (doc/ТЗ_чекпоинты_conform.md)
    # опции (как в align/CLI)
    fps_ref: str | None = None
    fps_dub: str | None = None
    free_start: bool = True
    fill_silence: bool = True          # «вырезы оригиналом»: тишину озвучки заполнить рефом ПОСЛЕ
                                       # band/muq (только синхронно; при аудио off неактивно)
    audio_band: bool = True            # ДЕФОЛТ: anchor-пайплайн, карта DSP 48 полос (GPU, без модели)
    audio_muq: bool = False            # anchor-пайплайн, карта MuQ (GPU, опц. transformers)
    apply_cuts: bool = True            # band/muq: применять резкую правку резов (иначе только дрейф ≤2%)
    drift_speed_pct: float = 1.25      # band/muq: потолок скорости изменения сдвига кривой дрейфа, %/с
    ref_atrack: int = 0                # ⭐ 5.1: аудиодорожка РЕФА (звуковой эталон band/заливки)
    dub_atracks: list[int] | None = None   # ⭐ 5.1: дорожка каждой озвучки (параллельно dubs; None → все 0)
    autostart: bool = True             # False → задача встаёт в PAUSED и ждёт ручного пуска
    # состояние
    status: str = QUEUED
    progress: float = 0.0          # 0..1 общий
    stage: str = ""               # decode/extract/align/geom/resample/write
    stage_pct: float = 0.0         # 0..1 ВНУТРИ текущей фазы (для панели)
    detail: str = ""               # живая деталь фазы (напр. "12000 кадров, 1300 к/с")
    dub_index: int = 0
    dub_total: int = 0
    cur_dub: str = ""
    results: list[DubResult] = Field(default_factory=list)
    error: str | None = None
    elapsed_s: float = 0.0
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _dir_size(p: Path) -> int:
    try:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except OSError:
        return 0


def _purge_tmp(job: ConformJob) -> int:
    """Удалить временные файлы завершённой задачи. Возврат: освобождено байт.

    Что удаляем: кеш серии (CK1/CK2/CK3-чекпоинты, гигабайты) и наши каталоги
    `_tmp` рядом с исходниками (memmap-раскладки декода). Что НЕ трогаем НИКОГДА:
    выходной каталог задачи с аудиофайлами и `_plots` — прямое требование
    пользователя. При отмене штатная чистка `conform_episode` не отрабатывает
    (серия прервана) — этот проход её и заменяет.
    """
    out = Path(job.out_dir)
    targets: list[Path] = []
    if job.cache_dir:
        targets.append(Path(job.cache_dir))
    targets.append(Path(job.ref).parent / "_conform_cache")     # дефолтная раскладка серии
    for src in [job.ref, *job.dubs]:
        targets.append(Path(src).parent / "_tmp")

    freed = 0
    seen: set[str] = set()
    for t in targets:
        key = str(t.resolve()) if t.exists() else str(t)
        if key in seen:
            continue
        seen.add(key)
        if not t.is_dir():
            continue
        if _is_inside(out, t) or t.resolve() == out.resolve():   # защита результатов
            logger.warning("conform purge: пропуск {} — внутри выходного каталога", t)
            continue
        size = _dir_size(t)
        shutil.rmtree(t, ignore_errors=True)
        if t.exists():
            logger.warning("conform purge: каталог не удалён полностью: {}", t)
        freed += size - _dir_size(t)
    return freed


def _suspect(r: PairResult) -> bool:
    if r.skipped:                      # готовое не пересчитывали — не оцениваем как брак
        return False
    if getattr(r, "mode", "av") == "audio":
        # аудио-only: полей зрения (cos/assigned) нет по построению — судим по слуху:
        # покрытие якорями и остаток доводки (главный критерий ±80мс).
        return (not r.ok) or r.audio_coverage < 0.5 or abs(r.audio_resid_ms) > 80.0
    return (not r.ok) or r.cos_median < 0.30 or r.assigned_pct < 90.0


class ConformQueue:
    """Реестр серий-задач + один пул. Потокобезопасен (RLock). Параллелизм режет
    диспетчер по порогу `_limit` (меняется на лету), не размер пула."""

    def __init__(self, output_dir: Path, limit: int = DEFAULT_LIMIT) -> None:
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / _PERSIST
        self._lock = threading.RLock()
        self._items: dict[str, ConformJob] = {}
        self._stops: dict[str, threading.Event] = {}
        self._groups: dict[str, procreg.ProcGroup] = {}   # живые подпроцессы задач (надёжная отмена)
        self._limit = limit
        self._active = 0
        self._counter = itertools.count(1)
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="conform")

    # ── публичный API ──

    def enqueue(self, spec: dict) -> ConformJob:
        with self._lock:
            n = next(self._counter)
            job = ConformJob(id=f"cf{n:05d}", seq=n, **spec)
            job.dub_total = len(job.dubs)
            if not job.autostart:
                job.status = PAUSED
            self._items[job.id] = job
            self._stops[job.id] = threading.Event()
            self._persist_locked()
            self._pump_locked()
            return job.model_copy(deep=True)

    def list_items(self) -> list[ConformJob]:
        with self._lock:
            return [j.model_copy(deep=True) for j in self._items.values()]

    def get_item(self, jid: str) -> ConformJob | None:
        with self._lock:
            j = self._items.get(jid)
            return j.model_copy(deep=True) if j else None

    def cancel(self, jid: str) -> bool:
        """Отмена задачи. Для RUNNING — НАДЁЖНАЯ: помимо стоп-флага (проверяется между
        этапами) немедленно убивает дерево живых подпроцессов задачи, иначе длинный
        ffmpeg-декод доработал бы до конца. Для терминальной — удаление записи."""
        group = None
        with self._lock:
            j = self._items.get(jid)
            if j is None:
                return False
            if j.status in _TERMINAL:
                self._items.pop(jid, None)
                self._stops.pop(jid, None)
                self._groups.pop(jid, None)
            else:
                ev = self._stops.get(jid)
                if ev is not None:
                    ev.set()
                group = self._groups.get(jid)
                self._set(j, CANCELLED)
            self._persist_locked()
        if group is not None:                     # kill вне блокировки (ждёт смерти процессов)
            n = group.kill_all()
            if n:
                logger.info("conform {}: отмена — убито процессов: {}", jid, n)
        return True

    def pause(self, jid: str) -> bool:
        """QUEUED → PAUSED (задача ждёт ручного пуска). Running не трогаем — для него отмена."""
        with self._lock:
            j = self._items.get(jid)
            if j is None or j.status != QUEUED:
                return False
            self._set(j, PAUSED)
            self._persist_locked()
        return True

    def start(self, jid: str) -> bool:
        """PAUSED → QUEUED + попытка немедленного запуска (кнопка ▶ у строки)."""
        with self._lock:
            j = self._items.get(jid)
            if j is None or j.status != PAUSED:
                return False
            self._set(j, QUEUED)
            self._persist_locked()
            self._pump_locked()
        return True

    def shutdown(self) -> int:
        """Завершение работы приложения: остановить ВСЕ активные задачи и убить их
        подпроцессы. Без этого закрытие окна оставляет ffmpeg-сирот, продолжающих
        писать файлы. Возврат: сколько процессов убито."""
        with self._lock:
            groups = list(self._groups.values())
            for ev in self._stops.values():
                ev.set()
            for j in self._items.values():
                if j.status == RUNNING:
                    self._set(j, CANCELLED)
            self._persist_locked()
        killed = 0
        for g in groups:
            killed += g.kill_all()
        if killed:
            logger.info("conform shutdown: убито процессов: {}", killed)
        return killed

    def clear_done(self) -> dict:
        """Убрать ЗАВЕРШЁННЫЕ задачи из очереди И удалить их временные файлы.
        Выходные аудиофайлы и графики (_plots) НЕ трогаются — решение пользователя
        2026-08-06. Активные задачи не задеваются."""
        with self._lock:
            gone = [j.model_copy(deep=True) for j in self._items.values() if j.status in _TERMINAL]
            for j in gone:
                self._items.pop(j.id, None)
                self._stops.pop(j.id, None)
                self._groups.pop(j.id, None)
            self._persist_locked()
        freed = 0
        for j in gone:
            freed += _purge_tmp(j)
        return {"cleared": len(gone), "freed_mb": round(freed / (1 << 20), 1)}

    def retry(self, jid: str) -> bool:
        with self._lock:
            j = self._items.get(jid)
            if j is None or j.status not in _TERMINAL:
                return False
            j.error = None
            j.results = []
            j.progress = 0.0
            j.stage = ""
            j.dub_index = 0
            j.cur_dub = ""
            j.elapsed_s = 0.0
            self._stops[jid] = threading.Event()
            self._set(j, QUEUED)
            self._persist_locked()
            self._pump_locked()
        return True

    def clear(self) -> int:
        """Остановить ВСЕ задачи и удалить ВСЕ записи очереди. Возврат: сколько удалено.
        Running-воркеры получают сигнал стоп (по should_stop между озвучками/при декоде),
        их финальный блок увидит пустой реестр и просто свернётся."""
        with self._lock:
            n = len(self._items)
            for ev in self._stops.values():
                ev.set()
            self._items.clear()
            self._stops.clear()
            self._persist_locked()
        return n

    def get_settings(self) -> dict:
        with self._lock:
            return {"limit": self._limit, "active": self._active}

    def set_settings(self, limit: int | None = None) -> dict:
        with self._lock:
            if limit is not None:
                self._limit = max(0, int(limit))
            self._persist_locked()
            self._pump_locked()
            return {"limit": self._limit, "active": self._active}

    # ── диспетчер ──

    def _pump_locked(self) -> None:
        while self._active < self._limit:
            nxt = self._next(QUEUED)
            if nxt is None:
                break
            self._set(nxt, RUNNING)
            nxt.progress = 0.0
            self._active += 1
            self._stops.setdefault(nxt.id, threading.Event())
            self._pool.submit(self._run, nxt.id)

    def _next(self, status: str) -> ConformJob | None:
        c = [j for j in self._items.values() if j.status == status]
        c.sort(key=lambda j: j.seq)
        return c[0] if c else None

    def _set(self, job: ConformJob, status: str) -> None:
        job.status = status
        job.updated_at = _now()

    # ── воркер (вне _lock; состояние трогаем под _lock) ──

    def _run(self, jid: str) -> None:
        with self._lock:
            j = self._items.get(jid)
            if j is None or j.status != RUNNING:
                self._active = max(0, self._active - 1)
                self._pump_locked()
                return
            spec = j.model_copy(deep=True)
            stop = self._stops.get(jid)
            group = procreg.ProcGroup()          # реестр подпроцессов ЭТОЙ задачи
            self._groups[jid] = group
        procreg.bind(group)                      # потоко-локально: чужие задачи не задеты

        def progress(p) -> None:
            with self._lock:
                jj = self._items.get(jid)
                if jj is None:
                    return
                jj.stage = p.stage
                jj.stage_pct = p.pct
                jj.detail = p.detail
                jj.dub_index = p.dub_index
                jj.dub_total = p.dub_total
                jj.cur_dub = p.dub_name
                # Полоса = (реф + dub_total) РАВНЫХ долей по 1/(N+1). Доля 0 — реф; доли 1..N —
                # озвучки. Внутри доли — УПОРЯДОЧЕННЫЕ диапазоны этапов В ПОРЯДКЕ ВЫПОЛНЕНИЯ
                # (монотонно, без дыр, без скачков назад). Каждый этап: intra = lo + (hi−lo)·pct;
                # все этапы эмитят pct плавно (бэкенды инструментированы по итерациям циклов).
                # geom — между coarse и band (если зрение слепло; иначе диапазон пропускается ВПЕРЁД).
                slices = jj.dub_total + 1
                if p.dub_index == 0:                  # РЕФ: декод видео + извлечение аудио
                    lo, hi = {"decode": (0.0, 0.85), "extract": (0.85, 1.0)}.get(p.stage, (0.0, 0.85))
                else:                                 # ОЗВУЧКА: этапы по порядку (decode доминирует)
                    lo, hi = {"decode":   (0.00, 0.52),   # SRM дубля (по кадрам)
                              "coarse":   (0.52, 0.55),   # грубый проход
                              "geom":     (0.55, 0.62),   # геом-разбор (кроп/зум/полосы)
                              "band":     (0.62, 0.78),   # полоса Drop-DTW (по чанкам)
                              "extract":  (0.78, 0.88),   # декод аудио дубля (по кадрам)
                              "resample": (0.88, 0.93),   # ресэмпл/варп (по блокам)
                              "audio":    (0.93, 0.97),   # доводка band/muq
                              "write":    (0.97, 1.00)}.get(p.stage, (0.62, 0.78))
                intra = lo + (hi - lo) * min(1.0, max(0.0, p.pct))
                jj.progress = min(0.999, (p.dub_index + intra) / slices)
                jj.updated_at = _now()

        def on_pair(res: PairResult) -> None:
            if res.ok:
                logger.info("conform {} озвучка {} готова: назначено {:.1f}%, остаток {:.1f} мс, "
                            "покрытие {:.3f}, выход {}", jid, res.dub, res.assigned_pct,
                            res.audio_resid_ms, res.audio_coverage, res.out_path)
            else:
                logger.error("conform {} озвучка {} НЕ удалась: {}", jid, res.dub,
                             res.error or "без сообщения")
            with self._lock:
                jj = self._items.get(jid)
                if jj is None:
                    return
                jj.results.append(DubResult(
                    dub=res.dub, ok=res.ok, mode=getattr(res, "mode", "av"), skipped=res.skipped,
                    out_path=(str(res.out_path) if res.out_path else None),
                    assigned_pct=res.assigned_pct, slope=res.slope, cos_median=res.cos_median,
                    real_cuts=len(res.real_cuts), filled_cuts=res.filled_cuts,
                    edge_recovered=res.edge_recovered, audio_resid_ms=res.audio_resid_ms,
                    blind_zones=res.blind_zones,
                    dropped_intro_s=res.dropped_intro_s,
                    audio_cuts=res.audio_cuts, audio_max_step_ms=res.audio_max_step_ms,
                    audio_coverage=res.audio_coverage, audio_span_ms=res.audio_span_ms,
                    geom_used=res.geom_used, geom_n_in=res.geom_n_in,
                    geom_sx=res.geom_sx, geom_sy=res.geom_sy,
                    plots=[PlotRef(**p) for p in res.plots],
                    suspect=_suspect(res),
                    warnings=res.warnings, critical=res.critical,
                    error=res.error, elapsed_s=res.elapsed_s))
                # +1 доля — реф (slices = dub_total + 1): после k готовых озвучек → (k+1)/(N+1)
                jj.progress = min(0.999, (len(jj.results) + 1) / (jj.dub_total + 1))
                jj.updated_at = _now()
                self._persist_locked()

        def should_stop() -> bool:
            return stop is not None and stop.is_set()

        status, err = DONE, None
        t0 = time.perf_counter()
        logger.info("conform {} старт: реф={} озвучек={} выход={}",
                    jid, Path(spec.ref).name, len(spec.dubs), spec.out_dir)
        try:
            conform_episode(
                spec.ref, spec.dubs, spec.out_dir,
                cache_dir=spec.cache_dir,
                keep_tmp=spec.keep_tmp,             # tmp-чекпоинты: всё промежуточное (реф+дубль)
                low_mem=True,                       # 3.1: фичи через memmap — RAM не растёт с длиной
                fps_ref=_pfps(spec.fps_ref), fps_dub=_pfps(spec.fps_dub),
                free_start=spec.free_start, fill_silence=spec.fill_silence,
                audio_band=spec.audio_band, audio_muq=spec.audio_muq,
                apply_cuts=spec.apply_cuts,
                drift_speed_pct=spec.drift_speed_pct,
                ref_atrack=spec.ref_atrack, dub_atracks=spec.dub_atracks,
                progress=progress, should_stop=should_stop, on_pair=on_pair)
        except Exception as e:  # noqa: BLE001
            status, err = FAILED, str(e)
            logger.exception("conform job {} упал", jid)

        procreg.bind(None)
        with self._lock:
            self._groups.pop(jid, None)
            jj = self._items.get(jid)
            if jj is not None and jj.status == RUNNING:   # не перетирать CANCELLED
                jj.error = err
                jj.elapsed_s = time.perf_counter() - t0
                self._set(jj, status)
                if status == DONE:
                    jj.progress = 1.0
            self._active = max(0, self._active - 1)
            self._persist_locked()
            self._pump_locked()

    # ── персист / restore ──

    def _persist_locked(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            payload = {"items": [j.model_dump(mode="json") for j in self._items.values()],
                       "limit": self._limit, "updated_at": _now()}
            fd, tmp = tempfile.mkstemp(prefix="_conform.", suffix=".tmp", dir=str(self.output_dir))
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception as e:  # noqa: BLE001
            logger.warning("conform persist failed: {}", e)

    def restore(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning("conform restore failed: {}", e)
            return
        with self._lock:
            self._limit = int(data.get("limit", self._limit))
            maxseq = 0
            for d in data.get("items") or []:
                try:
                    job = ConformJob(**d)
                except Exception:  # noqa: BLE001
                    continue
                if job.status == RUNNING:              # оборвано рестартом → переиграть
                    job.status = QUEUED
                    job.progress = 0.0
                    job.stage = ""
                    job.dub_index = 0
                    job.cur_dub = ""
                    job.results = []
                self._items[job.id] = job
                self._stops[job.id] = threading.Event()
                maxseq = max(maxseq, job.seq)
            self._counter = itertools.count(maxseq + 1)
            self._pump_locked()
