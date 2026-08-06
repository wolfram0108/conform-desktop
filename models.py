"""Датаклассы conform: фичи, результаты, прогресс. JSON-дружелюбны (для API)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SrmFeatures:
    """SRM-вектора всех кадров одного видео + его эффективный fps.

    `pts` — ось времени кадров (сек, от нуля первого кадра), заполняется ТОЛЬКО для
    реального VFR (features.vfr_time_axis): там время кадра = pts[i], а НЕ индекс/fps.
    None (CFR, подавляющее большинство) → все потребители идут старым путём индекс/fps
    бит-в-бит. Решение принимает алгоритм по данным (невязка меток к равномерной сетке),
    это НЕ тумблер."""

    srm: np.ndarray            # (N, 18432) float16
    fps: float
    src: Path | None = None
    pts: np.ndarray | None = None   # (N,) float64, сек; только VFR

    @property
    def n_frames(self) -> int:
        return int(len(self.srm))

    @property
    def duration_s(self) -> float:
        if self.pts is not None and len(self.pts):
            return float(self.pts[-1]) + (1.0 / self.fps if self.fps else 0.0)
        return len(self.srm) / self.fps if self.fps else 0.0


@dataclass
class Progress:
    """Снимок прогресса для коллбэка (демон превращает в % для WEB)."""

    stage: str                 # "decode" | "align" | "resample" | "write"
    pct: float                 # 0..1 внутри этапа
    detail: str = ""
    dub_index: int = 0
    dub_total: int = 0
    dub_name: str = ""


@dataclass
class PairResult:
    """Паспорт качества одной пары ref↔dub (печатается в лог; в файлы пока не пишем)."""

    dub: str
    out_path: Path | None       # путь к выходному аудиофайлу (формат — деталь записи; None = не создан)
    ok: bool
    mode: str = "av"                   # "av" = зрение+звук (видео-дубль) | "audio" = аудио-only
                                       # (озвучка без видеопотока — только аудио-слой; поля зрения
                                       # assigned_pct/cos_median/slope в этом режиме НЕ имеют смысла)
    skipped: bool = False              # уже было готово (файл на диске) — не пересчитывали
    fps_ref: float = 0.0
    fps_dub: float = 0.0
    n_frames: int = 0
    duration_s: float = 0.0
    assigned_pct: float = 0.0          # % кадров озвучки, нашедших место на REF
    slope: float = 0.0                 # наклон offset (ref/dub), ~fps_ref/fps_dub
    cos_median: float = 0.0            # медиана cos назначенных пар
    monotonic_violations: int = 0
    real_cuts: list[tuple[float, float]] = field(default_factory=list)  # (t1,t2) сек на REF
    real_cuts_s: float = 0.0           # суммарная длительность вырезов (тишина/заполнение)
    filled_cuts: int = 0               # секунд тишины озвучки залито синхронным рефом (fill_silence, после band/muq)
    blind_zones: int = 0               # drop-зоны, опознанные как слепые (не режем)
    blind_restored: int = 0            # кадров возвращено в карту (R_ins)
    edge_recovered: int = 0            # кадров возвращено доп-проходом на краях (опенинг/концовка)
    geom_used: bool = False            # сработал геом-разбор (кроп/зум/анаморф/полосы) — зрение слепло без него
    geom_n_in: int = 0                 # геом: inlier-якорей в консенсусе (надёжность регистрации)
    geom_sx: float = 0.0               # геом: масштаб по X (анаморф = sx≠sy)
    geom_sy: float = 0.0               # геом: масштаб по Y
    telecine_ref: bool = False         # реф = запечённый 3:2-телесин (NTSC рип без IVTC) → каденс прорежен
    telecine_dub: bool = False         # дубль = запечённый 3:2-телесин → каденс прорежен
    tele_score: float = 0.0            # сила сигнатуры телесина (выступ периода-5; >0.05 = телесин)
    tc_dropped: int = 0                # кадров SRM выброшено прореживанием каденса (мягкий IVTC)
    audio_resid_ms: float = 0.0        # остаточный аудио-сдвиг ПОСЛЕ доводки (медиана |·|, мс; меньше=лучше; 0=слой не делался)
    dropped_intro_s: float = 0.0       # сколько секунд начала озвучки выпало (free_start)
    audio_cuts: int = 0                # band/muq: дискретных правок стыков (резов) найдено
    audio_max_step_ms: float = 0.0     # band/muq: крупнейший рез, мс
    audio_coverage: float = 0.0        # band/muq: доля трека с надёжными якорями (0..1)
    audio_span_ms: float = 0.0         # band/muq: полный диапазон движения сдвига, мс
    plots: list[dict] = field(default_factory=list)  # PNG-графики укладки: [{kind,name,t,v_ms}]
    warnings: list[str] = field(default_factory=list)
    critical: list[str] = field(default_factory=list)  # КРАСНОЕ: дефект, не исправляемый авто (студийный A/V-десинк)
    error: str | None = None
    elapsed_s: float = 0.0


@dataclass
class EpisodeResult:
    """Результат серии: реф + список PairResult."""

    ref: str
    out_dir: Path
    pairs: list[PairResult] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def n_ok(self) -> int:
        return sum(1 for p in self.pairs if p.ok)

    @property
    def n_fail(self) -> int:
        return sum(1 for p in self.pairs if not p.ok)
