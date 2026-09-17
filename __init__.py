"""Модуль аудио-conform: выравнивание озвучки по таймлайну референса.

Слои:
  • kernel/   — числовое ЯДРО (перенос 1-в-1 из исследовательского прототипа, НЕ трогаем);
  • features  — видео → SRM-вектора (декод + свёртка);
  • align     — одна ПАРА ref↔dub → выходной аудиофайл (вычищенный conform v8);
  • episode   — одна СЕРИЯ (реф 1 раз + список озвучек);
  • cache     — переиспользование SRM (реф в RAM, диск опционально);
  • models    — датаклассы результата/прогресса;
  • config    — пути (ffmpeg/ffprobe, кэш).

Очередь и HTTP-API (Этапы 2–3) добавляются поверх как отдельные слои.
"""

from __future__ import annotations

from track_muxer.conform.models import (
    EpisodeResult,
    PairResult,
    Progress,
    SrmFeatures,
)

__all__ = [
    "EpisodeResult",
    "PairResult",
    "Progress",
    "SrmFeatures",
    "build_srm",
    "conform_episode",
    "conform_features",
    "conform_pair",
]


def __getattr__(name: str):  # noqa: D401 — ленивая загрузка тяжёлых слоёв (torch/numba/scipy)
    if name == "build_srm":
        from track_muxer.conform.features import build_srm
        return build_srm
    if name in ("conform_pair", "conform_features"):
        from track_muxer.conform import align
        return getattr(align, name)
    if name == "conform_episode":
        from track_muxer.conform.episode import conform_episode
        return conform_episode
    raise AttributeError(name)
