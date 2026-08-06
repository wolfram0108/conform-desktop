"""Сборка FastAPI-приложения conform-desktop.

Отличия от демона track-muxer: НЕТ core/players/download/mux — только conform.
Каталог данных (очередь `_conform.json`, дефолтный выход) — РЯДОМ с exe
(portable-требование CHARTER), переопределяется env `CONFORM_DATA_DIR`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI

from track_muxer.conform.queue import ConformQueue
from track_muxer.conform.routes import make_conform_router


def data_dir() -> Path:
    """Portable-каталог данных: env CONFORM_DATA_DIR, иначе appdata/ рядом с exe
    (frozen) или с корнем репозитория (запуск из исходников)."""
    env = os.environ.get("CONFORM_DATA_DIR")
    if env:
        return Path(env)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "appdata"
    return Path(__file__).resolve().parents[1] / "appdata"


def create_app(base: Path | None = None) -> FastAPI:
    base = Path(base) if base else data_dir()
    base.mkdir(parents=True, exist_ok=True)

    cq = ConformQueue(output_dir=base)
    cq.restore()                                   # оборванные running → queued

    app = FastAPI(title="conform-desktop", docs_url=None, redoc_url=None)
    app.include_router(make_conform_router(cq))

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "app": "conform-desktop"}

    return app
