"""Мост «страница → оболочка»: то, чего веб-страница не умеет сама.

Три вещи: выбрать файлы/каталог системным диалогом, показать файл в проводнике,
открыть ссылку в браузере. Всё остальное страница делает через обычный API ядра.

Диалоги обязаны выполняться в GUI-потоке, поэтому запрос из страницы прокидывается
в поток Qt через сигнал и ждёт ответа.
"""

from __future__ import annotations

import queue
import subprocess
from pathlib import Path

from fastapi import APIRouter
from PySide6.QtCore import QObject, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileDialog
from pydantic import BaseModel

FILE_FILTER = ("Видео и аудио (*.mkv *.mp4 *.avi *.m2ts *.ts *.webm *.flac *.mka "
               "*.mp3 *.wav *.aac *.opus *.ogg *.m4a);;Все файлы (*)")


class _Bridge(QObject):
    """Выполняет диалоги в GUI-потоке по запросу из потока сервера."""

    _ask = Signal(str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ask.connect(self._run, Qt.ConnectionType.QueuedConnection)

    @Slot(str, object)
    def _run(self, kind: str, box: queue.Queue) -> None:
        try:
            if kind == "dir":
                p = QFileDialog.getExistingDirectory(None, "")
                box.put([p] if p else [])
            elif kind == "ref":
                p, _ = QFileDialog.getOpenFileName(None, "", "", FILE_FILTER)
                box.put([p] if p else [])
            else:
                ps, _ = QFileDialog.getOpenFileNames(None, "", "", FILE_FILTER)
                box.put(list(ps))
        except Exception:  # noqa: BLE001 — окно не должно падать из-за диалога
            box.put([])

    def pick(self, kind: str) -> list[str]:
        box: queue.Queue = queue.Queue()
        self._ask.emit(kind, box)
        try:
            return box.get(timeout=300)
        except queue.Empty:
            return []


class _PickBody(BaseModel):
    kind: str = "dubs"          # ref | dubs | dir


class _PathBody(BaseModel):
    path: str


class _UrlBody(BaseModel):
    url: str


def make_shell_router(bridge: _Bridge) -> APIRouter:
    r = APIRouter(prefix="/ui", tags=["shell"])

    @r.post("/pick")
    def pick(body: _PickBody) -> dict:
        return {"paths": bridge.pick(body.kind)}

    @r.post("/reveal")
    def reveal(body: _PathBody) -> dict:
        """Показать файл в проводнике (или открыть каталог, если файла уже нет)."""
        p = Path(body.path)
        if p.exists():
            subprocess.Popen(["explorer", "/select,", str(p)])
        elif p.parent.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.parent)))
        return {"ok": True}

    @r.post("/open_url")
    def open_url(body: _UrlBody) -> dict:
        QDesktopServices.openUrl(QUrl(body.url))
        return {"ok": True}

    return r


def create_bridge(parent=None) -> _Bridge:
    return _Bridge(parent)
