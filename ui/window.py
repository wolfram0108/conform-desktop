"""Окно приложения: рамка Qt + страница интерфейса на встроенном Chromium.

Почему так (решение 2026-08-07 после провала ручной вёрстки на виджетах): интерфейс
описывается HTML/CSS — тем же языком, что и утверждённый макет, — и рисуется движком,
который едет ВНУТРИ дистрибутива. Отсюда два выигрыша: вид одинаков на любой машине
(ничего не берётся из системы) и правки предсказуемы.

Отладка: при `CONFORM_UI_DEBUG=1` включается порт удалённой отладки, и к живому окну
можно подключиться теми же средствами, что и к браузеру — кликать, читать состояние
элементов, снимать скриншоты. Это и есть «видеть ровно то, что видит пользователь».
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QMainWindow


def web_dir() -> Path:
    """Каталог со страницей интерфейса (внутри дистрибутива при сборке)."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "ui" / "web"      # noqa: SLF001 — раскладка PyInstaller
    return Path(__file__).resolve().parent / "web"


class _Page(QWebEnginePage):
    """Страница приложения: внешние ссылки уходят в системный браузер, ошибки — в лог."""

    def acceptNavigationRequest(self, url: QUrl, nav_type, is_main_frame: bool) -> bool:  # noqa: N802
        if is_main_frame and url.scheme() in ("http", "https") and "/conform/plot/" in url.path():
            QDesktopServices.openUrl(url)             # интерактивный график — в браузер
            return False
        return super().acceptNavigationRequest(url, nav_type, is_main_frame)

    def javaScriptConsoleMessage(self, level, message: str, line: int, source: str) -> None:  # noqa: N802
        print(f"[ui] {source}:{line} {message}", flush=True)


class MainWindow(QMainWindow):
    """Окно с интерфейсом. api_base — адрес локального API для страницы."""

    def __init__(self, api_base: str, settings=None) -> None:
        super().__init__()
        self.cfg = settings
        self.setWindowTitle("conform-desktop")
        w, h = 1180, 760
        if settings is not None:
            w = int(settings.value("ui/geometry_w", w))
            h = int(settings.value("ui/geometry_h", h))
        self.resize(w, h)
        self.setMinimumSize(900, 600)

        self.view = QWebEngineView(self)
        self.page = _Page(self.view)
        self.view.setPage(self.page)
        s = self.page.settings()
        s.setAttribute(QWebEngineSettings.WebAttribute.ShowScrollBars, True)
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        s.setAttribute(QWebEngineSettings.WebAttribute.FocusOnNavigationEnabled, True)
        self.setCentralWidget(self.view)

        # страница берётся с ТОГО ЖЕ локального сервера, что и API — один источник,
        # поэтому запросы к API не блокируются политикой безопасности браузера
        self.view.load(QUrl(f"{api_base}/app/index.html?api={api_base}"))

    def closeEvent(self, e) -> None:  # noqa: N802 — Qt-API
        if self.cfg is not None:
            self.cfg.setValue("ui/geometry_w", self.width())
            self.cfg.setValue("ui/geometry_h", self.height())
            self.cfg.sync()
        super().closeEvent(e)


def enable_debug_port() -> str | None:
    """Включить порт удалённой отладки страницы (для автотестов интерфейса).

    Ставится ДО создания QApplication — иначе Chromium не подхватит параметр.
    """
    port = os.environ.get("CONFORM_UI_DEBUG_PORT") or ("9333" if os.environ.get("CONFORM_UI_DEBUG") else None)
    if not port:
        return None
    flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = f"{flags} --remote-allow-origins=* ".strip()
    os.environ["QTWEBENGINE_REMOTE_DEBUGGING"] = port
    return port
