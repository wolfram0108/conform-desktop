"""Запуск приложения: `python -m ui`.

Поднимает embedded API (uvicorn) фоновым потоком ЭТОГО ЖЕ процесса на
127.0.0.1:<порт> и открывает Qt-окно. Если порт уже отвечает нашим /health —
переиспользуем живой сервер (второй экземпляр приложения).
"""

from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from server.app import create_app, data_dir
from ui.client import Api
from ui.main_window import MainWindow

PORT = int(os.environ.get("CONFORM_PORT", "8799"))


def _health_ok(base: str) -> bool:
    try:
        with urllib.request.urlopen(base + "/health", timeout=1) as r:
            return b"conform-desktop" in r.read()
    except Exception:  # noqa: BLE001
        return False


def _start_server() -> None:
    import uvicorn
    cfg = uvicorn.Config(create_app(), host="127.0.0.1", port=PORT, log_level="warning")
    uvicorn.Server(cfg).run()      # в не-главном потоке uvicorn сам пропускает signal-handlers


def main() -> int:
    base = f"http://127.0.0.1:{PORT}"
    if not _health_ok(base):
        threading.Thread(target=_start_server, daemon=True, name="api").start()
        for _ in range(100):                       # ждём готовности (torch-импорты небыстрые)
            if _health_ok(base):
                break
            time.sleep(0.2)

    app = QApplication(sys.argv)
    app.setApplicationName("conform-desktop")
    cfg = QSettings(str(data_dir() / "ui.ini"), QSettings.IniFormat)
    win = MainWindow(Api(base), cfg)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
