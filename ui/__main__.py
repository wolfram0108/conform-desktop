"""Запуск приложения: `python -m ui` (dev) или frozen conform-desktop.exe.

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
from pathlib import Path

# ── frozen-бутстрап ДО импорта conform (config.py резолвит ffmpeg при импорте) ──
if getattr(sys, "frozen", False):
    _d = Path(sys.executable).resolve().parent
    for _c in (_d / "_internal", _d):
        if (_c / "ffmpeg.exe").exists():
            os.environ.setdefault("TM_FFMPEG", str(_c / "ffmpeg.exe"))
            os.environ.setdefault("TM_FFPROBE", str(_c / "ffprobe.exe"))
            break
    # windowed-exe: stdout/stderr = None — любая запись в них (loguru, uvicorn, traceback)
    # валит процесс. Подменяем ОБА безусловно на файл рядом с exe, до импорта чего-либо.
    _log_dir = _d / "appdata"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _log = open(_log_dir / "ui.log", "a", buffering=1, encoding="utf-8", errors="replace")
    if sys.stdout is None or not hasattr(sys.stdout, "write"):
        sys.stdout = _log
    if sys.stderr is None or not hasattr(sys.stderr, "write"):
        sys.stderr = _log

from PySide6.QtCore import QSettings                      # noqa: E402
from PySide6.QtWidgets import QApplication                # noqa: E402

from server.app import create_app, data_dir               # noqa: E402
from ui.client import Api                                 # noqa: E402
from ui.main_window import MainWindow                     # noqa: E402

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
    rc = app.exec()
    # Жёсткий выход обязателен: ConformQueue держит non-daemon ThreadPoolExecutor —
    # обычный sys.exit() оставляет процесс (и консоль) висеть после закрытия окна.
    # ⚠ Запущенные ffmpeg-подпроцессы при этом осиротеют — их аккуратное убийство
    # придёт вместе с «надёжной отменой» (этап 10а, kill process-tree задачи).
    os._exit(rc)


if __name__ == "__main__":
    main()
