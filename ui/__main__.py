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

# ── ГАРАНТИЯ: ни одного консольного окна из ЛЮБОГО кода процесса ──
# Точечных флагов в ядре мало: окно может открыть сторонняя библиотека, служебный
# taskkill или любой будущий вызов. Патчим subprocess на уровне процесса ДО импорта
# ядра — каждый дочерний процесс стартует скрытым и без консоли, поэтому фокус
# никогда не уходит из окна приложения.
if os.name == "nt":
    import subprocess as _sp

    _CREATE_NO_WINDOW = 0x08000000
    _orig_popen = _sp.Popen

    class _HiddenPopen(_orig_popen):          # noqa: D101 — техническая обёртка
        def __init__(self, *a, **kw):
            kw["creationflags"] = kw.get("creationflags", 0) | _CREATE_NO_WINDOW
            si = kw.get("startupinfo") or _sp.STARTUPINFO()
            si.dwFlags |= _sp.STARTF_USESHOWWINDOW
            si.wShowWindow = 0                # SW_HIDE
            kw["startupinfo"] = si
            super().__init__(*a, **kw)

    _sp.Popen = _HiddenPopen

from PySide6.QtCore import QSettings                      # noqa: E402
from PySide6.QtWidgets import QApplication                # noqa: E402

from server.app import create_app, data_dir, get_queue    # noqa: E402
from ui import winjob                                     # noqa: E402
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
    # Рубеж №1 (страховка ОС): все потомки умрут вместе с процессом ЛЮБЫМ способом —
    # включая taskkill /F и падение, когда наш код выхода отработать не успевает.
    winjob.enable_kill_on_close()

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
    _shutdown()
    # Жёсткий выход: ConformQueue держит non-daemon ThreadPoolExecutor — обычный
    # возврат оставил бы процесс висеть после закрытия окна.
    os._exit(rc)


def _shutdown() -> None:
    """Закрытие окна = конец работы: гасим задачи и НЕ оставляем ffmpeg-сирот.

    Рубежи: (2) очередь убивает подпроцессы своих задач штатно; (3) контрольный kill
    дерева ЭТОГО процесса — на случай процессов в обход реестра, потому что os._exit
    детей не забирает. Рубеж (1) — job object ОС, включён в main().
    """
    try:
        q = get_queue()
        if q is not None:
            q.shutdown()
    except Exception:  # noqa: BLE001 — выход не должен падать
        pass
    if os.name == "nt":
        try:
            import subprocess
            subprocess.run(["taskkill", "/PID", str(os.getpid()), "/T", "/F"],
                           capture_output=True, check=False,
                           creationflags=0x08000000)     # без консольного окна
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
