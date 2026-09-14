"""Application entry: `python -m ui` (dev) or the frozen conform-desktop.exe.

Starts the embedded API (uvicorn) in a background thread of this process on
127.0.0.1:<port> and opens the Qt window. If the port already answers our /health,
that live server is reused (second instance of the application).
"""

from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

_FROZEN = bool(getattr(sys, "frozen", False))
_HOME = (Path(sys.executable).resolve().parent if _FROZEN
         else Path(__file__).resolve().parent.parent)     # exe dir or repository root

# Frozen bootstrap before importing conform: config.py resolves ffmpeg at import time.
if _FROZEN:
    for _c in (_HOME / "_internal", _HOME):
        if (_c / "ffmpeg.exe").exists():
            os.environ.setdefault("TM_FFMPEG", str(_c / "ffmpeg.exe"))
            os.environ.setdefault("TM_FFPROBE", str(_c / "ffprobe.exe"))
            break

# Any windowed launch (frozen exe or pythonw) has stdout/stderr = None: loguru, uvicorn's
# logging setup (sys.stdout.isatty()) and tracebacks then kill threads silently.
if sys.stdout is None or sys.stderr is None:
    _log_dir = _HOME / "appdata"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _log = open(_log_dir / "ui.log", "a", buffering=1, encoding="utf-8", errors="replace")
    if sys.stdout is None:
        sys.stdout = _log
    if sys.stderr is None:
        sys.stderr = _log

# No console window from any code of this process: a library, a service taskkill or any
# future call could open one, so subprocess is patched process-wide before the core loads.
if os.name == "nt":
    import subprocess as _sp

    _CREATE_NO_WINDOW = 0x08000000
    _orig_popen = _sp.Popen

    class _HiddenPopen(_orig_popen):          # noqa: D101 — technical wrapper
        def __init__(self, *a, **kw):
            kw["creationflags"] = kw.get("creationflags", 0) | _CREATE_NO_WINDOW
            si = kw.get("startupinfo") or _sp.STARTUPINFO()
            si.dwFlags |= _sp.STARTF_USESHOWWINDOW
            si.wShowWindow = 0                # SW_HIDE
            kw["startupinfo"] = si
            super().__init__(*a, **kw)

    _sp.Popen = _HiddenPopen

from loguru import logger                                 # noqa: E402
from PySide6.QtCore import QSettings                      # noqa: E402
from PySide6.QtWidgets import QApplication                # noqa: E402

from server.app import create_app, data_dir, get_queue    # noqa: E402
from ui import winjob                                     # noqa: E402
from ui.shell_api import create_bridge, make_shell_router  # noqa: E402
from ui.window import MainWindow, enable_debug_port, web_dir  # noqa: E402

PORT = int(os.environ.get("CONFORM_PORT", "8799"))


def _setup_log() -> Path:
    """Core log next to the application: `appdata/conform.log`.

    `ui.log` only receives what libraries write to the standard streams; core messages
    (job stages, errors with traces, output of failed external processes) need their own
    file, otherwise a failed hour-long run leaves nothing but a bare "Broken pipe".
    """
    log_dir = _HOME / "appdata"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "conform.log"
    logger.add(str(path), rotation="10 MB", retention=5, encoding="utf-8",
               enqueue=True,          # jobs run in threads: records must not interleave
               backtrace=True, diagnose=False, level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}")
    logger.info("conform-desktop запущен (порт {}), журнал: {}", PORT, path)
    return path


def _health_ok(base: str) -> bool:
    try:
        with urllib.request.urlopen(base + "/health", timeout=1) as r:
            return b"conform-desktop" in r.read()
    except Exception:  # noqa: BLE001
        return False


_APP = None                       # FastAPI application of this process (core + shell bridge)


def _start_server() -> None:
    import uvicorn
    cfg = uvicorn.Config(_APP, host="127.0.0.1", port=PORT, log_level="warning")
    uvicorn.Server(cfg).run()      # off the main thread uvicorn skips signal handlers itself


def main() -> int:
    global _APP
    # OS-level guard: every child dies with this process however it ends, including
    # taskkill /F and crashes, when our own exit code never gets to run.
    winjob.enable_kill_on_close()
    _setup_log()                   # first, so early failures are not mute
    enable_debug_port()            # page debug port must precede QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("conform-desktop")

    base = f"http://127.0.0.1:{PORT}"
    if not _health_ok(base):
        bridge = create_bridge(app)               # file dialogs run in the GUI thread
        _APP = create_app()
        _APP.include_router(make_shell_router(bridge))
        # The page is served by the same server: same origin, so the browser allows API
        # calls (they are blocked by the security policy when loaded from file://).
        from fastapi.staticfiles import StaticFiles
        _APP.mount("/app", StaticFiles(directory=str(web_dir()), html=True), name="ui")
        api = threading.Thread(target=_start_server, daemon=True, name="api")
        api.start()
        for _ in range(150):                      # readiness wait: torch imports are slow
            if _health_ok(base):
                break
            if not api.is_alive():                # a dead server must be named, not waited for
                logger.error("встроенный сервер не запустился на порту {}: см. appdata/ui.log", PORT)
                break
            time.sleep(0.2)

    cfg = QSettings(str(data_dir() / "ui.ini"), QSettings.IniFormat)
    win = MainWindow(base, cfg)
    win.show()
    rc = app.exec()
    _shutdown()
    # Hard exit: ConformQueue holds a non-daemon ThreadPoolExecutor, a plain return
    # would leave the process alive after the window closes.
    os._exit(rc)


def _shutdown() -> None:
    """Closing the window ends the work: stop jobs and leave no orphaned ffmpeg.

    The queue kills its jobs' subprocesses; then the tree of this process is killed as a
    backstop for processes outside the registry, because os._exit does not reap children.
    The OS job object enabled in main() is the last line.
    """
    try:
        q = get_queue()
        if q is not None:
            q.shutdown()
    except Exception:  # noqa: BLE001 — exit must not fail
        pass
    if os.name == "nt":
        try:
            import subprocess
            subprocess.run(["taskkill", "/PID", str(os.getpid()), "/T", "/F"],
                           capture_output=True, check=False,
                           creationflags=0x08000000)     # no console window
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
