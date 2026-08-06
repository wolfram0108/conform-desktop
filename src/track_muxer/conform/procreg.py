"""Реестр подпроцессов задачи — НАДЁЖНАЯ отмена (этап 10а миссии standalone).

Зачем: стоп-флаг проверяется МЕЖДУ этапами, а долгий ffmpeg-декод (SRM видео,
извлечение аудио, запись FLAC) внутри этапа доработает до конца — отмена на
минуты «залипает», а закрытие приложения оставляет процесс сиротой.

Как: воркер очереди привязывает к СВОЕМУ потоку группу (`bind`), все запуски
ffmpeg идут через `popen`/`run` этого модуля и регистрируются в ней; `kill_all`
убивает дерево процессов немедленно. Привязка потоко-локальная — процессы чужих
задач не задеваются. Без привязки (CLI, тесты) модуль работает как обычный
subprocess.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading

_local = threading.local()

# Windows: не показывать консольное окно дочернего процесса. В windowed-EXE (Qt без
# консоли) КАЖДЫЙ запуск ffmpeg/ffprobe иначе мигает чёрным окном — при десятках
# probe-вызовов на задачу это «куча консолей» на экране.
_CREATE_NO_WINDOW = 0x08000000


class Cancelled(RuntimeError):
    """Запуск отклонён: группа уже убита (задача отменяется)."""


def _kill_tree(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    try:
        if os.name == "nt":
            # taskkill /T — вместе с потомками (ffmpeg может порождать свои)
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                           capture_output=True, check=False)
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001 — процесс мог уже умереть
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            pass
    try:
        p.wait(timeout=5)
    except Exception:  # noqa: BLE001
        pass


class ProcGroup:
    """Живые процессы одной задачи."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._procs: set[subprocess.Popen] = set()
        self._killed = False

    @property
    def killed(self) -> bool:
        with self._lock:
            return self._killed

    def add(self, p: subprocess.Popen) -> None:
        with self._lock:
            if self._killed:                       # гонка: убили, пока стартовали
                _kill_tree(p)
                raise Cancelled("задача отменена")
            self._procs.add(p)

    def discard(self, p: subprocess.Popen) -> None:
        with self._lock:
            self._procs.discard(p)

    def kill_all(self) -> int:
        """Убить все процессы группы. Возврат: сколько было живых."""
        with self._lock:
            self._killed = True
            procs = list(self._procs)
            self._procs.clear()
        for p in procs:
            _kill_tree(p)
        return len(procs)


def bind(group: ProcGroup | None) -> None:
    """Привязать группу к ТЕКУЩЕМУ потоку (вызывает воркер очереди)."""
    _local.group = group


def current() -> ProcGroup | None:
    return getattr(_local, "group", None)


def popen(cmd, **kw) -> subprocess.Popen:
    """subprocess.Popen + регистрация в группе текущего потока + без консольного окна."""
    g = current()
    if g is not None and g.killed:
        raise Cancelled("задача отменена")
    if os.name == "nt":
        kw["creationflags"] = kw.get("creationflags", 0) | _CREATE_NO_WINDOW
    else:
        kw.setdefault("start_new_session", True)   # своя process group → killpg
    p = subprocess.Popen(cmd, **kw)
    if g is not None:
        g.add(p)
    return p


def done(p: subprocess.Popen) -> None:
    """Снять процесс с учёта (после wait/communicate)."""
    g = current()
    if g is not None:
        g.discard(p)


def run(cmd, **kw) -> subprocess.CompletedProcess:
    """Замена subprocess.run для ВСЕХ вызовов ffmpeg/ffprobe в conform: без консольного
    окна (windowed-EXE) и с учётом в группе задачи (надёжная отмена)."""
    inp = kw.pop("input", None)
    timeout = kw.pop("timeout", None)
    check = kw.pop("check", False)
    if kw.pop("capture_output", False):            # совместимость с subprocess.run
        kw.setdefault("stdout", subprocess.PIPE)
        kw.setdefault("stderr", subprocess.PIPE)
    p = popen(cmd, **kw)
    try:
        out, err = p.communicate(inp, timeout=timeout)
    finally:
        done(p)
    cp = subprocess.CompletedProcess(cmd, p.returncode, out, err)
    if check:
        cp.check_returncode()
    return cp
