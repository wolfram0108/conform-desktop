"""Registry of a task's subprocesses — RELIABLE cancellation.

Why: the stop flag is checked BETWEEN stages, but a long-running ffmpeg step
(video SRM decode, audio extraction, FLAC encoding) inside a stage runs to
completion regardless — cancellation can stick for minutes, and closing the
app leaves the process orphaned.

How: the queue worker binds a group (`bind`) to ITS OWN thread; every ffmpeg
run goes through this module's `popen`/`run` and registers itself in that
group; `kill_all` kills the whole process tree immediately. The binding is
thread-local, so other tasks' processes are unaffected. Without a binding
(CLI, tests) the module behaves like plain subprocess.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading

_local = threading.local()

# Windows: hide the child process console window. In a windowed EXE (Qt with no console
# of its own) every ffmpeg/ffprobe run would otherwise flash a black window; a task can
# make dozens of probe calls, which would pile up into a screenful of consoles.
_CREATE_NO_WINDOW = 0x08000000


class Cancelled(RuntimeError):
    """A run was rejected: the group was already killed (the task is being cancelled)."""


def start_mark(pid: int) -> int | None:
    """When the process with this pid started, in the system's own units; None if there is none.

    A pid alone does not name a process: once it exits, the number goes to the next one started.
    The pid together with its start moment does, so a process is killed by pid only while the mark
    still matches the one taken when it was started."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            h = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return None
            try:
                c, e, kt, ut = (wintypes.FILETIME() for _ in range(4))
                if not k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e),
                                           ctypes.byref(kt), ctypes.byref(ut)):
                    return None
                return (c.dwHighDateTime << 32) | c.dwLowDateTime
            finally:
                k32.CloseHandle(h)
        with open(f"/proc/{pid}/stat", "rb") as f:
            fields = f.read().rsplit(b")", 1)[1].split()   # the name may hold spaces and brackets
        return int(fields[19])                              # field 22 of stat: starttime
    except (OSError, ValueError, IndexError):
        return None


def kill_pid(pid: int, mark: int | None = None) -> bool:
    """Kill the tree of a process known only by its pid — one this process did not start.

    With `mark`, only the process that still carries that start mark dies; a process that has since
    taken the pid is left alone. Every process `popen` starts leads its own group, so the group id is
    the pid and the whole tree goes at once. -> whether a kill was attempted."""
    if mark is not None and start_mark(pid) != mark:
        return False
    try:
        if os.name == "nt":
            # taskkill /T also kills the descendants ffmpeg can spawn of its own;
            # CREATE_NO_WINDOW is required here too, or taskkill itself flashes a console.
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, check=False,
                           creationflags=_CREATE_NO_WINDOW)
        else:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001 — the process may already be dead
        pass
    return True


def _kill_tree(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    kill_pid(p.pid)
    if p.poll() is None:
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            pass
    try:
        p.wait(timeout=5)
    except Exception:  # noqa: BLE001
        pass


class ProcGroup:
    """The live processes of one task."""

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
            if self._killed:                       # a race: the group was killed while this process was starting
                _kill_tree(p)
                raise Cancelled("задача отменена")
            self._procs.add(p)

    def discard(self, p: subprocess.Popen) -> None:
        with self._lock:
            self._procs.discard(p)

    def kill_all(self) -> int:
        """Kill every process in the group. Returns how many were alive."""
        with self._lock:
            self._killed = True
            procs = list(self._procs)
            self._procs.clear()
        for p in procs:
            _kill_tree(p)
        return len(procs)


_observer = None      # called (pid, alive) as subprocesses start and finish


def observe(fn) -> None:
    """Report every subprocess start (`fn(pid, True)`) and finish (`fn(pid, False)`) to `fn`.

    The conform worker reports them to the daemon: if the worker dies, its subprocesses survive it
    in their own sessions, and only the daemon is left to know which pids to kill."""
    global _observer
    _observer = fn


def bind(group: ProcGroup | None) -> None:
    """Bind a group to the CURRENT thread (called by the queue worker)."""
    _local.group = group


def current() -> ProcGroup | None:
    return getattr(_local, "group", None)


def popen(cmd, **kw) -> subprocess.Popen:
    """subprocess.Popen, plus registration in the current thread's group, plus no console window."""
    g = current()
    if g is not None and g.killed:
        raise Cancelled("задача отменена")
    if os.name == "nt":
        kw["creationflags"] = kw.get("creationflags", 0) | _CREATE_NO_WINDOW
    else:
        kw.setdefault("start_new_session", True)   # its own process group, so killpg can reach it
    p = subprocess.Popen(cmd, **kw)
    if g is not None:
        g.add(p)                                   # a killed group kills p and raises Cancelled
    if _observer is not None:
        _observer(p.pid, True)
    return p


def done(p: subprocess.Popen) -> None:
    """Remove the process from tracking (after wait/communicate)."""
    g = current()
    if g is not None:
        g.discard(p)
    if _observer is not None:
        _observer(p.pid, False)


def run(cmd, **kw) -> subprocess.CompletedProcess:
    """A drop-in for subprocess.run for ALL ffmpeg/ffprobe calls in conform: no console
    window (windowed EXE), tracked in the task's group (reliable cancellation)."""
    inp = kw.pop("input", None)
    timeout = kw.pop("timeout", None)
    check = kw.pop("check", False)
    if kw.pop("capture_output", False):            # compatibility with subprocess.run
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
