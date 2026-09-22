# -*- coding: utf-8 -*-
"""Memory-usage markers at pipeline steps (diagnostics, off by default).

Why a separate module. Memory use must not grow with the input's duration. Violations of this
surface only on long files and all look the same -- "the process ate all the memory" -- with
nothing pointing at the culprit. The process's working set is no help here: it mixes its own
arrays with the pages of memory-mapped files and grows even where the data genuinely lives on
disk.

So this measures the process's OWN memory (PrivateUsage) at specific points in the pipeline.
Enabled by the environment variable `CONFORM_MEMLOG=1`; costs nothing in ordinary operation.

Usage:
    from track_muxer.conform.memlog import memlog
    memlog("<what has just happened>")
"""

from __future__ import annotations

import os

from loguru import logger

_ENV = "CONFORM_MEMLOG"


def enabled() -> bool:
    return os.environ.get(_ENV) == "1"


def memlog(tag: str) -> None:
    """Log the process's own memory and its working set."""
    if not enabled() or os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class _PMCEX(ctypes.Structure):      # PROCESS_MEMORY_COUNTERS_EX — PrivateUsage needed
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]

        # ⚠ Types are mandatory: without them the process handle is truncated to 32 bits
        # and the call silently returns zero.
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = wintypes.HANDLE
        fn = getattr(k, "K32GetProcessMemoryInfo", None) or ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMCEX), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        c = _PMCEX(); c.cb = ctypes.sizeof(_PMCEX)
        if not fn(k.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return
        logger.info("[память] {:<38} своя {:6.2f} ГБ | рабочий набор {:6.2f} ГБ", tag,
                    c.PrivateUsage / 2**30, c.WorkingSetSize / 2**30)
    except Exception:  # noqa: BLE001 — diagnostics must not break the pipeline
        pass
