# -*- coding: utf-8 -*-
"""Отметки расхода памяти по шагам конвейера (диагностика, по умолчанию выключена).

Зачем отдельный модуль. Закон проекта: расход памяти НЕ должен расти с длительностью
входа. Нарушения этого закона вылезают только на длинных файлах и выглядят одинаково —
«процесс съел всю память», без указания на виновника. Рабочий набор процесса тут не
помощник: он смешивает собственные массивы со страницами отображённых файлов и растёт
даже там, где данные честно лежат на диске.

Поэтому меряем СОБСТВЕННУЮ память процесса (PrivateUsage) в конкретных точках
конвейера. Включается переменной окружения `CONFORM_MEMLOG=1`, в обычной работе не
стоит ничего.

Использование:
    from track_muxer.conform.memlog import memlog
    memlog("после декода аудио")
"""

from __future__ import annotations

import os

from loguru import logger

_ENV = "CONFORM_MEMLOG"


def enabled() -> bool:
    return os.environ.get(_ENV) == "1"


def memlog(tag: str) -> None:
    """Записать в журнал собственную память процесса и рабочий набор."""
    if not enabled() or os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class _PMCEX(ctypes.Structure):      # PROCESS_MEMORY_COUNTERS_EX — нужен PrivateUsage
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

        # ⚠ Типы обязательны: без них дескриптор процесса усекается до 32 бит и вызов
        # молча возвращает ноль (наступал на это 2026-08-07).
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
    except Exception:  # noqa: BLE001 — диагностика не должна ронять работу
        pass
