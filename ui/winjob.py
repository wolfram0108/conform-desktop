"""Windows Job Object: ОС гарантированно убивает потомков вместе с приложением.

Зачем, если есть штатное гашение при выходе (`ConformQueue.shutdown` + kill дерева):
штатный путь работает только когда приложение УСПЕВАЕТ отработать выход. Он бессилен
при `taskkill /F`, падении процесса или выключении из диспетчера задач — а именно там
и остаются ffmpeg-зомби, продолжающие писать файлы и держать GPU.

Job Object с флагом KILL_ON_JOB_CLOSE решает это на уровне ядра ОС: процесс приложения
помещается в job, все его потомки наследуют job автоматически, и когда последний
дескриптор job закрывается (то есть процесс приложения умирает ЛЮБЫМ способом), система
уничтожает всех, кто в нём остался. Чужие процессы (ffmpeg, запущенный другой программой)
в job не входят и не задеваются.

Дескриптор job намеренно НЕ закрывается и держится в модульной переменной всё время
жизни процесса — его закрытие и есть сигнал «убить всех».
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

_job = None                                  # держим дескриптор живым до конца процесса

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOBOBJECTEXTENDEDLIMITINFORMATION = 9


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong)]


class _BASIC_LIMIT(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class _EXTENDED_LIMIT(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BASIC_LIMIT),
                ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


def enable_kill_on_close() -> bool:
    """Поместить текущий процесс в job «убить всех при закрытии». True — включено."""
    global _job
    if os.name != "nt" or _job is not None:
        return _job is not None
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return False
        info = _EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, _JOBOBJECTEXTENDEDLIMITINFORMATION,
                                           ctypes.byref(info), ctypes.sizeof(info)):
            k32.CloseHandle(job)
            return False
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            k32.CloseHandle(job)
            return False
        _job = job                            # НЕ закрывать: закрытие = убить всех
        return True
    except Exception:  # noqa: BLE001 — отсутствие job не должно мешать запуску
        return False
