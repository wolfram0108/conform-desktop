# -*- coding: utf-8 -*-
"""Надёжное удаление временных каталогов с файлами, подключёнными как память.

Зачем отдельный модуль. Промежуточные буферы конформа (зрение референса и озвучки,
декодированный звук, выходной буфер) живут в файлах и подключаются к процессу как
память. На Windows такой файл НЕЛЬЗЯ удалить, пока подключение не закрыто: удаление
молча не срабатывает, а `shutil.rmtree(..., ignore_errors=True)` эту неудачу проглатывает.

Итог на практике (2026-08-07): после УСПЕШНОГО прогона фильма на сетевом диске оставался
каталог зрения референса на 4.2 ГБ — и так с каждым прогоном, пока диск не кончится.

Здесь подключение сначала закрывается явно, затем каталог удаляется, а неудача попадает
в журнал, а не исчезает.
"""

from __future__ import annotations

import gc
import shutil
from pathlib import Path

from loguru import logger


def close_maps(*arrays) -> None:
    """Закрыть подключения к файлам у переданных массивов (если это они)."""
    for a in arrays:
        if a is None:
            continue
        m = getattr(a, "_mmap", None)
        if m is None:                       # объект-обёртка: поищем массивы внутри
            for name in ("srm", "arr", "data"):
                inner = getattr(a, name, None)
                m = getattr(inner, "_mmap", None) if inner is not None else None
                if m is not None:
                    break
        try:
            if m is not None:
                m.close()
        except Exception:  # noqa: BLE001 — уже закрыт
            pass


def drop_dir(path, *arrays) -> bool:
    """Закрыть подключения и удалить каталог. -> удалось ли (в журнал при неудаче)."""
    if path is None:
        return True
    p = Path(path)
    close_maps(*arrays)
    gc.collect()                             # отпустить ссылки, которые держат файл открытым
    shutil.rmtree(p, ignore_errors=True)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)  # вторая попытка после сборки мусора
    if p.exists():
        try:
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**30
        except OSError:
            size = -1
        logger.warning("временный каталог не удалён: {} ({:.2f} ГБ) — файл ещё занят", p, size)
        return False
    return True
