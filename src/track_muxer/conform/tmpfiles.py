# -*- coding: utf-8 -*-
"""Temporary directories of a run: one owner, and removal that survives memory-mapped files.

The intermediate buffers of conform (frame features, decoded audio, the output buffer) live in
files mapped into the process. On Windows such a file cannot be removed while the mapping is
open: the removal fails silently and `shutil.rmtree(..., ignore_errors=True)` swallows it, so
gigabytes stay behind after every run. Here the mapping is closed first, the directory is removed
next, and a failure goes to the log instead of vanishing.
"""

from __future__ import annotations

import gc
import shutil
import tempfile
from pathlib import Path

from loguru import logger


TMP_DIR = "_tmp"


class Workspace:
    """Owner of one run's temporary directories.

    Every temporary directory of the run is created under one root next to the data, and the root
    is removed as a whole when the run ends, whatever its outcome. A directory made anywhere else
    has no owner: nothing removes it when a step in between fails.
    """

    def __init__(self, data_dir: Path | str, prefix: str) -> None:
        self._base = Path(data_dir) / TMP_DIR
        self._prefix = prefix
        self.root: Path | None = None        # made on first use: a run that needs no files writes nothing
        self._held: list = []

    def sub(self, prefix: str) -> Path:
        """A fresh directory under the root; it lives until the owner closes."""
        if self.root is None:
            self._base.mkdir(parents=True, exist_ok=True)
            self.root = Path(tempfile.mkdtemp(prefix=f"{self._prefix}_", dir=str(self._base)))
        return Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=str(self.root)))

    def adopt(self, *arrays) -> None:
        """Arrays backed by files under the root that may still be referenced when the owner
        closes (a frame kept alive by a propagating exception); closed before the removal."""
        self._held.extend(arrays)

    def close(self) -> bool:
        """Release the adopted arrays and remove the root. -> whether it is gone."""
        held, self._held = self._held, []
        root, self.root = self.root, None
        close_maps(*held)
        return drop_dir(root)


def close_maps(*arrays) -> None:
    """Close the file mappings of the given arrays, where they have one."""
    for a in arrays:
        if a is None:
            continue
        m = getattr(a, "_mmap", None)
        if m is None:                       # a wrapper object: look for the arrays inside
            for name in ("srm", "arr", "data"):
                inner = getattr(a, name, None)
                m = getattr(inner, "_mmap", None) if inner is not None else None
                if m is not None:
                    break
        try:
            if m is not None:
                m.close()
        except Exception:  # noqa: BLE001 — already closed
            pass


def drop_dir(path, *arrays) -> bool:
    """Close the mappings and remove the directory. -> whether it is gone (logged when not)."""
    if path is None:
        return True
    p = Path(path)
    close_maps(*arrays)
    gc.collect()                             # release the references that keep a file open
    shutil.rmtree(p, ignore_errors=True)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)  # a second attempt after the collection
    if p.exists():
        try:
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**30
        except OSError:
            size = -1
        logger.warning("временный каталог не удалён: {} ({:.2f} ГБ) — файл ещё занят", p, size)
        return False
    return True
