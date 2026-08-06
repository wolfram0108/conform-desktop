"""Скан структуры для панели выравнивания. Важна ТОЛЬКО вложенность каталогов:
  <каталог>/<подпапка = серия>/<видеофайл = озвучка>
Имена произвольны. Для фильма видео лежат прямо в каталоге → одна «серия».

Авто-референс серии: файл с подстрокой `_[ref]_` в имени; если такого нет —
эвристика: единственный файл без `__rus__` (напр. BDRip). Иначе реф не определён
(задаётся в панели «Обзором»). Статус готовности: есть ли `_aligned/<stem>.wav`.
Подпапки/файлы, начинающиеся с `_` (служебные `_aligned`, `_conform_cache`), игнорируются.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

_VID = {".mp4", ".mkv", ".avi", ".mov", ".ts", ".webm", ".m4v"}
_REF_MARK = "_[ref]_"


def _videos(d: Path) -> list[Path]:
    return sorted(
        [f for f in d.iterdir()
         if f.is_file() and f.suffix.lower() in _VID and not f.name.startswith("_")],
        key=lambda f: f.name.lower(),
    )


class VideoFile(BaseModel):
    name: str
    is_ref: bool = False           # авто-реф (по _[ref]_ или эвристике)
    is_sub: bool = False           # субтитры (__sub__) — не озвучка
    done: bool = False             # есть _aligned/<stem>.wav


class SeriesScan(BaseModel):
    dir: str
    name: str
    ref_auto: str | None = None    # имя файла авто-рефа (или None)
    files: list[VideoFile] = []    # все видео папки (озвучки + реф)
    done: int = 0                  # озвучек уже выровнено
    total: int = 0                 # озвучек всего (без рефа и субтитров)


class CatalogItem(BaseModel):
    path: str
    name: str
    series: int                    # подпапок-серий с видео
    videos: int                    # суммарно видеофайлов


def _scan_series(d: Path) -> SeriesScan:
    vids = _videos(d)
    ref_auto = next((f.name for f in vids if _REF_MARK in f.name.lower()), None)
    if ref_auto is None:
        non_dub = [f.name for f in vids
                   if "__rus__" not in f.name.lower() and "__sub__" not in f.name.lower()]
        if len(non_dub) == 1:
            ref_auto = non_dub[0]
    aligned = d / "_aligned"
    files: list[VideoFile] = []
    done = total = 0
    for f in vids:
        is_ref = (f.name == ref_auto)
        is_sub = "__sub__" in f.name.lower()
        is_done = (aligned / f"{f.stem}.flac").exists() or (aligned / f"{f.stem}.wav").exists()
        files.append(VideoFile(name=f.name, is_ref=is_ref, is_sub=is_sub, done=is_done))
        if not is_ref and not is_sub:
            total += 1
            if is_done:
                done += 1
    return SeriesScan(dir=str(d), name=d.name, ref_auto=ref_auto,
                      files=files, done=done, total=total)


def scan_catalog(path: str | Path) -> list[SeriesScan]:
    """Каталог → серии. Подпапки с видео = серии; если видео прямо в каталоге — фильм (одна серия)."""
    root = Path(path)
    if not root.exists() or not root.is_dir():
        return []
    subdirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")],
        key=lambda p: p.name.lower(),
    )
    series_dirs = [d for d in subdirs if _videos(d)]
    if not series_dirs and _videos(root):
        series_dirs = [root]                       # фильм: видео прямо в каталоге
    return [_scan_series(d) for d in series_dirs]


def list_catalogs(root: str | Path) -> list[CatalogItem]:
    """Подкаталоги downloads-корня как кандидаты-каталоги (для выбора в панели)."""
    root = Path(root)
    out: list[CatalogItem] = []
    if not root.exists() or not root.is_dir():
        return out
    for d in sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")],
                    key=lambda p: p.name.lower()):
        series = scan_catalog(d)
        nvid = sum(len(s.files) for s in series)
        if nvid == 0:
            continue
        out.append(CatalogItem(path=str(d), name=d.name, series=len(series), videos=nvid))
    return out
