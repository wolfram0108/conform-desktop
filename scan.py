"""Scans a folder structure for the alignment panel. ONLY directory nesting matters:
  <catalog>/<subfolder = episode>/<video file = dub>
Names are arbitrary. For a movie, videos sit directly in the catalog folder → one "episode".

Auto-detected reference for an episode: a file with `_[ref]_` in its name; if there is none, a
heuristic applies: the single file without `__rus__` (e.g. a BDRip). Otherwise the reference is
left undetermined (set in the panel via "Overview"). Ready status: whether `_aligned/<output
name>.flac` or `.wav` exists (the output name comes from `naming.output_stem`).
Subfolders/files whose name starts with `_` (the service folders `_aligned`, `_conform_cache`) are
ignored.
"""

from __future__ import annotations

from pathlib import Path


from track_muxer.conform.naming import VIDEO_EXTS, output_stem
from track_muxer.conform.schema import ApiModel

_REF_MARK = "_[ref]_"


def _videos(d: Path) -> list[Path]:
    return sorted(
        [f for f in d.iterdir()
         if f.is_file() and f.suffix.lower() in VIDEO_EXTS and not f.name.startswith("_")],
        key=lambda f: f.name.lower(),
    )


class VideoFile(ApiModel):
    name: str
    is_ref: bool = False           # auto-detected ref (by _[ref]_ or the heuristic)
    is_sub: bool = False           # subtitles (__sub__), not a dub
    done: bool = False             # the conform output of this file's first audio track exists in _aligned
    done_tracks: list[int] = []    # audio tracks of this file whose conform output exists


class SeriesScan(ApiModel):
    dir: str
    name: str
    ref_auto: str | None = None    # auto-detected ref's file name (or None)
    files: list[VideoFile] = []    # all videos in the folder (dubs + ref)
    done: int = 0                  # dubs already aligned
    total: int = 0                 # total dubs (excluding the ref and subtitles)


class CatalogItem(ApiModel):
    path: str
    name: str
    series: int                    # subfolders that are episodes with video
    videos: int                    # total video files


def _done_tracks(stem: str, outputs: set[str]) -> list[int]:
    """Audio tracks of a file that have an output, by the naming rule: the first track is the
    stem itself, track N is the stem with __aN."""
    stem = stem.casefold()
    prefix = stem + "__a"
    later = sorted(int(o[len(prefix):]) for o in outputs if o.startswith(prefix) and o[len(prefix):].isdigit())
    return ([0] if stem in outputs else []) + later


def _scan_series(d: Path) -> SeriesScan:
    vids = _videos(d)
    ref_auto = next((f.name for f in vids if _REF_MARK in f.name.lower()), None)
    if ref_auto is None:
        non_dub = [f.name for f in vids
                   if "__rus__" not in f.name.lower() and "__sub__" not in f.name.lower()]
        if len(non_dub) == 1:
            ref_auto = non_dub[0]
    aligned = d / "_aligned"
    listing = [f.name for f in d.iterdir()]
    outputs = ({p.stem.casefold() for p in aligned.iterdir()
                if p.suffix.lower() in (".flac", ".wav") and p.stat().st_size > 0}
               if aligned.is_dir() else set())
    files: list[VideoFile] = []
    done = total = 0
    for f in vids:
        is_ref = (f.name == ref_auto)
        is_sub = "__sub__" in f.name.lower()
        done_tracks = _done_tracks(output_stem(f, siblings=listing), outputs)
        files.append(VideoFile(name=f.name, is_ref=is_ref, is_sub=is_sub, done=0 in done_tracks,
                               done_tracks=done_tracks))
        if not is_ref and not is_sub:
            total += 1
            if 0 in done_tracks:
                done += 1
    return SeriesScan(dir=str(d), name=d.name, ref_auto=ref_auto,
                      files=files, done=done, total=total)


def scan_catalog(path: str | Path) -> list[SeriesScan]:
    """Catalog folder → episodes. Subfolders with video are episodes; video directly in the catalog
    folder means a movie (one episode)."""
    root = Path(path)
    if not root.exists() or not root.is_dir():
        return []
    subdirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")],
        key=lambda p: p.name.lower(),
    )
    series_dirs = [d for d in subdirs if _videos(d)]
    if not series_dirs and _videos(root):
        series_dirs = [root]                       # a movie: video sits directly in the folder
    return [_scan_series(d) for d in series_dirs]


def list_catalogs(root: str | Path) -> list[CatalogItem]:
    """Subfolders of the downloads root as candidate catalogs (for the panel's picker)."""
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
