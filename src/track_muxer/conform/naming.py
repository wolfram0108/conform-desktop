"""Output names of a conform job: one rule shared by the writer, the scanner and the enqueue check.

An output is named after its source file. Two sources that would land on one output path must
never overwrite each other silently: either the rule tells them apart, or the job is refused
before any work starts.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

VIDEO_EXTS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".ts", ".webm", ".m4v"})
AUDIO_EXTS = frozenset({".flac", ".mka", ".mp3", ".aac", ".m4a", ".ac3", ".eac3", ".dts",
                        ".wav", ".ogg", ".opus", ".wma"})
MEDIA_EXTS = VIDEO_EXTS | AUDIO_EXTS


def _listing(directory: Path) -> list[str]:
    try:
        return [f.name for f in directory.iterdir()]
    except OSError:
        return []


def _has_twin(dub: Path, siblings: Iterable[str]) -> bool:
    """Whether the dub's directory holds another media file with the same stem."""
    stem, name = dub.stem.casefold(), dub.name.casefold()
    for other in siblings:
        o = Path(other)
        if (o.name.casefold() != name and o.stem.casefold() == stem
                and o.suffix.lower() in MEDIA_EXTS):
            return True
    return False


def output_stem(dub: Path | str, atrack: int = 0, siblings: Iterable[str] | None = None) -> str:
    """Stem of the conform output for `dub` and its audio track.

    A file whose directory holds a twin differing only by extension carries the extension in the
    stem. The rule reads the dub's own directory, not the job, so the writer and the scanner get
    the same name whatever set of dubs a job holds. `siblings` is that directory's listing when
    the caller already has it. A track other than the first is a separate output of the same file.
    """
    dub = Path(dub)
    stem = dub.stem
    if _has_twin(dub, _listing(dub.parent) if siblings is None else siblings):
        stem += f"__{dub.suffix.lstrip('.').lower()}"
    if atrack:
        stem += f"__a{atrack}"
    return stem


def plan_outputs(dubs: Iterable[Path | str], atracks: list[int] | None = None,
                 out_dir: Path | str | None = None, ref: Path | str | None = None) -> list[str]:
    """Output stems of a job, parallel to `dubs`; ValueError when the job cannot keep them apart.

    Refused: two sources with one output, and an output directory that holds a source. The naming
    rule reads the directories of the sources, so the product must never write its outputs there:
    an output would pass for a twin of its own source on the next run.
    Names are compared without case: the outputs may live on a case-insensitive file system.
    """
    dubs = [Path(d) for d in dubs]
    if out_dir is not None:
        out = Path(out_dir).resolve()
        for src in ([Path(ref)] if ref is not None else []) + dubs:
            if src.resolve().parent == out:
                raise ValueError(f"каталог выхода совпадает с каталогом источника {src}: "
                                 f"выходы пишутся в отдельный каталог")
    listings: dict[Path, list[str]] = {}
    stems: list[str] = []
    owner: dict[str, Path] = {}
    for i, dub in enumerate(dubs):
        atrack = int(atracks[i]) if (atracks and i < len(atracks)) else 0
        if dub.parent not in listings:
            listings[dub.parent] = _listing(dub.parent)
        stem = output_stem(dub, atrack, listings[dub.parent])
        first = owner.setdefault(stem.casefold(), dub)
        if first is not dub:
            raise ValueError(f"два источника дают один выход «{stem}»: {first} и {dub}")
        stems.append(stem)
    return stems
