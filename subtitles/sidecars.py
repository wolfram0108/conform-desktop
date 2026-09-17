"""The sidecar naming convention: a subtitle file belongs to the video it is named after."""

from __future__ import annotations

from pathlib import Path

# Extensions a text track can be saved under.
TEXT_TRACK_EXTS = ("vtt", "ass")


def sidecar_name(video_stem: str, label: str, language: str, ext: str) -> str:
    """<video stem>.<label>.<language>.<ext>; label and language must already be file-name safe."""
    return f"{video_stem}.{label}.{language}.{ext}"


def text_track_sidecars(video: Path) -> list[Path]:
    """Subtitle sidecars of a video: files next to it named by sidecar_name."""
    prefix = video.stem + "."
    return sorted(
        p for p in video.parent.iterdir()
        if p.is_file() and p.name.startswith(prefix)
        and p.suffix.lower().lstrip(".") in TEXT_TRACK_EXTS)


def text_track_tail(video: Path, sidecar: Path) -> str:
    """The part of a sidecar name after the video stem: '.<label>.<language>.<ext>'."""
    return sidecar.name[len(video.stem):]
