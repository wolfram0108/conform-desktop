"""SubRip codec. Payload is format-neutral text: plain lines with optional <i>/<b>/<u>."""

from __future__ import annotations

import re

from track_muxer.conform.subtitles.vtt import Cue

_RE_TIMING = re.compile(
    r"^\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})\s+-->\s+(\d+):(\d{2}):(\d{2})[,.](\d{3})")


def _stamp(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def dump_srt(cues) -> str:
    """Cues -> SubRip text, numbered in the given order. A cue without text is not written:
    SubRip treats an empty block as the end of the previous one."""
    out: list[str] = []
    n = 0
    for c in cues:
        lines = [ln for ln in c.payload if ln.strip()]
        if not lines:
            continue
        n += 1
        out += [str(n), f"{_stamp(c.start)} --> {_stamp(c.end)}", *lines, ""]
    return "\n".join(out)


def parse_srt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        k = next((i for i, ln in enumerate(lines) if _RE_TIMING.match(ln)), None)
        if k is None:
            continue
        g = _RE_TIMING.match(lines[k]).groups()
        start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000.0
        end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000.0
        cues.append(Cue(start=start, end=end, payload=tuple(lines[k + 1:])))
    return cues
