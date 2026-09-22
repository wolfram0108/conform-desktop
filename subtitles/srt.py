"""SubRip writer. Payload is format-neutral text: plain lines with optional <i>/<b>/<u>."""

from __future__ import annotations


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

