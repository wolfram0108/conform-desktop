"""WebVTT codec: cues in, cues out; everything that is not timing is carried through verbatim."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, replace

# WebVTT allows the hours field to be omitted; minutes and seconds are always two digits.
_RE_TIMING = re.compile(
    r"^\s*(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})\s+-->\s+(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})(.*)$")


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    payload: tuple[str, ...]
    settings: str = ""
    ident: str = ""

    def at(self, start: float, end: float) -> "Cue":
        return replace(self, start=start, end=end)


@dataclass(frozen=True)
class VttDocument:
    # Non-cue blocks (header line, NOTE/STYLE/REGION) in source order.
    blocks: tuple[tuple[str, ...], ...]
    cues: tuple[Cue, ...]

    def with_cues(self, cues) -> "VttDocument":
        return replace(self, cues=tuple(cues))


def _seconds(h: str | None, m: str, s: str, ms: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _stamp(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def parse_vtt(text: str) -> VttDocument:
    """Text -> document. Raises ValueError when the mandatory WEBVTT header is missing."""
    lines = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines or not lines[0].startswith("WEBVTT"):
        raise ValueError("not a WebVTT file: no WEBVTT header")
    blocks: list[tuple[str, ...]] = []
    cues: list[Cue] = []
    cur: list[str] = []
    # A cue payload may directly follow another cue without a blank line in damaged files,
    # so a timing line always opens a new block.
    chunks: list[list[str]] = []
    for ln in lines:
        if not ln.strip():
            if cur:
                chunks.append(cur)
                cur = []
            continue
        if _RE_TIMING.match(ln) and cur and any(_RE_TIMING.match(x) for x in cur):
            chunks.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        chunks.append(cur)
    for chunk in chunks:
        k = next((i for i, ln in enumerate(chunk) if _RE_TIMING.match(ln)), None)
        if k is None or k > 1:
            blocks.append(tuple(chunk))
            continue
        g = _RE_TIMING.match(chunk[k]).groups()
        cues.append(Cue(start=_seconds(*g[0:4]), end=_seconds(*g[4:8]),
                        settings=g[8].strip(), ident=chunk[0] if k == 1 else "",
                        payload=tuple(chunk[k + 1:])))
    return VttDocument(blocks=tuple(blocks), cues=tuple(cues))


# Tags every text subtitle format understands; the rest of WebVTT markup has no meaning outside it.
_RE_FOREIGN_TAG = re.compile(r"<(?!/?[ibu]>)[^<>]*>")


def neutral_cues(doc: VttDocument) -> tuple[list[Cue], int]:
    """Cues with format-neutral payload (plain text + <i>/<b>/<u>). -> (cues, settings dropped)."""
    out = [replace(c, settings="", ident="",
                   payload=tuple(html.unescape(_RE_FOREIGN_TAG.sub("", ln)) for ln in c.payload))
           for c in doc.cues]
    return out, sum(1 for c in doc.cues if c.settings)


def dump_vtt(doc: VttDocument) -> str:
    out: list[str] = []
    for b in doc.blocks:
        out += [*b, ""]
    if not doc.blocks:
        out += ["WEBVTT", ""]
    for c in doc.cues:
        if c.ident:
            out.append(c.ident)
        timing = f"{_stamp(c.start)} --> {_stamp(c.end)}"
        out.append(f"{timing} {c.settings}" if c.settings else timing)
        out += [*c.payload, ""]
    return "\n".join(out)
