"""Advanced SubStation Alpha reader: dialogue events in, format-neutral cues out.

Only what the format defines as text on screen becomes a cue. `Comment:` events are never shown;
text inside drawing mode ({\\p1} … {\\p0}) is a vector shape, not text. Of the override tags only
italic, bold and underline have a meaning in every text subtitle format; the rest is styling and
positioning that the neutral form cannot carry.
"""

from __future__ import annotations

import re

from track_muxer.conform.subtitles.vtt import Cue

_RE_TIME = re.compile(r"^\s*(\d+):(\d{1,2}):(\d{1,2})[.:](\d{1,3})\s*$")
_RE_OVERRIDE = re.compile(r"\{([^{}]*)\}")
_RE_TAG = re.compile(r"\\([a-zA-Z]+|\d+[a-zA-Z]+)([^\\]*)")
# Line breaks of the format: \N is hard, \n is soft (a break unless the wrap style is "smart").
_RE_BREAK = re.compile(r"\\[Nn]")
_NEUTRAL = {"i": "i", "b": "b", "u": "u"}


def _seconds(stamp: str) -> float:
    m = _RE_TIME.match(stamp)
    if m is None:
        raise ValueError(f"bad ASS time: {stamp!r}")
    h, mi, s, frac = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(frac) / 10 ** len(frac)


def _neutral_text(text: str) -> tuple[tuple[str, ...], bool]:
    """Event text -> (payload lines, whether styling was dropped)."""
    out: list[str] = []
    open_tags: list[str] = []
    drawing = False
    dropped = False
    pos = 0
    for m in _RE_OVERRIDE.finditer(text):
        if not drawing:
            out.append(text[pos:m.start()])
        pos = m.end()
        for tag, arg in _RE_TAG.findall(m.group(1)):
            arg = arg.strip()
            if tag == "p":
                drawing = arg not in ("", "0")
            elif tag.startswith("r"):
                # \\r and \\r<style name>: a reset ends every override; what the named style would set
                # is styling. No other override tag begins with r.
                arg = tag[1:] + arg
                out += [f"</{name}>" for name in reversed(open_tags)]
                open_tags.clear()
                dropped = dropped or bool(arg)
            elif tag in _NEUTRAL and arg in ("0", "1"):
                name = _NEUTRAL[tag]
                if arg == "1" and name not in open_tags:
                    open_tags.append(name)
                    out.append(f"<{name}>")
                elif arg == "0" and name in open_tags:
                    # Tags close in the reverse order of opening to stay well nested; the ones
                    # closed on the way are still in force and open again.
                    k = open_tags.index(name)
                    inner = open_tags[k + 1:]
                    out += [f"</{n}>" for n in reversed(open_tags[k:])]
                    out += [f"<{n}>" for n in inner]
                    open_tags[k:] = inner
            else:
                dropped = True
    if not drawing:
        out.append(text[pos:])
    out += [f"</{name}>" for name in reversed(open_tags)]
    flat = "".join(out).replace("\\h", " ")
    lines = tuple(ln.strip() for ln in _RE_BREAK.split(flat))
    return tuple(ln for ln in lines if ln), dropped


def parse_ass(text: str) -> tuple[list[Cue], int]:
    """Text -> (cues with format-neutral payload, number of cues whose styling was dropped).

    Raises ValueError when there is no [Events] section with a Format line naming Start, End
    and Text: without them the file does not say where its times and texts are.
    """
    lines = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    section = ""
    fields: list[str] | None = None
    cues: list[Cue] = []
    n_styled = 0
    for raw in lines:
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line.lower()
            continue
        if section != "[events]" or ":" not in line:
            continue
        kind, _, rest = line.partition(":")
        kind = kind.strip().lower()
        if kind == "format":
            fields = [f.strip().lower() for f in rest.split(",")]
            continue
        if kind != "dialogue":
            continue
        if fields is None or not {"start", "end", "text"} <= set(fields):
            raise ValueError("ASS events have no Format line naming Start, End and Text")
        # Text is the last field and may itself contain commas.
        values = rest.split(",", len(fields) - 1)
        if len(values) != len(fields):
            continue
        event = dict(zip(fields, values))
        payload, dropped = _neutral_text(event["text"])
        n_styled += bool(dropped)
        cues.append(Cue(start=_seconds(event["start"]), end=_seconds(event["end"]), payload=payload))
    if fields is None:
        raise ValueError("not an ASS file: no [Events] section with a Format line")
    return cues, n_styled
