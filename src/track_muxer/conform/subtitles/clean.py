"""Remove what is not text by the definition of the format it came from."""

from __future__ import annotations

import re

_RE_TAG = re.compile(r"</?[ibu]>")
# ASS vector drawing: a move to a point, then drawing commands with their numbers, nothing else.
# In ASS such a line is a shape ({\p1}); a converter that leaked it made it look like text.
_NUM = r"-?\d+(?:\.\d+)?"
_RE_ASS_DRAWING = re.compile(rf"^m(?:\s+{_NUM}){{2,}}(?:\s+[nlbspc](?:\s+{_NUM})*)+$")


def is_ass_drawing(payload) -> bool:
    text = " ".join(_RE_TAG.sub("", ln).strip() for ln in payload).strip()
    return bool(_RE_ASS_DRAWING.match(text))


def drop_empty(cues) -> tuple[list, int]:
    """-> (cues with text, number removed). A cue without text shows nothing on screen."""
    kept = [c for c in cues if any(_RE_TAG.sub("", ln).strip() for ln in c.payload)]
    return kept, len(cues) - len(kept)


def clean_cues(cues) -> tuple[list, dict]:
    """Every removal that loses no text, with its count: -> (cues, {reason: removed})."""
    cues, n_draw = drop_ass_drawings(cues)
    cues, n_empty = drop_empty(cues)
    cues, n_dup = drop_exact_duplicates(cues)
    return cues, {"dropped_drawings": n_draw, "dropped_empty": n_empty, "dropped_duplicates": n_dup}


def drop_exact_duplicates(cues) -> tuple[list, int]:
    """-> (cues, number removed). A cue repeating another's time and text to the letter adds
    nothing but a second copy on screen; flattened ASS layers produce them."""
    seen: set[tuple] = set()
    kept = []
    for c in cues:
        key = (c.start, c.end, c.payload)
        if key not in seen:
            seen.add(key)
            kept.append(c)
    return kept, len(cues) - len(kept)


def drop_ass_drawings(cues) -> tuple[list, int]:
    """-> (cues that are text, number of drawing cues removed)."""
    kept = [c for c in cues if not is_ass_drawing(c.payload)]
    return kept, len(cues) - len(kept)
