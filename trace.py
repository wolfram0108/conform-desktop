"""Decision trace of one ref<->dub pair.

Every branch conform takes on a measured value is recorded with the inputs it read, the
thresholds it compared against and the state of the features the inputs were measured on.
The trace is the audit of the run: it travels with PairResult to the API and is saved next
to the outputs, so a rejected or odd pair can be explained from the record alone.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from loguru import logger

TRACE_DIR = "_trace"                 # subfolder of the output dir holding <stem>.json per pair


def _plain(v):
    """JSON-safe copy: numpy scalars/arrays and paths become builtins; arrays are summarised."""
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, np.ndarray):
        return {"n": int(v.size), "median": float(np.median(v)) if v.size else None}
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, float):
        return round(v, 6)
    return v


class Trace:
    """Append-only record list; `decide` is for branches, `event` for measurements without one."""

    def __init__(self, pair: str) -> None:
        self.pair = pair
        self._t0 = time.perf_counter()
        self.records: list[dict] = []

    def _add(self, kind: str, name: str, *, state: str, source: str, fields: dict) -> dict:
        if not name or not state:
            raise ValueError(f"trace record needs name and state: {name!r}/{state!r}")
        rec = {"t": round(time.perf_counter() - self._t0, 2), "kind": kind, "name": name,
               "state": state, "source": source, **_plain(fields)}
        self.records.append(rec)
        logger.info("trace {} | {} {} [{}] {}", self.pair, kind, name, state,
                    json.dumps({k: v for k, v in rec.items() if k not in ("t", "kind", "name", "state")},
                               ensure_ascii=False, default=str))
        return rec

    def decide(self, name: str, *, state: str, verdict: str, inputs: dict | None = None,
               thresholds: dict | None = None, source: str = "computed") -> dict:
        """A branch taken: what was compared (inputs vs thresholds) and which way it went."""
        if not verdict:
            raise ValueError(f"decision {name!r} needs a verdict")
        return self._add("decision", name, state=state, source=source,
                         fields={"inputs": inputs or {}, "thresholds": thresholds or {}, "verdict": verdict})

    def event(self, name: str, *, state: str, source: str = "computed", **fields) -> dict:
        """A measurement or a step outcome that later decisions may read."""
        return self._add("event", name, state=state, source=source, fields=fields)

    def to_list(self) -> list[dict]:
        return list(self.records)

    def save(self, out_path: Path) -> Path | None:
        """Write the trace as JSON beside the pair's outputs; a failure here must not fail the pair."""
        p = Path(out_path).parent / TRACE_DIR / f"{Path(out_path).stem}.json"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"pair": self.pair, "records": self.records},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
            return p
        except OSError as e:
            logger.warning("trace {}: not saved to {}: {}", self.pair, p, e)
            return None
