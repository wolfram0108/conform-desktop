"""Stage-scoped progress reporting.

One Reporter per stage; `sub(lo, hi)` carves a range for a sub-step, so every
inner loop advances the same 0..1 of its stage instead of the bar sitting still
between hand-placed landmarks. Emission is rate-limited (step and interval) so hot
loops may report each iteration; landmarks (`mark`) and stage ends always pass.
"""

from __future__ import annotations

import time

from track_muxer.conform.models import Progress

MIN_STEP = 0.005          # do not report changes finer than half a percent
MIN_INTERVAL_S = 0.25     # consumers poll about once a second; more is noise


class Reporter:
    """Callable `(frac, detail=None)` mapping a sub-step's 0..1 into its stage range."""

    __slots__ = ("_sink", "stage", "_meta", "_lo", "_hi", "_detail", "_shared")

    def __init__(self, sink, stage: str, meta=(0, 0, ""), lo: float = 0.0, hi: float = 1.0,
                 detail: str = "", _shared: dict | None = None) -> None:
        self._sink = sink
        self.stage = stage
        self._meta = tuple(meta)
        self._lo = float(lo)
        self._hi = float(hi)
        self._detail = detail
        # Shared across sub-ranges: throttling and monotonicity belong to the stage, not the step.
        self._shared = _shared if _shared is not None else {"pct": -1.0, "t": 0.0}

    @classmethod
    def of(cls, sink, stage: str, meta=(0, 0, ""), detail: str = "") -> "Reporter | None":
        """None sink → None, keeping the `if on_prog is not None` idiom of the kernel hooks."""
        return None if sink is None else cls(sink, stage, meta, detail=detail)

    def sub(self, lo: float, hi: float, detail: str | None = None) -> "Reporter":
        span = self._hi - self._lo
        return Reporter(self._sink, self.stage, self._meta, self._lo + span * lo, self._lo + span * hi,
                        self._detail if detail is None else detail, self._shared)

    def _pct(self, frac: float) -> float:
        return self._lo + (self._hi - self._lo) * min(1.0, max(0.0, float(frac)))

    def _emit(self, pct: float, detail: str | None) -> None:
        s = self._shared
        pct = max(pct, s["pct"])                 # a stage never moves backwards
        if detail is not None:
            self._detail = detail
        s["pct"] = pct
        s["t"] = time.perf_counter()
        self._sink(Progress(self.stage, pct, self._detail, *self._meta))

    def mark(self, frac: float, detail: str | None = None) -> None:
        """Landmark: start of a sub-step. Always emitted so the shown detail is never stale."""
        self._emit(self._pct(frac), detail)

    def __call__(self, frac: float, detail: str | None = None) -> None:
        pct = self._pct(frac)
        s = self._shared
        if pct >= 1.0 and s["pct"] < 1.0:        # stage end always passes
            self._emit(pct, detail)
            return
        if pct - s["pct"] < MIN_STEP or time.perf_counter() - s["t"] < MIN_INTERVAL_S:
            return
        self._emit(pct, detail)


def part(rep: Reporter | None, lo: float, hi: float, detail: str | None = None) -> Reporter | None:
    """Sub-range of an optional reporter: None stays None, so hooks pass through untouched."""
    return None if rep is None else rep.sub(lo, hi, detail)
