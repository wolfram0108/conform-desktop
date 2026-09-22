"""Conform dataclasses: features, results, progress. JSON-friendly (for the API)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SrmFeatures:
    """SRM vectors of every frame of one video, plus its effective fps.

    `pts` is the frame time axis (seconds, from the first frame's zero), filled in ONLY for
    genuine VFR (features.vfr_time_axis): there, a frame's time is pts[i], not index/fps. None
    (CFR, the vast majority) means every consumer takes the index/fps path, bit for bit. The
    algorithm decides from the data (the residual of timestamps against a uniform grid) -- this is
    not a switch."""

    srm: np.ndarray            # (N, 18432) float16
    fps: float
    src: Path | None = None
    pts: np.ndarray | None = None   # (N,) float64, seconds; VFR only

    @property
    def n_frames(self) -> int:
        return int(len(self.srm))

    @property
    def duration_s(self) -> float:
        if self.pts is not None and len(self.pts):
            return float(self.pts[-1]) + (1.0 / self.fps if self.fps else 0.0)
        return len(self.srm) / self.fps if self.fps else 0.0


@dataclass
class Progress:
    """A progress snapshot for the callback (the daemon turns it into a % for the web panel)."""

    stage: str                 # decode | coarse | geom | band | extract | resample | audio | write
    pct: float                 # 0..1 within the stage
    detail: str = ""
    dub_index: int = 0
    dub_total: int = 0
    dub_name: str = ""


@dataclass
class PairResult:
    """Quality passport of one ref<->dub pair (printed to the log; not yet written to files)."""

    dub: str
    out_path: Path | None       # path to the output audio file (format is a recording detail; None = not created)
    ok: bool
    atrack: int = 0                    # audio track of the dub file this result belongs to
    mode: str = "av"                   # "av" = vision+audio (a video dub) | "audio" = audio only
                                       # (a dub without a video stream: only the audio layer runs; the
                                       # vision fields assigned_pct/cos_median/slope are meaningless here)
    skipped: bool = False              # already done (file present on disk): not recomputed
    fps_ref: float = 0.0
    fps_dub: float = 0.0
    n_frames: int = 0
    duration_s: float = 0.0
    assigned_pct: float = 0.0          # % of dub frames that found a place on REF
    slope: float = 0.0                 # offset slope (ref/dub), ~fps_ref/fps_dub
    cos_median: float = 0.0            # median cosine of the assigned pairs
    monotonic_violations: int = 0
    real_cuts: list[tuple[float, float]] = field(default_factory=list)  # (t1,t2) seconds on REF
    real_cuts_s: float = 0.0           # total duration of excisions (silence/fill), seconds
    filled_cuts: int = 0               # seconds of dub silence filled from the synced ref (fill_silence, after band/muq)
    blind_zones: int = 0               # drop zones identified as blind (not cut)
    blind_restored: int = 0            # frames restored into the map (R_ins)
    edge_recovered: int = 0            # frames restored by the extra edge pass (opening/ending)
    mirror_used: bool = False          # dub frames are horizontally mirrored; matched on mirrored SRM features
    geom_used: bool = False            # geometry recovery fired (crop/zoom/anamorphic/bars): vision would be blind without it
    geom_n_in: int = 0                 # geometry: inlier anchors in the consensus (registration reliability)
    geom_sx: float = 0.0               # geometry: X scale (anamorphic when sx != sy)
    geom_sy: float = 0.0               # geometry: Y scale
    telecine_ref: bool = False         # ref is baked-in 3:2 telecine (an NTSC rip without IVTC): the cadence is thinned
    telecine_dub: bool = False         # dub is baked-in 3:2 telecine: the cadence is thinned
    tele_score: float = 0.0            # strength of the telecine signature (period-5 peak; >0.05 = telecine)
    tc_dropped: int = 0                # SRM frames dropped by cadence thinning (soft IVTC)
    audio_resid_ms: float = 0.0        # residual audio shift after refinement (median |.|, ms; smaller is better; 0 = layer skipped)
    dropped_intro_s: float = 0.0       # how many seconds of the dub's start were dropped (free_start)
    audio_cuts: int = 0                # band/muq: number of discrete seam corrections (cuts) found
    audio_max_step_ms: float = 0.0     # band/muq: largest cut, ms
    audio_coverage: float = 0.0        # band/muq: fraction of the track with reliable anchors (0..1)
    audio_span_ms: float = 0.0         # band/muq: full range of shift movement, ms
    plots: list[dict] = field(default_factory=list)  # alignment PNG plots: [{kind,name,t,v_ms}]
    warnings: list[str] = field(default_factory=list)
    critical: list[str] = field(default_factory=list)  # red: output sync not trustworthy by the audio layer's result
    error: str | None = None
    elapsed_s: float = 0.0
    trace: list[dict] = field(default_factory=list)  # decision trace (conform.trace): every branch with its inputs
    # Sidecar subtitles carried onto the reference timeline: [{source, file, cues_in, cues_out, dropped_drawings, dropped_empty, dropped_duplicates, dropped_settings, error}].
    text_tracks: list[dict] = field(default_factory=list)


@dataclass
class EpisodeResult:
    """Result of an episode: the ref plus a list of PairResult."""

    ref: str
    out_dir: Path
    pairs: list[PairResult] = field(default_factory=list)
    elapsed_s: float = 0.0
    # The reference's own sidecar subtitles, moved onto its first-frame axis.
    ref_text_tracks: list[dict] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return sum(1 for p in self.pairs if p.ok)

    @property
    def n_fail(self) -> int:
        return sum(1 for p in self.pairs if not p.ok)
