"""The conform queue -- a flat registry of EPISODE jobs and the dispatcher of their worker. Modeled on
core/pipeline.py (but simpler): the unit is an episode (1 ref + a list of dubs). State is the source
of truth, exposed to the outside ONLY through the API. Persisted to `_conform.json`; on restore, jobs
left running become queued.

The jobs themselves run in a separate process, the conform worker (conform/worker.py), which alone
touches the GPU. The queue starts it when a job begins and lets it go when no job is running, so a
daemon with nothing to do holds nothing on the card — a CUDA context dies only with its process.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Protocol

from loguru import logger
from pydantic import Field

from track_muxer.conform import procreg
from track_muxer.conform.models import PairResult
from track_muxer.conform.schema import ApiModel
from track_muxer.conform.task_params import AudioMethod, DriftSpeedPct, DRIFT_SPEED_DEFAULT

QUEUED = "queued"
PAUSED = "paused"                # waits for a manual start (the ▶ button on the queue row)
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
_TERMINAL = {DONE, FAILED, CANCELLED}
_PERSIST = "_conform.json"
DEFAULT_LIMIT = 1                # episodes run one at a time (the threshold can change on the fly)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


ConformStatus = Literal["queued", "paused", "running", "done", "failed", "cancelled"]


class PlotRef(ApiModel):
    """Reference to a PNG alignment plot (band/muq). The file lives at <out_dir>/_plots/<name>."""

    kind: Literal["track", "html", "cut"]   # the track picture, its interactive page; "cut" is met in stored jobs only
    name: str                       # png file name
    t: float | None = None          # cut time, s (for kind=cut)
    v_ms: float | None = None       # cut size, ms (for kind=cut)


class MediaInfo(ApiModel):
    """Passport of a file for display and time estimates: metadata only, the file is not read through."""

    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frames: int = 0
    has_video: bool = True


class JobOp(ApiModel):
    """One finished or running operation of a job, for showing the course of the work.

    The order of operations is fixed, so a client only needs to know which of them are done and
    how long they took. `slice_i` = 0 is the preparation of the reference, shared by the whole
    job; 1..N is the matching audio track.
    """

    slice_i: int                     # 0 = reference, otherwise the track number (1-based)
    op: str                          # operation code: decode/extract/coarse/geom/band/resample/audio/write
    sec: float = 0.0                 # actual duration; for a running operation, how long it has run so far
    state: Literal["run", "done", "failed"] = "run"


class TextTrackReport(ApiModel):
    """What became of one text track lying next to a source."""

    source: str
    file: str | None = None          # the written file; None: nothing was written
    cues_in: int = 0
    cues_out: int = 0
    dropped_drawings: int = 0
    dropped_empty: int = 0
    dropped_duplicates: int = 0
    dropped_settings: int = 0
    error: str | None = None


class DubResult(ApiModel):
    """Passport of one dub (a snapshot for the API)."""

    dub: str
    atrack: int = 0                 # audio track of the file this result belongs to
    ok: bool = False
    mode: Literal["av", "audio"] = "av"   # vision and sound, or a dub without a video stream
    skipped: bool = False           # already done (file present on disk)
    out_path: str | None = None     # path to the output audio file (format is a recording detail)
    out_size: int = 0               # output size, bytes (0 = no file / could not be read)
    assigned_pct: float = 0.0
    slope: float = 0.0
    cos_median: float = 0.0
    real_cuts: int = 0
    filled_cuts: int = 0            # excisions filled with the reference's original sound
    edge_recovered: int = 0        # frames recovered by the extra edge pass (opening/ending)
    audio_resid_ms: float = 0.0    # residual audio shift after the audio layer's pass (ms; lower is better)
    blind_zones: int = 0
    dropped_intro_s: float = 0.0
    audio_cuts: int = 0            # band/muq: number of discrete seam fixes (cuts)
    audio_max_step_ms: float = 0.0 # band/muq: the largest cut, ms
    audio_coverage: float = 0.0    # band/muq: fraction of the track with reliable anchors (0..1)
    audio_span_ms: float = 0.0     # band/muq: range of shift movement, ms
    mirror_used: bool = False      # dub frames are horizontally mirrored; matched on mirrored features
    geom_used: bool = False        # geometric correction applied (crop/zoom/anamorphic/bars) — vision was blind without it
    geom_n_in: int = 0             # geometry: inlier anchors in the consensus (registration reliability)
    geom_sx: float = 0.0           # geometry: X scale (anamorphic when sx≠sy)
    geom_sy: float = 0.0           # geometry: Y scale
    plots: list[PlotRef] = Field(default_factory=list)  # PNG alignment plots
    suspect: bool = False          # auto-flagged as suspect (cos<0.3 / assigned<90 / error)
    warnings: list[str] = Field(default_factory=list)  # "pay attention" warnings, with reasons
    critical: list[str] = Field(default_factory=list)  # red: output sync not trustworthy by the audio layer's result
    error: str | None = None
    elapsed_s: float = 0.0
    trace: list[dict] = Field(default_factory=list)  # decision trace of the pair (conform.trace)
    # Sidecar subtitles carried onto the reference timeline, one report per text track.
    text_tracks: list[TextTrackReport] = Field(default_factory=list)


class ConformJob(ApiModel):
    """One episode job. Serialized to JSON (API + persistence)."""

    id: str
    seq: int = 0
    label: str = ""
    ref: str
    dubs: list[str]
    out_dir: str
    cache_dir: str | None = None       # cache subdirectory (<episode>/_conform_cache)
    keep_tmp: bool = False             # tmp checkpoints: keep ALL intermediates (reference SRM + dub
                                       # CK1/2/3) for a rerun without GPU decode; OFF clears the cache after the episode
    # Task parameters: a choice about the method or the content of the result, never a fix for a track.
    audio_method: AudioMethod = AudioMethod.BAND    # meter of the audio layer
    drift_speed_pct: DriftSpeedPct = DRIFT_SPEED_DEFAULT   # how fast the laid sound may follow a drift, % per second
    fill_silence: bool = True          # fill the silence left in the dub with the reference sound
    ref_atrack: int = 0                # audio track of the reference (the sound standard for band/fill)
    dub_atracks: list[int] | None = None   # audio track of each dub (parallel to dubs; None means all 0)
    autostart: bool = True             # False: the job comes up PAUSED and waits for a manual start
    # state
    status: ConformStatus = QUEUED
    progress: float = 0.0          # 0..1 overall
    stage: str = ""               # name of the running phase: free text, for display only
    stage_pct: float = 0.0         # 0..1 inside the current phase (for the panel)
    detail: str = ""               # live detail of the phase (e.g. "12000 frames, 1300 fps")
    dub_index: int = 0
    dub_total: int = 0
    cur_dub: str = ""
    ref_info: MediaInfo | None = None                # reference passport (for the header and the estimate)
    dub_infos: list[MediaInfo] = Field(default_factory=list)   # source file passports
    ops: list[JobOp] = Field(default_factory=list)   # operation log: what ran and how long it took
    results: list[DubResult] = Field(default_factory=list)
    error: str | None = None
    elapsed_s: float = 0.0
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _dir_size(p: Path) -> int:
    try:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except OSError:
        return 0


def _purge_tmp(job: ConformJob) -> int:
    """Remove the temporary files of a finished job. Returns: bytes freed.

    Removes: the episode's cache (CK1/CK2/CK3 checkpoints, gigabytes) and this job's own `_tmp`
    directories next to the sources (memmap decode layouts). NEVER touched: the job's output
    directory with the audio files and `_plots`. On cancel, `conform_episode`'s regular cleanup does
    not run (the episode was interrupted), so this pass replaces it.
    """
    out = Path(job.out_dir)
    targets: list[Path] = []
    if job.cache_dir:
        targets.append(Path(job.cache_dir))
    targets.append(Path(job.ref).parent / "_conform_cache")     # default episode cache layout
    for src in [job.ref, *job.dubs]:
        targets.append(Path(src).parent / "_tmp")

    freed = 0
    seen: set[str] = set()
    for t in targets:
        key = str(t.resolve()) if t.exists() else str(t)
        if key in seen:
            continue
        seen.add(key)
        if not t.is_dir():
            continue
        if _is_inside(out, t) or t.resolve() == out.resolve():   # protect the results
            logger.warning("conform purge: пропуск {} — внутри выходного каталога", t)
            continue
        size = _dir_size(t)
        shutil.rmtree(t, ignore_errors=True)
        if t.exists():
            logger.warning("conform purge: каталог не удалён полностью: {}", t)
        freed += size - _dir_size(t)
    return freed


def suspect(r: PairResult) -> bool:
    if r.skipped:                      # a job already done was not recomputed — don't flag it as suspect
        return False
    if getattr(r, "mode", "av") == "audio":
        # audio-only: vision fields (cos/assigned) don't exist by design — judge by hearing:
        # anchor coverage and the audio layer's residual shift (the main criterion is ±80 ms).
        return (not r.ok) or r.audio_coverage < 0.5 or abs(r.audio_resid_ms) > 80.0
    return (not r.ok) or r.cos_median < 0.30 or r.assigned_pct < 90.0


class Feed(Protocol):
    """Where the queue tells what happens to its jobs. The application supplies it; the queue
    calls sync() wherever it persists the jobs and progress() while a job runs."""

    def sync(self, items: list[dict]) -> None: ...

    def progress(self, item_id: str, fields: dict) -> None: ...


class _NoFeed:
    """Nobody listens: the queue used on its own."""

    def sync(self, items: list[dict]) -> None:
        pass

    def progress(self, item_id: str, fields: dict) -> None:
        pass


class _Worker:
    """The daemon's end of one conform worker process (conform/worker.py).

    A reader thread takes the worker's events off the pipe and hands them to the queue; when the pipe
    ends, the worker is gone, and the queue hears it as an event too. The worker's live subprocesses
    are kept on record here with their start marks: a worker that dies leaves them running in their
    own sessions, and this record is the only way left to stop them."""

    # Guards, not synchronisation. A worker told to retire has no job left and exits in seconds
    # (the CUDA teardown); one still alive this long after the retire is stuck, and is killed.
    RETIRE_GRACE_S = 60.0
    # Once its pipe has ended a worker is exiting anyway; this long is how long its exit code is
    # awaited before the process is killed and reported with whatever code it has.
    GONE_GRACE_S = 5.0

    def __init__(self, on_event: Callable, on_gone: Callable) -> None:
        import multiprocessing as mp

        from track_muxer.conform import worker

        ctx = mp.get_context("spawn")          # never fork a threaded daemon
        self._conn, child = ctx.Pipe(duplex=True)
        self.proc = ctx.Process(target=worker.serve, args=(child,), name="conform-worker", daemon=True)
        self.proc.start()
        child.close()                           # only the worker holds it now: its exit ends the pipe
        self.retired = False
        self.jobs: set[str] = set()             # runs sent here that have not reported an outcome
        self.procs: dict[int, int | None] = {}  # live subprocess pid -> its start mark
        # Set by the reader once it has waited the process out. Only the reader waits on the process:
        # two threads reaping one child race for its status, and the loser reads no exit code.
        self._gone = threading.Event()
        self._ended = threading.Event()         # the pipe has ended: the worker is dead or leaving
        self._send_lock = threading.Lock()
        threading.Thread(target=self._read, args=(on_event, on_gone), daemon=True,
                         name="conform-link").start()

    def send(self, msg: tuple) -> None:
        with self._send_lock:
            try:
                self._conn.send(msg)
            except (OSError, EOFError):         # a dead worker is reported by the reader, not here
                pass

    def alive(self) -> bool:
        """Whether runs may still go here. Judged by the pipe, not by asking the process: asking reaps
        a child from this thread, and the reader then finds no exit status to report."""
        return not self._ended.is_set()

    def retire(self) -> None:
        """Tell the worker to leave. The guard counts from here, not from the end of the pipe: a
        worker that hangs before it closes the pipe would otherwise never be caught."""
        self.retired = True
        self.send(("retire",))
        threading.Thread(target=self._guard_retire, daemon=True, name="conform-retire").start()

    def _guard_retire(self) -> None:
        if not self._gone.wait(self.RETIRE_GRACE_S):
            logger.error("conform: исполнитель pid {} не вышел за {:.0f} с после retire — убит",
                         self.proc.pid, self.RETIRE_GRACE_S)
            self.proc.kill()

    def kill(self) -> None:
        if self.proc.is_alive():
            self.proc.kill()

    def kill_orphans(self) -> int:
        """Kill the subprocesses the worker left behind; a pid since given to another process is spared."""
        n = sum(procreg.kill_pid(pid, mark) for pid, mark in list(self.procs.items()))
        self.procs.clear()
        return n

    def _read(self, on_event: Callable, on_gone: Callable) -> None:
        from track_muxer.conform.worker import EVENTS

        while True:
            try:
                msg = self._conn.recv()
            except (EOFError, OSError):
                self._ended.set()
                break
            if not isinstance(msg, tuple) or not msg or msg[0] not in EVENTS:
                logger.error("conform: неизвестное событие исполнителя {!r}", msg)   # the catalogue is the contract
                continue
            if msg[0] == "proc":
                _, pid, alive, mark = msg
                if alive:
                    self.procs[pid] = mark
                else:
                    self.procs.pop(pid, None)
                continue
            # A handler that fails must not end this thread: the pipe would go unread, the worker
            # would block on a full pipe holding the card, and no run would ever report an outcome.
            try:
                on_event(self, msg)
            except Exception:  # noqa: BLE001
                logger.exception("conform: событие исполнителя {!r} не обработано", msg[0])
        if not self.retired:
            self.proc.join(self.GONE_GRACE_S)
        if self.proc.is_alive() and not self.retired:
            logger.error("conform: исполнитель pid {} закрыл канал, но не вышел — убит", self.proc.pid)
            self.proc.kill()
        self.proc.join()                        # after a retire the guard thread bounds this wait
        self._gone.set()
        try:
            on_gone(self, self.proc.exitcode)   # None only if another reaper took the status first
        except Exception:  # noqa: BLE001
            logger.exception("conform: уход исполнителя pid {} не обработан", self.proc.pid)


def _code(exitcode: int | None) -> str:
    return "код не получен" if exitcode is None else f"код {exitcode}"


class _Clock:
    """When a running job started, and when its current operation began (daemon time)."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.op_t0 = self.t0


class ConformQueue:
    """Registry of episode jobs plus the dispatcher of their worker. Thread-safe (RLock).
    Concurrency is capped by the `_limit` threshold (changeable on the fly)."""

    def __init__(self, output_dir: Path, limit: int = DEFAULT_LIMIT,
                 feed: Feed | None = None, spawn: Callable | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.feed: Feed = feed if feed is not None else _NoFeed()
        self.path = self.output_dir / _PERSIST
        self._lock = threading.RLock()
        self._items: dict[str, ConformJob] = {}
        # A run is one attempt of a job in the worker. Cancel-then-retry puts two runs of one job in
        # flight; events are applied to the job only when they come from its current run.
        self._runs = itertools.count(1)
        self._run_of: dict[str, str] = {}       # job id -> its current run id
        self._job_of: dict[str, str] = {}       # run id -> job id, while the run is in flight
        self._clocks: dict[str, _Clock] = {}    # run id -> its clock
        self._limit = limit
        self._active = 0
        self._counter = itertools.count(1)
        self._spawn = spawn or _Worker          # a test hands in a worker of its own
        self._worker = None                     # the live worker; None while no job runs

    # -- public API --

    def enqueue(self, spec: dict) -> ConformJob:
        with self._lock:
            n = next(self._counter)
            job = ConformJob(id=f"cf{n:05d}", seq=n, **spec)
            job.dub_total = len(job.dubs)
            if not job.autostart:
                job.status = PAUSED
            self._items[job.id] = job
            self._persist_locked()
            self._pump_locked()
            return job.model_copy(deep=True)

    def list_items(self) -> list[ConformJob]:
        with self._lock:
            return [j.model_copy(deep=True) for j in self._items.values()]

    def get_item(self, jid: str) -> ConformJob | None:
        with self._lock:
            j = self._items.get(jid)
            return j.model_copy(deep=True) if j else None

    def cancel(self, jid: str) -> bool:
        """Cancel a job. For RUNNING, this is RELIABLE: the worker raises the job's stop flag
        (checked between stages) and kills its subprocesses at once, otherwise a long ffmpeg decode
        would run to completion. For a terminal job, this removes its record."""
        with self._lock:
            j = self._items.get(jid)
            if j is None:
                return False
            if j.status in _TERMINAL:
                self._items.pop(jid, None)
            else:
                rid = self._run_of.get(jid)
                if j.status == RUNNING and rid is not None and self._worker is not None:
                    self._worker.send(("stop", rid))
                self._set(j, CANCELLED)
            self._persist_locked()
        return True

    def pause(self, jid: str) -> bool:
        """QUEUED -> PAUSED (the job waits for a manual start). RUNNING is left untouched -- for that, use cancel."""
        with self._lock:
            j = self._items.get(jid)
            if j is None or j.status != QUEUED:
                return False
            self._set(j, PAUSED)
            self._persist_locked()
        return True

    def start(self, jid: str) -> bool:
        """PAUSED -> QUEUED, then an immediate attempt to start (the ▶ button on the queue row)."""
        with self._lock:
            j = self._items.get(jid)
            if j is None or j.status != PAUSED:
                return False
            self._set(j, QUEUED)
            self._persist_locked()
            self._pump_locked()
        return True

    def shutdown(self) -> int:
        """Application shutdown: stop ALL active jobs, the worker and its subprocesses. Without this,
        closing the window leaves orphaned ffmpeg processes that keep writing files. Returns: how
        many processes were killed."""
        with self._lock:
            for j in self._items.values():
                if j.status == RUNNING:
                    self._set(j, CANCELLED)
            self._persist_locked()
            w, self._worker = self._worker, None
        if w is None:
            return 0
        w.kill()
        killed = 1 + w.kill_orphans()
        logger.info("conform shutdown: убито процессов: {}", killed)
        return killed

    def clear_done(self) -> dict:
        """Remove FINISHED jobs from the queue AND delete their temporary files. Output audio files
        and plots (_plots) are NEVER touched. Active jobs are left untouched."""
        with self._lock:
            gone = [j.model_copy(deep=True) for j in self._items.values() if j.status in _TERMINAL]
            for j in gone:
                self._items.pop(j.id, None)
            self._persist_locked()
        freed = 0
        for j in gone:
            freed += _purge_tmp(j)
        return {"cleared": len(gone), "freed_mb": round(freed / (1 << 20), 1)}

    def retry(self, jid: str) -> bool:
        with self._lock:
            j = self._items.get(jid)
            if j is None or j.status not in _TERMINAL:
                return False
            j.error = None
            j.results = []
            j.ops = []
            j.progress = 0.0
            j.stage = ""
            j.dub_index = 0
            j.cur_dub = ""
            j.elapsed_s = 0.0
            self._set(j, QUEUED)
            self._persist_locked()
            self._pump_locked()
        return True

    def clear(self) -> int:
        """Stop ALL jobs and delete ALL queue records. Returns: how many were removed. Running jobs
        are stopped in the worker; their outcomes find an empty registry and are simply dropped."""
        with self._lock:
            n = len(self._items)
            if self._worker is not None:
                for j in self._items.values():
                    rid = self._run_of.get(j.id)
                    if j.status == RUNNING and rid is not None:
                        self._worker.send(("stop", rid))
            self._items.clear()
            self._persist_locked()
        return n

    def get_settings(self) -> dict:
        with self._lock:
            return {"limit": self._limit, "active": self._active}

    def set_settings(self, limit: int | None = None) -> dict:
        with self._lock:
            if limit is not None:
                self._limit = max(0, int(limit))
            self._persist_locked()
            self._pump_locked()
            return {"limit": self._limit, "active": self._active}

    # -- dispatcher --

    def _pump_locked(self) -> None:
        while self._active < self._limit:
            nxt = self._next(QUEUED)
            if nxt is None:
                break
            self._set(nxt, RUNNING)
            nxt.progress = 0.0
            self._active += 1
            if self._worker is None or not self._worker.alive():
                # A worker that died has not been reported yet: its own end fails its runs; a new
                # run goes to a new worker rather than into a dead pipe.
                self._worker = self._spawn(self._on_event, self._on_gone)
            rid = f"{nxt.id}.{next(self._runs)}"
            self._run_of[nxt.id] = rid
            self._job_of[rid] = nxt.id
            self._clocks[rid] = _Clock()
            self._worker.jobs.add(rid)
            self._worker.send(("run", rid, nxt.model_dump(mode="json")))
        self._publish_locked()      # a start is a change of state even when nothing is written to disk

    def _retire_if_idle_locked(self) -> None:
        """No job is running: the worker leaves, and the GPU goes back to the driver with it. Decided
        under the same lock that starts jobs, so a `run` never reaches a worker on its way out."""
        if self._active == 0 and self._worker is not None:
            w, self._worker = self._worker, None
            w.retire()

    def _publish_locked(self) -> None:
        self.feed.sync([j.model_dump(mode="json") for j in self._items.values()])

    def _next(self, status: str) -> ConformJob | None:
        c = [j for j in self._items.values() if j.status == status]
        c.sort(key=lambda j: j.seq)
        return c[0] if c else None

    def _set(self, job: ConformJob, status: str) -> None:
        job.status = status
        job.updated_at = _now()

    # -- the worker's events (from its reader thread; state is touched only under _lock) --

    def _current(self, rid: str) -> tuple[str | None, ConformJob | None]:
        """The job a run belongs to, and the job itself only while this run is its current one."""
        jid = self._job_of.get(rid)
        if jid is None or self._run_of.get(jid) != rid:
            return jid, None
        return jid, self._items.get(jid)

    def _on_event(self, w, msg: tuple) -> None:
        kind, rid = msg[0], msg[1]
        if kind == "progress":
            self._on_progress(rid, msg[2])
        elif kind == "pair":
            self._on_pair(rid, msg[2])
        elif kind == "infos":
            with self._lock:
                _, jj = self._current(rid)
                if jj is not None:
                    try:
                        infos = msg[2]
                        jj.ref_info = MediaInfo(**infos[0]) if infos[0] else None
                        jj.dub_infos = [MediaInfo(**i) if i else MediaInfo() for i in infos[1:]]
                    except Exception as e:  # noqa: BLE001 — display only, it must not cost the run
                        logger.warning("conform {}: паспорта файлов не приняты: {}", jj.id, e)
        elif kind == "finished":
            self._on_finished(w, rid, msg[2], msg[3])

    def _op_mark(self, jj: ConformJob, clock: _Clock, slice_i: int, op: str) -> None:
        """Mark which operation is running now, and close out the previous one's timing.

        The panel shows the name, state, and duration of EVERY operation; there is no way to
        produce them after the fact, so they are recorded as they happen, at each transition.
        """
        if not op:
            return
        now = time.perf_counter()
        last = jj.ops[-1] if jj.ops else None
        if last is not None and last.slice_i == slice_i and last.op == op:
            last.sec = now - clock.op_t0          # same operation — update how long it has run
            return
        if last is not None:
            last.sec = now - clock.op_t0
            if last.state == "run":
                last.state = "done"
        clock.op_t0 = now
        jj.ops.append(JobOp(slice_i=slice_i, op=op))

    def _on_progress(self, rid: str, p: dict) -> None:
        with self._lock:
            jid, jj = self._current(rid)
            clock = self._clocks.get(rid)
            if jj is None or clock is None:
                return
            stage, pct, dub_index = p["stage"], p["pct"], p["dub_index"]
            self._op_mark(jj, clock, dub_index, stage)
            jj.stage = stage
            jj.stage_pct = pct
            jj.detail = p["detail"]
            jj.dub_index = dub_index
            jj.dub_total = p["dub_total"]
            jj.cur_dub = p["dub_name"]
            # The bar = (reference + dub_total) EQUAL shares of 1/(N+1). Share 0 is the reference;
            # shares 1..N are the dubs. Inside a share, the stage ranges are ORDERED by EXECUTION
            # ORDER (monotonic, no gaps, no steps back). Per stage: intra = lo + (hi-lo)*pct; every
            # stage emits pct smoothly (backends are instrumented per loop iteration). geom sits
            # between coarse and band (when vision went blind; otherwise the range is skipped FORWARD).
            slices = jj.dub_total + 1
            if dub_index == 0:                    # reference: video decode + audio extraction
                lo, hi = {"decode": (0.0, 0.85), "extract": (0.85, 1.0)}.get(stage, (0.0, 0.85))
            else:                                 # dub: stages in order (decode dominates)
                lo, hi = {"decode":   (0.00, 0.52),   # dub SRM (per frame)
                          "coarse":   (0.52, 0.55),   # coarse pass
                          "geom":     (0.55, 0.62),   # geometry analysis (crop/zoom/bars)
                          "band":     (0.62, 0.78),   # Drop-DTW band (per chunk)
                          "extract":  (0.78, 0.88),   # dub audio decode (per frame)
                          "resample": (0.88, 0.93),   # resample/warp (per block)
                          "audio":    (0.93, 0.97),   # audio layer's pass
                          "write":    (0.97, 1.00)}.get(stage, (0.62, 0.78))
            intra = lo + (hi - lo) * min(1.0, max(0.0, pct))
            jj.progress = min(0.999, (dub_index + intra) / slices)
            jj.elapsed_s = time.perf_counter() - clock.t0   # grows as it runs, not only at the end
            jj.updated_at = _now()
            self.feed.progress(jid, {
                "status": jj.status, "progress": jj.progress, "stage": jj.stage,
                "stage_pct": jj.stage_pct, "detail": jj.detail, "dub_index": jj.dub_index,
                "dub_total": jj.dub_total, "cur_dub": jj.cur_dub, "elapsed_s": jj.elapsed_s,
                "ops": [op.model_dump(mode="json") for op in jj.ops]})

    def _on_pair(self, rid: str, result: dict) -> None:
        with self._lock:
            _, jj = self._current(rid)
            clock = self._clocks.get(rid)
            if jj is None:
                return
            res = DubResult(**result)
            if jj.ops and clock is not None:      # the track finished — close its last operation
                jj.ops[-1].sec = time.perf_counter() - clock.op_t0
                jj.ops[-1].state = "done" if res.ok else "failed"
                clock.op_t0 = time.perf_counter()
            jj.results.append(res)
            # +1 share for the reference (slices = dub_total + 1): after k finished dubs -> (k+1)/(N+1)
            jj.progress = min(0.999, (len(jj.results) + 1) / (jj.dub_total + 1))
            jj.updated_at = _now()
            self._persist_locked()

    def _end_run_locked(self, rid: str) -> tuple[_Clock | None, ConformJob | None]:
        """Forget a run that has ended. -> its clock, and its job if the run was the job's current one."""
        _, jj = self._current(rid)
        jid = self._job_of.pop(rid, None)
        if jid is not None and self._run_of.get(jid) == rid:
            del self._run_of[jid]
        return self._clocks.pop(rid, None), jj

    def _on_finished(self, w, rid: str, status: str, err: str | None) -> None:
        with self._lock:
            w.jobs.discard(rid)
            clock, jj = self._end_run_locked(rid)
            if jj is not None and jj.status == RUNNING:   # don't overwrite CANCELLED
                jj.error = err
                if clock is not None:
                    jj.elapsed_s = time.perf_counter() - clock.t0
                self._set(jj, DONE if status == DONE else FAILED)
                if status == DONE:
                    jj.progress = 1.0
            self._active = max(0, self._active - 1)
            self._persist_locked()
            self._pump_locked()
            self._retire_if_idle_locked()

    def _on_gone(self, w, exitcode: int | None) -> None:
        """The worker's process ended. After a retire that is the plan; with jobs still on it, it
        died, and each of them fails with the reason instead of hanging as RUNNING."""
        orphans = w.kill_orphans()
        with self._lock:
            lost = sorted(w.jobs)
            w.jobs.clear()
            for rid in lost:
                _, jj = self._end_run_locked(rid)
                if jj is not None and jj.status == RUNNING:
                    jj.error = f"процесс conform завершился аварийно ({_code(exitcode)})"
                    self._set(jj, FAILED)
            self._active = max(0, self._active - len(lost))
            if self._worker is w:
                self._worker = None
            if lost:
                self._persist_locked()
                self._pump_locked()
                self._retire_if_idle_locked()
        if lost:
            logger.error("conform: исполнитель pid {} погиб ({}), задания провалены: {}; "
                         "убито осиротевших процессов: {}", w.proc.pid, _code(exitcode), ", ".join(lost), orphans)
        else:
            logger.info("conform: исполнитель pid {} завершён ({}) — видеопамять возвращена "
                        "драйверу{}", w.proc.pid, _code(exitcode),
                        f"; убито осиротевших процессов: {orphans}" if orphans else "")

    # -- persist / restore --

    def _persist_locked(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            payload = {"items": [j.model_dump(mode="json") for j in self._items.values()],
                       "limit": self._limit, "updated_at": _now()}
            self.feed.sync(payload["items"])
            fd, tmp = tempfile.mkstemp(prefix="_conform.", suffix=".tmp", dir=str(self.output_dir))
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception as e:  # noqa: BLE001
            logger.warning("conform persist failed: {}", e)

    def restore(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning("conform restore failed: {}", e)
            return
        with self._lock:
            self._limit = int(data.get("limit", self._limit))
            maxseq = 0
            for d in data.get("items") or []:
                try:
                    # A job stored with the former pair of switches keeps the user's choice of meter.
                    if "audio_method" not in d and d.get("audio_muq"):
                        d["audio_method"] = "muq"
                    job = ConformJob(**d)
                except Exception:  # noqa: BLE001
                    continue
                if job.status == RUNNING:              # cut short by a restart -> replay it
                    job.status = QUEUED
                    job.progress = 0.0
                    job.stage = ""
                    job.dub_index = 0
                    job.cur_dub = ""
                    job.results = []
                    job.ops = []
                self._items[job.id] = job
                maxseq = max(maxseq, job.seq)
            self._counter = itertools.count(maxseq + 1)
            self._pump_locked()
