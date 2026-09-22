"""The conform worker: a process of its own that runs conform jobs and alone holds the GPU.

Why a separate process: a CUDA context lives exactly as long as the process that created it — torch
has no call that tears it down. If jobs ran inside the daemon, the daemon would sit on the card for
as long as it lives, together with every model it ever loaded. So the daemon never touches the GPU;
the queue starts this process when a job begins and tells it to leave when no job is running, and
the card is handed back whole when the process exits.

The daemon and the worker talk over one pipe, in messages (the catalogue is `COMMANDS` and
`EVENTS`). The queue owns the commands, the worker owns the events. Delivery is once and in order,
for the life of the worker; the state of the jobs is kept by the daemon, not here. Messages name a
RUN, not a job: a job cancelled and retried at once has two runs in flight, and the outcome of the
old one must not land on the new.

The worker is started with `spawn`. An application frozen into an executable must call
`multiprocessing.freeze_support()` first thing in its entry point, or the child re-runs the
application instead of this module.
"""

from __future__ import annotations

import dataclasses
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

# The pipe's vocabulary. Commands go daemon -> worker, events go worker -> daemon.
COMMANDS = {
    "run": "(\"run\", run_id, job: dict) — start a run of a job; `job` is a ConformJob dumped to JSON",
    "stop": "(\"stop\", run_id) — cancel it: the stop flag is raised and its subprocesses die at once",
    "retire": "(\"retire\",) — no job is running: finish and exit, handing the GPU back",
}
EVENTS = {
    "infos": "(\"infos\", run_id, [media passport | None]) — reference first, then each dub",
    "progress": "(\"progress\", run_id, Progress as dict)",
    "pair": "(\"pair\", run_id, DubResult as dict) — one dub finished, well or not",
    "finished": "(\"finished\", run_id, \"done\" | \"failed\", error | None) — the run's outcome",
    "proc": "(\"proc\", pid, alive, start_mark) — a subprocess started or finished",
}

_MAX_JOBS = 8          # threads for jobs; how many run at once is the queue's limit, not this


class _Pipe:
    """One end of the pipe shared by every job thread: a send is not atomic across threads."""

    def __init__(self, conn) -> None:
        self._conn = conn
        self._lock = threading.Lock()

    def send(self, msg: tuple) -> None:
        with self._lock:
            try:
                self._conn.send(msg)
            except (OSError, EOFError):      # the daemon is gone: nobody is left to tell
                pass


def dub_result(res) -> dict:
    """PairResult -> the DubResult the API shows, as a dict ready for the pipe."""
    from track_muxer.conform.queue import DubResult, PlotRef, suspect

    try:
        out_size = res.out_path.stat().st_size if res.out_path else 0
    except OSError:
        out_size = 0
    return DubResult(
        dub=res.dub, atrack=res.atrack, ok=res.ok, mode=getattr(res, "mode", "av"), skipped=res.skipped,
        out_path=(str(res.out_path) if res.out_path else None), out_size=out_size,
        assigned_pct=res.assigned_pct, slope=res.slope, cos_median=res.cos_median,
        real_cuts=len(res.real_cuts), filled_cuts=res.filled_cuts,
        edge_recovered=res.edge_recovered, audio_resid_ms=res.audio_resid_ms,
        blind_zones=res.blind_zones, dropped_intro_s=res.dropped_intro_s,
        audio_cuts=res.audio_cuts, audio_max_step_ms=res.audio_max_step_ms,
        audio_coverage=res.audio_coverage, audio_span_ms=res.audio_span_ms,
        mirror_used=res.mirror_used, geom_used=res.geom_used, geom_n_in=res.geom_n_in,
        geom_sx=res.geom_sx, geom_sy=res.geom_sy,
        plots=[PlotRef(**p) for p in res.plots],
        suspect=suspect(res),
        warnings=res.warnings, critical=res.critical,
        error=res.error, elapsed_s=res.elapsed_s, trace=list(res.trace),
        text_tracks=list(res.text_tracks),
    ).model_dump(mode="json")


def _job(rid: str, job: dict, stop: threading.Event, group, pipe: _Pipe) -> None:
    """One run of a job, start to outcome. Every exit path ends in exactly one `finished`: all that
    can raise — the imports and the job's own fields included — sits inside the `try`."""
    status, err = "done", None
    jid = rid
    try:
        from track_muxer.conform import features, gpu, procreg
        from track_muxer.conform.episode import conform_episode
        from track_muxer.conform.queue import ConformJob

        spec = ConformJob(**job)
        jid = spec.id
        procreg.bind(group)               # thread-local: the job's subprocesses land in its group
        logger.info("conform {} старт: реф={} озвучек={} выход={}",
                    jid, Path(spec.ref).name, len(spec.dubs), spec.out_dir)
        # Passports come from metadata before the work starts: the panel shows what is being worked
        # on and estimates the time left.
        try:
            pipe.send(("infos", rid, [features.probe_media_info(Path(p)) for p in [spec.ref, *spec.dubs]]))
        except Exception as e:  # noqa: BLE001 — display only, it must not hold the work up
            logger.warning("conform {}: не удалось прочитать паспорта файлов: {}", jid, e)

        def on_pair(res) -> None:
            if res.ok:
                logger.info("conform {} озвучка {} готова: назначено {:.1f}%, остаток {:.1f} мс, "
                            "покрытие {:.3f}, выход {}", jid, res.dub, res.assigned_pct,
                            res.audio_resid_ms, res.audio_coverage, res.out_path)
            else:
                logger.error("conform {} озвучка {} НЕ удалась: {}", jid, res.dub,
                             res.error or "без сообщения")
            pipe.send(("pair", rid, dub_result(res)))
            gpu.release()     # this dub is done: the card keeps only what running dubs still need

        conform_episode(
            spec.ref, spec.dubs, spec.out_dir,
            cache_dir=spec.cache_dir,
            keep_tmp=spec.keep_tmp,
            low_mem=True,                       # features via memmap — RAM does not grow with length
            audio_method=spec.audio_method, drift_speed_pct=spec.drift_speed_pct,
            fill_silence=spec.fill_silence,
            ref_atrack=spec.ref_atrack, dub_atracks=spec.dub_atracks,
            progress=lambda p: pipe.send(("progress", rid, dataclasses.asdict(p))),
            should_stop=stop.is_set, on_pair=on_pair)
    except BaseException as e:  # noqa: BLE001 — the outcome is reported, not raised
        status, err = "failed", str(e) or type(e).__name__
        logger.exception("conform job {} упал", jid)
    finally:
        try:
            from track_muxer.conform import gpu, procreg
            procreg.bind(None)
            gpu.release()
        except Exception:  # noqa: BLE001 — releasing must not cost the outcome
            logger.exception("conform job {}: уборка после прогона не удалась", jid)
        pipe.send(("finished", rid, status, err))


def serve(conn) -> None:
    """The worker's life: take commands until told to retire, then exit and free the GPU."""
    from track_muxer.conform import procreg

    pipe = _Pipe(conn)
    # Every subprocess goes on the daemon's record with its start mark: should this process die,
    # its ffmpeg runs outlive it in their own sessions, and the daemon kills them by that record.
    procreg.observe(lambda pid, alive: pipe.send(("proc", pid, alive, procreg.start_mark(pid))))
    pool = ThreadPoolExecutor(max_workers=_MAX_JOBS, thread_name_prefix="conform")
    stops: dict[str, threading.Event] = {}
    groups: dict[str, procreg.ProcGroup] = {}
    logger.info("conform: исполнитель запущен, pid {}", os.getpid())
    try:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):              # the daemon is gone: every run stops here
                logger.warning("conform: демон пропал — исполнитель останавливает задания")
                for ev in stops.values():
                    ev.set()
                for g in groups.values():
                    g.kill_all()
                break
            kind = msg[0]
            if kind not in COMMANDS:                 # the catalogue is the contract: nothing else is obeyed
                logger.error("conform: исполнитель получил неизвестную команду {!r}", kind)
                continue
            if kind == "run":
                _, rid, job = msg
                stops[rid] = threading.Event()
                groups[rid] = procreg.ProcGroup()
                pool.submit(_job, rid, job, stops[rid], groups[rid], pipe)
            elif kind == "stop":
                rid = msg[1]
                if rid in stops:
                    stops[rid].set()
                    n = groups[rid].kill_all()
                    if n:
                        logger.info("conform {}: отмена — убито процессов: {}", rid, n)
            elif kind == "retire":
                break
    finally:
        pool.shutdown(wait=True)
        try:
            conn.close()                             # the daemon sees the end at once, not at exit
        except OSError:
            pass
        logger.info("conform: исполнитель завершается — видеопамять уходит драйверу")
