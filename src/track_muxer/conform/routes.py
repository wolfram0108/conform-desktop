"""The HTTP router for conform (a FastAPI APIRouter). A thin layer over ConformQueue --
all the logic lives in the queue/core. Mounted in api/app.py via include_router.
"""

from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import ConfigDict

from track_muxer.conform.features import probe_audio_tracks
from track_muxer.conform.naming import plan_outputs
from track_muxer.conform.queue import ConformJob, ConformQueue
from track_muxer.conform.schema import ApiModel
from track_muxer.conform.task_params import AudioMethod, DriftSpeedPct, DRIFT_SPEED_DEFAULT, TaskParamLimits
from track_muxer.conform.scan import (
    CatalogItem,
    SeriesScan,
    list_catalogs,
    scan_catalog,
)


class _Body(ApiModel):
    """Base of the request bodies of conform. A field the daemon does not know is refused, not
    ignored: a client that still sends a removed setting learns that it has no effect. The base is
    local because this package is carried into another application as is and imports nothing
    outside itself."""

    model_config = ConfigDict(extra="forbid")


class ConformEnqueueBody(_Body):
    """Body of POST /conform/enqueue: one episode, a reference and its dubs."""

    ref: str
    dubs: list[str]
    out_dir: str | None = None         # None → <output>/_conform_out/<label>
    cache_dir: str | None = None       # cache subdirectory (<episode>/_conform_cache)
    keep_tmp: bool = False             # tmp checkpoints: keep ALL intermediates (ref SRM + dub
                                       # CK1/2/3) -> rerun without GPU decode; OFF -> cache is cleared after the episode
    label: str | None = None           # label (defaults to the reference folder's name)
    # Task parameters: a choice about the method or the content of the result, never a fix for a track.
    audio_method: AudioMethod = AudioMethod.BAND    # meter of the audio layer
    drift_speed_pct: DriftSpeedPct = DRIFT_SPEED_DEFAULT   # how fast the laid sound may follow a drift, % per second
    fill_silence: bool = True          # fill the silence left in the dub with the reference sound
    autostart: bool = True             # False -> the job waits for a manual start (the play button in the queue)
    ref_atrack: int = 0                # the REFERENCE audio track (sound baseline for band alignment/fill)
    dub_atracks: list[int] | None = None   # the audio track of EACH dub (parallel to dubs;
                                       # None -> all 0). A "virtual dub" is the same file
                                       # (even the reference itself) on a different track.


class ConformSettingsBody(_Body):
    limit: int | None = None           # concurrency limit (episodes at once)


class AudioTrack(ApiModel):
    """One audio track of a file, for choosing which one to align."""

    index: int                 # 0-based among the audio tracks
    codec: str
    channels: int
    layout: str | None
    lang: str | None
    title: str | None
    default: bool


class ConformDevice(ApiModel):
    device: str
    name: str
    gpu: bool


class ConformSettings(ApiModel):
    limit: int
    active: int
    params: TaskParamLimits = TaskParamLimits()


class ConformJobCancelled(ApiModel):
    cancelled: str


class ConformJobRetried(ApiModel):
    retried: str


class ConformJobStarted(ApiModel):
    started: str


class ConformJobPaused(ApiModel):
    paused: str


class ConformCleared(ApiModel):
    cleared: int


class ConformClearedDone(ApiModel):
    cleared: int
    freed_mb: float


def make_conform_router(cq: ConformQueue) -> APIRouter:
    r = APIRouter(prefix="/conform", tags=["conform"])

    @r.post("/enqueue", response_model=ConformJob)
    def enqueue(body: ConformEnqueueBody) -> ConformJob:
        ref = Path(body.ref)
        if not ref.exists():
            raise HTTPException(400, f"нет референса: {body.ref}")
        missing = [d for d in body.dubs if not Path(d).exists()]
        if missing:
            raise HTTPException(400, f"нет озвучек: {', '.join(missing)}")
        if not body.dubs:
            raise HTTPException(400, "пустой список озвучек")
        if body.dub_atracks is not None and len(body.dub_atracks) != len(body.dubs):
            raise HTTPException(400, f"dub_atracks ({len(body.dub_atracks)}) не параллелен dubs ({len(body.dubs)})")
        label = body.label or ref.parent.name
        out_dir = body.out_dir or str(cq.output_dir / "_conform_out" / label)
        try:
            plan_outputs(body.dubs, body.dub_atracks, out_dir, ref)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        spec = {
            "ref": str(ref), "dubs": [str(Path(d)) for d in body.dubs],
            "out_dir": out_dir, "cache_dir": body.cache_dir,
            "keep_tmp": body.keep_tmp, "label": label,
            "audio_method": body.audio_method, "drift_speed_pct": body.drift_speed_pct,
            "fill_silence": body.fill_silence,
            "ref_atrack": body.ref_atrack, "dub_atracks": body.dub_atracks,
            "autostart": body.autostart,
        }
        return cq.enqueue(spec)

    @r.get("/atracks")
    def atracks(path: str = Query(..., description="file whose audio tracks are listed for choosing one")) -> list[AudioTrack]:
        p = Path(path)
        if not p.exists():
            raise HTTPException(400, f"нет файла: {path}")
        return [AudioTrack(**t) for t in probe_audio_tracks(p)]

    @r.get("/device")
    def device() -> ConformDevice:
        """The device the audio layer runs on, for the GPU/CPU badge."""
        try:
            import torch
            if torch.cuda.is_available():
                return ConformDevice(device="cuda", name=torch.cuda.get_device_name(0), gpu=True)
        except Exception:  # noqa: BLE001 -- torch/CUDA unavailable -> CPU
            pass
        return ConformDevice(device="cpu", name="CPU", gpu=False)

    @r.get("/jobs", response_model=list[ConformJob])
    def jobs() -> list[ConformJob]:
        return cq.list_items()

    @r.get("/jobs/{jid}", response_model=ConformJob)
    def job(jid: str) -> ConformJob:
        j = cq.get_item(jid)
        if j is None:
            raise HTTPException(404, "нет такой задачи")
        return j

    @r.get("/plot/{jid}/{name}")
    def plot(jid: str, name: str):
        """An alignment plot: the PNG preview or the interactive HTML page, <job.out_dir>/_plots/<name>.
        The name is validated."""
        j = cq.get_item(jid)
        if j is None:
            raise HTTPException(404, "нет такой задачи")
        is_png = name.endswith(".png"); is_html = name.endswith(".html")
        if "/" in name or "\\" in name or ".." in name or not (is_png or is_html):
            raise HTTPException(400, "плохое имя файла")
        base = (Path(j.out_dir) / "_plots").resolve()
        p = (base / name).resolve()
        if p.parent != base or not p.exists():
            raise HTTPException(404, "нет такого графика")
        return FileResponse(p, media_type=("image/png" if is_png else "text/html"))

    @r.delete("/jobs/{jid}")
    def cancel(jid: str) -> ConformJobCancelled:
        if not cq.cancel(jid):
            raise HTTPException(404, "нет такой задачи")
        return ConformJobCancelled(cancelled=jid)

    @r.post("/jobs/{jid}/retry")
    def retry(jid: str) -> ConformJobRetried:
        if not cq.retry(jid):
            raise HTTPException(409, "нельзя перезапустить (нет задачи или ещё активна)")
        return ConformJobRetried(retried=jid)

    @r.post("/jobs/{jid}/start")
    def start(jid: str) -> ConformJobStarted:
        """Start a paused job by hand."""
        if not cq.start(jid):
            raise HTTPException(409, "нельзя запустить (нет задачи или она не на паузе)")
        return ConformJobStarted(started=jid)

    @r.post("/jobs/{jid}/pause")
    def pause(jid: str) -> ConformJobPaused:
        """Take a job off automatic feed: queued -> paused. A running job is cancelled, not paused."""
        if not cq.pause(jid):
            raise HTTPException(409, "нельзя поставить на паузу (нет задачи или она уже идёт)")
        return ConformJobPaused(paused=jid)

    @r.post("/clear_done")
    def clear_done() -> ConformClearedDone:
        """Remove finished jobs together with their temporary files (caches and checkpoints).
        Output audio files and plots are never touched."""
        return ConformClearedDone(**cq.clear_done())

    @r.post("/clear")
    def clear() -> ConformCleared:
        return ConformCleared(cleared=cq.clear())

    @r.get("/settings")
    def get_settings() -> ConformSettings:
        # The domains of the task parameters travel with the settings: a client shows them, never restates them.
        return ConformSettings(**cq.get_settings())

    @r.put("/settings")
    def put_settings(body: ConformSettingsBody) -> ConformSettings:
        return ConformSettings(**cq.set_settings(body.limit))

    @r.get("/catalogs", response_model=list[CatalogItem])
    def catalogs() -> list[CatalogItem]:
        """Subdirectories of the daemon's downloads: the candidates to choose a catalogue from."""
        return list_catalogs(cq.output_dir)

    @r.get("/scan", response_model=list[SeriesScan])
    def scan(path: str = Query(..., description="a catalogue: its episodes, dubs, the reference found and what is done")) -> list[SeriesScan]:
        return scan_catalog(path)

    return r
