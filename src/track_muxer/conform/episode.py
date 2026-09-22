"""One EPISODE: 1 reference + a list of dubs. The reference is decoded ONCE and reused
in RAM for every dub; optionally cached to disk (`cache_dir`) for reuse across RUNS (a new
dub can be synced precisely without re-decoding the reference).

Output: out_dir/<output name>.<ext> for each dub (the name comes from `naming.plan_outputs`;
the format is a detail of writing). skip_existing — dubs already done (the file is already on
disk) are skipped. keep_tmp=False → the cache subdirectory is removed on completion; True → it
stays (everything intermediate: the reference SRM + the dub's CK1/2/3).
"""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from track_muxer.conform import cache as cache_mod
from track_muxer.conform import naming
from track_muxer.conform import subs_transfer
from track_muxer.conform import tmpfiles
from track_muxer.conform.memlog import memlog
from track_muxer.conform.align import conform_pair, decode_reference_audio
from track_muxer.conform.config import FFMPEG
from track_muxer.conform.decode_backend import decode_backend
from track_muxer.conform.features import build_srm, probe_resolution
from track_muxer.conform.models import EpisodeResult, PairResult, SrmFeatures
from track_muxer.conform.progress import Reporter


def conform_episode(
    ref_video: Path | str,
    dub_videos: list[Path | str],
    out_dir: Path | str,
    *,
    ref_features: SrmFeatures | None = None,
    cache_dir: Path | str | None = None,
    skip_existing: bool = True,
    ffmpeg: str = FFMPEG,
    low_mem: bool = False,
    keep_tmp: bool = False,            # keep tmp checkpoints (CK1 dub SRM/CK2 audio/...) in cache_dir
    ref_atrack: int = 0,               # index of the ref's audio track (the audio reference for band/fill)
    dub_atracks: list[int] | None = None,   # audio track index for each dub (parallel to
                                       # dub_videos; None or a shorter list falls back to track 0). A
                                       # "virtual dub" is the same file (even the ref itself) on another track.
    progress=None,
    should_stop=None,
    on_pair=None,
    **pair_opts,
) -> EpisodeResult:
    """Align every dub of the episode onto the timeline of ref_video. -> EpisodeResult.

    pair_opts go to conform_pair/conform_features: the task parameters audio_method,
    drift_speed_pct and fill_silence.
    on_pair(PairResult) is called after each dub: live progress for the queue.

    The episode owns the reference's temporary files: they are removed whatever the outcome.

    Output names come from `naming.plan_outputs`: a track other than the first and a twin file
    differing only by extension get their own outputs; two sources landing on one output raise
    ValueError before any work.
    """
    ref_video = Path(ref_video)
    scratch = tmpfiles.Workspace(ref_video.parent, "episode") if low_mem else None
    try:
        return _conform_episode(
            ref_video, dub_videos, out_dir, scratch, ref_features=ref_features,
            cache_dir=cache_dir, skip_existing=skip_existing, ffmpeg=ffmpeg, low_mem=low_mem,
            keep_tmp=keep_tmp, ref_atrack=ref_atrack, dub_atracks=dub_atracks, progress=progress,
            should_stop=should_stop, on_pair=on_pair, **pair_opts)
    finally:
        if scratch is not None:
            scratch.close()


def _conform_episode(ref_video: Path, dub_videos, out_dir, scratch: tmpfiles.Workspace | None, *,
                     ref_features, cache_dir, skip_existing, ffmpeg, low_mem, keep_tmp,
                     ref_atrack, dub_atracks, progress, should_stop, on_pair, **pair_opts) -> EpisodeResult:
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dubs = [Path(d) for d in dub_videos]
    total = len(dubs)
    stems = naming.plan_outputs(dubs, dub_atracks, out_dir, ref_video)    # refuses the job before any decode

    # Reference: ref_features -> disk cache -> decode (+ save to cache).
    _decode = Reporter.of(progress, "decode", (0, total, "реф"), f"референс {ref_video.name}")
    if _decode is not None:
        _decode.mark(0.0)
    ref = ref_features
    if ref is None and cache_dir is not None:
        ref = cache_mod.load_srm(cache_dir, ref_video)        # memmap, if present in the cache
    if ref is None:
        with decode_backend(probe_resolution(ref_video), ffmpeg) as _be:   # 1080+ -> GPU (NVDEC ceiling), otherwise CPU
            if low_mem and cache_dir is not None:             # build the ref streaming straight into the cache -> memmap
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                ref = build_srm(ref_video, ffmpeg=ffmpeg, reporter=_decode,
                                should_stop=should_stop,
                                mmap_path=cache_mod.srm_file(cache_dir, ref_video), backend=_be)
                cache_mod.save_meta(cache_dir, ref_video, len(ref.srm), ref.fps)
            elif low_mem:                                     # no cache: a file-backed SRM in the episode's scratch
                ref = build_srm(ref_video, ffmpeg=ffmpeg, reporter=_decode,
                                should_stop=should_stop,
                                mmap_path=scratch.sub("srmref") / "ref.f16", backend=_be)
                scratch.adopt(ref)
            else:
                ref = build_srm(ref_video, ffmpeg=ffmpeg, reporter=_decode,
                                should_stop=should_stop, backend=_be)
                if cache_dir is not None:
                    cache_mod.save_srm(cache_dir, ref_video, ref)

    # The reference's own subtitles only move from the container axis to the first-frame axis.
    ref_dur = (float(ref.pts[-1]) + 1.0 / ref.fps) if ref.pts is not None else len(ref.srm) / ref.fps
    ref_text_tracks = subs_transfer.transfer_sidecars(
        ref_video, out_dir, ref_video.stem, subs_transfer.identity_time_map(ref.fps, ref_dur))

    # The reference sound is decoded once per episode and shared by every dub: the reference can be
    # gigabytes, and decoding it again would read the whole file for each dub. Decoded lazily, on the
    # first pair that is really processed; in low_mem it is a file in the episode's scratch, so memory
    # does not grow with the duration. The audio layer and the final fill of silence both read it.
    ref_audio = None
    ref_audio_failed = False      # the episode owns this decode: once failed, it is not tried again per dub

    pairs: list[PairResult] = []
    stopped = False
    for i, dub_video in enumerate(dubs, 1):
        if should_stop is not None and should_stop():
            stopped = True
            break
        atrack = int(dub_atracks[i - 1]) if (dub_atracks and i - 1 < len(dub_atracks)) else 0
        stem = stems[i - 1]
        out_path = out_dir / f"{stem}.flac"                     # conform output: FLAC, 16-bit / 44.1 kHz
        old = out_dir / f"{stem}.wav"                           # a .wav from before the switch to .flac also counts as done
        done = next((p for p in (out_path, old) if p.exists() and p.stat().st_size > 0), None)
        if skip_existing and done is not None:
            res = PairResult(dub=dub_video.name, out_path=done, ok=True, skipped=True)
        else:
            if ref_audio is None and not ref_audio_failed:        # the reference sound is decoded once
                try:
                    ref_audio = decode_reference_audio(
                        ref_video, ffmpeg, atrack=ref_atrack, low_mem=low_mem, scratch=scratch,
                        reporter=Reporter.of(progress, "extract", (0, total, "реф"), "аудио рефа (1 раз)"))
                    if scratch is not None:
                        scratch.adopt(ref_audio)
                    if len(ref_audio) == 0:                       # a track that decodes to nothing is no sound
                        raise ValueError("the reference audio track is empty")
                except Exception:  # noqa: BLE001 — no usable sound: every pair goes on without the audio layer
                    ref_audio, ref_audio_failed = None, True
            memlog('перед парой (после реф-аудио и SRM)')
            meta = (i, total, dub_video.name)
            try:
                res = conform_pair(ref, dub_video, out_path, ffmpeg=ffmpeg,
                                   progress=progress, should_stop=should_stop,
                                   progress_meta=meta, ref_audio=ref_audio,
                                   ref_atrack=ref_atrack, dub_atrack=atrack,
                                   low_mem=low_mem, cache_dir=cache_dir, keep_tmp=keep_tmp,
                                   **pair_opts)
            except Exception as e:  # noqa: BLE001 — one dub must not take the episode down
                # conform_pair records and saves its own trace; this catches only what escaped it.
                logger.exception("conform: озвучка {} упала на серии {}", dub_video.name, ref_video.name)
                res = PairResult(dub=dub_video.name, out_path=None, ok=False, error=str(e))
        # Finished pairs are served too: the stored map makes the transfer free of matching.
        if res.ok and res.out_path is not None:
            tmap = subs_transfer.load_time_map(out_path)
            if tmap is not None:
                res.text_tracks = subs_transfer.transfer_sidecars(dub_video, out_dir, stem, tmap)
        res.atrack = atrack
        pairs.append(res)
        if on_pair is not None:
            try:
                on_pair(res)
            except Exception:  # noqa: BLE001
                pass

    ref_audio = None        # the episode is over: drop the reference sound

    # Cache cleanup when checkpoints are not kept (and the episode was not interrupted). keep_tmp
    # covers keeping everything intermediate (ref SRM + dub CK1/2/3): ON keeps the cache, OFF cleans it up after the episode.
    if cache_dir is not None and not keep_tmp and not stopped:
        cache_mod.clear_dir(cache_dir)

    return EpisodeResult(ref=ref_video.name, out_dir=out_dir, pairs=pairs,
                         elapsed_s=time.perf_counter() - t0, ref_text_tracks=ref_text_tracks)
