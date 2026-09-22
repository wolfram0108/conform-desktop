"""SRM features: video -> per-frame vectors (ffmpeg decode + KB/D1 convolution).

The encoding is fixed: gray 128x72, the KB and D1 kernels, clip +-3, per-channel L2 norm,
concat x1/sqrt(2), float16, ffmpeg `-vsync 0` (passthrough -- frame index = position). Any
deviation here breaks bit-exact matching against a cached SRM or wav.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from track_muxer.conform.config import FFMPEG, FFPROBE
from track_muxer.conform import procreg
from track_muxer.conform.models import SrmFeatures
from track_muxer.conform.progress import Reporter

GW, GH = 128, 72
KB = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], np.float32)
D1 = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float32)
T = 3.0
_RBLOCK = 256


def probe_duration(video: Path, ffprobe: str = FFPROBE) -> float | None:
    r = procreg.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def _parse_hhmmss(s: str | None) -> float | None:
    """'00:25:11.410000000' -> seconds (float). Invalid input -> None."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        parts = s.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(s)
    except ValueError:
        return None


def probe_video_duration(video: Path, ffprobe: str = FFPROBE) -> float | None:
    """The real duration of the VIDEO STREAM (NOT the container's format=duration). The container
    value can be inflated by a broken Segment Duration in mkv or a long trailing audio/subtitle
    tail -- then fps=frames/duration lies (observed: format 1748 s against a video stream of
    1511 s -> fps 25.9 instead of 29.97, stretching the output and breaking audio touch-up).
    Priority: video stream duration -> its DURATION tag -> format.duration (fallback)."""
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    try:
        v = float(r.stdout.strip())
        if v > 0:
            return v
    except ValueError:
        pass
    r = procreg.run(                                   # video stream's DURATION tag ('00:25:11.41' — reliable in mkv)
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream_tags=DURATION", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    v = _parse_hhmmss(r.stdout)
    if v and v > 0:
        return v
    return probe_duration(video, ffprobe)                 # fallback: container duration (when the video stream is silent)


def probe_audio_channels(video: Path, ffprobe: str = FFPROBE,
                         atrack: int = 0) -> tuple[int, str | None]:
    """Channel count and layout of audio track `atrack` (default 0), so 2.0/5.1/7.1 carries through
    to the output. -> (channels, channel_layout|None). Any failure -> a safe fallback (2, None).

    Parsing must use ONLY the JSON output. ffprobe's text format (`-of csv`) emits a TRAILING
    COMMA on streams with extra data: a plain file gives `2,stereo`, but an HDR remux gives
    `6,5.1(side),`. A layout parsed with that trailing comma makes ffmpeg reject it
    (`Unable to parse "ch_layout" option value "5.1(side)," as channel layout`) and die at the
    start of encoding, which otherwise surfaces only as a broken pipe well into a long run."""
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", f"a:{int(atrack)}",
         "-show_entries", "stream=channels,channel_layout", "-of", "json", str(video)],
        capture_output=True, text=True,
    )
    try:
        streams = json.loads(r.stdout or "{}").get("streams") or []
        ch = int(streams[0].get("channels"))
    except (ValueError, TypeError, IndexError, json.JSONDecodeError):
        return (2, None)
    layout = str(streams[0].get("channel_layout") or "").strip().strip(",")
    return (max(1, ch), layout if layout and layout != "unknown" else None)


def _stream_starts(video: Path, ffprobe: str) -> list[tuple[str, float]] | None:
    """[(codec_type, start_time)] of every stream in container order; None when ffprobe output is unusable."""
    r = procreg.run(
        [ffprobe, "-v", "error", "-show_entries", "stream=codec_type,start_time", "-of", "json", str(video)],
        capture_output=True, text=True,
    )
    try:
        return [(str(s.get("codec_type")), float(s.get("start_time") or 0.0))
                for s in json.loads(r.stdout or "{}").get("streams") or []]
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def probe_av_delay(video: Path, ffprobe: str = FFPROBE, atrack: int = 0) -> float:
    """Container delay video_start − audio_start (s) of the file, 0.0 without one of the streams.
    Each stream is decoded from its own zero, so the audio must be moved by this value to sit
    on the video's time axis (positive: audio leads the video and its head is trimmed)."""
    starts = _stream_starts(video, ffprobe)
    if starts is None:
        return 0.0
    v = next((t for kind, t in starts if kind == "video"), None)
    audio = [t for kind, t in starts if kind == "audio"]
    a = audio[int(atrack)] if 0 <= int(atrack) < len(audio) else None
    if v is None or a is None:
        return 0.0
    return float(v - a)


def probe_video_start(video: Path, ffprobe: str = FFPROBE) -> float:
    """Video start on the container timeline (s): first video frame minus the earliest stream start.
    Sidecar subtitles are timed on the container timeline, conform's axis starts at the first frame."""
    starts = _stream_starts(video, ffprobe)
    v = next((t for kind, t in starts or [] if kind == "video"), None)
    if v is None:
        return 0.0
    return float(v - min(t for _, t in starts))


def av_delay_filters(delay_s: float) -> list[str]:
    """ffmpeg audio filters that lay the audio on the video axis for a container delay.
    Positive delay trims the audio head, negative pads it with silence; |delay| ≤ 1 ms is no-op."""
    if delay_s > 0.001:
        return [f"atrim=start={delay_s:.6f}", "asetpts=PTS-STARTPTS"]
    if delay_s < -0.001:
        return [f"adelay={int(round(-delay_s * 1000))}:all=1"]
    return []


def probe_audio_tracks(video: Path, ffprobe: str = FFPROBE) -> list[dict]:
    """The list of ALL audio tracks in the file, for track selection in the UI (the reference
    track / dub tracks). Each entry:
    {index (0-based among audio streams), codec, channels, layout, lang, title, default}.
    Error or no audio -> []."""
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", "a",
         "-show_entries",
         "stream=codec_name,channels,channel_layout:stream_disposition=default"
         ":stream_tags=language,title",
         "-of", "json", str(video)],
        capture_output=True, text=True,
    )
    try:
        streams = json.loads(r.stdout or "{}").get("streams") or []
    except json.JSONDecodeError:
        return []
    out: list[dict] = []
    for i, s in enumerate(streams):
        tags = s.get("tags") or {}
        layout = (s.get("channel_layout") or "").strip()
        out.append({
            "index": i,
            "codec": s.get("codec_name") or "",
            "channels": int(s.get("channels") or 0),
            "layout": layout if layout and layout != "unknown" else None,
            "lang": (tags.get("language") or "").strip() or None,
            "title": (tags.get("title") or "").strip() or None,
            "default": bool((s.get("disposition") or {}).get("default")),
        })
    return out


def probe_resolution(video: Path, ffprobe: str = FFPROBE) -> int:
    """Frame height of the FIRST video stream, for choosing the decode backend (CPU/GPU). Error -> 0 (-> CPU)."""
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=height", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    try:
        return int(r.stdout.strip())
    except (ValueError, AttributeError):
        return 0


def parse_fps(s: str | None) -> float | None:
    """'24000/1001' or '23.976' -> frames per second; anything else -> None.

    The only parser of a frame rate given as text, for ffprobe output and for user input alike:
    a number or a ratio of two numbers, never an evaluated expression. The result is finite and
    positive or it is None."""
    s = (s or "").strip()
    try:
        if "/" in s:
            a, b = s.split("/")
            v = float(a) / float(b)
        else:
            v = float(s)
    except (ValueError, ZeroDivisionError):
        return None
    return v if (math.isfinite(v) and v > 0) else None


def probe_fps(video: Path, ffprobe: str = FFPROBE) -> float | None:
    """fps of the video stream (r_frame_rate, e.g. '24000/1001'), for ESTIMATING the frame count in progress reporting."""
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    return parse_fps(r.stdout)


def probe_frame_count_hints(video: Path, ffprobe: str = FFPROBE) -> tuple[int | None, float | None]:
    """(nb_frames, avg_frame_rate) of the video stream, from METADATA, with no pass over the file.

    Why bypass `r_frame_rate`: it can be GARBAGE. Measured on 92 real web-player files (cvh):
    r_frame_rate=48.0 and 90000.0 (the latter is the MPEG-TS 90 kHz timebase leaking into the
    field) against a real 23.976 -> the completeness gate below treated the expectation as
    2x/3750x inflated and dropped an entire honest episode. Meanwhile `nb_frames`, when present,
    was filled in and matched the video packet count EXACTLY on all 71 files that had it
    (0 discrepancies).
    `avg_frame_rate` is NOT an honest source on VFR: on a variable-rate Matroska it stays nominal
    (measured: 24000/1001 against a real 3237 frames over 180 s = 17.98) -- so when `nb_frames` is
    missing, packets are counted honestly instead (`probe_packet_count`)."""
    # JSON, not csv: ffprobe outputs fields in its OWN internal order, not the order requested
    # (checked: `stream=nb_frames,avg_frame_rate` -> csv gives "avg,nb"), so positional parsing is fragile.
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,avg_frame_rate", "-of", "json", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        st = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    except (json.JSONDecodeError, IndexError):
        return None, None
    avg = parse_fps(st.get("avg_frame_rate"))
    try:
        nb = int(st.get("nb_frames"))
    except (TypeError, ValueError):
        nb = None
    return (nb if (nb and nb > 0) else None), avg


def probe_media_info(path: Path, ffprobe: str = FFPROBE) -> dict:
    """File passport from ONE metadata probe: duration, frame size, rate, frame count.

    Used by the panel to show what is being worked on and estimate the remaining time (the
    estimate scales with minutes of material and decode gigapixels).

    Metadata only, with NO pass over the file: costs a fraction of a second. The frame count can
    be missing (VFR, broken headers) -- then it is estimated as duration times rate, and if the
    rate is also missing it stays None. That is acceptable here: the value feeds a time estimate,
    not the layout. Any failure -> an empty dict, and the caller works without it.

    -> {duration_s, width, height, fps, frames, has_video}
    """
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration"
         ":format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        data = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    st = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}

    def _f(x):
        try:
            v = float(x)
            return v if v > 0 else None
        except (TypeError, ValueError):
            return None

    dur = _f(st.get("duration")) or _f(fmt.get("duration"))
    fps = parse_fps(st.get("avg_frame_rate")) or parse_fps(st.get("r_frame_rate"))
    try:
        frames = int(st.get("nb_frames"))
    except (TypeError, ValueError):
        frames = 0
    if frames <= 0:
        frames = int(dur * fps) if (dur and fps) else 0
    w, h = st.get("width") or 0, st.get("height") or 0
    return {"duration_s": dur or 0.0, "width": int(w), "height": int(h),
            "fps": fps or 0.0, "frames": frames, "has_video": bool(w and h)}


def probe_packet_count(video: Path, ffprobe: str = FFPROBE) -> int | None:
    """The HONEST video packet count: one pass over the file WITHOUT decoding pixels.

    The only source that does not lie on VFR (where `nb_frames` is missing and `avg_frame_rate`
    is nominal). Cost measured: 0.04 s on a 180 s clip, 0.68 s on a 182 MB MP4, 4.3 s on a 1.5 GB
    MKV -- against the 40-60 s of the SRM decode itself, i.e. <=10% overhead, and only paid when
    `nb_frames` is missing."""
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        n = int((r.stdout or "").strip().rstrip(",").splitlines()[0])
    except (ValueError, IndexError):
        return None
    return n if n > 0 else None


def probe_has_video(video: Path, ffprobe: str = FFPROBE) -> bool:
    """Whether the file has a REAL video stream. Bare audio (flac/mka/mp3/aac...) -> False,
    the signal for conform's AUDIO-ONLY branch.
    A cover image (attached_pic on mp3/flac) is formally a video stream but is NOT video: excluded
    by disposition. A probe failure -> True (conservative: let the video path fail with a clear
    error rather than silently fall back to audio mode)."""
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", "v",
         "-show_entries", "stream=codec_type:stream_disposition=attached_pic",
         "-of", "json", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        streams = json.loads(r.stdout or "{}").get("streams") or []
    except json.JSONDecodeError:
        return True
    for st in streams:
        if int((st.get("disposition") or {}).get("attached_pic", 0)) == 0:
            return True
    return False


def probe_frame_pts(video: Path, ffprobe: str = FFPROBE) -> np.ndarray | None:
    """PTS of every VIDEO PACKET (seconds), sorted ascending -- exactly the frame display times.
    Measured on a VFR clip: sorted(packet pts_time) == frame pts_time BIT-FOR-BIT
    (0.000 ms difference), at a cost of 0.05 s against 2.80 s for a per-frame decode pass; 1.2 s
    on a real 700 MB MKV. Any N/A / garbage / empty -> None (fail cautiously)."""
    r = procreg.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    vals: list[float] = []
    for x in (r.stdout or "").split():
        x = x.strip().rstrip(",")
        if not x:
            continue
        try:
            vals.append(float(x))
        except ValueError:                # N/A or other junk means there is no axis
            return None
    if not vals:
        return None
    return np.sort(np.asarray(vals, np.float64))


def vfr_time_axis(video: Path, n_frames: int, ffprobe: str = FFPROBE) -> np.ndarray | None:
    """Frame time axis (seconds FROM THE FIRST FRAME, t[0]=0), ONLY for real VFR; otherwise None.

    None means downstream consumers compute time as index/fps, bit-exact with the CFR case by
    construction. The VFR criterion is the residual against the BEST-FIT uniform grid
    (step = t[-1]/(n-1)), threshold 1.5 frames. Why not the median step: Matroska quantizes
    timestamps to whole milliseconds (42/41 alternating for a true 41.708), so the median (42.00)
    drifts by 10 s over an episode and reports false VFR on an honest CFR file. Measured on 5
    files: a quantized CFR mkv gives a residual of 0.02 frame; real VFR encodes give 539-1425
    frames -- the gap between the two classes spans 4 orders of magnitude.

    len(pts) != n_frames (broken packets, a decode abort) -> None: the packet-to-frame stitch by
    index must be 1:1, or the axis is not trusted."""
    if n_frames < 2:
        return None
    pts = probe_frame_pts(video, ffprobe)
    if pts is None or len(pts) != n_frames:
        return None
    t = pts - pts[0]
    slope = float(t[-1]) / (n_frames - 1)
    if slope <= 0:
        return None
    dev = np.abs(t - np.arange(n_frames) * slope)
    return t if float(dev.max()) > 1.5 * slope else None


def build_srm(
    video: Path | str,
    fps: float | None = None,
    *,
    ffmpeg: str = FFMPEG,
    ffprobe: str = FFPROBE,
    progress=None,
    should_stop=None,
    progress_meta: tuple[int, int, str] = (0, 0, ""),
    mmap_path: Path | None = None,
    crop: str | None = None,
    backend: str = "cpu",
    reporter: Reporter | None = None,
) -> SrmFeatures:
    """Decode the video and convolve -> SrmFeatures. fps=None means auto (frames/duration).

    Progress of the decode goes to `reporter` (stage Reporter); `progress`+`progress_meta`
    build one for stage "decode" when no reporter is given.
    should_stop() == True interrupts the decode (RuntimeError("stopped")).
    mmap_path given: features are STREAMED to a raw f16 file (not accumulated in RAM), and srm
    is returned as an np.memmap (read-only). Values are bit-exact with the in-RAM path.
    crop='W:H:X:Y' (geometry correction, conform.geom): crop the frame BEFORE scale=128:72,
    matching the dub's framing to the reference when crop/zoom/anamorphic stretch/letterboxing
    would otherwise blind the SRM. crop=None (default) leaves the frame uncropped, bit-exact.
    backend='cuda' decodes on the GPU (NVDEC, `-hwaccel cuda`); the 128x72 scale STAYS on the CPU
    (no output_format cuda), so frames are bit-exact with 'cpu' (the decode is deterministic) and
    the cache does not depend on the backend. conform.decode_backend picks the backend by
    resolution against the NVDEC ceiling.
    """
    video = Path(video)
    dur = probe_video_duration(video, ffprobe)                # video stream, not the container
    # fps of the video stream — for estimating the frame count: the progress bar AND the decode
    # completeness gate (below).
    fps_est = fps if fps else probe_fps(video, ffprobe)
    nb_meta, avg_fps = probe_frame_count_hints(video, ffprobe)   # honest sources for the gate
    rep = reporter if reporter is not None else Reporter.of(progress, "decode", progress_meta)

    vf = (f"crop={crop},scale={GW}:{GH},format=gray" if crop
          else f"scale={GW}:{GH},format=gray")
    pre = ["-hwaccel", "cuda"] if backend == "cuda" else []   # GPU decode; scale stays on CPU -> bit-exact
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", *pre, "-i", str(video), "-an",
           "-vf", vf, "-vsync", "0", "-f", "rawvideo", "-"]
    fb = GW * GH
    D = GW * GH * 2                       # frame vector dimension (rk[9216]+rd[9216])
    p = procreg.popen(cmd, stdout=subprocess.PIPE, bufsize=fb * _RBLOCK)
    out_f = open(mmap_path, "wb") if mmap_path is not None else None   # stream to disk (keeps RAM low)
    vecs: list[np.ndarray] = []           # used only when mmap_path is None
    n_written = 0
    buf = b""
    t0 = time.perf_counter()
    # Expected frame count for the completeness gate (below) and the progress bar. Sources are
    # listed from most to least honest. `r_frame_rate` is ONLY the last fallback — it can be
    # garbage (e.g. 48.0 or 90000.0 against a real 23.976), which made the gate drop a whole
    # honest episode; `avg_frame_rate` is also NOT honest on VFR (a nominal 24000/1001 against a
    # real 17.98), so when nb_frames is missing we count packets instead (costs ≤10% of the
    # decode time, see probe_packet_count).
    if nb_meta:
        n_expect = nb_meta                                   # exact frame count from the container
    else:
        n_pkt = probe_packet_count(video, ffprobe)            # honest count (VFR/Matroska)
        if n_pkt:
            n_expect = n_pkt
        elif dur and avg_fps:
            n_expect = int(round(dur * avg_fps))
        elif dur and fps_est:
            n_expect = int(round(dur * fps_est))             # the last-resort fallback
        else:
            n_expect = 0
    try:
        while True:
            if should_stop is not None and should_stop():
                p.kill()
                raise RuntimeError("stopped")
            chunk = p.stdout.read(fb * _RBLOCK)
            if not chunk:
                break
            buf += chunk
            k = len(buf) // fb
            if k:
                block = np.frombuffer(buf[:k * fb], np.uint8).reshape(k, GH, GW).astype(np.float32)
                buf = buf[k * fb:]
                bvs = np.empty((k, D), np.float16)
                for j in range(k):
                    g = block[j]
                    rk = np.clip(ndimage.convolve(g, KB, mode="reflect"), -T, T).ravel()
                    rd = np.clip(ndimage.convolve(g, D1, mode="reflect"), -T, T).ravel()
                    rk /= np.linalg.norm(rk) + 1e-6
                    rd /= np.linalg.norm(rd) + 1e-6
                    bvs[j] = (np.concatenate([rk, rd]) * 0.7071068).astype(np.float16)
                if out_f is not None:
                    out_f.write(bvs.tobytes())     # to disk — only the current block is kept in RAM
                else:
                    vecs.append(bvs)
                n_written += k
                if rep is not None:
                    frac = (n_written / n_expect) if n_expect else 0.0
                    kps = n_written / (time.perf_counter() - t0 + 1e-9)
                    rep(min(frac, 0.999), f"{n_written} кадров, {kps:,.0f} к/с")
    finally:
        p.stdout.close()
        p.wait(); procreg.done(p)
        if out_f is not None:
            out_f.close()

    # -- Decode integrity gate: an ffmpeg abort (NVDEC/OOM/broken stream) over a pipe is
    #    indistinguishable from the end of the file. Without this gate a truncated decode
    #    silently became a "successful" SRM (e.g. 4357 of 34552 frames, fps=n/dur=3.02),
    #    poisoning the cache and failing every dub of the episode. --
    if p.returncode not in (0, None):
        raise RuntimeError(f"декод SRM упал (ffmpeg rc={p.returncode}) на кадре {n_written}: {video.name}")
    # 0.9: a real abort loses a large fraction of the file (seen as low as 12.6%); honest frame
    # loss stays within a few percent. The expectation is computed from nb_frames/avg_fps (see
    # above), never from r_frame_rate — a garbage r_frame_rate would otherwise drop an honest
    # file (seen giving 0.4995 and 0.0003 of the expected count).
    if n_expect and n_written < 0.9 * n_expect:
        raise RuntimeError(
            f"декод SRM неполон: {n_written} из ~{n_expect} кадров — обрыв декодера ({video.name})")

    if mmap_path is not None:
        arr = (np.memmap(mmap_path, dtype=np.float16, mode="r", shape=(n_written, D))
               if n_written else np.zeros((0, D), np.float16))
    else:
        arr = np.concatenate(vecs) if vecs else np.zeros((0, D), np.float16)
    if fps is None:
        if dur is None:
            dur = probe_video_duration(video, ffprobe)    # video stream, not the bloated container
        fps = (len(arr) / dur) if dur else 0.0
    # A real frame time axis exists only for VFR (otherwise None, and downstream computes time
    # as index/fps, bit-exact). On VFR this also makes fps honest: the average frame density over the
    # axis, not n/dur from the container duration (used by density windows and the ref/dub
    # ratio downstream).
    pts_ax = vfr_time_axis(video, len(arr), ffprobe)
    if pts_ax is not None and pts_ax[-1] > 0:
        fps = (len(arr) - 1) / float(pts_ax[-1])
    return SrmFeatures(srm=arr, fps=float(fps), src=video, pts=pts_ax)
