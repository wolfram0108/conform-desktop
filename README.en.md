# conform-desktop

[Русский](README.md)

The application conforms audio tracks to the timeline of a given video file. Input: a
reference (the file defining the timeline) and any number of source files with audio
tracks. Output: FLAC tracks synchronous with the reference, ready for muxing.

There are no adjustment parameters: which corrections a given track needs is determined by
measurement.

---

# Installing on Windows

Requirements: Windows 10 or 11 (x64), 4.6 GB for the installation and about 8 GB for
intermediate files. An NVIDIA GPU is preferable; without one the same pipeline runs on the
CPU — slower, same result. Administrator rights are not needed.

### Step 1. Create a folder for the application

For example `D:\conform`. Everything is installed inside it; nothing is written to system
directories.

### Step 2. Download the launcher

From [Releases](../../releases) — `conform-setup.exe` (13 MB). Put it in that folder.

### Step 3. Run the launcher and press "Install"

A window lists the components. The launcher downloads them (about 1.5 GB), verifies
checksums and unpacks.

**If downloading through the application is inconvenient** (slow link, no access), fetch
the files in advance by any means — a browser, a download manager, another machine:

1. take `manifest.json` and the three `conform-*.7z` archives from the Releases page;
2. create a `packages` folder next to `conform-setup.exe` and put the files there;
3. run the launcher, press "Проверить наличие файлов" (check for files), then "Install".

No network access is required in that case. Files are identified by checksum, so renaming
during transfer does not matter.

### Step 4. Wait for the installation check

After unpacking, the launcher runs a control clip through the full processing pipeline and
compares the checksum of the result with the reference value. The log must show:

```
Результат обработки совпал с эталонным: установка исправна.
(the result matched the reference: the installation is sound)
```

This verifies that the installation works, not merely that files exist: incomplete
unpacking or an incompatible set of libraries is caught here.

### Step 5. Press "Run"

The application window opens. Afterwards you can start it either from the launcher or
directly — `app\conform-desktop.exe`.

### What gets downloaded

| Component | Contents | Archive |
|---|---|---|
| `conform-runtime` | Compute runtime: torch, CUDA libraries | 1199 MB |
| `conform-app` | Application, interface, processing core | 254 MB |
| `conform-ffmpeg` | Media decoder | 109 MB |

### Updating

Download `conform-setup.exe` of the new release and run it **in the same folder**. It
compares what is installed against the new release and downloads only what changed —
usually `conform-app` (254 MB); the runtime and the decoder stay in place.

---

# Running on Linux

There is no prebuilt Linux distribution yet — run from sources. Verified on Ubuntu 24.04
with CUDA.

### Step 1. System packages

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg \
    libgl1 libegl1 libxkbcommon0 libxkbfile1 libxcb-cursor0 libnss3 libnspr4 \
    libasound2t64 libxdamage1 libxrandr2 libxcomposite1 libxtst6 libgbm1 \
    libcups2t64 libdrm2 libpango-1.0-0 libatk1.0-0t64 libatk-bridge2.0-0t64 \
    libatspi2.0-0t64 libgl1-mesa-dri
```

### Step 2. Python environment

```bash
git clone https://github.com/wolfram0108/conform-desktop.git
cd conform-desktop
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Check that the GPU is visible:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`False` is not an error — processing will run on the CPU.

### Step 3. Run with a window

```bash
export PYTHONPATH="$PWD:$PWD/src"
export TM_FFMPEG=/usr/bin/ffmpeg TM_FFPROBE=/usr/bin/ffprobe
.venv/bin/python -m ui
```

### Step 4. Run without a graphical session

Over ssh, or on a machine without a display, no window can be created, but the application
can run as a local service with the interface opened in a browser:

```bash
export PYTHONPATH="$PWD:$PWD/src"
export TM_FFMPEG=/usr/bin/ffmpeg TM_FFPROBE=/usr/bin/ffprobe
export QT_QPA_PLATFORM=offscreen LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
export QTWEBENGINE_CHROMIUM_FLAGS='--no-sandbox --disable-gpu --single-process'
.venv/bin/python -m ui
```

Then open `http://127.0.0.1:8799/app/index.html`.

The variables are mandatory: without software OpenGL the embedded page engine aborts while
creating a graphics context. `--no-sandbox` is only needed when running as `root`.

---

# Your first job

1. **Task** tab, **Reference** field — choose the file that defines the timeline. If it has
   several audio tracks, select the one to be used as the audio reference.
2. **Source files** — add files with the button or by dragging them in. For each file select
   the audio tracks to process. Files without a video stream — plain audio files — are
   accepted as well.
3. **Output folder** — where results are written.
4. **Add to queue**. With "start immediately" enabled processing begins at once; otherwise
   the job waits for the ▶ button.
5. **Queue** tab — progress. The arrow next to the name expands the details: operations with
   their durations, tracks with measured values, plots, and a button that opens the output
   folder.

The result is FLAC files in the output folder; the channel layout of the source track is
preserved (2.0, 5.1, 7.1).

---

# What the measurements mean

| Value | Meaning | What to look for |
|---|---|---|
| residual offset | Audio shift remaining after processing | Normally a few milliseconds |
| measurement coverage | Fraction of the duration where audio analysis had something to lock onto | Low means the files share little common audio |
| matched | Fraction of frames that found a counterpart | A low value means the source does not match the reference |
| frame similarity | Closeness of descriptors of matched frames | Low on heavily re-encoded sources |
| discontinuities | Number of timeline jumps and the largest one | Correspond to insertions and cuts |
| filled from reference | Duration filled with reference audio | Segments where the source has no audio |

Values outside the expected range are flagged for review. A failed track does not stop the
job: the remaining tracks are processed and the reason is shown next to the track it
belongs to.

Performance, measured on an RTX 4090: a 23-minute 1080p reference with one source track
takes 2 min 34 s.

---

# How it works

Two measurement stages.

**Frame correspondence** establishes structure. A descriptor is computed for every frame;
reference and source frames are matched within a search band, which recovers the time rate
(including PAL speed-up) and the boundaries of insertions and cuts. If cropping or scaling
gets in the way, the frame transform is estimated and descriptors are recomputed.

**Audio refinement** then works on the already conformed stream and finds what the picture
cannot show: residual offset, drift, local discontinuities — traces of remixing, a
different edition of the soundtrack, or source defects. Silent segments where the reference
has audio are filled from the synchronised reference.

The order of operations is deterministic. Only geometric correction is optional — it runs
when coarse matching fails.

# Limitations

* Audio-to-picture misalignment introduced when the source itself was mixed is not
  corrected: relative to the reference this is not a desynchronisation, and the application
  reports it as a source defect.
* Non-linear tempo changes in audio without a matching change in the video are not
  recovered by the audio stage.
* For sources without a video stream only audio refinement is available, so structural
  differences are not reconstructed.

# Building and repository layout

| Path | Contents |
|---|---|
| `src/track_muxer/conform/` | Processing core. Not edited here: synchronised from the canonical repository via `git subtree`, see `docs/SYNC.md` |
| `server/` | Local HTTP API on top of the core |
| `ui/` | Qt interface, Russian and English |
| `build/` | Distribution build and pruning, see `build/README.md` |
| `installer/` | Thin launcher, splitting the distribution into components, control material, see `installer/README.md` |

# License

GPL-3.0, see [LICENSE](LICENSE).

The MuQ model (`OpenMuQ/MuQ-large-msd-iter`) is optional and not bundled. When enabled, its
weights are downloaded from HuggingFace under CC-BY-NC 4.0, which permits non-commercial
use of results only.
