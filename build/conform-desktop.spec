# -*- mode: python ; coding: utf-8 -*-
# Продуктовый spec conform-desktop.exe (windowed, --onedir).
# Сборка (из standalone-venv, из каталога build/):
#   <venv>/Scripts/python.exe -m PyInstaller conform-desktop.spec --noconfirm
# ffmpeg/ffprobe берутся из FFMPEG_DIR (env) — cuda-сборки, кладутся в дистрибутив.

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).resolve().parent                 # корень conform-desktop (build/ → ..)
SRC = ROOT / "src"
FFDIR = Path(os.environ.get("FFMPEG_DIR", r"D:\Temp\track_muxer"))

datas, binaries, hidden = [], [], []
for pkg in ["torch", "torchaudio", "torchvision", "transformers", "muq",
            "librosa", "nnAudio", "x_clip", "kornia", "numba", "llvmlite",
            "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hidden += h

# страница интерфейса (HTML/CSS/JS) — сам интерфейс, без неё окно пустое
datas += [(str(p), "ui/web") for p in (ROOT / "ui" / "web").iterdir() if p.is_file()]

hidden += collect_submodules("track_muxer.conform")
hidden += collect_submodules("ui")
hidden += collect_submodules("server")
hidden += ["cv2", "matplotlib", "matplotlib.pyplot", "plotly.graph_objects",
           "scipy", "fastapi", "uvicorn",
           "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebChannel"]

a = Analysis(
    ["entry_desktop.py"],
    pathex=[str(ROOT), str(SRC)],
    binaries=binaries + [(str(FFDIR / "ffmpeg.exe"), "."), (str(FFDIR / "ffprobe.exe"), ".")],
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "PyQt5", "PyQt6", "IPython", "pytest",
              "PySide6.Qt3DCore", "PySide6.QtQuick3D", "PySide6.QtCharts",
              "PySide6.QtMultimedia", "PySide6.QtPdf"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="conform-desktop",
    console=False,                   # windowed: без консоли; логи → appdata/ui.log
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    upx=False,
    name="conform-desktop",
)
