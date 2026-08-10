# Сборка тонкого загрузчика. Он намеренно лёгкий: ни torch, ни Qt, ни ядра conform —
# только окно (tkinter входит в состав Python), распаковщик 7-Zip и материал самопроверки.
# Тяжёлые части приезжают отдельными архивами и проверяются по контрольной сумме.
from pathlib import Path

ROOT = Path(SPECPATH)
SEVENZIP = Path(r"C:\Program Files\7-Zip")

datas = [
    (str(SEVENZIP / "7z.exe"), "."),
    (str(SEVENZIP / "7z.dll"), "."),
    (str(ROOT / "selfcheck.json"), "."),
    (str(ROOT / "selfcheck" / "ref_check.mkv"), "."),
    (str(ROOT / "selfcheck" / "dub_check.mkv"), "."),
]

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    excludes=["torch", "numpy", "scipy", "PySide6", "matplotlib", "PIL", "cv2",
              "transformers", "pandas", "sklearn", "numba", "llvmlite"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="conform-setup",
    console=False,          # окно, а не консоль
    upx=False,
    icon=None,
)
