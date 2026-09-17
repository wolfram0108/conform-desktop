"""Прунинг собранного дистрибутива: убрать то, что продукту не нужно.

Зачем. PyInstaller собирает пакеты «как есть»: вместе с рабочими DLL в дистрибутив
попадают артефакты линковки (`.lib`), заголовки C++, тесты, отладочные ресурсы движка
страницы и переводы на сотню языков. Это гигабайты, которые пользователь скачивает
и хранит зря.

⛔ Рамка: вариант «без GPU» ЗАПРЕЩЁН. Дистрибутив ОДИН универсальный — на машине с
NVIDIA работает через CUDA, без неё тот же полный путь на процессоре. Поэтому ни одна
группа ниже не отключает функциональность: срезается только то, что не исполняется
ни в одном из двух режимов.

Группы (`--levels`), по возрастанию риска:

  A  мусор сборки        `.lib`, `torch/include`, `torch/bin`, тесты пакетов
  W  ресурсы движка      отладочные `.pak`, инструменты разработчика, чужие локали
  D  неиспользуемый Qt   Quick/Qml/3D/Charts и прочие модули, которых нет в карте загрузки
  B  CUDA без импортёров DLL, которых нет ни в одной таблице импорта torch
  E  прочее              видео-модуль OpenCV, AVIF, CTC-декодер, лишние шрифты
  C1 cudnn_adv           слой RNN/attention — не зовут ни LoFTR, ни MuQ
  C2 cudnn engines       ⛔ НЕ ПРИМЕНЯТЬ: замер показал, что без них LoFTR падает

Проверено прогонами (SHA-256 выхода совпал с эталоном на band, MuQ, geom и без GPU):
рабочий набор групп — `A,W,D,B,E,C1`, это −1464 МБ (6073 → 4610 МБ).

Критерий безопасности для A/W/D/B/E — файл не открывался за живой прогон (карта загруженных
модулей) И не является статической зависимостью
оставшегося. Для C1/C2 карта бесполезна: `torch/__init__.py` грузит ВСЁ из `torch/lib`
по маске, поэтому там судья только один — прогон после удаления.

Запуск (по умолчанию ничего не удаляет, только считает):
    python build/prune_dist.py --dist dist/conform-desktop --levels A,W,D,B,E
    python build/prune_dist.py --dist dist/conform-desktop --levels A,W,D,B,E --apply
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from pathlib import Path

# Языки, которые продукт показывает (интерфейс ru/en) — их переводы остаются.
KEEP_LOCALES = ("ru", "en", "en-US", "en-GB")

# ── правила: (группа, комментарий, предикат по пути относительно _internal) ──
# Пути сравниваются в нижнем регистре с прямыми слэшами.

_A_GLOBS = [
    "torch/lib/*.lib",                     # артефакт линковки: цикл загрузки берёт только *.dll
    "torch/include/*",                     # заголовки C++ — нужны сборке расширений, не запуску
    "torch/bin/*",                         # дубли DLL + protoc
    "torch/testing/*", "torch/utils/benchmark/*",
    "numba/tests/*", "llvmlite/tests/*", "numpy/tests/*", "scipy/*/tests/*",
    "sklearn/tests/*", "pandas/tests/*",
    "*.pdb", "*.cmake", "*.a",
]

_W_GLOBS = [
    "pyside6/resources/*.debug.pak",       # отладочные дубликаты обычных .pak
    "pyside6/resources/*.debug.bin",
    "pyside6/resources/qtwebengine_devtools_resources.pak",   # панель разработчика в окне
]

# ⚠ Урок эксперимента (2026-08-07): список модулей Qt «на глаз» составлять НЕЛЬЗЯ.
# Движок страницы построен на Qt Quick — удаление Qt6Quick/Qt6Qml роняет его импорт,
# хотя интерфейс у нас на виджетах. Поэтому группа D определяется НЕ списком, а картой
# загрузки живого прогона: удаляется то, что процесс не открыл ни разу.
#
# Замер (карта загрузки живого прогона, 2026-08-07): продукт открывает из PySide6 ровно
# эти модули. Список зафиксирован здесь, чтобы сборка не зависела от стенда; перепроверить
# его можно ключом --loaded-map с новой картой.
_D_KEEP_NAMES = {
    "qt6core", "qt6gui", "qt6widgets", "qt6network", "qt6opengl", "qt6positioning",
    "qt6printsupport", "qt6qml", "qt6qmlmeta", "qt6qmlmodels", "qt6qmlworkerscript",
    "qt6quick", "qt6quickwidgets", "qt6webchannel",         # ⭐ движок страницы стоит на Qt Quick
    "qt6webenginecore", "qt6webenginewidgets",
    "qtcore", "qtgui", "qtwidgets", "qtnetwork", "qtprintsupport", "qtwebchannel",
    "qtwebenginecore", "qtwebenginewidgets",
    "pyside6.abi3", "msvcp140", "msvcp140_1", "msvcp140_2", "vcruntime140",
    "vcruntime140_1", "concrt140", "msvcp140_codecvt_ids",
}

# Белый список — то, что карта не видит, но продукту нужно в отложенных сценариях:
_D_KEEP_GLOBS = [
    "pyside6/opengl32sw.dll",              # софтовый OpenGL: машины без GL-драйвера (RDP,
                                           # виртуалки) — иначе окно движка останется чёрным
    "pyside6/plugins/platforms/*",         # платформенные плагины окна
    "pyside6/plugins/styles/*",            # оформление окна
    "pyside6/plugins/imageformats/*",      # иконки и картинки графиков
    "pyside6/plugins/iconengines/*",
    "pyside6/plugins/tls/*",               # https: загрузка весов модели
    "pyside6/plugins/networkinformation/*",
    "pyside6/qtwebengineprocess.exe",      # отдельный процесс движка
]
_D_DIRS: list[str] = []

_E_GLOBS = [
    "cv2/opencv_videoio_ffmpeg*.dll",      # VideoCapture не используется: декод — ffmpeg-процессом
    "pil/_avif*.pyd",                      # графика продукта — PNG
    "torchaudio/lib/libctc_prefix_decoder*",   # распознавание речи не нужно
    "plotly/package_data/widgetbundle.js",     # виджет для тетрадей
]

_B_NAMES = [                               # нет ни в одной таблице импорта (см. §2 отчёта)
    "cusolvermg64_11", "curand64_10", "cufftw64_11", "nvtoolsext64_1",
    "zlibwapi", "libiompstubs5md",
]
_B_GLOBS = ["torch/lib/cupti64_*.dll"]

_C1_NAMES = ["cudnn_adv64_9"]

# ⛔ ЗАПРЕТ по результатам замера 2026-08-07 (три прогона geom-пары `Fronda ep01`):
#   без `cudnn_engines_precompiled64_9` (562 МБ) — geom_used=false, назначено 0%;
#   без `cudnn_heuristic64_9` (85 МБ)          — то же самое.
# Гипотеза «cuDNN уйдёт на компиляцию ядер на лету через nvrtc» ОПРОВЕРГНУТА: LoFTR
# падает внутри try/except, geom-второй-шанс не срабатывает, и ВАЛИДНЫЙ дубль
# отбрасывается как «чужое видео». Группа оставлена только для повторной проверки —
# в сборке НЕ применять.
_C2_NAMES = ["cudnn_engines_precompiled64_9", "cudnn_heuristic64_9"]


def _rel(p: Path, root: Path) -> str:
    return str(p.relative_to(root)).replace("\\", "/").lower()


def _locale_keep(name: str) -> bool:
    """Оставить ли файл перевода: имя вида `qt_ru.qm` / `ru.pak` / `en-US.pak`."""
    stem = Path(name).stem.lower()
    stem = stem[3:] if stem.startswith("qt_") else stem
    stem = stem.split("_")[-1] if stem.startswith(("qtbase", "qtdeclarative")) else stem
    return any(stem == k.lower() or stem.startswith(k.lower() + "-") for k in KEEP_LOCALES)


def classify(rel: str, name: str, loaded: set[str]) -> str | None:
    """Вернуть группу, к которой относится файл, или None (файл остаётся).

    `loaded` — множество путей (относительно `_internal`, нижний регистр) из карты
    загрузки живого прогона. Для группы D это единственный судья; для torch/lib карта
    бесполезна (там всё грузится по маске) — те группы заданы точечно.
    """
    stem = Path(name).stem.lower()

    if any(fnmatch.fnmatch(rel, g) for g in _A_GLOBS):
        return "A"

    if any(fnmatch.fnmatch(rel, g) for g in _W_GLOBS):
        return "W"
    if rel.startswith("pyside6/translations/qtwebengine_locales/") and not _locale_keep(name):
        return "W"
    if (rel.startswith("pyside6/translations/") and rel.endswith(".qm")
            and not _locale_keep(name)):
        return "W"

    if (rel.startswith("pyside6/")
            and rel not in loaded and stem not in _D_KEEP_NAMES
            and not any(fnmatch.fnmatch(rel, g) for g in _D_KEEP_GLOBS)
            and rel.endswith((".dll", ".pyd", ".exe", ".qml", ".qmltypes", ".qmlc", ".metainfo"))):
        return "D"

    if stem in _B_NAMES or any(fnmatch.fnmatch(rel, g) for g in _B_GLOBS):
        return "B"

    if any(fnmatch.fnmatch(rel, g) for g in _E_GLOBS):
        return "E"

    if stem in _C1_NAMES:
        return "C1"
    if stem in _C2_NAMES:
        return "C2"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", required=True, help="каталог дистрибутива (с _internal внутри)")
    ap.add_argument("--levels", default="A,W,D,B,E,C1", help="группы через запятую")
    ap.add_argument("--apply", action="store_true", help="действительно удалить (иначе только счёт)")
    ap.add_argument("--copy-to", help="сначала скопировать дистрибутив сюда и резать копию")
    ap.add_argument("--loaded-map", help="JSON карты загрузки (стенд baseline_dllmap.py) — "
                                         "судья для группы D")
    a = ap.parse_args()

    loaded: set[str] = set()
    if a.loaded_map:
        import json
        doc = json.loads(Path(a.loaded_map).read_text(encoding="utf-8"))
        for item in doc.get("loaded", []):
            p = item["path"].replace("\\", "/").lower()
            loaded.add(p[len("_internal/"):] if p.startswith("_internal/") else p)
        print(f"карта загрузки: {len(loaded)} модулей ({a.loaded_map})")

    dist = Path(a.dist).resolve()
    if a.copy_to:
        dst = Path(a.copy_to).resolve()
        if dst.exists():
            print(f"ОШИБКА: {dst} уже существует", file=sys.stderr)
            return 2
        print(f"копирую {dist} → {dst} …", flush=True)
        shutil.copytree(dist, dst)
        dist = dst

    internal = dist / "_internal"
    if not internal.is_dir():
        print(f"ОШИБКА: не вижу {internal}", file=sys.stderr)
        return 2

    want = {s.strip().upper() for s in a.levels.split(",") if s.strip()}
    stats: dict[str, list[int]] = {}
    victims: list[Path] = []

    for p in internal.rglob("*"):
        if not p.is_file():
            continue
        g = classify(_rel(p, internal), p.name, loaded)
        if not g:
            continue
        sz = p.stat().st_size
        s = stats.setdefault(g, [0, 0])
        s[0] += 1
        s[1] += sz
        if g in want:
            victims.append(p)

    total_before = sum(f.stat().st_size for f in dist.rglob("*") if f.is_file())
    print(f"\nдистрибутив: {dist}")
    print(f"размер сейчас: {total_before / 2**20:,.0f} МБ\n")
    print(f"{'группа':>6} {'файлов':>8} {'МБ':>10}   применяется")
    cut = 0
    for g in ("A", "W", "D", "B", "E", "C1", "C2"):
        if g not in stats:
            continue
        n, sz = stats[g]
        mark = "ДА" if g in want else "—"
        if g in want:
            cut += sz
        print(f"{g:>6} {n:>8} {sz / 2**20:>10,.1f}   {mark}")
    print(f"\nсрезается: {cut / 2**20:,.1f} МБ, станет {(total_before - cut) / 2**20:,.0f} МБ")

    if not a.apply:
        print("\n(режим счёта; для удаления добавить --apply)")
        return 0

    removed = 0
    for p in victims:
        try:
            sz = p.stat().st_size
            p.unlink()
            removed += sz
        except OSError as e:
            print(f"  не удалось {p}: {e}", flush=True)
    for d in sorted((d for d in internal.rglob("*") if d.is_dir()),
                    key=lambda x: -len(x.parts)):
        try:
            d.rmdir()                       # пустые каталоги после чистки
        except OSError:
            pass
    total_after = sum(f.stat().st_size for f in dist.rglob("*") if f.is_file())
    print(f"\nудалено {removed / 2**20:,.1f} МБ; дистрибутив {total_after / 2**20:,.0f} МБ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
