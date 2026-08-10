"""Нарезка готового дистрибутива на компоненты + манифест для тонкого загрузчика.

Зачем нарезка. Наш собственный код весит около двух мегабайт, а дистрибутив — 4.5 ГБ,
и 74% этого веса — torch с библиотеками CUDA, которые меняются раз в полгода. Если
раздавать одним куском, человек скачивает полтора гигабайта ради правки в пару
мегабайт. Разделение по частоте изменений это и лечит.

Манифест намеренно описывает ФАЙЛЫ (имя, размер, SHA-256), а не адреса. Архив можно
положить куда угодно — в облако, на флешку, в сетевую папку; загрузчик ищет его рядом
с собой и проверяет по контрольной сумме. Ссылки (`url`) — лишь подсказка для тех, кто
хочет, чтобы скачалось само.

Запуск:
    python installer/split_dist.py --dist dist/conform-desktop --out installer/packages
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEVENZIP = Path(r"C:\Program Files\7-Zip\7z.exe")

# Состав компонентов. Порядок важен: файл попадает в ПЕРВЫЙ подошедший компонент,
# остаток уходит в "app". Так список остаётся коротким и не разъезжается со сборкой.
COMPONENTS = [
    {"name": "runtime", "title": "Вычислительный рантайм (torch, CUDA)",
     "match": ["_internal/torch/", "_internal/torchvision/", "_internal/torchaudio/",
               "_internal/nvidia/", "_internal/functorch/"]},
    {"name": "ffmpeg", "title": "Декодер медиафайлов (ffmpeg)",
     "match": ["_internal/ffmpeg.exe", "_internal/ffprobe.exe"]},
    {"name": "app", "title": "Приложение", "match": None},   # всё остальное
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(rel: str) -> str:
    r = rel.replace("\\", "/")
    for c in COMPONENTS:
        if c["match"] and any(r.startswith(m) for m in c["match"]):
            return c["name"]
    return "app"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default="dist/conform-desktop")
    ap.add_argument("--out", default="installer/packages")
    ap.add_argument("--version", default=time.strftime("%Y.%m.%d"))
    ap.add_argument("--base-url", default="", help="откуда качать, если хочется автоматически")
    args = ap.parse_args()

    dist = Path(args.dist).resolve()
    out = Path(args.out).resolve()
    if not dist.is_dir():
        print(f"нет каталога {dist}"); return 2
    out.mkdir(parents=True, exist_ok=True)

    # 1. разложить пути по компонентам
    buckets: dict[str, list[str]] = {c["name"]: [] for c in COMPONENTS}
    total = 0
    for f in dist.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(dist))
            buckets[classify(rel)].append(rel)
            total += f.stat().st_size
    print(f"дистрибутив: {total / 2**20:,.0f} МБ, файлов {sum(len(v) for v in buckets.values()):,}")

    # 2. упаковать каждый компонент отдельно (пути — относительно корня дистрибутива,
    #    чтобы распаковка была простым «развернуть поверх»)
    manifest = {"version": args.version, "product": "conform-desktop", "components": []}
    for c in COMPONENTS:
        files = buckets[c["name"]]
        if not files:
            continue
        size = sum((dist / f).stat().st_size for f in files)
        arc = out / f"conform-{c['name']}-{args.version}.7z"
        if arc.exists():
            arc.unlink()
        lst = out / f"_{c['name']}.lst"
        lst.write_text("\n".join(files), encoding="utf-8")
        print(f"  упаковка {c['name']}: {len(files):,} файлов, {size / 2**20:,.0f} МБ …", flush=True)
        t0 = time.time()
        r = subprocess.run([str(SEVENZIP), "a", "-t7z", "-mx=5", "-bso0", "-bsp0",
                            str(arc), f"@{lst}"], cwd=str(dist), check=False)
        lst.unlink(missing_ok=True)
        if r.returncode != 0 or not arc.exists():
            print(f"  ОШИБКА упаковки {c['name']}"); return 3
        print(f"    → {arc.name}: {arc.stat().st_size / 2**20:,.0f} МБ за {time.time() - t0:.0f}с",
              flush=True)
        manifest["components"].append({
            "name": c["name"], "title": c["title"],
            "file": arc.name, "size": arc.stat().st_size, "sha256": sha256(arc),
            "unpacked_mb": round(size / 2**20), "files": len(files),
            "url": (args.base_url.rstrip("/") + "/" + arc.name) if args.base_url else "",
        })

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"\nманифест: {out / 'manifest.json'}")
    for c in manifest["components"]:
        print(f"  {c['name']:8} {c['size'] / 2**20:7,.0f} МБ  {c['sha256'][:16]}…  {c['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
