"""HTTP-роутер conform (FastAPI APIRouter). Тонкий слой над ConformQueue —
вся логика в очереди/ядре. Подключается в api/app.py через include_router.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from track_muxer.conform.queue import ConformJob, ConformQueue
from track_muxer.conform.scan import (
    CatalogItem,
    SeriesScan,
    list_catalogs,
    scan_catalog,
)


class ConformEnqueueBody(BaseModel):
    """Тело POST /conform/enqueue — одна серия (1 реф + список озвучек)."""

    ref: str
    dubs: list[str]
    out_dir: str | None = None         # None → <output>/_conform_out/<label>
    cache_dir: str | None = None       # подкаталог кеша (<серия>/_conform_cache)
    keep_tmp: bool = False             # tmp-чекпоинты: сохранить ВСЁ промежуточное (реф SRM + дубль
                                       # CK1/2/3) → повтор без GPU-декода; OFF → кеш чистится после серии
    label: str | None = None           # ярлык (по умолч. имя папки рефа)
    fps_ref: str | None = None         # None → авто; '24000/1001' и т.п.
    fps_dub: str | None = None
    free_start: bool = True
    fill_silence: bool = True          # «вырезы оригиналом»: тишину озвучки заполнить рефом ПОСЛЕ
                                       # band/muq (только на синхронной дорожке; при аудио off неактивно)
    audio_band: bool = True            # ДЕФОЛТ: anchor-пайплайн, карта DSP 48 полос (GPU, без модели)
    audio_muq: bool = False            # anchor-пайплайн, карта MuQ (GPU, опц. transformers)
    apply_cuts: bool = True            # band/muq: применять резкую правку резов (иначе только дрейф ≤2%)
    drift_speed_pct: float = 1.25      # band/muq: потолок скорости изменения сдвига кривой дрейфа, %/с
    ref_atrack: int = 0                # ⭐ 5.1: аудиодорожка РЕФА (звуковой эталон band/заливки)
    dub_atracks: list[int] | None = None   # ⭐ 5.1: дорожка КАЖДОЙ озвучки (параллельно dubs;
                                       # None → все 0). «Виртуальный дубль» = тот же файл
                                       # (хоть сам реф) с другой дорожкой.


class ConformSettingsBody(BaseModel):
    limit: int | None = None           # порог параллельности (серий разом)


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
        spec = {
            "ref": str(ref), "dubs": [str(Path(d)) for d in body.dubs],
            "out_dir": out_dir, "cache_dir": body.cache_dir,
            "keep_tmp": body.keep_tmp, "label": label,
            "fps_ref": body.fps_ref, "fps_dub": body.fps_dub,
            "free_start": body.free_start, "fill_silence": body.fill_silence,
            "audio_band": body.audio_band, "audio_muq": body.audio_muq,
            "apply_cuts": body.apply_cuts,
            "drift_speed_pct": body.drift_speed_pct,
            "ref_atrack": body.ref_atrack, "dub_atracks": body.dub_atracks,
        }
        return cq.enqueue(spec)

    @r.get("/atracks")
    def atracks(path: str = Query(..., description="файл: список аудиодорожек для выбора в UI")) -> list[dict]:
        p = Path(path)
        if not p.exists():
            raise HTTPException(400, f"нет файла: {path}")
        from track_muxer.conform.features import probe_audio_tracks
        return probe_audio_tracks(p)

    @r.get("/device")
    def device() -> dict:
        """Устройство аудио-слоя band/muq (для бейджа GPU/CPU в панели)."""
        try:
            import torch
            if torch.cuda.is_available():
                return {"device": "cuda", "name": torch.cuda.get_device_name(0), "gpu": True}
        except Exception:  # noqa: BLE001 — torch/CUDA недоступны → CPU
            pass
        return {"device": "cpu", "name": "CPU", "gpu": False}

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
        """График укладки (PNG-превью/cut ИЛИ интерактивный HTML для модалки):
        <job.out_dir>/_plots/<name>. Имя валидируется."""
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
    def cancel(jid: str) -> dict:
        if not cq.cancel(jid):
            raise HTTPException(404, "нет такой задачи")
        return {"cancelled": jid}

    @r.post("/jobs/{jid}/retry")
    def retry(jid: str) -> dict:
        if not cq.retry(jid):
            raise HTTPException(409, "нельзя перезапустить (нет задачи или ещё активна)")
        return {"retried": jid}

    @r.post("/clear")
    def clear() -> dict:
        return {"cleared": cq.clear()}

    @r.get("/settings")
    def get_settings() -> dict:
        return cq.get_settings()

    @r.put("/settings")
    def put_settings(body: ConformSettingsBody) -> dict:
        return cq.set_settings(body.limit)

    @r.get("/catalogs", response_model=list[CatalogItem])
    def catalogs() -> list[CatalogItem]:
        """Подкаталоги downloads демона — кандидаты-каталоги для выбора."""
        return list_catalogs(cq.output_dir)

    @r.get("/scan", response_model=list[SeriesScan])
    def scan(path: str = Query(..., description="каталог: серии + озвучки + авто-реф + готовность")) -> list[SeriesScan]:
        return scan_catalog(path)

    return r
