"""GPU memory of the conform worker while it runs: handing the cache back between dubs.

The caching allocator of torch keeps every segment it has ever taken from the driver. Jobs run side
by side in one worker process, so the peak of a finished dub would otherwise stay reserved while the
others go on, and the next process that needs the card (an NVDEC session of a decode) fails to create
its context. The worker calls `release()` when a dub or a job ends: an event, not a timer.

This is not how the card is freed at the end: the CUDA context and the loaded models live as long
as the process, and the worker process exits once the queue has no job running (conform/worker.py).

No torch or no CUDA — every call here is a no-op, so the CPU path is never held up by it.
"""

from __future__ import annotations

from loguru import logger


def _cuda():
    """The torch module when a CUDA device is really there, otherwise None."""
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 — no torch in the build, or a broken driver: the CPU path stands
        return None


def release() -> int:
    """Give the driver back every cached block no live tensor is using. -> bytes freed."""
    t = _cuda()
    if t is None:
        return 0
    before = t.cuda.memory_reserved()
    t.cuda.empty_cache()
    freed = before - t.cuda.memory_reserved()
    if freed:
        logger.info("gpu: драйверу возвращено {:.0f} МБ, за процессом осталось {:.0f} МБ "
                    "(из них под данными {:.0f} МБ)",
                    freed / 1024 ** 2, t.cuda.memory_reserved() / 1024 ** 2,
                    t.cuda.memory_allocated() / 1024 ** 2)
    return freed
