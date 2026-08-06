"""HTTP-клиент локального API + обёртка фоновых вызовов для Qt.

Сеть НИКОГДА не трогается из GUI-потока: каждый вызов уходит в QThreadPool
(`call(...)`), результат приходит сигналом. Клиент — тонкий stdlib-urllib,
без зависимостей сверх манифеста.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class Api:
    """Синхронный клиент (вызывать только из воркеров)."""

    def __init__(self, base: str = "http://127.0.0.1:8799") -> None:
        self.base = base

    def _json(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def _bytes(self, path: str) -> bytes:
        with urllib.request.urlopen(self.base + path, timeout=30) as r:
            return r.read()

    # ── эндпоинты ядра ──
    def health(self) -> dict:
        return self._json("GET", "/health")

    def device(self) -> dict:
        return self._json("GET", "/conform/device")

    def atracks(self, path: str) -> list[dict]:
        return self._json("GET", "/conform/atracks?path=" + urllib.parse.quote(path))

    def enqueue(self, body: dict) -> dict:
        return self._json("POST", "/conform/enqueue", body)

    def jobs(self) -> list[dict]:
        return self._json("GET", "/conform/jobs")

    def cancel(self, jid: str) -> dict:
        return self._json("DELETE", f"/conform/jobs/{jid}")

    def get_settings(self) -> dict:
        return self._json("GET", "/conform/settings")

    def set_limit(self, limit: int) -> dict:
        return self._json("PUT", "/conform/settings", {"limit": limit})

    def start(self, jid: str) -> dict:
        return self._json("POST", f"/conform/jobs/{jid}/start")

    def pause(self, jid: str) -> dict:
        return self._json("POST", f"/conform/jobs/{jid}/pause")

    def clear_done(self) -> dict:
        return self._json("POST", "/conform/clear_done")

    def clear(self) -> dict:
        return self._json("POST", "/conform/clear")

    def plot_png(self, jid: str, name: str) -> bytes:
        return self._bytes(f"/conform/plot/{jid}/{name}")

    def plot_url(self, jid: str, name: str) -> str:
        """URL интерактивного HTML-графика — открывается в системном браузере."""
        return f"{self.base}/conform/plot/{jid}/{name}"


class _Signals(QObject):
    done = Signal(object)
    error = Signal(str)


class _Worker(QRunnable):
    def __init__(self, fn, args, kwargs) -> None:
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = _Signals()

    def run(self) -> None:  # noqa: D102
        try:
            self.signals.done.emit(self.fn(*self.args, **self.kwargs))
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode()).get("detail", str(e))
            except Exception:  # noqa: BLE001
                detail = str(e)
            self.signals.error.emit(str(detail))
        except Exception as e:  # noqa: BLE001
            self.signals.error.emit(str(e))


def call(fn, *args, done=None, error=None, **kwargs) -> None:
    """Выполнить fn(*args) в пуле потоков; done(result)/error(str) — в GUI-потоке."""
    w = _Worker(fn, args, kwargs)
    if done is not None:
        w.signals.done.connect(done)
    if error is not None:
        w.signals.error.connect(error)
    QThreadPool.globalInstance().start(w)
