"""Запуск локального API: `python -m server [порт]`.

Слушает ТОЛЬКО 127.0.0.1 (внутренняя шина Qt-приложения, не сетевой сервис).
Порт: аргумент CLI → env CONFORM_PORT → 8799.
"""

from __future__ import annotations

import os
import sys

import uvicorn

from server.app import create_app


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("CONFORM_PORT", "8799"))
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
