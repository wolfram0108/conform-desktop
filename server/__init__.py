"""Локальный HTTP API conform-desktop: минимальный FastAPI поверх ядра conform.

Внутренняя шина приложения (Qt UI = клиент), НЕ веб-морда: слушает только
127.0.0.1. Вся логика — в ядре (`track_muxer.conform.routes` + `ConformQueue`);
здесь только сборка приложения и portable-каталог данных рядом с exe.
"""
