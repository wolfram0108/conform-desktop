"""Qt-интерфейс conform-desktop (PySide6, ru/en).

Морда = клиент REST локального API (`server/`): всё состояние живёт в ядре
(ConformQueue), UI только отображает и дёргает HTTP — API-паритет, как у
веб-панели демона. Макет утверждён 2026-08-06:
doc/missions/conform-standalone-app/ui_mockup.html (в репозитории track-muxer).
"""
