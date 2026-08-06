# Сборка portable-дистрибутива

```bash
# из корня conform-desktop, окружением по requirements.txt:
cd build
set FFMPEG_DIR=D:\путь\где\лежат\ffmpeg.exe\и\ffprobe.exe
python -m PyInstaller conform-desktop.spec --noconfirm ^
    --workpath ..\_work --distpath ..\dist
```

⚠ **`--workpath` и `--distpath` — обязательны.** Без них PyInstaller кладёт рядом
ДВА одноимённых `conform-desktop.exe`: рабочий в `dist/` и промежуточный в
`build/` (workpath). Промежуточный НЕ запускается — «Failed to load Python DLL
python312.dll», и отличить их по имени невозможно. С этими флагами рабочий
дистрибутив один и лежит в `dist/`, мусор сборки — в `_work/` (обе папки в .gitignore).

**Готовый продукт: `dist/conform-desktop/conform-desktop.exe`** (~5.5 ГБ).
Архив доставки: `7z a -t7z -mx=5 conform-desktop.7z dist/conform-desktop` (~1.7 ГБ).

## Устройство сборки

| Файл | Что |
|---|---|
| `entry_desktop.py` | Точка входа frozen-exe (обёртка над `ui.__main__`) |
| `conform-desktop.spec` | `--onedir`, **windowed** (`console=False`), UPX выключен; `collect_all` для torch/transformers/muq-хвоста/kornia/numba; ffmpeg+ffprobe из `FFMPEG_DIR` |

**Portable-контракт:** `ffmpeg.exe`/`ffprobe.exe` лежат в `_internal/` рядом с exe и
подставляются в `TM_FFMPEG`/`TM_FFPROBE` **до** импорта ядра; данные приложения
(очередь, `ui.ini`, `ui.log`) — в `appdata/` рядом с exe.

## Грабли, на которые уже наступали (не повторять)

0. **Два одноимённых exe.** См. предупреждение о `--workpath`/`--distpath` выше:
   промежуточный `build/build/conform-desktop/conform-desktop.exe` внешне не
   отличим от рабочего, но падает с «Failed to load Python DLL python312.dll».
1. **`ROOT` в spec.** `SPECPATH` = каталог `build/`, поэтому корень репозитория —
   `Path(SPECPATH).resolve().parent`. Ошибка на один уровень вверх → в сборку не
   попадают пакеты `ui`/`server` → `ModuleNotFoundError: No module named 'ui'`,
   а windowed-exe умирает молча (ни окна, ни лога).
2. **Диагностика windowed-сборки.** Чтобы увидеть трейс, соберите тем же спеком
   копию с `console=True` под другим именем — обычный запуск ошибок не показывает.
3. **`sys.stdout`/`sys.stderr` = `None`** в windowed-режиме: любая запись (loguru,
   uvicorn, traceback) валит процесс. Подменяются на `appdata/ui.log` в самом начале
   `ui/__main__.py`, до импорта ядра.
4. **Выход из процесса.** `ConformQueue` держит non-daemon `ThreadPoolExecutor` —
   после закрытия окна обычный возврат из `app.exec()` оставляет процесс висеть.
   Поэтому `os._exit(rc)`. ⚠ Осиротевшие ffmpeg-подпроцессы закроет «надёжная
   отмена» (этап 10а: kill дерева процессов задачи).

## Приёмка сборки (замеры 2026-08-06)

| Проверка | Результат |
|---|---|
| Размер дистрибутива | 5578 МБ (onedir) |
| Старт до появления окна | 5 с |
| `GET /health` после старта | отвечает |
| Консольное окно | отсутствует |
| Закрытие окна → процесс | завершается, остаточных процессов 0, порт 8799 свободен |
