"""Двуязычие ru/en: плоский словарь строк + живое переключение.

Каждый виджет, показывающий текст, реализует `retranslate()`; переключение языка
обходит их без перезапуска. Ключи — по-английски (стабильные идентификаторы).
"""

from __future__ import annotations

_STRINGS: dict[str, dict[str, str]] = {
    "theme.tip": {"ru": "Тема: системная / светлая / тёмная", "en": "Theme: system / light / dark"},
    "tab.task": {"ru": "Задача", "en": "Task"},
    "tab.queue": {"ru": "Очередь", "en": "Queue"},
    "ref": {"ru": "Референс:", "en": "Reference:"},
    "ref.track": {"ru": "реф-дорожка:", "en": "ref track:"},
    "tracks.n": {"ru": "дорожек: {n}", "en": "tracks: {n}"},
    "tracks.chosen": {"ru": "дорожки", "en": "tracks"},
    "tracks.none": {"ru": "не выбрано", "en": "none selected"},
    "ref.rest_as_dubs": {"ru": "+ остальные дорожки в озвучки", "en": "+ other tracks as dubs"},
    "ref.rest_tip": {"ru": "Добавить все дорожки этого файла, кроме эталонной, одной строкой — "
                           "каждая станет отдельной озвучкой",
                     "en": "Add every track of this file except the reference one as a single row — "
                           "each becomes its own dub"},
    "dubs": {"ru": "Озвучки:", "en": "Dubs:"},
    "dubs.add": {"ru": "+ Добавить файлы", "en": "+ Add files"},
    "dubs.drop": {"ru": "или перетащите файлы в окно", "en": "or drop files onto the window"},
    "dub.audio_only": {"ru": "аудио-only", "en": "audio-only"},
    "dub.virtual": {"ru": "виртуальный дубль", "en": "virtual dub"},
    "dub.same_as_ref": {"ru": "= реф", "en": "= ref"},
    "dub.track": {"ru": "дорожка", "en": "track"},
    "out_dir": {"ru": "Выходной каталог:", "en": "Output folder:"},
    "browse": {"ru": "Обзор…", "en": "Browse…"},
    "settings": {"ru": "Настройки", "en": "Settings"},
    "set.analysis": {"ru": "Аудио-анализ:", "en": "Audio analysis:"},
    "set.band": {"ru": "band (стандарт)", "en": "band (default)"},
    "set.muq_note": {
        "ru": "MuQ — экспериментальная модель, нужна NVIDIA-карта. Веса загружаются при "
              "первом включении; лицензия CC-BY-NC 4.0 — только некоммерческое использование результата.",
        "en": "MuQ is an experimental model and requires an NVIDIA GPU. Weights download on "
              "first use; CC-BY-NC 4.0 license — non-commercial use of results only.",
    },
    "set.fill": {"ru": "Заполнять тишину озвучки референсом", "en": "Fill dub silence from reference"},
    "set.drift": {"ru": "Потолок скорости дрейфа:", "en": "Drift speed ceiling:"},
    "set.drift_unit": {"ru": "%/с", "en": "%/s"},
    "set.keep_tmp": {"ru": "Сохранять промежуточные файлы (повтор без пересчёта)",
                     "en": "Keep intermediate files (instant re-runs)"},
    "set.keep_tmp_dir": {"ru": "каталог для промежуточных файлов", "en": "folder for intermediate files"},
    "task.label": {"ru": "Название:", "en": "Label:"},
    "task.enqueue": {"ru": "Добавить в очередь", "en": "Add to queue"},
    "task.added": {"ru": "Задача добавлена в очередь — запустите её кнопкой ▶", "en": "Task added to the queue — press ▶ to start it"},
    "task.added_run": {"ru": "Задача добавлена и запущена", "en": "Task added and started"},
    "task.need_ref": {"ru": "Укажите файл референса", "en": "Choose a reference file"},
    "task.need_dubs": {"ru": "Добавьте хотя бы одну озвучку", "en": "Add at least one dub"},
    "task.n_dubs": {"ru": "озвучек в задаче: {n}", "en": "dubs in task: {n}"},
    "task.need_out": {"ru": "Укажите выходной каталог", "en": "Choose an output folder"},
    "q.parallel": {"ru": "параллельно:", "en": "parallel:"},
    "q.clear_done": {"ru": "Очистить готовые", "en": "Clear finished"},
    "q.empty": {"ru": "Очередь пуста — соберите задачу на вкладке «Задача»",
                "en": "Queue is empty — build a task on the “Task” tab"},
    "q.queued": {"ru": "в очереди", "en": "queued"},
    "q.paused": {"ru": "на паузе", "en": "paused"},
    "q.start_tip": {"ru": "Запустить задачу", "en": "Start the task"},
    "q.pause_tip": {"ru": "Снять с автозапуска (пауза)", "en": "Hold the task (pause)"},
    "q.cleared": {"ru": "Очищено задач: {n}, освобождено {mb} МБ", "en": "Cleared {n} tasks, freed {mb} MB"},
    "q.autostart": {"ru": "запускать сразу после добавления", "en": "start right after adding"},
    "q.done": {"ru": "готово", "en": "done"},
    "q.failed": {"ru": "ошибка", "en": "failed"},
    "q.cancelled": {"ru": "отменено", "en": "cancelled"},
    "q.dub_of": {"ru": "озвучка {i}/{n}", "en": "dub {i}/{n}"},
    "q.expand": {"ru": "▸ развернуть", "en": "▸ expand"},
    "q.collapse": {"ru": "▾ свернуть", "en": "▾ collapse"},
    "d.details": {"ru": "подробнее", "en": "details"},
    "d.folder": {"ru": "папка", "en": "folder"},
    "d.resid": {"ru": "остаток {v} мс", "en": "residual {v} ms"},
    "d.coverage": {"ru": "покрытие {v}", "en": "coverage {v}"},
    "d.suspect": {"ru": "⚠ проверьте результат", "en": "⚠ review the result"},
    "d.skipped": {"ru": "уже было готово", "en": "already done"},
    "m.resid": {"ru": "остаток", "en": "residual"},
    "m.coverage": {"ru": "покрытие", "en": "coverage"},
    "m.assigned": {"ru": "назначено", "en": "assigned"},
    "m.cos": {"ru": "cos", "en": "cos"},
    "m.slope": {"ru": "наклон", "en": "slope"},
    "m.cuts": {"ru": "резы аудио", "en": "audio cuts"},
    "m.max_step": {"ru": "макс", "en": "max"},
    "m.filled": {"ru": "заполнено рефом", "en": "filled from ref"},
    "m.geom": {"ru": "geom", "en": "geom"},
    "m.span": {"ru": "диапазон сдвига", "en": "shift span"},
    "m.mode_audio": {"ru": "режим: аудио-only", "en": "mode: audio-only"},
    "p.track": {"ru": "укладка — весь трек", "en": "alignment — full track"},
    "p.cut": {"ru": "рез @{t} с · {v} мс", "en": "cut @{t} s · {v} ms"},
    "p.open": {"ru": "открыть ↗", "en": "open ↗"},
    "stage.decode": {"ru": "декод", "en": "decode"},
    "stage.extract": {"ru": "аудио", "en": "audio"},
    "stage.coarse": {"ru": "грубый проход", "en": "coarse"},
    "stage.geom": {"ru": "geom", "en": "geom"},
    "stage.band": {"ru": "полоса", "en": "band"},
    "stage.resample": {"ru": "ресэмпл", "en": "resample"},
    "stage.audio": {"ru": "доводка", "en": "refine"},
    "stage.write": {"ru": "запись", "en": "write"},
    "err.api": {"ru": "Ошибка API: {e}", "en": "API error: {e}"},
    "cancel.tip": {"ru": "Отменить задачу (убивает её процессы)", "en": "Cancel the task (kills its processes)"},
    "unit.ms": {"ru": "мс", "en": "ms"},
    "unit.s": {"ru": "с", "en": "s"},
    "files.video": {"ru": "Видео и аудио (*.mkv *.mp4 *.avi *.m2ts *.ts *.webm *.flac *.mka *.mp3 *.wav *.aac *.opus *.ogg);;Все файлы (*)",
                    "en": "Video and audio (*.mkv *.mp4 *.avi *.m2ts *.ts *.webm *.flac *.mka *.mp3 *.wav *.aac *.opus *.ogg);;All files (*)"},
}

_lang = "ru"


def set_lang(lang: str) -> None:
    global _lang
    _lang = lang if lang in ("ru", "en") else "ru"


def get_lang() -> str:
    return _lang


def tr(key: str, **fmt) -> str:
    s = _STRINGS.get(key, {}).get(_lang) or _STRINGS.get(key, {}).get("ru") or key
    return s.format(**fmt) if fmt else s
