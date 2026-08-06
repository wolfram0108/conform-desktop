# conform-desktop

**Русский** | [English](#english)

Portable-приложение для Windows: автоматическое выравнивание аудиодорожек (озвучек,
дубляжей) по референс-видео. Кладёте референс и набор озвучек — получаете выровненные
`.flac`, готовые к сборке в контейнер. Полностью автоматически: без ручной подгонки,
без визуальной сверки.

> ⚠ **Статус: в разработке (pre-release).** Ядро выравнивания рабочее и проверенное;
> графический интерфейс (Qt) и готовые сборки — в процессе. Следите за Releases.

## Что умеет ядро

- **Зрение + слух.** Видео-сопоставление кадров (SRM) строит структуру: масштаб
  (PAL-ускорение), вставки, вырезы. Затем аудио-слой снимает то, чего видео не видит:
  константный сдвиг, дрейф, локальные рассинхроны — и заливает тишину озвучки
  синхронным референсом.
- **Сырой вход.** Файлы «как скачаны»: VFR, телесин 3:2, контейнерные задержки,
  дыры PTS — без предварительной подготовки, не хуже полного перекода в CFR.
- **Аудио-only режим.** Озвучка без видео (голый аудиофайл) выравнивается только
  аудио-слоем.
- **Многодорожечные файлы.** Выбор референс-дорожки, выбор дорожек озвучек,
  выравнивание дорожек внутри одного файла.
- **GPU и CPU.** Один дистрибутив: на NVIDIA — CUDA, без неё тот же полный пайплайн
  на CPU (медленнее, но идентично). Автоматически, без настроек.

## Установка (пока из исходников)

```
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

ffmpeg/ffprobe: положите бинари рядом или задайте пути через переменные окружения
`TM_FFMPEG` / `TM_FFPROBE`.

## Устройство репозитория

| Путь | Что |
|---|---|
| `src/track_muxer/conform/` | Ядро выравнивания. **Здесь не редактируется** — синхронизируется из канонического репозитория (`git subtree`, см. `docs/SYNC.md`) |
| `server/` | Локальный HTTP API поверх ядра (в работе) |
| `ui/` | Qt-интерфейс, ru/en (в работе) |
| `build/` | Сборка portable-дистрибутива, PyInstaller (в работе) |

## Лицензия

GPL-3.0 — см. [LICENSE](LICENSE).

Опциональная модель MuQ (`OpenMuQ/MuQ-large-msd-iter`) не входит в дистрибутив;
при включении фичи веса загружаются с HuggingFace и лицензированы **CC-BY-NC 4.0**
(только некоммерческое использование результата).

---

## English

Portable Windows app for automatic conforming of audio tracks (dubs, voice-overs)
against a reference video. Drop in a reference and a set of audio sources — get
aligned `.flac` files ready for muxing. Fully automatic: no manual adjustment,
no visual verification.

> ⚠ **Status: work in progress (pre-release).** The conforming core is functional
> and battle-tested; the Qt GUI and prebuilt distributions are underway. Watch Releases.

**Core features:** frame-level video matching (structure: speed-up, insertions, cuts)
followed by an audio layer that catches what video can't (constant offset, drift,
local desync) and fills silence gaps from the synced reference; raw input support
(VFR, 3:2 telecine, container delays, PTS gaps — no pre-normalization required);
audio-only mode for video-less tracks; multi-track files (reference track selection,
per-source track selection); one distribution for both GPU (CUDA) and CPU — the same
full pipeline, auto-detected.

**License:** GPL-3.0 (see [LICENSE](LICENSE)). The optional MuQ model weights are
not bundled; when enabled they are downloaded from HuggingFace under **CC-BY-NC 4.0**
(non-commercial use of results only).
