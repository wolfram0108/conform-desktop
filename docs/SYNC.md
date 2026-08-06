# Синхронизация ядра conform (git subtree)

Ядро `src/track_muxer/conform/` — НЕ редактируется в этом репозитории. Канон живёт в
приватном репозитории `track-muxer` (`src/track_muxer/conform/` — путь ОБЯЗАН совпадать,
иначе ломаются абсолютные импорты `from track_muxer.conform.… import …`).

Синк — одно-направленный, squash (история канона не тянется):

```bash
# первый раз (уже сделано):
git subtree add  --squash --prefix=src/track_muxer/conform <track-muxer-url-или-путь> main:src/track_muxer/conform  # см. ниже

# обновление ядра до текущего main track-muxer:
git subtree pull --squash --prefix=src/track_muxer/conform <track-muxer-url-или-путь> main
```

⚠ Стандартный `git subtree` тянет ветку ЦЕЛИКОМ и кладёт её корень в prefix. Нам нужен
ПОДКАТАЛОГ ветки (`src/track_muxer/conform` → `src/track_muxer/conform`). Это делается
через split на стороне канона:

```bash
# в клоне track-muxer: выделить историю подкаталога во временную ветку
git subtree split --prefix=src/track_muxer/conform main -b conform-split

# в conform-desktop: добавить/обновить из этой ветки
git subtree add  --squash --prefix=src/track_muxer/conform <путь-к-track-muxer> conform-split   # первый раз
git subtree pull --squash --prefix=src/track_muxer/conform <путь-к-track-muxer> conform-split   # обновление
```

Правило: любые изменения алгоритма → сначала в track-muxer (со стендами и регрессом),
затем subtree pull сюда. Патчи к ядру в этом репозитории не принимаются (PR в ядро —
только через канон).
