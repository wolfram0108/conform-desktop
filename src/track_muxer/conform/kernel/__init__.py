"""Числовое ЯДРО conform — перенос 1-в-1 из research/dropdtw_conform.

ВАЖНО: код этих модулей перенесён БЕЗ изменения логики и констант
(DSYN/OPEN/EXT/пороги и сама динамика Drop-DTW). Менялись ТОЛЬКО импорты
между модулями (на относительные) и убраны тестовые `main()` с привязкой к
синтетическому датасету и matplotlib. Эталон — оригиналы в
research/dropdtw_conform/{_probe_dropdtw,_band_align,_coarse_hough_srm}.py.
Приёмка переноса — совпадение wav по sha256 с conform v8.
"""

from __future__ import annotations
