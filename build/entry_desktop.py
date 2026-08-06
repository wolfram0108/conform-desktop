"""Точка входа frozen conform-desktop.exe: тонкая обёртка над ui.__main__.

Весь frozen-бутстрап (ffmpeg рядом с exe, лог-файл вместо stdout/stderr,
жёсткий выход) живёт в ui/__main__.py — здесь только вызов.
"""

from ui.__main__ import main

if __name__ == "__main__":
    main()
