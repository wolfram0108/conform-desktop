"""Темы оформления: токены из утверждённого макета (ui_mockup.html) → QSS.

Режимы: system (следует за Windows, живо — по colorSchemeChanged), light, dark.
Все виджеты стилизуются через QSS от одного словаря токенов — палитра меняется
целиком, без точечных правок стилей.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication

LIGHT = {
    "win": "#f7f8fa", "chrome": "#eceef1", "panel": "#ffffff", "ctl": "#fdfdfe",
    "bd": "#d3d7dd", "bdsoft": "#e3e6ea", "ink": "#20242a", "muted": "#5c6470",
    "acc": "#0a6ccb", "accink": "#ffffff", "ok": "#1a7f37",
    "warn": "#9a6700", "warnbg": "#fff3d6", "critbg": "#fde8e8", "crit": "#b42318",
    "thumb": "#f2f6fb",
}
DARK = {
    "win": "#22252b", "chrome": "#1b1e23", "panel": "#282c33", "ctl": "#2e333b",
    "bd": "#3c424c", "bdsoft": "#333941", "ink": "#e6e8ec", "muted": "#9aa2ad",
    "acc": "#4cc2ff", "accink": "#0b2233", "ok": "#3fb950",
    "warn": "#d29922", "warnbg": "#3a3020", "critbg": "#3d2222", "crit": "#f2555a",
    "thumb": "#20242b",
}


def system_is_dark() -> bool:
    return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark


def tokens(mode: str) -> dict:
    """mode: system | light | dark → словарь токенов."""
    if mode == "light":
        return LIGHT
    if mode == "dark":
        return DARK
    return DARK if system_is_dark() else LIGHT


def qss(t: dict) -> str:
    return f"""
QWidget {{
    background: {t['win']}; color: {t['ink']};
    font-family: "Segoe UI Variable", "Segoe UI"; font-size: 10.5pt;
}}
QFrame#toolbar {{ background: {t['chrome']}; border-bottom: 1px solid {t['bdsoft']}; }}
QPushButton#tab {{
    background: transparent; color: {t['muted']}; border: none;
    padding: 6px 16px; border-radius: 7px;
}}
QPushButton#tab:checked {{ background: {t['panel']}; color: {t['ink']}; font-weight: 600; }}
QLabel#badge, QLabel#gpuBadge {{
    background: {t['ctl']}; color: {t['muted']}; border: 1px solid {t['bd']};
    border-radius: 11px; padding: 2px 10px; font-size: 9pt;
}}
QLabel#gpuBadge {{ color: {t['ok']}; }}
QPushButton#seg {{
    background: {t['ctl']}; color: {t['muted']}; border: 1px solid {t['bd']};
    padding: 2px 9px; font-size: 9pt;
}}
QPushButton#seg:checked {{ background: {t['acc']}; color: {t['accink']}; border-color: {t['acc']}; }}
QPushButton#segL {{ border-top-left-radius: 7px; border-bottom-left-radius: 7px; }}
QPushButton#segR {{ border-top-right-radius: 7px; border-bottom-right-radius: 7px; margin-left: -1px; }}
QPushButton#icon {{
    background: {t['ctl']}; color: {t['muted']}; border: 1px solid {t['bd']};
    border-radius: 7px; padding: 2px 8px;
}}
QLineEdit, QComboBox, QAbstractSpinBox {{
    background: {t['ctl']}; border: 1px solid {t['bd']}; border-radius: 6px;
    padding: 5px 9px; selection-background-color: {t['acc']}; selection-color: {t['accink']};
}}
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus {{ border-color: {t['acc']}; }}
QLineEdit#path {{ font-family: "Cascadia Mono", Consolas; font-size: 9pt; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {t['panel']}; border: 1px solid {t['bd']};
    selection-background-color: {t['acc']}; selection-color: {t['accink']};
}}
QPushButton {{
    background: {t['ctl']}; border: 1px solid {t['bd']}; border-radius: 6px;
    padding: 5px 14px;
}}
QPushButton:hover {{ border-color: {t['acc']}; }}
QPushButton:disabled {{ color: {t['muted']}; border-color: {t['bdsoft']}; }}
QPushButton#primary {{
    background: {t['acc']}; color: {t['accink']}; border-color: {t['acc']}; font-weight: 600;
}}
QPushButton#ghost {{ border-style: dashed; color: {t['muted']}; }}
QPushButton#rowbtn {{ padding: 1px 7px; font-size: 9pt; color: {t['muted']}; }}
QPushButton#link {{
    background: transparent; border: none; color: {t['acc']}; padding: 2px 4px; font-size: 9.5pt;
}}
QLabel#muted, QCheckBox#muted {{ color: {t['muted']}; }}
QLabel#hint {{ color: {t['muted']}; font-size: 9pt; }}
QLabel#note {{ color: {t['muted']}; font-size: 8.5pt; }}
QLabel#mono {{ font-family: "Cascadia Mono", Consolas; font-size: 9pt; }}
QFrame#card, QFrame#group {{
    background: {t['panel']}; border: 1px solid {t['bd']}; border-radius: 8px;
}}
QFrame#group {{ border-color: {t['bdsoft']}; }}
QFrame#subrows {{ background: {t['win']}; border: none; border-top: 1px solid {t['bdsoft']}; }}
QFrame#dubrow, QFrame#drow {{ background: transparent; border: none; border-bottom: 1px solid {t['bdsoft']}; }}
QFrame#drowWarn {{ background: {t['warnbg']}; border: none; border-left: 3px solid {t['warn']}; }}
QFrame#drowCrit {{ background: {t['critbg']}; border: none; border-left: 3px solid {t['crit']}; }}
QFrame#detail {{ background: {t['win']}; border: none; border-top: 1px dashed {t['bdsoft']}; }}
QLabel#warnText {{ color: {t['warn']}; font-size: 9pt; }}
QLabel#critText {{ color: {t['crit']}; font-size: 9pt; }}
QLabel#tag {{
    background: {t['thumb']}; color: {t['muted']}; border: 1px solid {t['bdsoft']};
    border-radius: 10px; padding: 1px 8px; font-size: 8.5pt;
}}
QLabel#mkey {{ color: {t['muted']}; font-size: 8pt; letter-spacing: 0.5px; }}
QLabel#thumb {{ background: {t['thumb']}; border: 1px solid {t['bdsoft']}; border-radius: 5px; }}
QLabel#dot {{ font-size: 11pt; }}
QLabel#dot[state="run"] {{ color: {t['acc']}; }}
QLabel#dot[state="wait"] {{ color: {t['muted']}; }}
QLabel#dot[state="ok"] {{ color: {t['ok']}; }}
QLabel#dot[state="warn"] {{ color: {t['warn']}; }}
QLabel#dot[state="crit"] {{ color: {t['crit']}; }}
QProgressBar {{
    background: {t['thumb']}; border: 1px solid {t['bdsoft']}; border-radius: 5px;
    height: 8px; text-align: center; font-size: 1pt; color: transparent;
}}
QProgressBar::chunk {{ background: {t['acc']}; border-radius: 4px; }}
QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {t['bd']}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QRadioButton, QCheckBox {{ spacing: 8px; }}
QRadioButton::indicator, QCheckBox::indicator {{ width: 15px; height: 15px; }}
QToolTip {{ background: {t['panel']}; color: {t['ink']}; border: 1px solid {t['bd']}; }}
"""
