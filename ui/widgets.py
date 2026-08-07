r"""Мелкие виджеты, устойчивые к ДЛИННОМУ содержимому.

Реальные данные длиннее макетных: пути вида
`Z:\Anime\Series test\Башня Бога (2020) [tmdbid-97860]\…s01e02 BDRip 1080p.mkv`
и подписи дорожек вида `a2 · rus · Dejz, Derenn, Kari, Hekomi, MyAska [AniLibria]`.
Обычные QLabel/QComboBox требуют ширину ПО ТЕКСТУ и распирают форму за край окна —
поэтому имя обрезается многоточием, а полный текст живёт в подсказке.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QComboBox, QLabel, QMenu, QSizePolicy, QToolButton


class ElidedLabel(QLabel):
    """QLabel, который обрезает текст многоточием по своей ФАКТИЧЕСКОЙ ширине."""

    def __init__(self, text: str = "", mode: Qt.TextElideMode = Qt.ElideMiddle) -> None:
        super().__init__(text)
        self._full = text
        self._mode = mode
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(60)
        if text:
            self.setToolTip(text)

    def setText(self, text: str) -> None:  # noqa: N802 — Qt-API
        self._full = text
        self.setToolTip(text)
        self._apply()

    def fullText(self) -> str:  # noqa: N802 — Qt-стиль
        return self._full

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        self._apply()

    def _apply(self) -> None:
        fm = QFontMetrics(self.font())
        super().setText(fm.elidedText(self._full, self._mode, max(20, self.width() - 4)))


def tame_combo(combo: QComboBox, max_width: int = 340, chars: int = 8) -> QComboBox:
    """Не давать выпадающему списку растягивать форму по самому длинному пункту."""
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(chars)
    combo.setMaximumWidth(max_width)
    combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    combo.view().setTextElideMode(Qt.ElideRight)
    combo.currentIndexChanged.connect(lambda _i: combo.setToolTip(combo.currentText()))
    return combo


class MultiTrackPicker(QToolButton):
    """Выбор НЕСКОЛЬКИХ дорожек одного файла галочками.

    Сценарий пользователя: в файле 10 озвучек, одна берётся референсом, остальные 9
    надо выровнять по ней — они выбираются здесь разом, а не добавлением файла 9 раз.
    Каждая отмеченная дорожка становится отдельной озвучкой задачи.
    """

    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("picker")
        self.setPopupMode(QToolButton.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.setMaximumWidth(340)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._menu = QMenu(self)
        self.setMenu(self._menu)
        self._acts: list[QAction] = []
        self._labels: list[str] = []
        self._summary = "дорожки"
        self._none = "не выбрано"

    def set_texts(self, summary: str, none_text: str) -> None:
        """Подписи для локализации: «дорожки: 3 из 10» / «не выбрано»."""
        self._summary, self._none = summary, none_text
        self._refresh()

    def set_tracks(self, labels: list[str], checked: list[int] | None = None) -> None:
        self._menu.clear()
        self._acts.clear()
        self._labels = list(labels)
        checked = set(checked or [])
        for i, text in enumerate(labels):
            a = QAction(text, self._menu)
            a.setCheckable(True)
            a.setChecked(i in checked)
            a.toggled.connect(self._on_toggle)
            self._menu.addAction(a)
            self._acts.append(a)
        self._menu.addSeparator()
        all_a = QAction("выбрать все / снять все", self._menu)
        all_a.triggered.connect(self._toggle_all)
        self._menu.addAction(all_a)
        self._refresh()

    def selected(self) -> list[int]:
        return [i for i, a in enumerate(self._acts) if a.isChecked()]

    def _toggle_all(self) -> None:
        target = len(self.selected()) < len(self._acts)
        for a in self._acts:
            a.blockSignals(True)
            a.setChecked(target)
            a.blockSignals(False)
        self._on_toggle()

    def _on_toggle(self, *_a) -> None:
        self._refresh()
        self.changed.emit()

    def _refresh(self) -> None:
        sel = self.selected()
        if not sel:
            self.setText(self._none)
            self.setToolTip("")
            return
        if len(sel) == 1:
            self.setText(self._labels[sel[0]])
        else:
            self.setText(f"{self._summary}: {len(sel)} / {len(self._acts)}")
        self.setToolTip(chr(10).join(self._labels[i] for i in sel))
