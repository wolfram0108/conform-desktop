r"""Мелкие виджеты, устойчивые к ДЛИННОМУ содержимому.

Реальные данные длиннее макетных: пути вида
`Z:\Anime\Series test\Башня Бога (2020) [tmdbid-97860]\…s01e02 BDRip 1080p.mkv`
и подписи дорожек вида `a2 · rus · Dejz, Derenn, Kari, Hekomi, MyAska [AniLibria]`.
Обычные QLabel/QComboBox требуют ширину ПО ТЕКСТУ и распирают форму за край окна —
поэтому имя обрезается многоточием, а полный текст живёт в подсказке.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QComboBox, QLabel, QSizePolicy


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
