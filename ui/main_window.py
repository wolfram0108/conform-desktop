"""Главное окно: тулбар (вкладки, бейдж GPU, язык, тема) + поллинг очереди.

Тема: system (следует за Windows живо) → light → dark по кнопке-циклу.
Язык и тема сохраняются в ini рядом с exe (portable).
"""

from __future__ import annotations

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from ui import i18n, theme
from ui.client import Api, call
from ui.i18n import tr
from ui.queue_tab import QueueTab
from ui.task_tab import TaskTab

_THEME_ICONS = {"system": "◐", "light": "☀", "dark": "☾"}
_THEME_ORDER = ["system", "light", "dark"]


class MainWindow(QMainWindow):
    def __init__(self, api: Api, settings: QSettings) -> None:
        super().__init__()
        self.api = api
        self.cfg = settings
        self.setWindowTitle("conform-desktop")
        self.resize(int(self.cfg.value("ui/geometry_w", 1120)),
                    int(self.cfg.value("ui/geometry_h", 720)))   # длинные пути → окно пошире

        i18n.set_lang(str(self.cfg.value("ui/lang", "ru")))
        self.theme_mode = str(self.cfg.value("ui/theme", "system"))

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── тулбар ──
        bar = QFrame()
        bar.setObjectName("toolbar")
        root.addWidget(bar)
        h = QHBoxLayout(bar)
        h.setContentsMargins(12, 6, 12, 6)
        h.setSpacing(8)

        self.b_task = QPushButton(tr("tab.task"))
        self.b_queue = QPushButton(tr("tab.queue"))
        grp = QButtonGroup(self)
        for i, b in enumerate((self.b_task, self.b_queue)):
            b.setObjectName("tab")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            grp.addButton(b, i)
            h.addWidget(b)
        self.b_task.setChecked(True)
        grp.idClicked.connect(self._switch_tab)
        h.addStretch(1)

        self.gpu_badge = QLabel("…")
        self.gpu_badge.setObjectName("badge")
        self.gpu_badge.setMaximumWidth(280)
        h.addWidget(self.gpu_badge)

        self.b_ru = QPushButton("RU")
        self.b_en = QPushButton("EN")
        lg = QButtonGroup(self)
        for b, oid in ((self.b_ru, "segL"), (self.b_en, "segR")):
            b.setObjectName("seg")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            lg.addButton(b)
            h.addWidget(b)
        self.b_ru.setObjectName("seg")  # QSS: seg + позиционные скругления через segL/segR не
        self.b_en.setObjectName("seg")  # комбинируются с одним objectName — оставляем единый стиль
        (self.b_ru if i18n.get_lang() == "ru" else self.b_en).setChecked(True)
        self.b_ru.clicked.connect(lambda: self._set_lang("ru"))
        self.b_en.clicked.connect(lambda: self._set_lang("en"))

        self.b_theme = QPushButton(_THEME_ICONS.get(self.theme_mode, "◐"))
        self.b_theme.setObjectName("icon")
        self.b_theme.setCursor(Qt.PointingHandCursor)
        self.b_theme.clicked.connect(self._cycle_theme)
        h.addWidget(self.b_theme)

        # ── вкладки ──
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.gpu = True
        self.task_tab = TaskTab(api, gpu=True, cfg=self.cfg)
        self.queue_tab = QueueTab(api)
        self.stack.addWidget(self.task_tab)
        self.stack.addWidget(self.queue_tab)

        self.task_tab.toast.connect(lambda m: self.statusBar().showMessage(m, 5000))
        self.queue_tab.toast.connect(lambda m: self.statusBar().showMessage(m, 6000))
        self.task_tab.enqueued.connect(lambda _j: self._switch_tab(1, click=True))

        # ── живые данные ──
        call(self.api.device, done=self._on_device)
        self.poll = QTimer(self)
        self.poll.setInterval(1000)
        self.poll.timeout.connect(self._poll)
        self.poll.start()
        self._polling = False

        QGuiApplication.styleHints().colorSchemeChanged.connect(self._sys_scheme_changed)
        self.apply_theme()
        # Минимум окна = реальный минимум формы: Qt не позволит сузить его до состояния,
        # когда поля и кнопки уезжают за край (болезнь длинных путей/названий дорожек).
        QTimer.singleShot(0, self._apply_min_width)

    def _apply_min_width(self) -> None:
        """Не дать сузить окно до состояния, когда поля и кнопки уезжают за край."""
        form = self.task_tab.ref_edit.parentWidget()          # страница формы задачи
        need = form.minimumSizeHint().width() + 48            # поля + вертикальный скроллбар
        self.setMinimumWidth(max(720, min(1100, need)))

    # ── вкладки ──

    def _switch_tab(self, idx: int, click: bool = False) -> None:
        if click:
            (self.b_task, self.b_queue)[idx].setChecked(True)
        self.stack.setCurrentIndex(idx)

    # ── устройство ──

    def _on_device(self, d: dict) -> None:
        self.gpu = bool(d.get("gpu"))
        self.gpu_badge.setText(("GPU · " + d.get("name", "")) if self.gpu else "CPU")
        self.gpu_badge.setObjectName("gpuBadge" if self.gpu else "badge")
        self.task_tab.r_muq.setVisible(self.gpu)
        self.task_tab.l_muq_note.setVisible(self.gpu)
        self.apply_theme()      # objectName сменился — перекрасить

    # ── поллинг очереди ──

    def _poll(self) -> None:
        if self._polling:
            return
        self._polling = True
        call(self.api.jobs, done=self._on_jobs, error=self._on_poll_err)

    def _on_jobs(self, jobs: list[dict]) -> None:
        self._polling = False
        self.queue_tab.update_jobs(jobs)
        n = self.queue_tab.active_count(jobs)
        self.b_queue.setText(tr("tab.queue") + (f"  ●{n}" if n else ""))

    def _on_poll_err(self, _msg: str) -> None:
        self._polling = False

    # ── язык / тема ──

    def _set_lang(self, lang: str) -> None:
        i18n.set_lang(lang)
        self.cfg.setValue("ui/lang", lang)
        self.cfg.sync()
        self.b_task.setText(tr("tab.task"))
        self.b_queue.setText(tr("tab.queue"))
        self.task_tab.retranslate()
        self.queue_tab.retranslate()

    def _cycle_theme(self) -> None:
        i = _THEME_ORDER.index(self.theme_mode) if self.theme_mode in _THEME_ORDER else 0
        self.theme_mode = _THEME_ORDER[(i + 1) % len(_THEME_ORDER)]
        self.cfg.setValue("ui/theme", self.theme_mode)
        self.cfg.sync()
        self.b_theme.setText(_THEME_ICONS[self.theme_mode])
        self.apply_theme()

    def _sys_scheme_changed(self) -> None:
        if self.theme_mode == "system":
            self.apply_theme()

    def closeEvent(self, e) -> None:  # noqa: N802 — Qt-API
        """Дописать настройки на диск до жёсткого выхода процесса."""
        self.cfg.setValue("ui/geometry_w", self.width())
        self.cfg.setValue("ui/geometry_h", self.height())
        self.task_tab._save_cfg()
        self.cfg.sync()
        super().closeEvent(e)

    def apply_theme(self) -> None:
        QApplication.instance().setStyleSheet(theme.qss(theme.tokens(self.theme_mode)))
