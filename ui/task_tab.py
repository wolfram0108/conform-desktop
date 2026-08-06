"""Вкладка «Задача»: сборка задачи по утверждённому макету.

Референс (+ выбор реф-дорожки, если их >1) → список озвучек (видео с выбором
дорожки / голое аудио / виртуальный дубль) → выходной каталог → настройки
(свёрнуты по умолчанию) → «Добавить в очередь».
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QRadioButton, QScrollArea, QVBoxLayout, QWidget,
)

from ui.client import Api, call
from ui.i18n import tr

AUDIO_EXT = {".flac", ".mka", ".mp3", ".wav", ".aac", ".opus", ".ogg", ".m4a", ".ac3", ".dts"}


def _track_label(t: dict) -> str:
    """Подпись дорожки в комбо: 'a1 · rus · AniLiberty · 2.0'."""
    parts = [f"a{t['index']}"]
    if t.get("lang"):
        parts.append(t["lang"])
    if t.get("title"):
        parts.append(t["title"])
    if t.get("layout"):
        parts.append(t["layout"])
    elif t.get("channels"):
        parts.append(f"{t['channels']}ch")
    return " · ".join(parts)


class DubRow(QFrame):
    """Строка озвучки: имя + (комбо дорожки | метка аудио-only) + удалить."""

    removed = Signal(object)

    def __init__(self, path: str, api: Api) -> None:
        super().__init__()
        self.setObjectName("dubrow")
        self.path = path
        self.is_audio = Path(path).suffix.lower() in AUDIO_EXT
        self._tracks: list[dict] = []

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 10, 6)
        lay.setSpacing(10)

        self.name = QLabel(Path(path).name)
        self.name.setObjectName("mono")
        self.name.setToolTip(path)
        lay.addWidget(self.name, 1)

        self.ref_tag = QLabel(tr("dub.same_as_ref"))
        self.ref_tag.setObjectName("hint")
        self.ref_tag.hide()
        lay.addWidget(self.ref_tag)

        self.combo = QComboBox()
        self.combo.hide()
        lay.addWidget(self.combo)

        self.tag = QLabel(tr("dub.audio_only") if self.is_audio else "")
        self.tag.setObjectName("tag")
        self.tag.setVisible(self.is_audio)
        lay.addWidget(self.tag)

        btn = QPushButton("✕")
        btn.setObjectName("rowbtn")
        btn.clicked.connect(lambda: self.removed.emit(self))
        lay.addWidget(btn)

        if not self.is_audio:
            call(api.atracks, path, done=self._on_tracks)

    def _on_tracks(self, tracks: list[dict]) -> None:
        self._tracks = tracks
        if len(tracks) > 1:
            self.combo.addItems([_track_label(t) for t in tracks])
            self.combo.show()

    def atrack(self) -> int:
        return self.combo.currentIndex() if self.combo.isVisible() else 0

    def mark_virtual(self, is_ref_file: bool) -> None:
        """Пометить строку как виртуальный дубль (тот же файл, что референс)."""
        self.ref_tag.setVisible(is_ref_file)
        if is_ref_file and not self.is_audio:
            self.tag.setText(tr("dub.virtual"))
            self.tag.show()

    def retranslate(self) -> None:
        self.ref_tag.setText(tr("dub.same_as_ref"))
        if self.is_audio:
            self.tag.setText(tr("dub.audio_only"))
        elif self.tag.isVisible():
            self.tag.setText(tr("dub.virtual"))


class TaskTab(QWidget):
    """Форма сборки задачи. enqueued(job_dict) — задача принята ядром."""

    enqueued = Signal(dict)
    toast = Signal(str)

    def __init__(self, api: Api, gpu: bool = True, cfg=None) -> None:
        super().__init__()
        self.api = api
        self.gpu = gpu
        self.cfg = cfg
        self.dub_rows: list[DubRow] = []
        self.setAcceptDrops(True)
        self._build()
        self._load_cfg()

    # ── каркас ──

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        page = QWidget()
        scroll.setWidget(page)
        root = QVBoxLayout(page)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        def frow(label_key: str) -> tuple[QLabel, QHBoxLayout]:
            lay = QHBoxLayout()
            lay.setSpacing(10)
            lab = QLabel(tr(label_key))
            lab.setObjectName("muted")
            lab.setFixedWidth(150)
            lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lay.addWidget(lab)
            root.addLayout(lay)
            return lab, lay

        # референс
        self.l_ref, lay = frow("ref")
        self.ref_edit = QLineEdit()
        self.ref_edit.setObjectName("path")
        self.ref_edit.editingFinished.connect(self._ref_changed)
        lay.addWidget(self.ref_edit, 1)
        self.b_ref = QPushButton(tr("browse"))
        self.b_ref.clicked.connect(self._pick_ref)
        lay.addWidget(self.b_ref)

        # реф-дорожка (видна только при >1)
        self.l_rtrack, lay = frow("ref.track")
        self.ref_combo = QComboBox()
        lay.addWidget(self.ref_combo)
        self.ref_hint = QLabel("")
        self.ref_hint.setObjectName("hint")
        lay.addWidget(self.ref_hint)
        lay.addStretch(1)
        self._rtrack_row = (self.l_rtrack, self.ref_combo, self.ref_hint)
        for w in self._rtrack_row:
            w.hide()

        # озвучки
        self.l_dubs, lay = frow("dubs")
        self.l_dubs.setAlignment(Qt.AlignRight | Qt.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(8)
        lay.addLayout(col, 1)
        self.dub_list = QFrame()
        self.dub_list.setObjectName("card")
        self.dub_col = QVBoxLayout(self.dub_list)
        self.dub_col.setContentsMargins(0, 0, 0, 0)
        self.dub_col.setSpacing(0)
        self.dub_list.hide()
        col.addWidget(self.dub_list)
        add_lay = QHBoxLayout()
        self.b_add = QPushButton(tr("dubs.add"))
        self.b_add.setObjectName("ghost")
        self.b_add.clicked.connect(self._pick_dubs)
        add_lay.addWidget(self.b_add)
        self.l_drop = QLabel(tr("dubs.drop"))
        self.l_drop.setObjectName("hint")
        add_lay.addWidget(self.l_drop)
        add_lay.addStretch(1)
        col.addLayout(add_lay)

        # выходной каталог
        self.l_out, lay = frow("out_dir")
        self.out_edit = QLineEdit()
        self.out_edit.setObjectName("path")
        lay.addWidget(self.out_edit, 1)
        self.b_out = QPushButton(tr("browse"))
        self.b_out.clicked.connect(self._pick_out)
        lay.addWidget(self.b_out)

        # настройки (сворачиваемые)
        _, lay = frow("")
        self.group = QFrame()
        self.group.setObjectName("group")
        glay = QVBoxLayout(self.group)
        glay.setContentsMargins(0, 0, 0, 0)
        glay.setSpacing(0)
        self.g_head = QPushButton("▸ " + tr("settings"))
        self.g_head.setObjectName("link")
        self.g_head.setStyleSheet("text-align:left; padding:8px 14px; font-weight:600;")
        self.g_head.clicked.connect(self._toggle_settings)
        glay.addWidget(self.g_head)
        self.g_body = QWidget()
        gb = QVBoxLayout(self.g_body)
        gb.setContentsMargins(16, 4, 16, 12)
        gb.setSpacing(9)
        glay.addWidget(self.g_body)
        self.g_body.hide()
        lay.addWidget(self.group, 1)

        arow = QHBoxLayout()
        arow.setSpacing(20)
        self.l_analysis = QLabel(tr("set.analysis"))
        self.l_analysis.setObjectName("muted")
        arow.addWidget(self.l_analysis)
        self.r_band = QRadioButton(tr("set.band"))
        self.r_band.setChecked(True)
        arow.addWidget(self.r_band)
        self.r_muq = QRadioButton("MuQ")
        self.r_muq.setVisible(self.gpu)      # без NVIDIA опция скрыта (CHARTER)
        arow.addWidget(self.r_muq)
        arow.addStretch(1)
        gb.addLayout(arow)
        self.l_muq_note = QLabel(tr("set.muq_note"))
        self.l_muq_note.setObjectName("note")
        self.l_muq_note.setWordWrap(True)
        self.l_muq_note.setVisible(self.gpu)
        gb.addWidget(self.l_muq_note)

        self.c_fill = QCheckBox(tr("set.fill"))
        self.c_fill.setChecked(True)
        gb.addWidget(self.c_fill)

        drow = QHBoxLayout()
        drow.setSpacing(10)
        self.l_drift = QLabel(tr("set.drift"))
        self.l_drift.setObjectName("muted")
        drow.addWidget(self.l_drift)
        self.s_drift = QDoubleSpinBox()
        self.s_drift.setRange(0.1, 10.0)
        self.s_drift.setSingleStep(0.25)
        self.s_drift.setValue(1.25)
        drow.addWidget(self.s_drift)
        self.l_drift_u = QLabel(tr("set.drift_unit"))
        self.l_drift_u.setObjectName("muted")
        drow.addWidget(self.l_drift_u)
        drow.addStretch(1)
        gb.addLayout(drow)

        self.c_tmp = QCheckBox(tr("set.keep_tmp"))
        gb.addWidget(self.c_tmp)
        trow = QHBoxLayout()
        trow.setContentsMargins(24, 0, 0, 0)
        self.tmp_edit = QLineEdit()
        self.tmp_edit.setObjectName("path")
        self.tmp_edit.setEnabled(False)
        self.tmp_edit.setMaximumWidth(360)
        trow.addWidget(self.tmp_edit)
        self.b_tmp = QPushButton(tr("browse"))
        self.b_tmp.setEnabled(False)
        self.b_tmp.clicked.connect(self._pick_tmp)
        trow.addWidget(self.b_tmp)
        trow.addStretch(1)
        gb.addLayout(trow)
        self.c_tmp.toggled.connect(self.tmp_edit.setEnabled)
        self.c_tmp.toggled.connect(self.b_tmp.setEnabled)

        # нижний ряд: название + кнопка
        bottom = QHBoxLayout()
        self.c_autostart = QCheckBox(tr("q.autostart"))
        self.c_autostart.setChecked(True)
        bottom.addWidget(self.c_autostart)
        bottom.addStretch(1)
        self.l_label = QLabel(tr("task.label"))
        self.l_label.setObjectName("muted")
        bottom.addWidget(self.l_label)
        self.label_edit = QLineEdit()
        self.label_edit.setMaximumWidth(260)
        bottom.addWidget(self.label_edit)
        self.b_go = QPushButton(tr("task.enqueue"))
        self.b_go.setObjectName("primary")
        self.b_go.clicked.connect(self._enqueue)
        bottom.addWidget(self.b_go)
        root.addLayout(bottom)
        root.addStretch(1)

    # ── сохранение настроек между запусками ──

    def _load_cfg(self) -> None:
        c = self.cfg
        if c is None:
            return
        def b(key, default):
            v = c.value(key, default)
            return v if isinstance(v, bool) else str(v).lower() in ("true", "1")
        self.out_edit.setText(str(c.value("task/out_dir", "")))
        self.tmp_edit.setText(str(c.value("task/cache_dir", "")))
        self.c_fill.setChecked(b("task/fill_silence", True))
        self.c_tmp.setChecked(b("task/keep_tmp", False))
        self.c_autostart.setChecked(b("task/autostart", True))
        if self.gpu and b("task/muq", False):
            self.r_muq.setChecked(True)
        try:
            self.s_drift.setValue(float(c.value("task/drift", 1.25)))
        except (TypeError, ValueError):
            pass

    def _save_cfg(self) -> None:
        c = self.cfg
        if c is None:
            return
        c.setValue("task/out_dir", self.out_edit.text().strip())
        c.setValue("task/cache_dir", self.tmp_edit.text().strip())
        c.setValue("task/fill_silence", self.c_fill.isChecked())
        c.setValue("task/keep_tmp", self.c_tmp.isChecked())
        c.setValue("task/autostart", self.c_autostart.isChecked())
        c.setValue("task/muq", self.r_muq.isChecked())
        c.setValue("task/drift", self.s_drift.value())

    # ── референс ──

    def _pick_ref(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, tr("ref"), "", tr("files.video"))
        if p:
            self.ref_edit.setText(p)
            self._ref_changed()

    def _ref_changed(self) -> None:
        p = self.ref_edit.text().strip()
        for w in self._rtrack_row:
            w.hide()
        self.ref_combo.clear()
        if p:
            if not self.label_edit.text().strip():
                self.label_edit.setText(Path(p).stem[:60])
            call(self.api.atracks, p, done=self._on_ref_tracks)
        for r in self.dub_rows:
            r.mark_virtual(r.path == p)

    def _on_ref_tracks(self, tracks: list[dict]) -> None:
        if len(tracks) > 1:
            self.ref_combo.addItems([_track_label(t) for t in tracks])
            dflt = next((i for i, t in enumerate(tracks) if t.get("default")), 0)
            self.ref_combo.setCurrentIndex(dflt)
            self.ref_hint.setText(tr("tracks.n", n=len(tracks)))
            for w in self._rtrack_row:
                w.show()

    # ── озвучки ──

    def _pick_dubs(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, tr("dubs"), "", tr("files.video"))
        for p in paths:
            self._add_dub(p)

    def _add_dub(self, path: str) -> None:
        row = DubRow(path, self.api)
        row.removed.connect(self._remove_dub)
        row.mark_virtual(path == self.ref_edit.text().strip())
        self.dub_rows.append(row)
        self.dub_col.addWidget(row)
        self.dub_list.show()

    def _remove_dub(self, row: DubRow) -> None:
        self.dub_rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        if not self.dub_rows:
            self.dub_list.hide()

    def dragEnterEvent(self, e) -> None:  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:  # noqa: N802
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if not p:
                continue
            if not self.ref_edit.text().strip():
                self.ref_edit.setText(p)
                self._ref_changed()
            else:
                self._add_dub(p)

    # ── прочее ──

    def _pick_out(self) -> None:
        p = QFileDialog.getExistingDirectory(self, tr("out_dir"))
        if p:
            self.out_edit.setText(p)

    def _pick_tmp(self) -> None:
        p = QFileDialog.getExistingDirectory(self, tr("set.keep_tmp"))
        if p:
            self.tmp_edit.setText(p)

    def _toggle_settings(self) -> None:
        vis = not self.g_body.isVisible()
        self.g_body.setVisible(vis)
        self.g_head.setText(("▾ " if vis else "▸ ") + tr("settings"))

    # ── отправка ──

    def _enqueue(self) -> None:
        ref = self.ref_edit.text().strip()
        out = self.out_edit.text().strip()
        if not ref:
            self.toast.emit(tr("task.need_ref"))
            return
        if not self.dub_rows:
            self.toast.emit(tr("task.need_dubs"))
            return
        if not out:
            self.toast.emit(tr("task.need_out"))
            return
        body = {
            "ref": ref,
            "dubs": [r.path for r in self.dub_rows],
            "dub_atracks": [r.atrack() for r in self.dub_rows],
            "ref_atrack": self.ref_combo.currentIndex() if self.ref_combo.count() else 0,
            "out_dir": out,
            "label": self.label_edit.text().strip() or None,
            "fill_silence": self.c_fill.isChecked(),
            "audio_band": not self.r_muq.isChecked(),
            "audio_muq": self.r_muq.isChecked(),
            "drift_speed_pct": self.s_drift.value(),
            "autostart": self.c_autostart.isChecked(),
            "keep_tmp": self.c_tmp.isChecked(),
            "cache_dir": self.tmp_edit.text().strip() or None if self.c_tmp.isChecked() else None,
        }
        self.b_go.setEnabled(False)
        call(self.api.enqueue, body, done=self._on_enqueued, error=self._on_error)

    def _on_enqueued(self, job: dict) -> None:
        self.b_go.setEnabled(True)
        self._save_cfg()
        self.toast.emit(tr("task.added"))
        self.enqueued.emit(job)
        # форма НЕ очищается целиком: типовой сценарий — следующая серия тем же составом;
        # чистим только озвучки
        for r in list(self.dub_rows):
            self._remove_dub(r)
        self.label_edit.clear()

    def _on_error(self, msg: str) -> None:
        self.b_go.setEnabled(True)
        self.toast.emit(tr("err.api", e=msg))

    # ── язык ──

    def retranslate(self) -> None:
        self.l_ref.setText(tr("ref"))
        self.l_rtrack.setText(tr("ref.track"))
        self.l_dubs.setText(tr("dubs"))
        self.l_out.setText(tr("out_dir"))
        for b in (self.b_ref, self.b_out, self.b_tmp):
            b.setText(tr("browse"))
        self.b_add.setText(tr("dubs.add"))
        self.l_drop.setText(tr("dubs.drop"))
        self.g_head.setText(("▾ " if self.g_body.isVisible() else "▸ ") + tr("settings"))
        self.l_analysis.setText(tr("set.analysis"))
        self.r_band.setText(tr("set.band"))
        self.l_muq_note.setText(tr("set.muq_note"))
        self.c_fill.setText(tr("set.fill"))
        self.l_drift.setText(tr("set.drift"))
        self.l_drift_u.setText(tr("set.drift_unit"))
        self.c_tmp.setText(tr("set.keep_tmp"))
        self.c_autostart.setText(tr("q.autostart"))
        self.l_label.setText(tr("task.label"))
        self.b_go.setText(tr("task.enqueue"))
        if self.ref_combo.count():
            self.ref_hint.setText(tr("tracks.n", n=self.ref_combo.count()))
        for r in self.dub_rows:
            r.retranslate()
