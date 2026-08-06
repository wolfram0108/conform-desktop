"""Вкладка «Очередь»: карточки задач с живым прогрессом и результатами.

Данные приходят из поллинга `GET /conform/jobs` (раз в секунду, MainWindow).
Карточки обновляются НА МЕСТЕ (без пересборки списка) — раскрытые аккордеоны
и скролл не сбрасываются. В строке готовой озвучки — остаток + покрытие;
остальные метрики и предпросмотры графиков — в аккордеоне «подробнее»
(решение пользователя 2026-08-06).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from ui.client import Api, call
from ui.i18n import tr

_STATUS_DOT = {"running": "run", "queued": "wait", "done": "ok",
               "failed": "crit", "cancelled": "wait"}


def _dot(state: str) -> QLabel:
    d = QLabel("●")
    d.setObjectName("dot")
    d.setProperty("state", state)
    return d


def _set_dot(d: QLabel, state: str) -> None:
    if d.property("state") != state:
        d.setProperty("state", state)
        d.style().unpolish(d)
        d.style().polish(d)


def _fmt_elapsed(s: float) -> str:
    s = int(s)
    return f"{s // 60}м{s % 60:02d}с" if s >= 60 else f"{s}с"


class _PngDialog(QDialog):
    """Крупный просмотр PNG-графика внутри приложения."""

    def __init__(self, parent, title: str, png: bytes) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        pm = QPixmap()
        pm.loadFromData(png)
        lab = QLabel()
        if pm.width() > 1400:
            pm = pm.scaledToWidth(1400, Qt.SmoothTransformation)
        lab.setPixmap(pm)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(lab)
        lay.addWidget(scroll)
        self.resize(min(1440, pm.width() + 40), min(860, pm.height() + 40))


class DubResultRow(QWidget):
    """Строка озвучки в готовой задаче + аккордеон «подробнее»."""

    def __init__(self, api: Api, jid: str, res: dict) -> None:
        super().__init__()
        self.api = api
        self.jid = jid
        self.res = res
        self._detail_built = False

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        level = "crit" if (res.get("critical") or not res.get("ok")) else \
                ("warn" if res.get("suspect") else "ok")
        self.row = QFrame()
        self.row.setObjectName({"ok": "drow", "warn": "drowWarn", "crit": "drowCrit"}[level])
        col.addWidget(self.row)
        lay = QHBoxLayout(self.row)
        lay.setContentsMargins(30, 6, 12, 6)
        lay.setSpacing(10)

        lay.addWidget(_dot(level))
        name = Path(res.get("out_path") or res["dub"]).name
        n = QLabel(name)
        n.setObjectName("mono")
        n.setToolTip(res.get("out_path") or res["dub"])
        lay.addWidget(n, 1)

        if res.get("mode") == "audio":
            tag = QLabel(tr("dub.audio_only"))
            tag.setObjectName("tag")
            lay.addWidget(tag)
        if res.get("skipped"):
            tag = QLabel(tr("d.skipped"))
            tag.setObjectName("tag")
            lay.addWidget(tag)

        # главные числа строки: остаток + покрытие (решение Б)
        if res.get("ok"):
            m = QLabel(tr("d.resid", v=f"{res.get('audio_resid_ms', 0.0):.1f}") + " · "
                       + tr("d.coverage", v=f"{res.get('audio_coverage', 0.0):.2f}"))
            m.setObjectName("hint")
            lay.addWidget(m)
        if level == "warn":
            w = QLabel(tr("d.suspect"))
            w.setObjectName("warnText")
            w.setToolTip("\n".join(res.get("warnings") or []))
            lay.addWidget(w)
        if level == "crit":
            w = QLabel((res.get("critical") or [res.get("error") or ""])[0][:60])
            w.setObjectName("critText")
            w.setToolTip(res.get("error") or "\n".join(res.get("critical") or []))
            lay.addWidget(w)

        if res.get("ok"):
            self.b_detail = QPushButton("▸ " + tr("d.details"))
            self.b_detail.setObjectName("link")
            self.b_detail.clicked.connect(self._toggle_detail)
            lay.addWidget(self.b_detail)
        if res.get("out_path"):
            b = QPushButton(tr("d.folder"))
            b.setObjectName("link")
            b.clicked.connect(lambda: QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(Path(res["out_path"]).parent))))
            lay.addWidget(b)

        self.detail = QFrame()
        self.detail.setObjectName("detail")
        self.detail.hide()
        col.addWidget(self.detail)

    # ── аккордеон ──

    def _toggle_detail(self) -> None:
        vis = not self.detail.isVisible()
        if vis and not self._detail_built:
            self._build_detail()
        self.detail.setVisible(vis)
        self.b_detail.setText(("▾ " if vis else "▸ ") + tr("d.details"))

    def _build_detail(self) -> None:
        self._detail_built = True
        r = self.res
        lay = QVBoxLayout(self.detail)
        lay.setContentsMargins(30, 10, 14, 12)
        lay.setSpacing(10)

        grid = QGridLayout()
        grid.setHorizontalSpacing(26)
        grid.setVerticalSpacing(6)
        cells: list[tuple[str, str]] = [
            (tr("m.resid"), f"{r.get('audio_resid_ms', 0):.1f} мс"),
            (tr("m.coverage"), f"{r.get('audio_coverage', 0):.2f}"),
        ]
        if r.get("mode") == "audio":
            cells.append((tr("m.mode_audio"), ""))
        else:
            cells += [
                (tr("m.assigned"), f"{r.get('assigned_pct', 0):.0f} %"),
                (tr("m.cos"), f"{r.get('cos_median', 0):.2f}"),
                (tr("m.slope"), f"{r.get('slope', 0):.4f}"),
            ]
        cuts = r.get("audio_cuts", 0)
        cells.append((tr("m.cuts"),
                      f"{cuts} · {tr('m.max_step')} {r.get('audio_max_step_ms', 0):.0f} мс" if cuts else "0"))
        cells.append((tr("m.filled"), f"{r.get('filled_cuts', 0)} с"))
        cells.append((tr("m.span"), f"{r.get('audio_span_ms', 0):.0f} мс"))
        if r.get("geom_used"):
            cells.append((tr("m.geom"),
                          f"sx {r.get('geom_sx', 0):.3f} · sy {r.get('geom_sy', 0):.3f} · in {r.get('geom_n_in', 0)}"))
        else:
            cells.append((tr("m.geom"), "—"))
        for i, (k, v) in enumerate(cells):
            kl = QLabel(k.upper())
            kl.setObjectName("mkey")
            vl = QLabel(v)
            grid.addWidget(kl, (i // 4) * 2, i % 4)
            grid.addWidget(vl, (i // 4) * 2 + 1, i % 4)
        lay.addLayout(grid)

        plots = r.get("plots") or []
        if plots:
            prow = QHBoxLayout()
            prow.setSpacing(12)
            for p in plots[:4]:
                prow.addWidget(self._preview(p))
            prow.addStretch(1)
            lay.addLayout(prow)

    def _preview(self, plot: dict) -> QWidget:
        """Предпросмотр графика: PNG-миниатюра (клик — крупно) + «открыть ↗» (HTML в браузер)."""
        box = QWidget()
        col = QVBoxLayout(box)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(3)

        img = QLabel()
        img.setObjectName("thumb")
        img.setFixedSize(250, 84)
        img.setAlignment(Qt.AlignCenter)
        img.setCursor(Qt.PointingHandCursor)
        name = plot["name"]
        call(self.api.plot_png, self.jid, name,
             done=lambda png, im=img: self._set_thumb(im, png))
        img.mousePressEvent = lambda e, nm=name: self._open_png(nm)
        col.addWidget(img)

        cap_txt = tr("p.track") if plot.get("kind") == "track" else \
            tr("p.cut", t=f"{plot.get('t') or 0:.0f}", v=f"{plot.get('v_ms') or 0:+.0f}")
        cap = QHBoxLayout()
        c = QLabel(cap_txt)
        c.setObjectName("note")
        cap.addWidget(c)
        if plot.get("kind") == "track":
            html = name.rsplit(".", 1)[0] + ".html"
            b = QPushButton(tr("p.open"))
            b.setObjectName("link")
            b.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self.api.plot_url(self.jid, html))))
            cap.addWidget(b)
        cap.addStretch(1)
        col.addLayout(cap)
        return box

    def _set_thumb(self, img: QLabel, png: bytes) -> None:
        pm = QPixmap()
        pm.loadFromData(png)
        img.setPixmap(pm.scaled(img.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))

    def _open_png(self, name: str) -> None:
        call(self.api.plot_png, self.jid, name,
             done=lambda png: _PngDialog(self, name, png).exec())


class JobCard(QFrame):
    """Карточка задачи: шапка (статус/прогресс/кнопки) + строки озвучек."""

    def __init__(self, api: Api, job: dict, on_cancel) -> None:
        super().__init__()
        self.setObjectName("card")
        self.api = api
        self.jid = job["id"]
        self._on_cancel = on_cancel
        self._results_key = None
        self._expanded = True

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        head = QFrame()
        col.addWidget(head)
        h = QHBoxLayout(head)
        h.setContentsMargins(14, 8, 10, 8)
        h.setSpacing(12)
        self.dot = _dot("wait")
        h.addWidget(self.dot)
        self.name = QLabel(job.get("label") or self.jid)
        self.name.setStyleSheet("font-weight:600;")
        h.addWidget(self.name)
        self.meta = QLabel("")
        self.meta.setObjectName("hint")
        h.addWidget(self.meta, 1)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setMinimumWidth(150)
        self.bar.setMaximumWidth(260)
        h.addWidget(self.bar)
        self.pct = QLabel("")
        self.pct.setObjectName("hint")
        h.addWidget(self.pct)
        self.b_toggle = QPushButton(tr("q.collapse"))
        self.b_toggle.setObjectName("link")
        self.b_toggle.clicked.connect(self._toggle)
        h.addWidget(self.b_toggle)
        self.b_cancel = QPushButton("✕")
        self.b_cancel.setObjectName("rowbtn")
        self.b_cancel.setToolTip(tr("cancel.tip"))
        self.b_cancel.clicked.connect(lambda: self._on_cancel(self.jid))
        h.addWidget(self.b_cancel)

        self.sub = QFrame()
        self.sub.setObjectName("subrows")
        self.sub_col = QVBoxLayout(self.sub)
        self.sub_col.setContentsMargins(0, 0, 0, 0)
        self.sub_col.setSpacing(0)
        self.sub.hide()
        col.addWidget(self.sub)

        self.update_job(job)

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self.sub.setVisible(self._expanded and self.sub_col.count() > 0)
        self.b_toggle.setText(tr("q.collapse") if self._expanded else tr("q.expand"))

    def update_job(self, job: dict) -> None:
        st = job["status"]
        _set_dot(self.dot, _STATUS_DOT.get(st, "wait"))
        self.name.setText(job.get("label") or self.jid)

        if st == "running":
            stage = tr("stage." + job.get("stage", "")) if job.get("stage") else ""
            parts = [stage]
            if job.get("dub_total"):
                parts.append(tr("q.dub_of", i=job.get("dub_index", 0), n=job["dub_total"]))
            if job.get("cur_dub"):
                parts.append(Path(job["cur_dub"]).name[:40])
            self.meta.setText(" · ".join(p for p in parts if p))
            self.bar.show()
            self.pct.show()
            self.bar.setValue(int(job.get("progress", 0) * 1000))
            self.pct.setText(f"{job.get('progress', 0) * 100:.0f}%")
        else:
            self.bar.hide()
            self.pct.hide()
            if st == "queued":
                self.meta.setText(tr("q.queued") + f" · {job.get('dub_total', 0)}")
            elif st == "done":
                ok = sum(1 for r in job.get("results") or [] if r.get("ok"))
                self.meta.setText(f"{tr('q.done')} · {_fmt_elapsed(job.get('elapsed_s', 0))}"
                                  f" · {ok}/{len(job.get('results') or [])} ok")
            elif st == "failed":
                self.meta.setText(tr("q.failed") + (f" · {job.get('error', '')[:70]}" if job.get("error") else ""))
            else:
                self.meta.setText(tr("q." + st) if st in ("cancelled",) else st)

        # строки озвучек пересобираются только при изменении результатов
        results = job.get("results") or []
        key = (len(results), tuple(r.get("ok") for r in results))
        if key != self._results_key:
            self._results_key = key
            while self.sub_col.count():
                w = self.sub_col.takeAt(0).widget()
                if w:
                    w.deleteLater()
            for r in results:
                self.sub_col.addWidget(DubResultRow(self.api, self.jid, r))
        self.sub.setVisible(self._expanded and bool(results))
        self.b_toggle.setVisible(bool(results))


class QueueTab(QWidget):
    """Список карточек. update_jobs() — из поллера MainWindow."""

    def __init__(self, api: Api) -> None:
        super().__init__()
        self.api = api
        self.cards: dict[str, JobCard] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 12, 20, 12)
        root.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(10)
        self.l_par = QLabel(tr("q.parallel"))
        self.l_par.setObjectName("muted")
        head.addWidget(self.l_par)
        self.limit = QComboBox()
        self.limit.addItems([str(i) for i in range(1, 9)])
        self.limit.currentTextChanged.connect(self._limit_changed)
        head.addWidget(self.limit)
        head.addStretch(1)
        self.b_clear = QPushButton(tr("q.clear_done"))
        self.b_clear.clicked.connect(self._clear_done)
        head.addWidget(self.b_clear)
        root.addLayout(head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root.addWidget(scroll)
        page = QWidget()
        scroll.setWidget(page)
        self.col = QVBoxLayout(page)
        self.col.setContentsMargins(0, 0, 4, 0)
        self.col.setSpacing(10)
        self.empty = QLabel(tr("q.empty"))
        self.empty.setObjectName("hint")
        self.empty.setAlignment(Qt.AlignCenter)
        self.col.addWidget(self.empty)
        self.col.addStretch(1)

        self._limit_loaded = False
        call(self.api.get_settings, done=self._on_settings)

    # ── данные ──

    def update_jobs(self, jobs: list[dict]) -> None:
        seen = set()
        for i, job in enumerate(reversed(jobs)):        # новые сверху
            jid = job["id"]
            seen.add(jid)
            card = self.cards.get(jid)
            if card is None:
                card = JobCard(self.api, job, self._cancel)
                self.cards[jid] = card
                self.col.insertWidget(i, card)
            else:
                card.update_job(job)
        for jid in list(self.cards):
            if jid not in seen:
                self.cards.pop(jid).deleteLater()
        self.empty.setVisible(not self.cards)

    def active_count(self, jobs: list[dict]) -> int:
        return sum(1 for j in jobs if j["status"] in ("running", "queued"))

    # ── действия ──

    def _cancel(self, jid: str) -> None:
        call(self.api.cancel, jid)

    def _clear_done(self) -> None:
        # терминальные записи удаляются поштучно (DELETE на терминальной = удаление);
        # уборка временных файлов — доработка ядра (этап 10а)
        for jid, card in list(self.cards.items()):
            if card.dot.property("state") in ("ok", "crit"):
                call(self.api.cancel, jid)

    def _on_settings(self, s: dict) -> None:
        self._limit_loaded = True
        v = str(s.get("limit", 1))
        if self.limit.currentText() != v and v in [self.limit.itemText(i) for i in range(self.limit.count())]:
            self.limit.blockSignals(True)
            self.limit.setCurrentText(v)
            self.limit.blockSignals(False)

    def _limit_changed(self, v: str) -> None:
        if self._limit_loaded:
            call(self.api.set_limit, int(v))

    def retranslate(self) -> None:
        self.l_par.setText(tr("q.parallel"))
        self.b_clear.setText(tr("q.clear_done"))
        self.empty.setText(tr("q.empty"))
