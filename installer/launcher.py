"""Тонкий загрузчик conform-desktop.

Назначение: привести рабочее место в состояние, пригодное для работы, и запустить
приложение. Тяжёлые компоненты (torch с CUDA, ffmpeg, само приложение) в загрузчик не
входят и поставляются отдельными архивами.

Ключевое свойство: манифест описывает компоненты по имени файла и контрольной сумме,
а не по адресу загрузки. Источник поэтому произвольный — облачное хранилище, съёмный
носитель, сетевой каталог. Файлы, полученные заранее и размещённые в каталоге
установки, распознаются автоматически; загрузка по сети нужна лишь тогда, когда файлов
на месте нет.

Порядок работы:
    манифест → поиск архивов (локально либо загрузка) → сверка SHA-256 → распаковка →
    проверка установки обработкой → запуск приложения.

Проверка установки не сводится к наличию файлов: контрольный материал проходит полный
конвейер обработки, а контрольная сумма результата сверяется с эталонной. Так выявляются
и неполная распаковка, и несовместимый набор библиотек.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import filedialog, ttk

# Консоль Windows может быть в однобайтовой кодировке (на англоязычной системе — cp1252),
# и печать русского текста роняет программу с UnicodeEncodeError. Переводим вывод в UTF-8
# с заменой непредставимых знаков: сообщение важнее, чем точность отдельного символа.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

APP_DIR_NAME = "app"            # каталог развёрнутого дистрибутива
PACKAGES_DIR_NAME = "packages"  # каталог с файлами установки
STATE_FILE = "installed.json"
MANIFEST = "manifest.json"
IS_WINDOWS = os.name == "nt"
EXE_NAME = "conform-desktop.exe" if IS_WINDOWS else "conform-desktop"
NO_WINDOW = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0   # флага нет вне Windows


def base_dir() -> Path:
    """Каталог рядом с загрузчиком (в собранном виде — рядом с exe)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def res_dir() -> Path:
    """Каталог ресурсов самого загрузчика (7-Zip, встроенный манифест, материал проверки)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", base_dir()))
    return Path(__file__).resolve().parent


def sha256(path: Path, progress=None) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
            done += len(chunk)
            if progress:
                progress(done / total if total else 1.0)
    return h.hexdigest()


def human(n: float) -> str:
    mb = n / 2**20
    return f"{mb / 1024:.2f} ГБ" if mb >= 1024 else f"{mb:.0f} МБ"


class Installer:
    """Установка без интерфейса: поиск, загрузка, сверка, распаковка, проверка обработкой."""

    def __init__(self, root: Path, log=print, progress=None):
        self.root = root
        self.app_dir = root / APP_DIR_NAME
        self.pkg_dirs = [root / PACKAGES_DIR_NAME, root]
        self.log = log
        self.progress = progress or (lambda frac, text="": None)
        self.manifest = self._load_manifest()

    # ── манифест ──
    def manifest_url(self) -> str:
        """Адрес состава выпуска задаётся при сборке (`manifest_url.txt`). Нужен для случая,
        когда у пользователя есть только загрузчик и ничего больше."""
        p = res_dir() / "manifest_url.txt"
        return p.read_text(encoding="utf-8").strip() if p.is_file() else ""

    def _load_manifest(self) -> dict:
        for p in [self.root / MANIFEST, *[d / MANIFEST for d in self.pkg_dirs], res_dir() / MANIFEST]:
            if p.is_file():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    self.log(f"Манифест повреждён: {p}")
        return {}

    def fetch_manifest(self) -> bool:
        """Получить состав выпуска из сети и сохранить рядом: дальше он доступен и офлайн."""
        url = self.manifest_url()
        if not url:
            return False
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as e:
            self.log(f"Не удалось получить состав выпуска: {e}")
            return False
        base = url.rsplit("/", 1)[0]
        for c in data.get("components", []):
            if not c.get("url"):                    # адреса компонентов — рядом с манифестом
                c["url"] = f"{base}/{c['file']}"
        (self.root / PACKAGES_DIR_NAME).mkdir(exist_ok=True)
        (self.root / PACKAGES_DIR_NAME / MANIFEST).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.manifest = data
        self.log(f"Состав выпуска получен: версия {data.get('version', '—')}, "
                 f"компонентов {len(data.get('components', []))}")
        return True

    def installed(self) -> dict:
        p = self.app_dir / STATE_FILE
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    # ── поиск файлов установки ──
    def find_archive(self, comp: dict) -> Path | None:
        """Поиск по имени в известных каталогах; при несовпадении имени — по размеру
        среди архивов (файл мог быть переименован при передаче)."""
        for d in self.pkg_dirs:
            p = d / comp["file"]
            if p.is_file() and p.stat().st_size == comp["size"]:
                return p
        for d in self.pkg_dirs:
            if not d.is_dir():
                continue
            for p in d.glob("*.7z"):
                if p.stat().st_size == comp["size"]:
                    return p
        return None

    def verify(self, path: Path, comp: dict) -> bool:
        self.log(f"Проверка целостности: {path.name} ({human(path.stat().st_size)})")
        got = sha256(path, lambda f: self.progress(f, f"проверка {path.name}"))
        ok = got.lower() == comp["sha256"].lower()
        self.log("  контрольная сумма совпала" if ok else f"  контрольная сумма не совпала: {got[:16]}…")
        return ok

    def download(self, comp: dict, dest: Path) -> bool:
        url = comp.get("url")
        if not url:
            self.log(f"источник для компонента «{comp['title']}» не задан: "
                     f"разместите {comp['file']} в каталоге установки")
            return False
        part = dest.with_suffix(dest.suffix + ".part")
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")     # докачка после обрыва
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length", 0)) + have
                mode = "ab" if have and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                with part.open(mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        self.progress(have / total if total else 0,
                                      f"загрузка {comp['file']} — {human(have)} из {human(total)}")
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            self.log(f"Загрузка прервана: {e}. Полученная часть сохранена, загрузку можно продолжить.")
            return False
        part.replace(dest)
        return True

    def unpack_tool(self) -> str:
        """Распаковщик: на Windows вложен в загрузчик, на прочих системах берётся из
        системы (7zz / 7za / 7z из p7zip). Проверка наличия — до начала установки."""
        if IS_WINDOWS:
            return str(res_dir() / "7z.exe")
        local = res_dir() / "7zz"
        if local.is_file():
            return str(local)
        for name in ("7zz", "7za", "7z"):
            found = shutil.which(name)
            if found:
                return found
        return ""

    def unpack(self, archive: Path) -> bool:
        exe = self.unpack_tool()
        if not exe:
            self.log("  Распаковщик не найден. Установите p7zip (пакет p7zip-full).")
            return False
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"Распаковка: {archive.name}")
        r = subprocess.run([exe, "x", "-y", f"-o{self.app_dir}", str(archive)],
                           capture_output=True, text=True, creationflags=NO_WINDOW)
        if r.returncode != 0:
            self.log(f"  Ошибка распаковки: {(r.stdout or r.stderr or '')[-300:]}")
            return False
        return True

    # ── установка целиком ──
    def missing(self) -> list[dict]:
        state = self.installed()
        return [c for c in self.manifest.get("components", [])
                if state.get(c["name"]) != c["sha256"]]

    def install(self, want_download: bool = True) -> bool:
        if not self.manifest.get("components"):
            # пустой состав — это не «всё готово», а отсутствие сведений о выпуске
            self.log("Состав выпуска неизвестен: manifest.json не найден и не получен из "
                     "сети. Разместите файлы установки в каталоге установки.")
            return False
        comps = self.missing()
        if not comps:
            self.log("Все компоненты установлены.")
            return True
        state = self.installed()
        for c in comps:
            self.log(f"Компонент «{c['title']}» — {human(c['size'])}")
            arc = self.find_archive(c)
            if arc is None and want_download:
                dest = (self.root / PACKAGES_DIR_NAME)
                dest.mkdir(exist_ok=True)
                if self.download(c, dest / c["file"]):
                    arc = dest / c["file"]
            if arc is None:
                self.log(f"  Файл {c['file']} не найден. Разместите его в подкаталоге "
                         f"«{PACKAGES_DIR_NAME}» или укажите каталог с файлами установки.")
                return False
            if not self.verify(arc, c):
                self.log("  Контрольная сумма не совпала: файл повреждён либо принадлежит "
                         "другой версии. Загрузите файл заново.")
                return False
            if not self.unpack(arc):
                return False
            state[c["name"]] = c["sha256"]
            state["version"] = self.manifest.get("version", "")
            (self.app_dir / STATE_FILE).write_text(
                json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            self.log(f"  Компонент установлен: {c['title']}")
        return True

    # ── проверка установки: не наличие файлов, а корректность обработки ──
    def selfcheck(self) -> bool:
        exe = self.app_dir / EXE_NAME
        if not exe.is_file():
            self.log("Проверка установки: исполняемый файл приложения не найден.")
            return False
        spec_path = res_dir() / "selfcheck.json"
        if not spec_path.is_file():
            self.log("Проверка установки: эталон недоступен, проверка пропущена.")
            return True
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        work = self.root / "_selfcheck"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        ref = res_dir() / spec["ref"]
        dub = res_dir() / spec["dub"]

        self.log("Проверка установки: обработка контрольного материала")
        proc = subprocess.Popen([str(exe)], cwd=str(self.app_dir), creationflags=NO_WINDOW)
        try:
            port = spec.get("port", 8799)
            api = f"http://127.0.0.1:{port}"
            if not self._wait_api(api, 180):
                self.log("  Приложение не отвечает."); return False
            job = self._api(api, "POST", "/conform/enqueue", {
                "label": "самопроверка", "ref": str(ref), "dubs": [str(dub)],
                "out_dir": str(work), "autostart": True})
            t0 = time.time()
            while True:
                jobs = self._api(api, "GET", "/conform/jobs")
                j = next((x for x in jobs if x["id"] == job["id"]), None)
                if j is None:
                    self.log("  Задача отсутствует в очереди."); return False
                if j["status"] not in ("running", "queued", "paused"):
                    break
                self.progress(j.get("progress", 0), "самопроверка — обработка")
                if time.time() - t0 > spec.get("timeout_s", 900):
                    self.log("  Превышено время ожидания."); return False
                time.sleep(2)
            res = (j.get("results") or [{}])[0]
            if not res.get("ok"):
                self.log(f"  Обработка завершилась ошибкой: {res.get('error')}")
                return False
            got = sha256(Path(res["out_path"]))
            if got.lower() != spec["sha256"].lower():
                self.log(f"  Результат не совпал с эталонным: {got[:16]}… вместо {spec['sha256'][:16]}…")
                return False
            self.log("  Результат обработки совпал с эталонным: установка исправна.")
            return True
        finally:
            proc.terminate()
            shutil.rmtree(work, ignore_errors=True)

    @staticmethod
    def _api(base: str, method: str, path: str, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8")
        return json.loads(raw) if raw else None

    def _wait_api(self, base: str, limit_s: float) -> bool:
        t0 = time.time()
        while time.time() - t0 < limit_s:
            try:
                self._api(base, "GET", "/conform/jobs")
                return True
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(1)
        return False

    def launch(self) -> bool:
        exe = self.app_dir / EXE_NAME
        if not exe.is_file():
            return False
        subprocess.Popen([str(exe)], cwd=str(self.app_dir))
        return True


class Window:
    """Окно установки: состав компонентов, состояние каждого, журнал, основное действие."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("conform-desktop — установка")
        self.root.geometry("760x520")
        self.root.minsize(620, 420)
        self.events: queue.Queue = queue.Queue()
        self.pending_refresh = False
        self.inst = Installer(base_dir(), log=self.log, progress=self.on_progress)
        self.busy = False
        self._build()
        self._pump()
        # У пользователя может не быть ничего, кроме загрузчика: тогда состав выпуска
        # запрашивается из сети. Запрос идёт ДО первой отрисовки состава — иначе окно
        # успевает сообщить «манифест не найден» о том, что уже загружается.
        self.fetching = False
        if not self.inst.manifest and self.inst.manifest_url():
            self._fetch_then_refresh()
        else:
            self.refresh()

    def _fetch_then_refresh(self) -> None:
        self.fetching = True
        self.state_lbl["text"] = "запрос состава выпуска…"
        self.log("Запрос состава выпуска…")
        self.root.update_idletasks()
        try:
            self.inst.fetch_manifest()
        finally:
            self.fetching = False
            self.refresh()

    def _build(self) -> None:
        pad = {"padx": 14, "pady": 6}
        top = ttk.Frame(self.root); top.pack(fill="x", **pad)
        ver = self.inst.manifest.get("version", "—")
        ttk.Label(top, text=f"conform-desktop {ver}", font=("Segoe UI", 12, "bold")).pack(side="left")
        self.state_lbl = ttk.Label(top, text="")
        self.state_lbl.pack(side="right")

        cols = ("part", "size", "status")
        self.tree = ttk.Treeview(self.root, columns=cols, show="headings", height=5)
        for c, w, t in (("part", 320, "Компонент"), ("size", 110, "Размер"), ("status", 260, "Состояние")):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="x", **pad)

        self.bar = ttk.Progressbar(self.root, mode="determinate", maximum=1000)
        self.bar.pack(fill="x", **pad)
        self.bar_lbl = ttk.Label(self.root, text="")
        self.bar_lbl.pack(fill="x", padx=14)

        self.text = tk.Text(self.root, height=10, wrap="word", relief="flat",
                            background="#f4f5f7", font=("Consolas", 9))
        self.text.pack(fill="both", expand=True, **pad)

        row = ttk.Frame(self.root); row.pack(fill="x", **pad)
        self.btn = ttk.Button(row, text="Установить", command=self.on_main)
        self.btn.pack(side="right")
        ttk.Button(row, text="Проверить наличие файлов", command=self.refresh).pack(side="right", padx=6)
        ttk.Button(row, text="Выбрать каталог с файлами установки…", command=self.pick_dir).pack(side="left")

    # ── вывод ──
    # Установка идёт в отдельном потоке, а виджеты Tk можно трогать только из главного:
    # сообщения складываются в очередь, а забирает их таймер главного потока.
    def log(self, msg: str) -> None:
        self.events.put(("log", msg))
        if threading.current_thread() is threading.main_thread():
            self._drain()

    def on_progress(self, frac: float, text: str = "") -> None:
        self.events.put(("progress", (frac, text)))
        if threading.current_thread() is threading.main_thread():
            self._drain()

    def _drain(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self.text.insert("end", payload + "\n")
                self.text.see("end")
            else:
                frac, text = payload
                self.bar["value"] = max(0, min(1000, int(frac * 1000)))
                if text:
                    self.bar_lbl["text"] = text
        self.root.update_idletasks()

    def _pump(self) -> None:
        self._drain()
        if self.pending_refresh:
            self.pending_refresh = False
            self.refresh()
        self.root.after(120, self._pump)

    def pick_dir(self) -> None:
        d = filedialog.askdirectory(title="Каталог с файлами установки")
        if d:
            self.inst.pkg_dirs.insert(0, Path(d))
            self.inst.manifest = self.inst._load_manifest()
            self.log(f"Каталог поиска файлов: {d}")
            self.refresh()

    def refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        comps = self.inst.manifest.get("components", [])
        if not comps:
            self.state_lbl["text"] = "состав выпуска недоступен"
            self.log("Состав выпуска не получен. Разместите manifest.json и файлы "
                     "компонентов в каталоге установки либо укажите каталог с ними.")
            self.btn["text"] = "Повторить запрос"
            self.btn["state"] = "normal" if self.inst.manifest_url() else "disabled"
            return
        state = self.inst.installed()
        need = 0
        for c in comps:
            if state.get(c["name"]) == c["sha256"]:
                st = "установлено"
            elif self.inst.find_archive(c) is not None:
                st = "файл найден, к установке"; need += c["size"]
            elif c.get("url"):
                st = "будет загружен"; need += c["size"]
            else:
                st = "файл не найден"; need += c["size"]
            self.tree.insert("", "end", values=(c["title"], human(c["size"]), st))
        self.state_lbl["text"] = "установка полная" if need == 0 else f"к установке: {human(need)}"
        self.btn["text"] = "Запустить" if need == 0 else "Установить"
        self.btn["state"] = "normal"

    # ── главная кнопка ──
    def on_main(self) -> None:
        if self.busy or self.fetching:
            return
        if not self.inst.manifest:                 # состава ещё нет — сначала получить его
            self._fetch_then_refresh()
            return
        if not self.inst.missing():
            self.log("Запуск приложения")
            if self.inst.launch():
                self.root.after(1200, self.root.destroy)
            else:
                self.log("Исполняемый файл приложения не найден.")
            return
        self.busy = True
        self.btn["state"] = "disabled"
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self) -> None:
        try:
            ok = self.inst.install()
            if ok:
                ok = self.inst.selfcheck()
            self.log("Установка завершена. Приложение готово к запуску." if ok
                     else "Установка не завершена: см. сообщения выше.")
        except Exception as e:  # noqa: BLE001 — окно не должно молча закрываться
            self.log(f"Ошибка установки: {e}")
        finally:
            self.busy = False
            self.on_progress(0, "")
            self.pending_refresh = True     # перерисовку делает главный поток

    def run(self) -> None:
        self.root.mainloop()


def _console_log(msg: str) -> None:
    """Печать, переживающая оконный режим: там стандартного вывода нет вовсе."""
    if sys.stdout is not None:
        print(msg, flush=True)
    with (base_dir() / "setup.log").open("a", encoding="utf-8") as f:
        f.write(msg + chr(10))


def main() -> int:
    if "--ui-selftest" in sys.argv:
        # Проверка окна: создаём его по-настоящему и нажимаем основную кнопку кодом.
        # Проверяются те же обработчики, что и при нажатии мышью, — но без участия человека.
        w = Window()
        w.root.update()
        rows = [w.tree.item(i)["values"] for i in w.tree.get_children()]
        print("состав в окне:", rows, flush=True)
        print("кнопка:", w.btn["text"], "| состояние:", w.state_lbl["text"], flush=True)
        w.btn.invoke()
        t0 = time.time()
        while (w.busy or time.time() - t0 < 3) and time.time() - t0 < 1800:
            w.root.update()
            time.sleep(0.2)
        print("после нажатия:", w.state_lbl["text"], "| кнопка:", w.btn["text"], flush=True)
        print("журнал:\n" + w.text.get("1.0", "end").strip(), flush=True)
        ok = not w.inst.missing()
        w.root.destroy()
        return 0 if ok else 1

    if "--check" in sys.argv:            # режим без интерфейса, для стенда
        offline = "--offline" in sys.argv    # запретить сеть: только принесённые файлы
        inst = Installer(base_dir(), log=_console_log)
        if not inst.manifest and not offline:
            inst.fetch_manifest()
        ok = inst.install(want_download=not offline) and inst.selfcheck()
        return 0 if ok else 1
    Window().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
