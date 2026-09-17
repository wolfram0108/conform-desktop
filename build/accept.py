# -*- coding: utf-8 -*-
"""Acceptance of a build or of the source tree: the control material goes through the local API and the
result is compared with the reference. Works the same on Windows and Linux.

Usage:
  accept.py judge                    examples with a known answer for the verdict function (no server)
  accept.py rehearse                 the whole tract on a recorded sample instead of a server
  accept.py run [--exe PATH] [--port N] [--fuse S]
                                     control job through the API; the raw record goes to _work/accept/
  accept.py bless RECORD.json        make a raw record the new reference (an explicit, separate act)

run starts `python -m server` from the source tree, or the given executable, waits until the API answers,
enqueues installer/selfcheck/{ref,dub}_check.mkv and judges the output. Exit codes: 0 accepted, 1 the
subject failed, 2 the instrument itself failed (nothing was observed, so there is no verdict).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "installer" / "selfcheck"
RAW = ROOT / "_work" / "accept"
REQUIRED = ("sha256", "ok", "resid_ms", "coverage", "elapsed_s", "startup_s", "subject")
# Twice the measured price with a margin for a frozen build: 7.7 s from source (2.6 s start-up, 3.4 s job).
DEFAULT_FUSE_S = 40


class InstrumentDefect(Exception):
    """The observation did not take place; a verdict on the subject would be a lie."""


def judge(record, reference):
    """Verdict on a complete record. The digest decides; metrics only explain a mismatch."""
    missing = [k for k in REQUIRED if k not in record]
    if missing:
        raise InstrumentDefect("в записи нет полей: " + ", ".join(missing))
    if not record["ok"]:
        return False, "обработка завершилась ошибкой: %s" % record.get("error")
    if record["sha256"].lower() == reference["sha256"].lower():
        return True, "результат совпал с эталонным"
    return False, ("результат не совпал с эталонным: %s… вместо %s…; рассогласование %.2f мс (эталон %.2f), "
                   "покрытие %.3f (эталон %.3f)" % (
                       record["sha256"][:16], reference["sha256"][:16],
                       record["resid_ms"], reference.get("resid_ms", float("nan")),
                       record["coverage"], reference.get("coverage", float("nan"))))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def api(base, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode("utf-8")
    return json.loads(raw) if raw else None


def start_subject(exe, port, data_dir, log):
    env = dict(os.environ, CONFORM_PORT=str(port), CONFORM_DATA_DIR=str(data_dir))
    if exe:
        cmd, cwd = [str(exe)], str(Path(exe).resolve().parent)
    else:
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")])
        cmd, cwd = [sys.executable, "-m", "server", str(port)], str(ROOT)
    return subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)


def observe(exe, port, fuse_s):
    """One control job through the API. Returns the raw record; raises InstrumentDefect otherwise."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    work = RAW / stamp
    out_dir = work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = "http://127.0.0.1:%d" % port
    t0 = time.time()
    with open(work / "subject.log", "wb") as log:
        proc = start_subject(exe, port, work / "appdata", log)
        try:
            while True:
                if proc.poll() is not None:
                    raise InstrumentDefect("предмет завершился до ответа API, код %s; журнал %s"
                                           % (proc.returncode, work / "subject.log"))
                try:
                    api(base, "GET", "/health")
                    break
                except (urllib.error.URLError, OSError, TimeoutError):
                    if time.time() - t0 > fuse_s:
                        raise InstrumentDefect("API не ответил за %d с; журнал %s" % (fuse_s, work / "subject.log"))
                    time.sleep(0.5)
            startup = time.time() - t0
            job = api(base, "POST", "/conform/enqueue", {
                "label": "приёмка", "ref": str(CHECK / "ref_check.mkv"), "dubs": [str(CHECK / "dub_check.mkv")],
                "out_dir": str(out_dir), "autostart": True})
            while True:
                found = next((j for j in api(base, "GET", "/conform/jobs") if j["id"] == job["id"]), None)
                if found is None:
                    raise InstrumentDefect("задача исчезла из очереди")
                if found["status"] not in ("running", "queued", "paused"):
                    break
                if time.time() - t0 > fuse_s:
                    raise InstrumentDefect("задача не завершилась за %d с, статус %s" % (fuse_s, found["status"]))
                time.sleep(1)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
    res = (found.get("results") or [{}])[0]
    out = res.get("out_path")
    record = {
        "stamp": stamp, "subject": str(exe) if exe else "source", "status": found["status"],
        "ok": bool(res.get("ok")), "error": res.get("error"),
        "sha256": sha256(out) if out and Path(out).is_file() else "",
        "resid_ms": res.get("audio_resid_ms"), "coverage": res.get("audio_coverage"),
        "elapsed_s": res.get("elapsed_s"), "startup_s": round(startup, 1),
        "total_s": round(time.time() - t0, 1), "result": res,
    }
    (work / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(work / "appdata", ignore_errors=True)
    return record, work / "record.json"


def reference():
    return json.loads((CHECK / "reference.json").read_text(encoding="utf-8"))


def cmd_judge(_):
    ref = {"sha256": "AB" * 32, "resid_ms": 1.0, "coverage": 0.9}
    good = dict(sha256="ab" * 32, ok=True, resid_ms=1.0, coverage=0.9, elapsed_s=6, startup_s=9, subject="x")
    cases = [(good, True), (dict(good, sha256="cd" * 32), False), (dict(good, ok=False, error="e"), False)]
    for record, expected in cases:
        verdict, why = judge(record, ref)
        assert verdict is expected, (record, why)
    try:
        judge({"sha256": "x"}, ref)
    except InstrumentDefect:
        pass
    else:
        raise AssertionError("неполная запись обязана считаться дефектом прибора")
    print("судья: 4 примера с известным ответом — верно")
    return 0


def cmd_rehearse(_):
    ref = reference()
    sample = dict(sha256=ref["sha256"], ok=True, resid_ms=ref["resid_ms"], coverage=ref["coverage"],
                  elapsed_s=ref["elapsed_s"], startup_s=0.0, subject="sample")
    print("репетиция:", judge(sample, ref), "| иной результат:", judge(dict(sample, sha256="0" * 64), ref)[0])
    return 0


def cmd_run(args):
    record, path = observe(args.exe, args.port, args.fuse)
    verdict, why = judge(record, reference())
    print("предмет: %s | запуск %.1f с, обработка %s с, всего %.1f с" % (
        record["subject"], record["startup_s"], record["elapsed_s"], record["total_s"]))
    print("рассогласование %s мс, покрытие %s, сумма %s" % (record["resid_ms"], record["coverage"], record["sha256"]))
    print("сырьё:", path)
    print("ПРИНЯТО:" if verdict else "НЕ ПРИНЯТО:", why)
    return 0 if verdict else 1


def cmd_bless(args):
    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    if not record.get("ok") or not record.get("sha256"):
        raise InstrumentDefect("эталоном становится только успешная запись с суммой")
    ref = reference()
    ref.update(sha256=record["sha256"], resid_ms=record["resid_ms"], coverage=record["coverage"],
               elapsed_s=record["elapsed_s"])
    (CHECK / "reference.json").write_text(json.dumps(ref, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("эталон обновлён:", record["sha256"])
    return 0


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("judge").set_defaults(fn=cmd_judge)
    sub.add_parser("rehearse").set_defaults(fn=cmd_rehearse)
    run = sub.add_parser("run")
    run.add_argument("--exe")
    run.add_argument("--port", type=int, default=8797)
    run.add_argument("--fuse", type=int, default=DEFAULT_FUSE_S)
    run.set_defaults(fn=cmd_run)
    bless = sub.add_parser("bless")
    bless.add_argument("record")
    bless.set_defaults(fn=cmd_bless)
    args = parser.parse_args()
    try:
        return args.fn(args)
    except InstrumentDefect as e:
        print("ДЕФЕКТ ПРИБОРА:", e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
