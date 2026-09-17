/* Логика интерфейса conform-desktop.
   Состояние живёт в ядре: страница только отображает и дёргает локальный API.
   Диалоги выбора файлов/каталогов — у оболочки Qt (эндпоинты /ui/pick_*). */

const API = new URLSearchParams(location.search).get("api") || "http://127.0.0.1:8799";
const $ = sel => document.querySelector(sel);
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };

const AUDIO_EXT = [".flac",".mka",".mp3",".wav",".aac",".opus",".ogg",".m4a",".ac3",".dts"];
const isAudio = p => AUDIO_EXT.some(x => p.toLowerCase().endsWith(x));
const baseName = p => p.split(/[\\/]/).pop();

async function api(method, path, body) {
  const r = await fetch(API + path, {
    method,
    headers: body ? {"Content-Type": "application/json"} : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return r.status === 204 ? null : r.json();
}

let toastTimer = null;
function toast(msg) {
  const box = $("#toast");
  box.textContent = msg;
  box.classList.add("on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => box.classList.remove("on"), 4000);
}

/* ═══════════ состояние формы ═══════════ */
const state = {
  ref: "", refTracks: [], refTrack: 0,
  dubs: [],            // {path, tracks:[], sel:[индексы], audioOnly}
  theme: "auto",
};

/* ═══════════ вкладки, тема, язык ═══════════ */
function showPage(name) {
  document.querySelectorAll(".page").forEach(p => p.classList.toggle("on", p.id === "page-" + name));
  document.querySelectorAll(".tab").forEach(b => b.classList.toggle("on", b.dataset.page === name));
}

function applyTheme(mode) {
  state.theme = mode;
  const root = document.documentElement;
  root.dataset.theme = mode === "auto" ? "auto" : mode;
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  root.classList.toggle("sys-dark", mode === "auto" && dark);
  $("#theme").textContent = mode === "auto" ? "◐" : (mode === "light" ? "☀" : "☾");
  save("theme", mode);
}

/* ═══════════ сохранение настроек (в ядре, файл рядом с exe) ═══════════ */
const PREFS = {};
function save(key, value) {
  PREFS[key] = value;
  localStorage.setItem("conform-ui", JSON.stringify(PREFS));
}
function loadPrefs() {
  try { Object.assign(PREFS, JSON.parse(localStorage.getItem("conform-ui") || "{}")); } catch (_) {}
  return PREFS;
}

/* ═══════════ форма задачи ═══════════ */
async function pickFiles(kind) {          // kind: ref | dubs | dir
  return api("POST", "/ui/pick", {kind});
}

async function setRef(path) {
  state.ref = path;
  $("#ref").value = path;
  if (!$("#label").value.trim()) $("#label").value = baseName(path).replace(/\.[^.]+$/, "").slice(0, 60);
  state.refTracks = path ? await api("GET", "/conform/atracks?path=" + encodeURIComponent(path)).catch(() => []) : [];
  const row = $("#ref-track-row"), sel = $("#ref-track");
  sel.innerHTML = "";
  if (state.refTracks.length > 1) {
    state.refTracks.forEach((tr, i) => sel.append(new Option(trackLabel(tr), i, tr.default, tr.default)));
    state.refTrack = Math.max(0, state.refTracks.findIndex(x => x.default));
    sel.value = String(state.refTrack);
    $("#ref-track-hint").textContent = t("tracks.n", {n: state.refTracks.length});
    row.hidden = false;
  } else {
    state.refTrack = 0;
    row.hidden = true;
  }
  renderDubs();
}

function trackLabel(tr) {
  const parts = ["a" + tr.index];
  if (tr.lang) parts.push(tr.lang);
  if (tr.title) parts.push(tr.title.length > 30 ? tr.title.slice(0, 29) + "…" : tr.title);
  if (tr.layout) parts.push(tr.layout);
  else if (tr.channels) parts.push(tr.channels + "ch");
  return parts.join(" · ");
}

async function addDub(path, preselect) {
  const audioOnly = isAudio(path);
  const tracks = audioOnly ? [] : await api("GET", "/conform/atracks?path=" + encodeURIComponent(path)).catch(() => []);
  state.dubs.push({path, tracks, sel: preselect || [0], audioOnly});
  renderDubs();
}

function renderDubs() {
  const box = $("#dubs");
  box.innerHTML = "";
  box.hidden = state.dubs.length === 0;
  state.dubs.forEach((d, i) => {
    const row = el("div", "dub");
    const name = el("span", "name", baseName(d.path));
    name.title = d.path;
    row.append(name);

    if (d.path === state.ref) row.append(el("span", "tag", t("dub.same_as_ref")));
    if (d.audioOnly) row.append(el("span", "tag", t("dub.audio_only")));
    else if (d.tracks.length > 1) row.append(trackPicker(d, i));
    if (d.path === state.ref && !d.audioOnly) row.append(el("span", "tag", t("dub.virtual")));

    const x = el("button", "x", "✕");
    x.onclick = () => { state.dubs.splice(i, 1); renderDubs(); };
    row.append(x);
    box.append(row);
  });
}

function closeTrackMenus() {
  document.querySelectorAll(".tracks-menu.on").forEach(m => m.classList.remove("on"));
}

function trackPicker(d, idx) {
  const wrap = el("span", "tracks");
  const btn = el("button", "btn small");
  const menu = el("div", "tracks-menu");
  const refresh = () => {
    btn.textContent = d.sel.length === 0 ? t("tracks.none")
      : (d.sel.length === 1 ? trackLabel(d.tracks[d.sel[0]])
         : t("tracks.chosen", {k: d.sel.length, n: d.tracks.length}));
    btn.title = d.sel.map(i => trackLabel(d.tracks[i])).join("\n");
  };
  d.tracks.forEach((tr, i) => {
    const lab = el("label");
    const cb = el("input"); cb.type = "checkbox"; cb.checked = d.sel.includes(i);
    cb.onchange = () => {
      d.sel = d.tracks.map((_, k) => k).filter(k => k === i ? cb.checked : d.sel.includes(k));
      refresh();
    };
    lab.append(cb, el("span", null, trackLabel(tr)));
    menu.append(lab);
  });
  const all = el("button", "btn small", t("tracks.all"));
  all.style.margin = "4px";
  all.onclick = e => {
    e.stopPropagation();
    const on = d.sel.length < d.tracks.length;
    d.sel = on ? d.tracks.map((_, i) => i) : [];
    menu.querySelectorAll("input").forEach((cb, i) => { cb.checked = d.sel.includes(i); });
    refresh();
  };
  menu.append(all);
  // клик ВНУТРИ меню не должен его закрывать: иначе выбор первой же дорожки схлопывает
  // список и выбрать несколько невозможно (снаружи выглядит как «список не открывается»)
  menu.onclick = e => e.stopPropagation();
  btn.onclick = e => {
    e.stopPropagation();
    const open = menu.classList.contains("on");
    closeTrackMenus();                       // одновременно открыто не больше одного меню
    if (!open) menu.classList.add("on");
  };
  wrap.append(btn, menu);
  refresh();
  return wrap;
}

/* ═══════════ отправка задачи ═══════════ */
async function enqueue() {
  if (!state.ref) return toast(t("task.need_ref"));
  if (!state.dubs.length) return toast(t("task.need_dubs"));
  const out = $("#out").value.trim();
  if (!out) return toast(t("task.need_out"));

  const dubs = [], atracks = [];
  for (const d of state.dubs) {
    const sel = d.audioOnly ? [0] : (d.sel.length ? d.sel : [0]);
    for (const a of sel) { dubs.push(d.path); atracks.push(a); }
  }
  const body = {
    ref: state.ref, dubs, dub_atracks: atracks,
    ref_atrack: state.refTrack,
    out_dir: out,
    label: $("#label").value.trim() || null,
    fill_silence: $("#fill").checked,
    audio_band: !$("#a-muq").checked,
    audio_muq: $("#a-muq").checked,
    drift_speed_pct: parseFloat($("#drift").value) || 1.25,
    keep_tmp: $("#keeptmp").checked,
    cache_dir: $("#keeptmp").checked ? ($("#tmpdir").value.trim() || null) : null,
    autostart: $("#autostart").checked,
  };
  try {
    const job = await api("POST", "/conform/enqueue", body);
    toast(t($("#autostart").checked ? "task.added_run" : "task.added", {n: job.dubs.length}));
    state.dubs = []; renderDubs(); $("#label").value = "";
    showPage("queue");
    refreshJobs();
  } catch (e) { toast(t("err.api", {e: e.message})); }
}

/* ═══════════ очередь ═══════════ */
const openDetails = new Set();          // какие строки раскрыты (переживает обновление)
const openJobs = new Set();             // какие задачи развёрнуты

/* Последовательность операций детерминирована, поэтому полосу прогресса можно
   разложить заранее. Веса — РОВНО те, по которым ядро считает progress
   (queue.py), иначе полоса и процент разойдутся.
   Доля 0 — подготовка референса, одна на задачу; доли 1..N — аудиодорожки. */
const REF_OPS = [{op: "decode", w: 85}, {op: "extract", w: 15}];
const TRACK_OPS = [
  {op: "decode",   w: 52},
  {op: "coarse",   w:  3},
  {op: "geom",     w:  7, opt: true},   // только если грубое сопоставление не удалось
  {op: "band",     w: 16},
  {op: "extract",  w: 10},
  {op: "resample", w:  5},
  {op: "audio",    w:  4},
  {op: "write",    w:  3},
];
const opsOf = sliceI => (sliceI === 0 ? REF_OPS : TRACK_OPS);
const opName = (sliceI, op) => t(sliceI === 0 ? "op.ref." + op : "op." + op);

/* Speed norms are MEASURED on two materials. Audio operations barely depend on the material
   (seconds per minute); decoding depends on resolution and is normalised to pixels
   (seconds per gigapixel). */
const RATE_PER_MIN = {coarse: 0.20, band: 0.70, extract: 0.29, resample: 0.35, audio: 0.43, write: 0.48};
const decodeRate = h => h >= 1500 ? 0.55 : h >= 900 ? 0.80 : 0.43;   // с / гигапиксель
const GEOM_SHARE = 0.2;              // геометрия ≈ пятая часть декода (замер: 2:18 против 12:40)

function opCost(op, info, fallback) {
  const m = ((info && info.duration_s) || (fallback && fallback.duration_s) || 0) / 60;
  if (op === "decode" || op === "geom") {
    if (!info || !info.frames || !info.width) return 0;
    const gpix = info.frames * info.width * info.height / 1e9;
    const c = gpix * decodeRate(info.height);
    return op === "geom" ? c * GEOM_SHARE : c;
  }
  return (RATE_PER_MIN[op] || 0) * m;
}

/* Остаток = хвост текущей операции по ЖИВОЙ скорости + оставшиеся операции этой доли
   + все операции ещё не начатых долей. Живая часть надёжнее таблиц, поэтому она первая. */
function estimateRemaining(j) {
  if (j.status !== "running") return null;
  const ref = j.ref_info || null;
  const infoOf = s => s === 0 ? ref : ((j.dub_infos || [])[s - 1] || ref);
  const plannedOps = (s, info) => {
    if (s === 0) return ["decode", "extract"];
    if (info && info.has_video === false) return ["extract", "audio", "write"];   // аудио-только
    return ["decode", "coarse", "band", "extract", "resample", "audio", "write"]; // geom — если понадобится
  };
  let left = 0;
  const cur = j.dub_index || 0;

  const rec = (j.ops || []).find(o => o.state === "run");
  if (rec && j.stage_pct > 0.02) left += rec.sec * (1 - j.stage_pct) / j.stage_pct;

  const curInfo = infoOf(cur);
  const done = new Set((j.ops || []).filter(o => o.slice_i === cur).map(o => o.op));
  plannedOps(cur, curInfo).forEach(op => { if (!done.has(op)) left += opCost(op, curInfo, ref); });

  for (let s = cur + 1; s < sliceCount(j); s++) {
    const info = infoOf(s);
    plannedOps(s, info).forEach(op => { left += opCost(op, info, ref); });
  }
  return left > 0 ? left : null;
}

function fmtSize(bytes) {
  if (!bytes) return "";
  const mb = bytes / (1024 * 1024);
  return mb >= 1024 ? (mb / 1024).toFixed(1) + " " + t("unit.gb") : Math.round(mb) + " " + t("unit.mb");
}

function fmtMedia(info) {          // «7:03 · 1920×1080 · 23.976 к/с · 10 141 кадр»
  if (!info) return "";
  const bits = [];
  if (info.duration_s) bits.push(fmtDur(info.duration_s));
  if (info.width) bits.push(info.width + "×" + info.height);
  if (info.fps) bits.push(info.fps.toFixed(3).replace(/\.?0+$/, "") + " " + t("unit.fps"));
  if (info.frames) bits.push(info.frames.toLocaleString(LANG === "en" ? "en-US" : "ru-RU") +
                             " " + plural(info.frames, "unit.frames"));
  if (!info.width) bits.push(t("dub.audio_only"));
  return bits.join(" · ");
}

function fmtDur(sec) {
  sec = Math.max(0, Math.round(sec || 0));
  const h = Math.floor(sec / 3600), m = Math.floor(sec % 3600 / 60), s = sec % 60;
  if (h) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

const jobDone = j => j.status === "done";
const trackCount = j => j.dub_total || (j.dubs || []).length || 1;
const sliceCount = j => 1 + trackCount(j);
/* какая доля обрабатывается сейчас: 0 — референс, 1..N — дорожка */
const curSlice = j => j.status === "running" ? (j.dub_index || 0) : -1;

/* Состояние операции берётся из журнала ядра (job.ops), а не угадывается по
   текущему этапу: только журнал знает, была ли необязательная операция пропущена. */
function opState(j, sliceI, op) {
  const rec = (j.ops || []).find(o => o.slice_i === sliceI && o.op === op);
  if (rec) return {state: rec.state === "run" ? "run" : rec.state, sec: rec.sec};
  // Записи нет — операция либо ещё предстоит, либо была пропущена. Пропущена, если
  // доля уже позади ИЛИ внутри неё началась операция, идущая ПОЗЖЕ по порядку.
  const logged = (j.ops || []).length > 0;
  if (!logged) return {state: "pending", sec: 0};   // журнала нет (задача из старой версии)
  const cur = curSlice(j);
  if (jobDone(j) || (cur >= 0 && sliceI < cur)) return {state: "skipped", sec: 0};
  const order = opsOf(sliceI).map(o => o.op);
  const reached = (j.ops || []).filter(o => o.slice_i === sliceI)
                               .some(o => order.indexOf(o.op) > order.indexOf(op));
  return {state: reached ? "skipped" : "pending", sec: 0};
}

function trackResult(j, sliceI) {          // результаты приходят в порядке обработки дорожек
  return (j.results || [])[sliceI - 1] || null;
}

function jobDot(status, results) {
  if (status === "running") return "run";
  // отказ ОДНОЙ дорожки не делает задачу сбойной: остальные обработаны
  if (status === "done") return results.every(r => !r.ok) ? "crit"
                              : results.some(r => !r.ok) ? "warn" : "ok";
  if (status === "failed") return "crit";
  return "wait";
}

const barSizes = new ResizeObserver(es => es.forEach(e => {
  const card = e.target.closest(".job");
  if (card) layoutBar(card);
}));

/* Карточки живут между обновлениями: полностью пересобираются только при смене
   «скелета» (состояние, число долей, число результатов), иначе обновляются на месте.
   Иначе список пересоздавался бы раз в секунду и срывал клики по кнопкам. */
const cards = new Map();
const cardSig = j => [j.status, sliceCount(j), (j.results || []).length].join("|");

function renderJobs(jobs) {
  const box = $("#jobs");
  if (!jobs.length) { box.replaceChildren(el("div", "empty", t("q.empty"))); cards.clear(); return; }
  const order = [...jobs].reverse();
  const alive = new Set(order.map(j => j.id));
  for (const [id, card] of cards) if (!alive.has(id)) { card.remove(); cards.delete(id); }

  order.forEach((j, i) => {
    let card = cards.get(j.id);
    try {
      if (!card || card.dataset.sig !== cardSig(j)) {
        const fresh = jobCard(j);
        fresh.dataset.sig = cardSig(j);
        if (card) card.replaceWith(fresh); else box.append(fresh);
        cards.set(j.id, fresh);
        card = fresh;
        layoutBar(card);
        const bar = card.querySelector(".pbar");
        if (bar) barSizes.observe(bar);   // подача подписей меняется вместе с шириной окна
      } else if (card.update) {
        card.update(j);
      }
    } catch (e) {
      // ошибка отрисовки ОДНОЙ задачи не должна прятать остальные
      console.error("не удалось отрисовать задачу", j && j.id, e);
      const stub = el("div", "job");
      stub.append(el("span", "dot crit"), el("span", "nm", (j && (j.label || j.id)) || "?"),
                  el("span", "metrics", t("q.render_error")));
      if (card) card.replaceWith(stub); else box.append(stub);
      cards.set(j.id, stub);
      card = stub;
    }
    if (box.children[i] !== card) box.insertBefore(card, box.children[i] || null);
  });
}

/* ── полоса: одна доля на референс + по доле на дорожку, внутри — операции ── */
function progressBar(j) {
  const slices = sliceCount(j), cur = curSlice(j);
  const bar = el("div", "pbar" + (jobDone(j) ? " done" : ""));
  bar.dataset.slices = slices;
  for (let s = 0; s < slices; s++) {
    if (s) bar.append(el("div", "brk"));
    const res = trackResult(j, s);
    const failed = s > 0 && res && !res.ok;      // отказ дорожки не касается остальных
    const reused = s > 0 && res && res.ok && res.skipped;   // выход уже был на диске
    opsOf(s).forEach(o => {
      const seg = el("div", "op");
      seg.style.setProperty("--w", o.w);
      const st = opState(j, s, o.op);
      let tip = (s === 0 ? t("q.reference") : t("q.track_n", {n: s})) + " · " + opName(s, o.op);
      if (reused) { seg.classList.add("skip"); tip += " — " + t("op.reused"); }
      else if (failed) { seg.classList.add("err"); tip += " — " + t("d.track_failed"); }
      else if (st.state === "done") { seg.classList.add("done"); tip += " · " + fmtDur(st.sec); }
      else if (st.state === "failed") seg.classList.add("err");
      else if (st.state === "skipped") { seg.classList.add("skip"); tip += " — " + t("op.skipped"); }
      else if (st.state === "run") {
        seg.classList.add("run");
        const f = el("i");
        f.style.width = Math.round((j.stage_pct || 0) * 100) + "%";
        seg.append(f);
      }
      seg.title = tip;
      bar.append(seg);
    });
  }
  return bar;
}

/* Подписи долей. Текст расставляет layoutBar — он один знает, сколько досталось места. */
function sliceLabels(j) {
  const slices = sliceCount(j);
  const row = el("div", "tnums");
  row.dataset.cur = curSlice(j);
  for (let s = 0; s < slices; s++) {
    if (s) row.append(el("i"));
    const res = trackResult(j, s);
    const cell = el("span", (curSlice(j) === s) ? "cur" : (s > 0 && res && !res.ok) ? "bad" : "");
    cell.style.flex = "100 1 0%";
    cell.title = s === 0 ? t("q.ref_prep") : trackTitle(j, s);
    row.append(cell);
  }
  return row;
}

/* Чем меньше места на долю, тем короче подпись — иначе при двадцати дорожках
   ярлыки налезут друг на друга. Решение по ИЗМЕРЕННОЙ ширине, не по ширине окна. */
function layoutBar(card) {
  const bar = card.querySelector(".pbar"); if (!bar) return;
  const slices = +bar.dataset.slices || 1;
  const per = (bar.clientWidth || 1) / slices;
  const lvl = per >= 88 ? 1 : per >= 30 ? 2 : per >= 10 ? 3 : 4;
  const setLvl = n => { n.classList.remove("lvl1", "lvl2", "lvl3", "lvl4"); n.classList.add("lvl" + lvl); };
  setLvl(bar);
  const nums = card.querySelector(".tnums"); if (!nums) return;
  setLvl(nums);
  const cur = +nums.dataset.cur;
  const tracks = slices - 1;
  [...nums.children].filter(n => n.tagName === "SPAN").forEach((cell, s) => {
    cell.replaceChildren();
    if (lvl === 1) cell.textContent = s === 0 ? t("q.reference") : t("q.track_n", {n: s});
    else if (lvl === 2) cell.textContent = s === 0 ? t("q.ref_short") : String(s);
    else if (cur >= 0 && s === cur) {
      const tag = el("em", null, s === 0 ? t("q.reference") : t("q.track_of", {i: s, n: tracks}));
      if (s <= 1) tag.className = "l"; else if (s >= slices - 2) tag.className = "r";
      cell.append(tag);
    }
  });
}

/* ── одна строка статуса: что идёт, у какой дорожки, сколько потрачено ── */
function statusLine(j) {
  const line = el("div", "stat");
  const tail = el("span", "tm");
  if (j.status === "running") {
    const s = j.dub_index || 0;
    line.append(el("span", "pulse"),
      el("span", "op", j.stage ? opName(s, j.stage) : "…"),
      el("span", "of", [s === 0 ? t("q.ref_prep") : t("q.track_of", {i: s, n: trackCount(j)}),
                        j.detail].filter(Boolean).join(" · ")));
    const parts = [t("q.elapsed", {v: fmtDur(j.elapsed_s)})];
    const left = estimateRemaining(j);                  // по нормативам + живой скорости
    if (left) parts.push(t("q.left", {v: fmtDur(left)}));
    tail.textContent = parts.join(" · ");
  } else if (j.status === "done") {
    const res = j.results || [];
    const bad = res.filter(r => !r.ok).length;
    const review = res.filter(r => r.ok && (r.suspect || (r.critical || []).length)).length;
    const extra = [bad ? t("q.with_errors", {n: bad}) : "",
                   review ? t("q.need_review", {n: review}) : ""].filter(Boolean);
    if (bad) line.classList.add(bad === res.length ? "crit" : "part");
    line.append(el("span", "op", bad ? t("q.finished_part") : t("q.finished")),
      el("span", "of", [t("q.ready_of", {ok: res.length - bad, n: res.length}), ...extra].join(" · ")));
    tail.textContent = t("q.took", {v: fmtDur(j.elapsed_s)});
  } else if (j.status === "failed") {
    line.classList.add("crit");
    line.append(el("span", "op", j.stage ? t("q.failed_at", {op: opName(j.dub_index || 0, j.stage)})
                                         : t("q.failed_plain")),
                el("span", "of", j.error || ""));
    tail.textContent = t("q.elapsed", {v: fmtDur(j.elapsed_s)});
  } else {
    const head = {queued: "q.in_queue", paused: "q.on_hold", cancelled: "q.was_cancelled"}[j.status];
    line.append(el("span", "op", t(head || "q.in_queue")),
      el("span", "of", [t("q.tracks_count", {n: trackCount(j)}),
                        j.status === "queued" ? t("q.will_start") : ""].filter(Boolean).join(" · ")));
    if (j.elapsed_s) tail.textContent = t("q.elapsed", {v: fmtDur(j.elapsed_s)});
  }
  line.append(tail);
  return line;
}

/* ── скрываемое: имя, состояние и время КАЖДОЙ операции ── */
function opsList(j, sliceI) {
  const grid = el("div", "ops");
  const res = trackResult(j, sliceI);
  const reused = sliceI > 0 && res && res.ok && res.skipped;
  opsOf(sliceI).forEach(o => {
    const st = reused ? {state: "reused", sec: 0} : opState(j, sliceI, o.op);
    const mark = {done: "✓", run: "◐", failed: "✕", skipped: "⌀", reused: "⌀", pending: "·"}[st.state];
    const cls = {done: "ok", run: "run", failed: "err"}[st.state] || "";
    const nm = o.opt && st.state === "pending" ? t("op.optional", {op: opName(sliceI, o.op)})
                                               : opName(sliceI, o.op);
    let val = t("op.pending");
    if (st.state === "done" || st.state === "failed") val = fmtDur(st.sec);
    else if (st.state === "skipped") val = t("op.skipped");
    else if (st.state === "reused") val = t("op.reused");
    else if (st.state === "run") {
      val = Math.round((j.stage_pct || 0) * 100) + " %" + (j.detail ? " · " + j.detail : "");
    }
    grid.append(el("div", "st " + cls, mark),
                el("div", "nmc " + (st.state === "run" ? "now" : st.state === "pending" ? "fut" : ""), nm),
                el("div", "tmc " + (st.state === "run" ? "live" : ""), val));
  });
  return grid;
}

function trackTitle(j, sliceI) {
  const res = trackResult(j, sliceI);
  const path = (res && res.dub) || (j.dubs || [])[sliceI - 1] || "";
  const atr = (j.dub_atracks || [])[sliceI - 1];
  const num = (atr == null ? 0 : atr) + 1;
  return t("q.track_n", {n: sliceI}) + " · " + baseName(path) + (num > 1 ? " · #" + num : "");
}

function jobCard(j) {
  const card = el("div", "job" + (j.status === "running" ? " act" : ""));
  const head = el("div", "head");
  head.append(el("span", "dot " + jobDot(j.status, j.results || [])),
              el("span", "nm", j.label || baseName(j.ref) || j.id));
  head.append(el("span", "pct", j.status === "running" ? Math.round((j.progress || 0) * 100) + " %" : ""));

  const caret = el("button", "caret", openJobs.has(j.id) ? "▾" : "▸");
  caret.title = t("d.details");
  head.append(caret);
  if (j.status === "paused") head.append(rowBtn("▶", t("q.start"), () => act("start", j.id)));
  if (j.status === "queued") head.append(rowBtn("⏸", t("q.pause"), () => act("pause", j.id)));
  head.append(rowBtn("✕", t("q.cancel"), () => act("cancel", j.id)));

  card.append(head, sliceLabels(j), progressBar(j), statusLine(j));

  const more = el("div", "more" + (openJobs.has(j.id) ? " on" : ""));
  const opsSliceOf = jj => { const s = curSlice(jj); return Math.max(0, s >= 0 ? s : (jj.results || []).length); };
  const opsHead = el("p", "sub");
  const opsBox = el("div");
  const fillOps = jj => {
    const s = opsSliceOf(jj);
    opsHead.textContent = t("q.ops_of", {who: s === 0 ? t("q.ref_prep") : trackTitle(jj, s)});
    opsBox.replaceChildren(opsList(jj, s));
  };
  fillOps(j);
  more.append(el("p", "sub", t("q.reference")), el("div", "hint", j.ref));
  const refLine = el("div", "hint", fmtMedia(j.ref_info));
  const refShow = jj => { refLine.textContent = fmtMedia(jj.ref_info); };  // паспорт приходит на старте
  more.append(refLine, opsHead, opsBox);
  more.append(el("p", "sub", jobDone(j) ? t("q.results") : t("q.tracks_n", {n: trackCount(j)})));
  const list = el("div", "trks");
  for (let i = 1; i <= trackCount(j); i++) list.append(trackRow(j, i));
  more.append(list);

  // обновление на месте: скелет тот же, меняются только числа и состояния
  card.update = jj => {
    card.classList.toggle("act", jj.status === "running");
    head.querySelector(".pct").textContent =
      jj.status === "running" ? Math.round((jj.progress || 0) * 100) + " %" : "";
    card.replaceChild(progressBar(jj), card.querySelector(".pbar"));
    card.replaceChild(statusLine(jj), card.querySelector(".stat"));
    layoutBar(card);
    if (more.classList.contains("on")) { fillOps(jj); refShow(jj); }
  };

  caret.onclick = () => {
    const on = more.classList.toggle("on");
    caret.textContent = on ? "▾" : "▸";
    on ? openJobs.add(j.id) : openJobs.delete(j.id);
    if (on) fillOps(j);
  };
  card.append(more);
  return card;
}

function rowBtn(txt, title, fn) {
  const b = el("button", "x", txt);
  b.title = title;
  b.onclick = fn;
  return b;
}

/* Строка дорожки: два исхода — результат или причина отказа. Дорожка, которой
   ещё нет в results, показывается ожидающей или обрабатываемой. */
function trackRow(job, sliceI) {
  const r = trackResult(job, sliceI);
  const cur = curSlice(job);
  const wrap = el("div", "trk");
  const line = el("div", "tline");

  const failed = r && !r.ok;
  const review = r && r.ok && (r.suspect || (r.critical || []).length);
  const dot = failed ? "crit" : review ? "warn" : r ? "ok" : (cur === sliceI ? "run" : "wait");
  line.append(el("span", "dot " + dot));

  const path = (r && r.dub) || (job.dubs || [])[sliceI - 1] || "";
  const atr = (job.dub_atracks || [])[sliceI - 1];
  const who = el("span", "who");
  who.append(el("b", null, t("q.track_n", {n: sliceI})),
             el("s", null, " " + baseName(path) + (atr ? " · #" + (atr + 1) : "")));
  who.title = path;
  line.append(who);

  if (r && r.ok) {
    line.append(el("span", "val",
      t("d.resid", {v: (r.audio_resid_ms || 0).toFixed(1)}) + " · " +
      t("d.coverage", {v: (r.audio_coverage || 0).toFixed(2)})));
  } else if (failed) {
    line.append(el("span", "val bad", t("d.track_failed")));
  } else {
    line.append(el("span", "val", cur === sliceI ? t("d.track_running") : t("d.track_waiting")));
  }
  if (r && r.mode === "audio") line.append(el("span", "pill", t("dub.audio_only")));
  if (r && r.skipped) line.append(el("span", "pill", t("d.skipped")));

  const detail = el("div", "detail");
  if (r && r.ok) {
    const key = job.id + "|" + r.dub + "|" + (r.out_path || "");
    if (openDetails.has(key)) detail.classList.add("on");
    const push = el("span", "push");
    const more = el("button", "link", (openDetails.has(key) ? "▾ " : "▸ ") + t("d.details"));
    more.onclick = () => {
      const on = detail.classList.toggle("on");
      on ? openDetails.add(key) : openDetails.delete(key);
      more.textContent = (on ? "▾ " : "▸ ") + t("d.details");
      if (on && !detail.dataset.built) buildDetail(detail, job, r);
    };
    push.append(more);
    if (r.out_path) {
      const f = el("button", "link", t("d.folder"));
      f.onclick = () => api("POST", "/ui/reveal", {path: r.out_path}).catch(() => {});
      push.append(f);
    }
    line.append(push);
    if (openDetails.has(key)) buildDetail(detail, job, r);
  }
  wrap.append(line);

  // причина отказа стоит ПРИ своей дорожке, а не общим журналом внизу
  if (failed) {
    // ⚠ пустой массив в JS «истинный»: выражение (r.critical || [r.error])[0] брало
    // ПУСТОЙ critical, давало undefined и роняло отрисовку — а с ней и весь список.
    const msg = (r.critical && r.critical.length ? r.critical[0] : r.error) || t("d.failed");
    wrap.append(el("div", "terr", String(msg)));
  }
  // находки — своей строкой под дорожкой, к которой относятся
  const pills = [];
  if (r && r.ok) {
    if (r.geom_used) pills.push({t: t("f.geom", {sx: (r.geom_sx || 0).toFixed(3)})});
    if (r.audio_cuts) pills.push({t: pluralF(r.audio_cuts, "f.cuts",
                                             {n: r.audio_cuts, ms: Math.round(r.audio_max_step_ms)})});
    if (r.filled_cuts) pills.push({t: t("f.filled", {n: r.filled_cuts})});
    if (r.blind_zones) pills.push({t: t("f.blind", {n: r.blind_zones}), c: "w"});
    (r.warnings || []).forEach(w => pills.push({t: w, c: "w"}));
    (r.critical || []).forEach(w => pills.push({t: w, c: "c"}));
    if (r.out_path) pills.push({t: baseName(r.out_path) + (r.out_size ? " · " + fmtSize(r.out_size) : "")});
  }
  if (pills.length) {
    const box = el("div", "tpills");
    pills.forEach(p => box.append(el("span", "pill " + (p.c || ""), p.t)));
    wrap.append(box);
  }
  wrap.append(detail);
  return wrap;
}

function buildDetail(box, job, r) {
  box.dataset.built = "1";
  const grid = el("div", "mgrid");
  const cell = (k, v) => { const c = el("div", "mcell"); c.append(el("b", null, k), el("span", null, v)); grid.append(c); };
  cell(t("m.resid"), (r.audio_resid_ms || 0).toFixed(1) + " " + t("unit.ms"));
  cell(t("m.coverage"), (r.audio_coverage || 0).toFixed(2));
  if (r.mode === "audio") cell(t("m.mode_audio"), t("dub.audio_only"));
  else {
    cell(t("m.assigned"), Math.round(r.assigned_pct || 0) + " %");
    cell(t("m.cos"), (r.cos_median || 0).toFixed(2));
    cell(t("m.slope"), (r.slope || 0).toFixed(4));
  }
  cell(t("m.cuts"), r.audio_cuts ? `${r.audio_cuts} · ${Math.round(r.audio_max_step_ms)} ${t("unit.ms")}` : "0");
  cell(t("m.filled"), (r.filled_cuts || 0) + " " + t("unit.s"));
  cell(t("m.span"), Math.round(r.audio_span_ms || 0) + " " + t("unit.ms"));
  cell(t("m.geom"), r.geom_used ? `sx ${(r.geom_sx || 0).toFixed(3)} · sy ${(r.geom_sy || 0).toFixed(3)}` : "—");
  box.append(grid);

  // Ядро отдаёт и картинки (track/cut), и интерактивные HTML-графики. Показываем
  // превью ТОЛЬКО картинок; html — это ссылка «открыть», а не изображение.
  const all = r.plots || [];
  const plots = all.filter(p => (p.name || "").endsWith(".png"));
  const htmlByName = new Map(all.filter(p => (p.name || "").endsWith(".html"))
                                .map(p => [p.name.replace(/\.html$/, ""), p.name]));
  if (plots.length) {
    const strip = el("div", "previews");
    plots.slice(0, 4).forEach(p => {
      const b = el("div", "preview");
      const img = el("img");
      img.src = `${API}/conform/plot/${job.id}/${p.name}`;
      img.onclick = () => { $("#modal-img").src = img.src; $("#modal").classList.add("on"); };
      b.append(img);
      const cap = el("b", null, p.kind === "cut"
        ? t("p.cut", {t: Math.round(p.t || 0), v: (p.v_ms > 0 ? "+" : "") + Math.round(p.v_ms || 0)})
        : t("p.track"));
      const html = htmlByName.get(p.name.replace(/\.png$/, ""));
      if (html) {                       // интерактивная версия существует — даём ссылку
        const open = el("button", "link", t("p.open"));
        open.onclick = () => api("POST", "/ui/open_url", {url: `${API}/conform/plot/${job.id}/${html}`}).catch(() => {});
        cap.append(" ", open);
      }
      b.append(cap);
      strip.append(b);
    });
    box.append(strip);
  }
}

async function act(what, jid) {
  const call = {start: () => api("POST", `/conform/jobs/${jid}/start`),
                pause: () => api("POST", `/conform/jobs/${jid}/pause`),
                cancel: () => api("DELETE", `/conform/jobs/${jid}`)}[what];
  try { await call(); await refreshJobs(); } catch (e) { toast(t("err.api", {e: e.message})); }
}

let polling = false;
async function refreshJobs() {
  if (polling) return;
  polling = true;
  try {
    const jobs = await api("GET", "/conform/jobs");
    renderJobs(jobs);
    const n = jobs.filter(j => ["running", "queued", "paused"].includes(j.status)).length;
    $("#tab-queue").innerHTML = t("tab.queue") + (n ? `<span class="n">${n}</span>` : "");
  } catch (_) { /* сервер ещё поднимается */ }
  polling = false;
}

/* ═══════════ запуск ═══════════ */
function bind() {
  // одна привязка на всё приложение: раньше слушатель вешался в trackPicker и копился
  // при каждой перерисовке списка озвучек
  document.addEventListener("click", closeTrackMenus);
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeTrackMenus(); });

  document.querySelectorAll(".tab").forEach(b => b.onclick = () => showPage(b.dataset.page));
  $("#theme").onclick = () => applyTheme({auto: "light", light: "dark", dark: "auto"}[state.theme]);
  $("#theme").title = t("theme.tip");
  $("#lang-ru").onclick = () => switchLang("ru");
  $("#lang-en").onclick = () => switchLang("en");

  $("#ref-browse").onclick = async () => { const p = await pickFiles("ref"); if (p && p.paths[0]) setRef(p.paths[0]); };
  $("#ref").onchange = () => setRef($("#ref").value.trim());
  $("#ref-track").onchange = e => { state.refTrack = parseInt(e.target.value, 10) || 0; };
  $("#rest-as-dubs").onclick = () => {
    if (state.refTracks.length < 2) return;
    const rest = state.refTracks.map((_, i) => i).filter(i => i !== state.refTrack);
    state.dubs = state.dubs.filter(d => d.path !== state.ref);
    state.dubs.push({path: state.ref, tracks: state.refTracks, sel: rest, audioOnly: false});
    renderDubs();
  };
  $("#dubs-add").onclick = async () => { const p = await pickFiles("dubs"); if (p) for (const f of p.paths) await addDub(f); };
  $("#out-browse").onclick = async () => { const p = await pickFiles("dir"); if (p && p.paths[0]) { $("#out").value = p.paths[0]; save("out", p.paths[0]); } };
  $("#tmp-browse").onclick = async () => { const p = await pickFiles("dir"); if (p && p.paths[0]) { $("#tmpdir").value = p.paths[0]; save("tmpdir", p.paths[0]); } };

  $("#settings-head").onclick = () => {
    const g = $("#settings"); const on = g.classList.toggle("open");
    $("#settings-head").textContent = (on ? "▾ " : "▸ ") + t("settings");
    save("settings_open", on);
  };
  $("#keeptmp").onchange = e => { $("#tmp-row").hidden = !e.target.checked; save("keeptmp", e.target.checked); };
  $("#fill").onchange = e => save("fill", e.target.checked);
  $("#autostart").onchange = e => save("autostart", e.target.checked);
  $("#a-muq").onchange = e => save("muq", e.target.checked);
  $("#out").onchange = e => save("out", e.target.value.trim());
  $("#tmpdir").onchange = e => save("tmpdir", e.target.value.trim());

  const step = d => {
    const v = Math.min(10, Math.max(0.1, (parseFloat($("#drift").value) || 1.25) + d));
    $("#drift").value = v.toFixed(2);
    save("drift", v);
  };
  $("#drift-up").onclick = () => step(0.25);
  $("#drift-down").onclick = () => step(-0.25);
  $("#drift").onchange = () => step(0);

  $("#enqueue").onclick = enqueue;
  $("#clear-done").onclick = async () => {
    try { const r = await api("POST", "/conform/clear_done"); toast(t("q.cleared", {n: r.cleared, mb: r.freed_mb})); await refreshJobs(); }
    catch (e) { toast(t("err.api", {e: e.message})); }
  };
  $("#limit").onchange = e => api("PUT", "/conform/settings", {limit: parseInt(e.target.value, 10)}).catch(() => {});
  $("#modal").onclick = () => $("#modal").classList.remove("on");

  // перетаскивание файлов
  document.addEventListener("dragover", e => e.preventDefault());
  document.addEventListener("drop", async e => {
    e.preventDefault();
    const paths = [...e.dataTransfer.files].map(f => f.path).filter(Boolean);
    for (const p of paths) { if (!state.ref) await setRef(p); else await addDub(p); }
  });
}

function switchLang(lang) {
  setLang(lang);
  save("lang", lang);
  $("#lang-ru").classList.toggle("on", lang === "ru");
  $("#lang-en").classList.toggle("on", lang === "en");
  $("#settings-head").textContent = ($("#settings").classList.contains("open") ? "▾ " : "▸ ") + t("settings");
  $("#tab-task").textContent = t("tab.task");
  renderDubs();
  refreshJobs();
}

async function boot() {
  const p = loadPrefs();
  bind();
  switchLang(p.lang || "ru");
  applyTheme(p.theme || "auto");
  if (p.out) $("#out").value = p.out;
  if (p.tmpdir) $("#tmpdir").value = p.tmpdir;
  $("#fill").checked = p.fill !== false;
  $("#keeptmp").checked = !!p.keeptmp;
  $("#tmp-row").hidden = !p.keeptmp;
  $("#autostart").checked = !!p.autostart;
  if (p.muq) $("#a-muq").checked = true;
  if (p.drift) $("#drift").value = Number(p.drift).toFixed(2);
  if (p.settings_open) $("#settings").classList.add("open");
  $("#settings-head").textContent = (p.settings_open ? "▾ " : "▸ ") + t("settings");

  for (let i = 1; i <= 8; i++) $("#limit").append(new Option(String(i), String(i)));
  try {
    const s = await api("GET", "/conform/settings");
    $("#limit").value = String(s.limit || 1);
  } catch (_) {}
  try {
    const d = await api("GET", "/conform/device");
    $("#device").textContent = d.gpu ? "GPU · " + d.name : "CPU";
    $("#device").classList.toggle("gpu", !!d.gpu);
    if (!d.gpu) { $("#a-muq").parentElement.querySelectorAll("#a-muq, label[for=a-muq]").forEach(x => x.remove()); $("#muq-note").remove(); }
  } catch (_) {}

  refreshJobs();
  setInterval(refreshJobs, 1000);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyTheme(state.theme));
}

document.addEventListener("DOMContentLoaded", boot);
