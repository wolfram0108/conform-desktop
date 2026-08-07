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

function jobDot(status, results) {
  if (status === "running") return "run";
  if (status === "done") return results.some(r => !r.ok) ? "crit" : "ok";
  if (status === "failed") return "crit";
  return "";
}

function renderJobs(jobs) {
  const box = $("#jobs");
  box.innerHTML = "";
  if (!jobs.length) { box.append(el("div", "empty", t("q.empty"))); return; }
  // каждая карточка — отдельно: ошибка отрисовки ОДНОЙ задачи не должна прятать
  // остальные (контейнер уже очищен, и пользователь увидел бы пустую очередь)
  [...jobs].reverse().forEach(j => {
    try {
      box.append(jobCard(j));
    } catch (e) {
      console.error("не удалось отрисовать задачу", j && j.id, e);
      const stub = el("div", "job");
      stub.append(el("span", "dot crit"), el("span", "jname", (j && (j.label || j.id)) || "?"),
                  el("span", "metrics", t("q.render_error")));
      box.append(stub);
    }
  });
}

function jobCard(j) {
  const card = el("div", "job");
  const row = el("div", "jrow");
  row.append(el("span", "dot " + jobDot(j.status, j.results || [])));
  row.append(el("span", "jname", j.label || j.id));

  const meta = el("span", "jmeta");
  if (j.status === "running") {
    const bits = [];
    if (j.stage) bits.push(t("stage." + j.stage));
    if (j.dub_total) bits.push(t("q.dub_of", {i: j.dub_index, n: j.dub_total}));
    if (j.cur_dub) bits.push(baseName(j.cur_dub));
    meta.textContent = bits.join(" · ");
  } else if (j.status === "done") {
    const res = j.results || [], ok = res.filter(r => r.ok).length;
    // задача, где НИ ОДНА озвучка не удалась, не должна выглядеть выполненной:
    // раньше она подписывалась «готово · 0/1 ok» и читалась как успех
    const head = res.length && ok === 0 ? t("q.done_failed") : t("q.done");
    meta.textContent = `${head} · ${Math.round(j.elapsed_s)} ${t("unit.s")} · ${ok}/${res.length} ok`;
    if (res.length && ok === 0) meta.style.color = "var(--crit)";
  } else if (j.status === "failed") {
    meta.textContent = t("q.failed") + (j.error ? " · " + j.error.slice(0, 80) : "");
  } else {
    meta.textContent = t("q." + j.status) + " · " + (j.dub_total || 0);
  }
  row.append(meta);

  if (j.status === "running") {
    const bar = el("span", "bar"); const fill = el("i");
    fill.style.width = Math.round((j.progress || 0) * 100) + "%";
    bar.append(fill); row.append(bar);
    row.append(el("span", "metrics", Math.round((j.progress || 0) * 100) + "%"));
  }

  if (j.status === "paused") row.append(rowBtn("▶", t("q.start"), () => act("start", j.id)));
  if (j.status === "queued") row.append(rowBtn("⏸", t("q.pause"), () => act("pause", j.id)));
  row.append(rowBtn("✕", t("q.cancel"), () => act("cancel", j.id)));
  card.append(row);

  if ((j.results || []).length) {
    const sub = el("div", "subrows");
    j.results.forEach(r => sub.append(dubRow(j, r)));
    card.append(sub);
  }
  return card;
}

function rowBtn(txt, title, fn) {
  const b = el("button", "x", txt);
  b.title = title;
  b.onclick = fn;
  return b;
}

function dubRow(job, r) {
  const wrap = el("div");
  const level = (r.critical && r.critical.length) || !r.ok ? "crit" : (r.suspect ? "warn" : "");
  const row = el("div", "drow " + level);
  row.append(el("span", "dot " + (level || "ok")));
  const name = el("span", "name", baseName(r.out_path || r.dub));
  name.title = r.out_path || r.dub;
  name.style.cssText = "flex:1;min-width:60px;font-family:Cascadia Mono,Consolas,monospace;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
  row.append(name);
  if (r.mode === "audio") row.append(el("span", "tag", t("dub.audio_only")));
  if (r.skipped) row.append(el("span", "tag", t("d.skipped")));
  if (r.ok) row.append(el("span", "metrics",
    t("d.resid", {v: (r.audio_resid_ms || 0).toFixed(1)}) + " · " + t("d.coverage", {v: (r.audio_coverage || 0).toFixed(2)})));
  if (level === "warn") { const w = el("span", "metrics", t("d.suspect")); w.style.color = "var(--warn)"; w.title = (r.warnings || []).join("\n"); row.append(w); }
  if (level === "crit") {
    // ⚠ пустой массив в JS «истинный»: выражение (r.critical || [r.error])[0] брало
    // ПУСТОЙ critical, давало undefined и роняло отрисовку — а вместе с ней ВЕСЬ список
    // очереди (контейнер уже очищен). Ломалась ровно та задача, у которой была ошибка.
    const msg = (r.critical && r.critical.length ? r.critical[0] : r.error) || t("d.failed");
    const w = el("span", "metrics", String(msg).slice(0, 60));
    w.style.color = "var(--crit)"; w.title = r.error || String(msg); row.append(w);
  }

  const key = job.id + "|" + r.dub + "|" + (r.out_path || "");
  const detail = el("div", "detail" + (openDetails.has(key) ? " on" : ""));
  if (r.ok) {
    const more = el("button", "link", (openDetails.has(key) ? "▾ " : "▸ ") + t("d.details"));
    more.onclick = () => {
      const on = detail.classList.toggle("on");
      on ? openDetails.add(key) : openDetails.delete(key);
      more.textContent = (on ? "▾ " : "▸ ") + t("d.details");
      if (on && !detail.dataset.built) buildDetail(detail, job, r);
    };
    row.append(more);
    if (openDetails.has(key)) buildDetail(detail, job, r);
  }
  if (r.out_path) {
    const f = el("button", "link", t("d.folder"));
    f.onclick = () => api("POST", "/ui/reveal", {path: r.out_path}).catch(() => {});
    row.append(f);
  }
  wrap.append(row, detail);
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
