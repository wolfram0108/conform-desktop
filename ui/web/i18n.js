/* Двуязычие ru/en. Ключи проставляются в разметке атрибутом data-i,
   динамические строки берутся функцией t(). */
const STRINGS = {
  "tab.task":        {ru:"Задача", en:"Task"},
  "tab.queue":       {ru:"Очередь", en:"Queue"},
  "ref":             {ru:"Референс:", en:"Reference:"},
  "ref.track":       {ru:"дорожка референса:", en:"reference track:"},
  "ref.rest_as_dubs":{ru:"+ остальные дорожки этого файла", en:"+ other tracks of this file"},
  "tracks.n":        {ru:"дорожек: {n}", en:"tracks: {n}"},
  "tracks.chosen":   {ru:"дорожки: {k} / {n}", en:"tracks: {k} / {n}"},
  "tracks.none":     {ru:"не выбрано", en:"none selected"},
  "tracks.all":      {ru:"выбрать все / снять все", en:"select all / clear"},
  "dubs":            {ru:"Исходные файлы:", en:"Source files:"},
  "dubs.add":        {ru:"+ Добавить файлы", en:"+ Add files"},
  "dubs.drop":       {ru:"или перетащите файлы в окно", en:"or drop files onto the window"},
  "dub.audio_only":  {ru:"аудио-only", en:"audio-only"},
  "dub.virtual":     {ru:"тот же файл, другая дорожка", en:"same file, another track"},
  "dub.same_as_ref": {ru:"= референс", en:"= reference"},
  "out_dir":         {ru:"Выходной каталог:", en:"Output folder:"},
  "browse":          {ru:"Обзор…", en:"Browse…"},
  "settings":        {ru:"Настройки", en:"Settings"},
  "set.analysis":    {ru:"Аудио-анализ:", en:"Audio analysis:"},
  "set.band":        {ru:"band (стандарт)", en:"band (default)"},
  "set.muq_note":    {ru:"MuQ — экспериментальная модель, нужна NVIDIA-карта. Веса загружаются при первом включении; лицензия CC-BY-NC 4.0 — только некоммерческое использование результата.",
                      en:"MuQ is experimental and needs an NVIDIA GPU. Weights download on first use; CC-BY-NC 4.0 — non-commercial use of results only."},
  "set.fill":        {ru:"Заполнять тишину звуком референса", en:"Fill silence from the reference"},
  "set.drift":       {ru:"Потолок скорости дрейфа:", en:"Drift speed ceiling:"},
  "set.drift_unit":  {ru:"%/с", en:"%/s"},
  "set.keep_tmp":    {ru:"Сохранять промежуточные файлы (повтор без пересчёта)", en:"Keep intermediate files (instant re-runs)"},
  "set.keep_tmp_dir":{ru:"каталог для промежуточных файлов", en:"folder for intermediate files"},
  "task.label":      {ru:"Название:", en:"Label:"},
  "task.enqueue":    {ru:"Добавить в очередь", en:"Add to queue"},
  "task.added":      {ru:"Задача добавлена — запустите её кнопкой ▶ (дорожек: {n})", en:"Task added — press ▶ to start it ({n} tracks)"},
  "task.added_run":  {ru:"Задача добавлена и запущена (дорожек: {n})", en:"Task added and started ({n} tracks)"},
  "task.need_ref":   {ru:"Укажите файл референса", en:"Choose a reference file"},
  "task.need_dubs":  {ru:"Добавьте хотя бы один исходный файл", en:"Add at least one source file"},
  "task.need_out":   {ru:"Укажите выходной каталог", en:"Choose an output folder"},
  "q.autostart":     {ru:"запускать сразу после добавления", en:"start right after adding"},
  "q.parallel":      {ru:"параллельно:", en:"parallel:"},
  "q.clear_done":    {ru:"Очистить готовые", en:"Clear finished"},
  "q.cleared":       {ru:"Очищено задач: {n}, освобождено {mb} МБ", en:"Cleared {n} tasks, freed {mb} MB"},
  "q.empty":         {ru:"Очередь пуста — соберите задачу на вкладке «Задача»", en:"Queue is empty — build a task on the “Task” tab"},
  "q.queued":        {ru:"в очереди", en:"queued"},
  "q.paused":        {ru:"на паузе", en:"paused"},
  "q.done":          {ru:"готово", en:"done"},
  "q.done_failed":   {ru:"без результата", en:"no output"},
  "q.render_error":  {ru:"строка задачи не отрисовалась — см. журнал", en:"row failed to render — see log"},
  "q.failed":        {ru:"ошибка", en:"failed"},
  "q.cancelled":     {ru:"отменено", en:"cancelled"},
  "q.start":         {ru:"Запустить задачу", en:"Start the task"},
  "q.pause":         {ru:"Снять с автозапуска", en:"Hold the task"},
  "q.cancel":        {ru:"Отменить (убивает её процессы)", en:"Cancel (kills its processes)"},
  "d.details":       {ru:"подробнее", en:"details"},
  "d.folder":        {ru:"папка", en:"folder"},
  "d.resid":         {ru:"рассогласование {v} мс", en:"residual offset {v} ms"},
  "d.coverage":      {ru:"покрытие {v}", en:"coverage {v}"},
  "d.suspect":       {ru:"⚠ проверьте результат", en:"⚠ review the result"},
  "d.failed":        {ru:"не удалось — см. журнал", en:"failed — see log"},
  "d.skipped":       {ru:"уже было готово", en:"already done"},
  /* показатели результата — термины из словаря миссии, без жаргона */
  "m.resid":         {ru:"остаточное рассогласование", en:"residual offset"},
  "m.coverage":      {ru:"покрытие измерением", en:"measurement coverage"},
  "m.assigned":      {ru:"сопоставлено", en:"matched"},
  "m.cos":           {ru:"сходство кадров", en:"frame similarity"},
  "m.slope":         {ru:"ход времени", en:"time rate"},
  "m.cuts":          {ru:"разрывы", en:"discontinuities"},
  "m.filled":        {ru:"заполнено референсом", en:"filled from reference"},
  "m.span":          {ru:"диапазон сдвига", en:"shift span"},
  "m.geom":          {ru:"геометрия", en:"geometry"},
  "m.mode_audio":    {ru:"режим", en:"mode"},
  "p.track":         {ru:"сдвиг звука — вся дорожка", en:"audio shift — full track"},
  "p.cut":           {ru:"разрыв @{t} с · {v} мс", en:"discontinuity @{t} s · {v} ms"},
  "p.open":          {ru:"открыть ↗", en:"open ↗"},
  /* Названия операций — из словаря терминов миссии (GLOSSARY_terms.md).
     Одна операция — один термин, одинаково в интерфейсе, журнале и отчётах. */
  "op.decode":       {ru:"Признаки кадров", en:"Frame features"},
  "op.extract":      {ru:"Декодирование звука", en:"Audio decoding"},
  "op.coarse":       {ru:"Грубое соответствие", en:"Coarse correspondence"},
  "op.geom":         {ru:"Геометрическая коррекция", en:"Geometric correction"},
  "op.band":         {ru:"Карта соответствия кадров", en:"Frame correspondence map"},
  "op.resample":     {ru:"Временное преобразование", en:"Temporal resampling"},
  "op.audio":        {ru:"Уточнение по звуку", en:"Audio refinement"},
  "op.write":        {ru:"Кодирование выхода", en:"Output encoding"},
  "op.ref.decode":   {ru:"Признаки кадров референса", en:"Reference frame features"},
  "op.ref.extract":  {ru:"Декодирование звука референса", en:"Reference audio decoding"},
  "op.optional":     {ru:"{op} (если потребуется)", en:"{op} (if needed)"},
  "op.skipped":      {ru:"не потребовалась", en:"not needed"},
  "op.reused":       {ru:"выход уже был готов", en:"output already existed"},
  "op.pending":      {ru:"—", en:"—"},

  /* ход выполнения */
  "q.reference":     {ru:"референс", en:"reference"},
  "q.ref_short":     {ru:"реф", en:"ref"},
  "q.ref_prep":      {ru:"подготовка референса", en:"preparing reference"},
  "q.track_n":       {ru:"дорожка {n}", en:"track {n}"},
  "q.track_of":      {ru:"дорожка {i} из {n}", en:"track {i} of {n}"},
  "q.ops":           {ru:"операции", en:"operations"},
  "q.ops_of":        {ru:"операции · {who}", en:"operations · {who}"},
  "q.tracks_n":      {ru:"дорожки задачи · {n}", en:"task tracks · {n}"},
  "q.results":       {ru:"результаты", en:"results"},
  "q.in_queue":      {ru:"В очереди", en:"Queued"},
  "q.on_hold":       {ru:"Снята с автозапуска", en:"On hold"},
  "q.was_cancelled": {ru:"Отменена", en:"Cancelled"},
  "q.finished":      {ru:"Завершена", en:"Completed"},
  "q.finished_part": {ru:"Завершена частично", en:"Completed partially"},
  "q.ready_of":      {ru:"{ok} из {n} дорожек готовы", en:"{ok} of {n} tracks ready"},
  "q.with_errors":   {ru:"{n} с ошибкой", en:"{n} failed"},
  "q.need_review":   {ru:"{n} требует проверки", en:"{n} need review"},
  "q.failed_at":     {ru:"Сбой на операции «{op}»", en:"Failed at “{op}”"},
  "q.failed_plain":  {ru:"Сбой задачи", en:"Task failed"},
  "q.elapsed":       {ru:"прошло {v}", en:"elapsed {v}"},
  "q.left":          {ru:"осталось ~ {v}", en:"~ {v} left"},
  "q.took":          {ru:"заняло {v}", en:"took {v}"},
  "q.tracks_count":  {ru:"дорожек: {n}", en:"tracks: {n}"},
  "q.will_start":    {ru:"старт по освобождении места", en:"starts when a slot frees up"},
  "d.track_failed":  {ru:"дорожка не обработана", en:"track not processed"},
  "d.track_running": {ru:"обрабатывается", en:"processing"},
  "d.track_waiting": {ru:"ждёт", en:"waiting"},
  "d.out_size":      {ru:"{f}", en:"{f}"},

  /* находки обработки — стоят при своей дорожке */
  "f.geom":          {ru:"геометрия скорректирована · sx {sx}", en:"geometry corrected · sx {sx}"},
  "f.cuts":          {ru:"{n} разрывов · макс {ms} мс", en:"{n} discontinuities · max {ms} ms"},
  "f.cuts.one":      {ru:"{n} разрыв · макс {ms} мс", en:"{n} discontinuity · max {ms} ms"},
  "f.cuts.few":      {ru:"{n} разрыва · макс {ms} мс", en:"{n} discontinuities · max {ms} ms"},
  "f.filled":        {ru:"заполнено референсом {n} с", en:"filled from reference {n} s"},
  "f.blind":         {ru:"зон без опоры: {n}", en:"unsupported zones: {n}"},
  "theme.tip":       {ru:"Тема: системная / светлая / тёмная", en:"Theme: system / light / dark"},
  "err.api":         {ru:"Ошибка: {e}", en:"Error: {e}"},
  "unit.ms":         {ru:"мс", en:"ms"},
  "unit.s":          {ru:"с", en:"s"},
  "unit.mb":         {ru:"МБ", en:"MB"},
  "unit.gb":         {ru:"ГБ", en:"GB"},
  "unit.fps":        {ru:"к/с", en:"fps"},
  /* три формы для русского счёта: 1 кадр, 2 кадра, 5 кадров */
  "unit.frames":     {ru:"кадров", en:"frames"},
  "unit.frames.one": {ru:"кадр", en:"frame"},
  "unit.frames.few": {ru:"кадра", en:"frames"},
};

let LANG = "ru";

/* Русский счёт требует трёх форм. Английский берёт «one» для единицы и общий ключ иначе. */
function pluralKey(n, key) {
  n = Math.abs(Math.round(n));
  if (LANG === "en") return n === 1 ? key + ".one" : key;
  const d10 = n % 10, d100 = n % 100;
  if (d10 === 1 && d100 !== 11) return key + ".one";
  if (d10 >= 2 && d10 <= 4 && (d100 < 12 || d100 > 14)) return key + ".few";
  return key;
}
const plural = (n, key) => t(pluralKey(n, key));
const pluralF = (n, key, vars) => t(pluralKey(n, key), vars);

function t(key, vars) {
  let s = (STRINGS[key] && STRINGS[key][LANG]) || (STRINGS[key] && STRINGS[key].ru) || key;
  if (vars) for (const [k, v] of Object.entries(vars)) s = s.replaceAll("{" + k + "}", v);
  return s;
}

function setLang(lang) {
  LANG = lang === "en" ? "en" : "ru";
  document.documentElement.lang = LANG;
  document.querySelectorAll("[data-i]").forEach(el => { el.textContent = t(el.dataset.i); });
  document.getElementById("tab-task").firstChild
    ? null : null;
  window.dispatchEvent(new CustomEvent("langchange"));
}
