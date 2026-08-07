/* Двуязычие ru/en. Ключи проставляются в разметке атрибутом data-i,
   динамические строки берутся функцией t(). */
const STRINGS = {
  "tab.task":        {ru:"Задача", en:"Task"},
  "tab.queue":       {ru:"Очередь", en:"Queue"},
  "ref":             {ru:"Референс:", en:"Reference:"},
  "ref.track":       {ru:"реф-дорожка:", en:"ref track:"},
  "ref.rest_as_dubs":{ru:"+ остальные дорожки в озвучки", en:"+ other tracks as dubs"},
  "tracks.n":        {ru:"дорожек: {n}", en:"tracks: {n}"},
  "tracks.chosen":   {ru:"дорожки: {k} / {n}", en:"tracks: {k} / {n}"},
  "tracks.none":     {ru:"не выбрано", en:"none selected"},
  "tracks.all":      {ru:"выбрать все / снять все", en:"select all / clear"},
  "dubs":            {ru:"Озвучки:", en:"Dubs:"},
  "dubs.add":        {ru:"+ Добавить файлы", en:"+ Add files"},
  "dubs.drop":       {ru:"или перетащите файлы в окно", en:"or drop files onto the window"},
  "dub.audio_only":  {ru:"аудио-only", en:"audio-only"},
  "dub.virtual":     {ru:"виртуальный дубль", en:"virtual dub"},
  "dub.same_as_ref": {ru:"= реф", en:"= ref"},
  "out_dir":         {ru:"Выходной каталог:", en:"Output folder:"},
  "browse":          {ru:"Обзор…", en:"Browse…"},
  "settings":        {ru:"Настройки", en:"Settings"},
  "set.analysis":    {ru:"Аудио-анализ:", en:"Audio analysis:"},
  "set.band":        {ru:"band (стандарт)", en:"band (default)"},
  "set.muq_note":    {ru:"MuQ — экспериментальная модель, нужна NVIDIA-карта. Веса загружаются при первом включении; лицензия CC-BY-NC 4.0 — только некоммерческое использование результата.",
                      en:"MuQ is experimental and needs an NVIDIA GPU. Weights download on first use; CC-BY-NC 4.0 — non-commercial use of results only."},
  "set.fill":        {ru:"Заполнять тишину озвучки референсом", en:"Fill dub silence from reference"},
  "set.drift":       {ru:"Потолок скорости дрейфа:", en:"Drift speed ceiling:"},
  "set.drift_unit":  {ru:"%/с", en:"%/s"},
  "set.keep_tmp":    {ru:"Сохранять промежуточные файлы (повтор без пересчёта)", en:"Keep intermediate files (instant re-runs)"},
  "set.keep_tmp_dir":{ru:"каталог для промежуточных файлов", en:"folder for intermediate files"},
  "task.label":      {ru:"Название:", en:"Label:"},
  "task.enqueue":    {ru:"Добавить в очередь", en:"Add to queue"},
  "task.added":      {ru:"Задача добавлена — запустите её кнопкой ▶ (озвучек: {n})", en:"Task added — press ▶ to start it ({n} dubs)"},
  "task.added_run":  {ru:"Задача добавлена и запущена (озвучек: {n})", en:"Task added and started ({n} dubs)"},
  "task.need_ref":   {ru:"Укажите файл референса", en:"Choose a reference file"},
  "task.need_dubs":  {ru:"Добавьте хотя бы одну озвучку", en:"Add at least one dub"},
  "task.need_out":   {ru:"Укажите выходной каталог", en:"Choose an output folder"},
  "q.autostart":     {ru:"запускать сразу после добавления", en:"start right after adding"},
  "q.parallel":      {ru:"параллельно:", en:"parallel:"},
  "q.clear_done":    {ru:"Очистить готовые", en:"Clear finished"},
  "q.cleared":       {ru:"Очищено задач: {n}, освобождено {mb} МБ", en:"Cleared {n} tasks, freed {mb} MB"},
  "q.empty":         {ru:"Очередь пуста — соберите задачу на вкладке «Задача»", en:"Queue is empty — build a task on the “Task” tab"},
  "q.queued":        {ru:"в очереди", en:"queued"},
  "q.paused":        {ru:"на паузе", en:"paused"},
  "q.done":          {ru:"готово", en:"done"},
  "q.failed":        {ru:"ошибка", en:"failed"},
  "q.cancelled":     {ru:"отменено", en:"cancelled"},
  "q.dub_of":        {ru:"озвучка {i}/{n}", en:"dub {i}/{n}"},
  "q.start":         {ru:"Запустить задачу", en:"Start the task"},
  "q.pause":         {ru:"Снять с автозапуска", en:"Hold the task"},
  "q.cancel":        {ru:"Отменить (убивает её процессы)", en:"Cancel (kills its processes)"},
  "d.details":       {ru:"подробнее", en:"details"},
  "d.folder":        {ru:"папка", en:"folder"},
  "d.resid":         {ru:"остаток {v} мс", en:"residual {v} ms"},
  "d.coverage":      {ru:"покрытие {v}", en:"coverage {v}"},
  "d.suspect":       {ru:"⚠ проверьте результат", en:"⚠ review the result"},
  "d.skipped":       {ru:"уже было готово", en:"already done"},
  "m.resid":         {ru:"остаток", en:"residual"},
  "m.coverage":      {ru:"покрытие", en:"coverage"},
  "m.assigned":      {ru:"назначено", en:"assigned"},
  "m.cos":           {ru:"cos", en:"cos"},
  "m.slope":         {ru:"наклон", en:"slope"},
  "m.cuts":          {ru:"резы аудио", en:"audio cuts"},
  "m.filled":        {ru:"заполнено рефом", en:"filled from ref"},
  "m.span":          {ru:"диапазон сдвига", en:"shift span"},
  "m.geom":          {ru:"geom", en:"geom"},
  "m.mode_audio":    {ru:"режим", en:"mode"},
  "p.track":         {ru:"укладка — весь трек", en:"alignment — full track"},
  "p.cut":           {ru:"рез @{t} с · {v} мс", en:"cut @{t} s · {v} ms"},
  "p.open":          {ru:"открыть ↗", en:"open ↗"},
  "stage.decode":    {ru:"декод", en:"decode"},
  "stage.extract":   {ru:"аудио", en:"audio"},
  "stage.coarse":    {ru:"грубый проход", en:"coarse"},
  "stage.geom":      {ru:"geom", en:"geom"},
  "stage.band":      {ru:"полоса", en:"band"},
  "stage.resample":  {ru:"ресэмпл", en:"resample"},
  "stage.audio":     {ru:"доводка", en:"refine"},
  "stage.write":     {ru:"запись", en:"write"},
  "theme.tip":       {ru:"Тема: системная / светлая / тёмная", en:"Theme: system / light / dark"},
  "err.api":         {ru:"Ошибка: {e}", en:"Error: {e}"},
  "unit.ms":         {ru:"мс", en:"ms"},
  "unit.s":          {ru:"с", en:"s"},
};

let LANG = "ru";

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
