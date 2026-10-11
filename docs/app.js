const $ = s => document.querySelector(s);
const num = v => { const n = parseFloat(v); return Number.isFinite(n) ? n : null; };
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const trunc = (s, n) => s && s.length > n ? s.slice(0, n) + '…' : (s ?? '');

let grades = [];          // rows from grades.tsv
let prices = {};          // model -> {i, o, cur} from llms_config.jsonl
let qsPerMillion = null;  // questions per 1M tokens, from real avg tokens/q
const USD_RUB = 85;       // same rate as src/top_charts.py
let byQ = {};             // question_id -> {question, expert_answer, dynamic_rubric, ...}
let manifest = [];        // model names with answers files
const ansCache = {};      // model -> {id: row}
const jevCache = {};      // model -> [rows]
const charts = {};

// shareable URL state: #tab=answers&q=9879&m=yandexgpt-lite&lang=ru
const getHash = () => Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
const setHash = o => {
  const p = new URLSearchParams(location.hash.slice(1));
  Object.entries(o).forEach(([k, v]) => v == null ? p.delete(k) : p.set(k, v));
  history.replaceState(null, '', p.size ? '#' + p : location.pathname);
};

let LANG = localStorage.getItem('ev_lang') || ((navigator.language || '').startsWith('ru') ? 'ru' : 'en');
let STR = {};
const t = k => STR[k] ?? k;
const tf = (k, o) => t(k).replace(/\{(\w+)\}/g, (_, m) => o[m] ?? m);
const dtLang = () => ({ search: t('dt.search'), info: t('dt.info') });

function parseTSV(text) {
  const lines = text.split('\n').filter(l => l.trim());
  const cols = lines[0].split('\t');
  const critIdx = cols.indexOf('criteria_scores');
  return lines.slice(1).map(l => {
    let p = l.split('\t');
    if (p.length > cols.length && critIdx >= 0) { // overflow tabs live inside criteria JSON
      p = [...p.slice(0, critIdx), p.slice(critIdx, critIdx + p.length - cols.length + 1).join('\t'),
           ...p.slice(critIdx + p.length - cols.length + 1)];
    }
    const o = {};
    cols.forEach((c, i) => {
      let v = p[i] ?? '';
      if (v.length > 1 && v.startsWith('"') && v.endsWith('"')) v = v.slice(1, -1).replace(/""/g, '"');
      o[c] = v;
    });
    return o;
  });
}

async function fetchJSONL(path) {
  const t = await (await fetch(path)).text();
  return t.split('\n').filter(l => l.trim()).map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
}
async function ensureAnswers(model) {
  if (!ansCache[model]) {
    const rows = await fetchJSONL(`data/${model}_answers.jsonl`);
    ansCache[model] = Object.fromEntries(rows.map(r => [String(r.id), r]));
  }
  return ansCache[model];
}
async function ensureJev(model) {
  if (!jevCache[model]) jevCache[model] = await fetchJSONL(`data/${model}_jev_grades.jsonl`).catch(() => []);
  return jevCache[model];
}

const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
const med = a => { const s = [...a].sort((x, y) => x - y), n = s.length;
  return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : null; };
const metricsOf = m => {
  const vals = grades.filter(g => g.model === m);
  const avg = k => mean(vals.map(v => num(v[k])).filter(v => v !== null));
  const fails = k => vals.reduce((s, v) => s + (1 - (num(v[k]) ?? 1)), 0);
  const nums = k => vals.map(v => num(v[k])).filter(v => v !== null);
  return { n: vals.length, score: avg('score_avg'), median: med(nums('score_avg')),
           pass: avg('pass_flag_avg'), cost: avg('cost_avg'),
           legAvg: avg('legality_avg'), factAvg: avg('factual_correctness_avg'),
           tokens: avg('total_tokens_avg'), elapsed: avg('elapsed_s_avg'),
           illegal: fails('legality_avg'), facterr: fails('factual_correctness_avg') };
};

// brand-ish color per vendor group — makes groups easy to tell apart
const VENDOR_COLORS = { 'Sber': '#21a038', 'Yandex': '#fc3f1d', 'OpenAI': '#0f0f0f',
                        'Anthropic': '#d97757', 'Open weights': '#8b5cf6' };

// draws point labels (e.g. model names) next to scatter dots
const VENDORS = ['Sber', 'Yandex', 'OpenAI', 'Anthropic', 'Open weights'];
const vendor = m => m.startsWith('GigaChat') ? 'Sber'
  : m.startsWith('yandex') ? 'Yandex'
  : m.includes('gpt') ? 'OpenAI'
  : (m.includes('claude') || m.includes('haiku')) ? 'Anthropic' : 'Open weights';
const selVendors = new Set(VENDORS);
let modelRows = [];
const filteredRows = () => modelRows.filter(r => selVendors.has(vendor(r.model)));

// score tier — serious wording, badge color carries the mood
const tierKey = v => v >= .7 ? 'sc.strong' : v >= .4 ? 'sc.mid' : v > 0 ? 'sc.weak' : 'sc.fail';
const scoreBadge = v => `<span class="badge text-bg-${v >= .7 ? 'success' : v >= .4 ? 'warning' : 'danger'}">${v.toFixed(2)}</span>` +
  ` <span class="score-tier">${t(tierKey(v))}</span>`;

const pointLabels = {
  id: 'pointLabels',
  afterDatasetsDraw(ch) {
    const { ctx } = ch;
    ctx.save(); ctx.fillStyle = '#8a94a3'; ctx.font = '9px Inter, sans-serif';
    const drawn = []; // skip labels that would overlap an already-drawn one
    ch.data.datasets.forEach((ds, di) => ch.getDatasetMeta(di).data.forEach((p, i) => {
      const r = ds.data[i];
      if (!r || !r.model) return;
      const box = { x: p.x + 5, y: p.y - 7, w: ctx.measureText(r.model).width, h: 11 };
      if (drawn.some(b => box.x < b.x + b.w && b.x < box.x + box.w && box.y < b.y + b.h && b.y < box.y + box.h)) return;
      ctx.fillText(r.model, box.x, p.y + 3);
      drawn.push(box);
    }));
    ctx.restore();
  },
};

function modelGrades(qid) { return grades.filter(g => String(g.question_id) === String(qid)); }

// ---------- tabs ----------
function activateTab(name) {
  document.querySelectorAll('[data-tab]').forEach(x => x.classList.toggle('active', x.dataset.tab === name));
  document.querySelectorAll('section[id^=tab-]').forEach(s => s.classList.add('d-none'));
  $('#tab-' + name)?.classList.remove('d-none');
}
document.querySelectorAll('[data-tab]').forEach(a => a.addEventListener('click', () => {
  activateTab(a.dataset.tab);
  setHash({ tab: a.dataset.tab });
}));

// ---------- models tab ----------
let dtModels, dtQuestions, dtAnswers, dtCriteria;
const rubRate = m => prices[m]?.cur === 'USD' ? USD_RUB : 1;
const rubCost = (m, c) => c === null ? null : c * rubRate(m);
const priceTip = m => {
  const p = prices[m];
  if (!p) return '';
  if (p.free) return t('tip.free');
  const sym = p.cur === 'USD' ? '$' : '₽';
  return `${t('tip.cost')}: ${p.i}${sym}/1M in · ${p.o}${sym}/1M out` +
         (p.cur === 'USD' ? ` · $1 = ${USD_RUB}₽` : '') +
         (qsPerMillion ? ` · ${tf('tip.perm', { n: qsPerMillion })}` : '');
};
const costCell = r => prices[r.model]?.free
  ? `<span class="fw-semibold" title="${esc(t('tip.free'))}">free*</span>`
  : `<span title="${esc(priceTip(r.model))}">${r.cost === null ? '—' : rubCost(r.model, r.cost).toFixed(2) + ' ₽'}</span>`;
const effOf = r => { const c = rubCost(r.model, r.cost);
  return (prices[r.model]?.free || !c || r.score === null) ? null : r.score / c; };

// special awards — recomputed on the filtered set so vendor toggles re-crown winners
const AWARD_ICONS = { 'aw.facts': '🧠', 'aw.legal': '🛡️', 'aw.cheap': '💸', 'aw.value': '⚡', 'aw.gray': '🚩' };
let awardMap = {};
function computeAwards(rows) {
  awardMap = {};
  const pick = fn => { let b = null;
    rows.forEach(r => { const v = fn(r); if (v !== null && v !== undefined && (!b || v > b.v)) b = { m: r.model, v }; });
    return b?.m; };
  const add = (m, k) => m && (awardMap[m] = [...(awardMap[m] || []), k]);
  add(pick(r => r.factAvg), 'aw.facts');
  add(pick(r => r.legAvg), 'aw.legal');
  add(pick(r => prices[r.model]?.free ? null : (r.cost !== null ? -rubCost(r.model, r.cost) : null)), 'aw.cheap');
  add(pick(effOf), 'aw.value');
  add(pick(r => r.illegal), 'aw.gray');
}
const awardChips = r => (awardMap[r.model] || []).map(k =>
  `<span class="award-chip" title="${esc(t(k))}">${AWARD_ICONS[k]} ${esc(t(k))}</span>`).join(' ');

function buildModels() {
  modelRows = [...new Set(grades.map(g => g.model))].map(m => ({ model: m, ...metricsOf(m) }));
  computeAwards(filteredRows());
  dtModels = new DataTable('#tblModels', {
    data: filteredRows(), order: [[1, 'desc']], language: dtLang(), paging: false, info: false,
    columns: [
      { title: t('col.model'), data: 'model', render: (d, type) =>
          type === 'display' ? `<span class="vdot me-1" style="background:${VENDOR_COLORS[vendor(d)]}"></span>${esc(d)}` : d },
      { title: t('col.score'), data: 'score', render: (d, type) =>
          type === 'display' ? (d === null ? '—' : scoreBadge(d)) : d },
      { title: t('col.median'), data: r => r.median?.toFixed(3) ?? '—' },
      { title: t('col.pass'), data: r => r.pass?.toFixed(3) ?? '—' },
      { title: t('col.costq'), data: 'cost', render: (d, type, r) =>
          type === 'display' ? costCell(r) : (prices[r.model]?.free ? -1 : rubCost(r.model, d)) },
      { title: t('col.valq'), data: null, render: (d, type, r) => {
          const e = effOf(r);
          return type === 'display' ? (e === null ? '—' : (e * 100).toFixed(1)) : e; } },
      { title: t('col.tokq'), data: r => r.tokens?.toFixed(0) ?? '—' },
      { title: t('col.awards'), data: r => awardChips(r), orderable: false },
    ],
  });
  renderModelCharts();
}

function renderModelCharts() {
  const rows = filteredRows();
  const sorted = [...rows].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  const venColor = r => VENDOR_COLORS[vendor(r.model)];
  computeAwards(rows);
  // all four charts share one height so the 2×2 grid stays uniform
  const chH = Math.max(430, sorted.length * 26 + 90);
  ['chScoreModel', 'chCostScore', 'chIllegal', 'chFactErr'].forEach(id => {
    const c = $('#' + id); if (!c) return;
    let w = c.parentElement;
    if (!w.classList.contains('ch-wrap')) {
      w = document.createElement('div'); w.className = 'ch-wrap';
      c.parentElement.insertBefore(w, c); w.appendChild(c);
    }
    w.style.height = chH + 'px';
  });
  $('#podium').innerHTML = sorted.slice(0, 3).map((r, i) => `
    <div class="podium-spot podium-r${i + 1}${i === 0 ? ' podium-first' : ''}">
      <div class="podium-medal">${['🥇', '🥈', '🥉'][i]}</div>
      <div class="podium-model"><span class="vdot" style="background:${venColor(r)}"></span>${esc(r.model)}</div>
      <div class="podium-score">${((r.score ?? 0) * 100).toFixed(1)}%</div>
    </div>`).join('');
  const venLegend = { display: true, position: 'bottom', onClick: () => {},
    labels: { boxWidth: 10, font: { size: 10 }, generateLabels: () => VENDORS.filter(v => selVendors.has(v))
      .map(v => ({ text: t('v.' + v), fillStyle: VENDOR_COLORS[v] })) } };
  draw('chScoreModel', {
    type: 'bar',
    data: { labels: sorted.map(r => r.model),
      datasets: [{ label: t('st.score'), data: sorted.map(r => r.score), backgroundColor: sorted.map(venColor) }] },
    options: { indexAxis: 'y', maintainAspectRatio: false,
      plugins: { legend: venLegend },
      scales: { x: { max: 1, title: { text: t('axis.score'), display: true } },
                y: { ticks: { autoSkip: false } } } },
  });
  draw('chCostScore', {
    type: 'scatter',
    data: { datasets: [{ label: t('col.model'),
      data: sorted.map(r => ({ x: rubCost(r.model, r.cost), y: r.score, model: r.model })),
      backgroundColor: sorted.map(venColor), pointRadius: 5 }] },
    options: { maintainAspectRatio: false, plugins: { legend: venLegend, tooltip: { callbacks: {
      label: c => `${c.raw.model}: ${c.raw.x?.toFixed(2)}₽ / ${(c.raw.y * 100).toFixed(1)}%` } } },
      scales: { x: { title: { text: t('axis.cost') + ', ₽', display: true } },
                y: { title: { text: t('axis.score'), display: true }, max: 1 } } },
    plugins: [pointLabels],
  });
  const failChart = (id, key) => {
    const s = [...rows].sort((a, b) => b[key] - a[key]);
    draw(id, {
      type: 'bar',
      data: { labels: s.map(r => r.model),
        datasets: [{ data: s.map(r => r[key]), backgroundColor: s.map(venColor) }] },
      options: { indexAxis: 'y', maintainAspectRatio: false, plugins: { legend: venLegend },
        scales: { x: { title: { text: t('axis.count'), display: true }, beginAtZero: true },
                  y: { ticks: { autoSkip: false } } } },
    });
  };
  failChart('chIllegal', 'illegal');
  failChart('chFactErr', 'facterr');
}

// ---------- questions tab ----------
function buildQuestions() {
  const rows = Object.entries(byQ).map(([id, q]) => {
    const gs = modelGrades(id);
    return { id, category: q.category, intent: q.intent, knowledge: q.knowledge,
             question: trunc(q.question, 140), models: gs.length,
             score: mean(gs.map(g => num(g.score_avg)).filter(v => v !== null)) };
  });
  dtQuestions = new DataTable('#tblQuestions', {
    data: rows, order: [[0, 'asc']], language: dtLang(),
    columns: [
      { title: t('col.id'), data: 'id' }, { title: t('col.category'), data: 'category' },
      { title: t('col.intent'), data: 'intent' }, { title: t('col.knowledge'), data: 'knowledge' },
      { title: t('col.question'), data: 'question' }, { title: t('col.nmodels'), data: 'models' },
      { title: t('col.meanscore'), data: r => r.score?.toFixed(3) ?? '—' },
    ],
  });
  $('#tblQuestions tbody').addEventListener('click', e => {
    const r = dtQuestions.row(e.target.closest('tr')).data();
    if (r) openDetail(r.id);
  });
  const scored = rows.filter(r => r.score !== null);
  const qline = r => `<div class="qline" data-q="${r.id}">${scoreBadge(r.score)} ${esc(trunc(byQ[r.id].question, 100))}</div>`;
  $('#qHardest').innerHTML = [...scored].sort((a, b) => a.score - b.score).slice(0, 5).map(qline).join('');
  $('#qEasiest').innerHTML = [...scored].sort((a, b) => b.score - a.score).slice(0, 5).map(qline).join('');
  // suspicious answers feed — lowest judge scores across all models, flag problems inline
  $('#qShame').innerHTML = grades.map(g => ({ g, s: num(g.score_avg) })).filter(x => x.s !== null)
    .sort((a, b) => a.s - b.s).slice(0, 5).map(({ g, s }) => {
      const fl = [(num(g.legality_avg) ?? 1) < 1 ? t('fl.leg.f') : '',
                  (num(g.factual_correctness_avg) ?? 1) < 1 ? t('fl.fact.f') : '',
                  (num(g.task_completion_avg) ?? 1) < 1 ? t('fl.task.f') : ''].filter(Boolean).join(' · ');
      return `<div class="qline" data-q="${esc(g.question_id)}" data-m="${esc(g.model)}">` +
        `<span class="badge text-bg-danger">${s.toFixed(2)}</span> <b>${esc(g.model)}</b> — ` +
        `${esc(trunc(g.question, 70))}${fl ? ` <span class="text-danger small">${fl}</span>` : ''}</div>`;
    }).join('');
  document.querySelectorAll('.qline').forEach(el => el.onclick = () => openDetail(el.dataset.q, el.dataset.m || undefined));
}

// ---------- answers tab ----------
async function buildAnswers(model) {
  $('#ansHint').textContent = tf('ans.loading', { m: model });
  const answers = await ensureAnswers(model);
  const rows = Object.values(answers).map(a => {
    const g = grades.find(x => x.model === model && String(x.question_id) === String(a.id));
    return { id: a.id, question: trunc(a.question, 120), answer: trunc(a.answer, 140),
             score: g ? num(g.score_avg) : null, cost: g ? num(g.cost_avg) : null,
             error: a.error && a.error !== 'None' ? trunc(a.error, 40) : '' };
  }).sort((a, b) => String(a.id).localeCompare(String(b.id)));
  const cols = [
    { title: t('col.id'), data: 'id' }, { title: t('col.question'), data: 'question' },
    { title: t('col.answer'), data: 'answer' },
    { title: t('col.score'), data: 'score', render: (d, type) =>
        type === 'display' ? (d === null ? '—' : scoreBadge(d)) : d },
    { title: t('col.cost'), data: r => r.cost?.toFixed(4) ?? '—' },
    { title: t('col.error'), data: 'error' },
  ];
  if (dtAnswers) { dtAnswers.clear().rows.add(rows).draw(); }
  else {
    dtAnswers = new DataTable('#tblAnswers', { data: rows, columns: cols, order: [[0, 'asc']], language: dtLang() });
    $('#tblAnswers tbody').addEventListener('click', e => {
      const r = dtAnswers.row(e.target.closest('tr')).data();
      if (r) openDetail(r.id, $('#selAnsModel').value);
    });
  }
  $('#ansHint').textContent = tf('ans.count', { n: Object.keys(answers).length });
}

// ---------- detail modal ----------
const modal = new bootstrap.Modal('#detailModal');
document.getElementById('detailModal').addEventListener('hidden.bs.modal', () => setHash({ q: null, m: null }));
async function openDetail(qid, model) {
  const q = byQ[qid] ?? {};
  const gs = modelGrades(qid);
  $('#dmTitle').textContent = `Q${qid} — ${q.category ?? ''}/${q.intent ?? ''}`;
  $('#dmQuestion').textContent = q.question ?? gs[0]?.question ?? '';
  $('#dmExpert').textContent = q.expert_answer || '—';
  try {
    const rub = JSON.parse(q.dynamic_rubric);
    $('#dmRubric').innerHTML = '<ol class="mb-0 ps-3">' + rub.map(i =>
      `<li class="mb-1">${esc(i.criterion)} <span class="badge text-bg-secondary">${i.points}</span></li>`).join('') + '</ol>';
  } catch { $('#dmRubric').textContent = q.dynamic_rubric || '—'; }
  const models = [...new Set([...gs.map(g => g.model), ...manifest])];
  $('#dmModel').innerHTML = models.map(m =>
    `<option ${m === model ? 'selected' : ''}>${esc(m)}</option>`).join('');
  $('#dmScoreChips').innerHTML = gs.map(g =>
    `<span class="badge text-bg-${num(g.score_avg) >= .7 ? 'success' : num(g.score_avg) >= .4 ? 'warning' : 'danger'} me-1"
      title="${t('m.clickview')}">${esc(g.model)} ${num(g.score_avg)?.toFixed(2) ?? '?'}</span>`).join('');
  $('#dmScoreChips').querySelectorAll('.badge').forEach((b, i) => {
    b.style.cursor = 'pointer';
    b.addEventListener('click', () => { $('#dmModel').value = gs[i].model; renderAnswer(qid); });
  });
  if (!model && gs.length) $('#dmModel').value = gs[0].model;
  $('#dmModel').onchange = () => renderAnswer(qid);
  renderAnswer(qid);
  setHash({ q: qid, m: $('#dmModel').value });
  modal.show();
}

async function renderAnswer(qid) {
  const model = $('#dmModel').value;
  setHash({ m: model });
  $('#dmChatModel').innerHTML = `<span class="vdot" style="background:${VENDOR_COLORS[vendor(model)]}"></span> ${esc(model)}`;
  $('#dmAnswer').textContent = t('loading');
  const [answers, jev] = await Promise.all([ensureAnswers(model).catch(() => ({})), ensureJev(model)]);
  const a = answers[String(qid)];
  $('#dmAnswer').innerHTML = a ? marked.parse(a.answer ?? '') : `<i class="text-secondary">${t('m.noanswer')}</i>`;
  const runs = jev.filter(r => String(r.id) === String(qid));
  const g = grades.find(x => x.model === model && String(x.question_id) === String(qid));

  // judge picker — a select only if more than one judge graded this answer
  const judges = [...new Set(runs.map(r => r.judge).concat(g?.judge ? [g.judge] : []))];
  if (judges.length > 1) {
    if (!$('#dmJudgeSel')) {
      $('#dmJudge').innerHTML = `<select id="dmJudgeSel" class="form-select form-select-sm d-inline-block w-auto">${judges.map(j => `<option>${esc(j)}</option>`).join('')}</select>`;
      $('#dmJudgeSel').onchange = () => renderAnswer(qid);
    }
  } else $('#dmJudge').textContent = judges[0] ?? '—';
  const judge = $('#dmJudgeSel')?.value ?? judges[0];
  const jruns = runs.filter(r => r.judge === judge);
  const run = jruns[0];

  // answer flags
  const flagIcon = v => (v === true || v === 1) ? '<span class="text-success fw-bold">✓</span>'
    : (v === false || v === 0) ? '<span class="text-danger fw-bold">✗</span>' : null;
  const flagKey = { legality: 'leg', factual_correctness: 'fact', task_completion: 'task' };
  $('#dmFlags').innerHTML = Object.keys(flagKey).map(k => {
    const v = run ? run[k] : num(g?.[k + '_avg']);
    const icon = flagIcon(v);
    const reason = run?.flag_reasons?.[k] || '';
    const label = t(`fl.${flagKey[k]}.${(v === false || v === 0) ? 'f' : 't'}`);
    return `<div class="mb-1">${icon ?? `<b>${v == null ? '—' : (+v).toFixed(2)}</b>`} ${label}` +
           (reason ? ` <span class="text-secondary">— ${esc(reason)}</span>` : '') + '</div>';
  }).join('');

  // quality table — one row per criterion, all runs of the chosen judge
  let critRows = jruns.flatMap(r => (r.scores ?? []).map(s => ({
    criterion: esc(s.criterion), awarded: s.awarded, max: s.max, comment: esc(s.reason ?? ''),
  }))).concat(jruns[0]?.scores?.length || !run ? [] : [{
    criterion: '', awarded: run.total_awarded, max: run.total_max,
    comment: esc(trunc(run.overall_comment, 200)),
  }]);
  if (!critRows.length && g?.criteria_scores) {
    try {
      critRows = JSON.parse(g.criteria_scores).map(s => ({
        criterion: esc(s.criterion), awarded: s.awarded, max: s.max, comment: esc(s.reason ?? ''),
      }));
    } catch { /* leave empty */ }
  }
  dtCriteria?.destroy();
  dtCriteria = new DataTable('#dmCriteria', {
    data: critRows, paging: false, searching: false, info: false,
    columns: [
      { title: t('col.criterion'), data: 'criterion' },
      { title: t('col.awarded'), data: r => `${(+r.awarded).toFixed(2)}/${r.max}` },
      { title: t('col.comment'), data: 'comment' },
    ],
  });
}

// ---------- chart builder ----------
const METRICS = ['score_avg','pass_flag_avg','task_completion_avg','legality_avg','factual_correctness_avg',
  'criteria_avg','cost_avg','total_tokens_avg','prompt_tokens_avg','completion_tokens_avg','elapsed_s_avg','n_runs'];
const DIMS = ['model','category','intent','knowledge','harness','judge'];
function agg(arr, how) {
  const v = arr.filter(x => x !== null);
  if (how === 'count') return arr.length;
  if (!v.length) return null;
  if (how === 'sum') return v.reduce((s, x) => s + x, 0);
  if (how === 'min') return Math.min(...v);
  if (how === 'max') return Math.max(...v);
  return mean(v);
}
function drawBuilder() {
  const metric = $('#cbMetric').value, how = $('#cbAgg').value,
        group = $('#cbGroup').value, split = $('#cbSplit').value, type = $('#cbType').value;
  const val = gv => agg(grades.filter(g => g[group] === gv).map(g => num(g[metric])), how);
  const groups = [...new Set(grades.map(g => g[group]))]
    .sort((a, b) => (val(b) ?? -Infinity) - (val(a) ?? -Infinity));
  const palette = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc948','#b07aa1','#ff9da7','#9c755f','#bab0ac'];
  let datasets, labels = groups;
  if (split === '—') {
    datasets = [{ label: `${t('opt.' + how)} · ${t('met.' + metric)}`,
      data: groups.map(gv => agg(grades.filter(g => g[group] === gv).map(g => num(g[metric])), how)),
      backgroundColor: groups.map((_, i) => palette[i % palette.length]) }];
  } else {
    const splits = [...new Set(grades.map(g => g[split]))];
    datasets = splits.map((sv, i) => ({ label: String(sv), backgroundColor: palette[i % palette.length],
      data: groups.map(gv => agg(grades.filter(g => g[group] === gv && g[split] === sv).map(g => num(g[metric])), how)) }));
  }
  draw('chCustom', {
    type: type === 'barH' ? 'bar' : type,
    data: { labels, datasets },
    options: { indexAxis: type === 'barH' ? 'y' : 'x', maintainAspectRatio: false,
      scales: type === 'doughnut' ? {} : { y: { beginAtZero: true } } },
  });
}
function draw(id, cfg) { charts[id]?.destroy(); charts[id] = new Chart($('#' + id), cfg); }

// ---------- init ----------
(async () => {
  const h = getHash();
  if (h.lang) LANG = h.lang;
  const [mf, gt, ds, str, cfg] = await Promise.all([
    fetch('data/manifest.json').then(r => r.json()),
    fetch('data/grades.tsv').then(r => r.text()),
    fetch('data/dataset.tsv').then(r => r.text()),
    fetch(`${LANG}.json`).then(r => r.json()).catch(() => ({})),
    fetch('data/llms_config.jsonl').then(r => r.text()).catch(() => ''),
  ]);
  manifest = mf;
  grades = parseTSV(gt);
  byQ = Object.fromEntries(parseTSV(ds).map(r => [String(r.id), r]));
  STR = str;
  prices = Object.fromEntries(cfg.split('\n').filter(l => l.trim()).map(l => {
    try { const c = JSON.parse(l);
      return [c.name, { i: c.price_1m_input, o: c.price_1m_output, cur: c.currency,
                        free: !c.price_1m_input && !c.price_1m_output }];
    } catch { return null; }
  }).filter(Boolean));
  document.documentElement.lang = LANG;
  document.title = t('title');
  $('#tabOracle').title = t('tip.oracle');
  $('#exLive').onclick = e => { e.preventDefault(); openDetail(9879); };
  document.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll('[data-i18n-html]').forEach(el => { el.innerHTML = t(el.dataset.i18nHtml); });
  $('#vendorFilter').innerHTML = `<span class="vf-label">${esc(t('v.title'))}</span>` +
    VENDORS.map(v => `<button class="vbtn active" data-v="${v}"><span class="vdot" style="background:${VENDOR_COLORS[v]}"></span>${esc(t('v.' + v))}</button>`).join('');
  document.querySelectorAll('#vendorFilter .vbtn').forEach(b => b.onclick = () => {
    b.classList.toggle('active');
    b.classList.contains('active') ? selVendors.add(b.dataset.v) : selVendors.delete(b.dataset.v);
    renderModelCharts();
    dtModels.clear().rows.add(filteredRows()).draw();
  });
  document.querySelectorAll('.lang-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.lang === LANG);
    b.onclick = () => { localStorage.setItem('ev_lang', b.dataset.lang); setHash({ lang: b.dataset.lang }); location.reload(); };
  });
  $('#loadStatus').textContent = tf('status', { rows: grades.length, q: Object.keys(byQ).length, m: manifest.length });
  const countUp = (el, end, fmt) => {
    const t0 = performance.now(), dur = 700;
    const step = now => {
      const p = Math.min(1, (now - t0) / dur), e = 1 - (1 - p) ** 3;
      el.textContent = fmt(end * e);
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };
  const scores = grades.map(g => num(g.score_avg)).filter(v => v !== null);
  const nModels = new Set(grades.map(g => g.model)).size;
  countUp($('#stModels'), nModels, v => Math.round(v));
  countUp($('#stQuestions'), Object.keys(byQ).length, v => Math.round(v));
  countUp($('#stScore'), (mean(scores) ?? 0) * 100, v => v.toFixed(1) + '%');
  const perModelCost = [...new Set(grades.map(g => g.model))]
    .filter(m => !prices[m]?.free)
    .map(m => rubCost(m, metricsOf(m).cost)).filter(v => v !== null);
  const avgCost = mean(perModelCost) ?? 0;
  countUp($('#stCost'), avgCost, v => '~' + v.toFixed(2) + ' ₽');
  const toks = grades.map(g => num(g.total_tokens_avg)).filter(v => v !== null);
  if (toks.length) qsPerMillion = Math.round(1e6 / mean(toks) / 10) * 10;
  $('#selAnsModel').innerHTML = manifest.map(m => `<option>${esc(m)}</option>`).join('');
  if (h.m && manifest.includes(h.m)) $('#selAnsModel').value = h.m;
  $('#selAnsModel').onchange = () => { setHash({ m: $('#selAnsModel').value }); buildAnswers($('#selAnsModel').value); };
  $('#cbMetric').innerHTML = METRICS.map(m => `<option value="${m}">${esc(t('met.' + m))}</option>`).join('');
  $('#cbGroup').innerHTML = DIMS.map(d => `<option>${d}</option>`).join('');
  $('#cbSplit').innerHTML = ['—', ...DIMS].map(d => `<option>${d}</option>`).join('');
  $('#btnDraw').onclick = drawBuilder;
  buildModels(); buildQuestions(); buildAnswers($('#selAnsModel').value); drawBuilder();
  if (h.tab) activateTab(h.tab);
  if (h.q) openDetail(h.q, h.m);
})().catch(e => { $('#loadStatus').textContent = t('failed') + ': ' + e.message; console.error(e); });
