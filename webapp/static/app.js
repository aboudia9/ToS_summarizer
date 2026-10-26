'use strict';

// Results stream in from the server (Server-Sent Events) and go through a small reveal
// queue, so cards pop in and summaries land one at a time even when the server sends a
// whole batch at once. Every change re-sorts the list and animates cards to their new
// spots (FLIP: measure First, change the DOM, measure Last, Invert, Play).

const $ = (id) => document.getElementById(id);
const SEVERITY = { clearly_unfair: 0, potentially_unfair: 1 };
const LABELS = {
  clearly_unfair: ['🚨', 'Clearly unfair', 'clear'],
  potentially_unfair: ['⚠️', 'Potentially unfair', 'potential'],
};
const STAGES = {
  loading_models: ['Warming up the models (first run after a restart)…', null],
  splitting: ['Splitting the text into sentences…', 0],
  classifying: ['Legal-BERT is reading {n} sentences…', 1],
  summarizing: ['BART is rewriting flagged clauses in plain English…', 2],
};
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

const state = {
  config: null,
  example: null,        // {slug, name, ..., text} of the example in the textarea, if any
  jobId: null,
  es: null,
  lastEventId: 0,
  retries: 0,
  done: false,
  cached: false,
  jobThreshold: 0.5,    // clauses below this weren't analyzed (live jobs only)
  items: new Map(),     // id -> {id, clause, label, unfair_prob, top_pick_unfair, summary, el}
  summarized: 0,
  queue: [],            // pending reveal steps
  revealing: false,
  startTime: 0,
  timer: null,
};

// ---------------- Setup ----------------

async function init() {
  const res = await fetch('/api/config');
  state.config = await res.json();
  const { examples, default_threshold, github } = state.config;

  $('github-link').href = github;
  $('threshold').value = default_threshold;
  showThreshold();

  for (const ex of examples) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'chip';
    btn.textContent = ex.name;
    btn.dataset.slug = ex.slug;
    btn.addEventListener('click', () => runExample(ex));
    $('example-buttons').append(btn);
  }

  $('tos-text').addEventListener('input', onTextInput);
  $('upload-btn').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', onUpload);
  $('analyze').addEventListener('click', () => submit());
  $('rerun').addEventListener('click', () => submit());
  $('threshold').addEventListener('input', () => { showThreshold(); render(); });
  $('show-potential').addEventListener('change', render);
  $('download').addEventListener('click', downloadCsv);
  document.addEventListener('visibilitychange', onVisible);
  window.addEventListener('online', onVisible);
  $('wait-link').addEventListener('click', () => {
    $('how').open = true;
    $('how').scrollIntoView({ behavior: 'smooth' });
  });
  // Narrow screens (phones, the portfolio iframe): keep the explainer collapsed at the top
  if (matchMedia('(max-width: 900px)').matches) $('how').open = false;
  updateCharCount();

  // A #job=... link (or a reload mid-analysis) picks the analysis back up
  const match = location.hash.match(/job=([\w-]+)/);
  if (match) {
    resetResults();
    showStatus('Reconnecting to your analysis…');
    connect(match[1]);
  }
}

function showThreshold() {
  $('threshold-val').textContent = Number($('threshold').value).toFixed(2);
}

// ---------------- Input ----------------

let actionSeq = 0;  // bumped by every example click / submit, so a slow one can't clobber a newer one

async function runExample(ex) {
  setError(null);
  const seq = ++actionSeq;
  const res = await fetch(`/api/examples/${ex.slug}`);
  if (!res.ok) return setError('Could not load that example.');
  const { text } = await res.json();
  if (seq !== actionSeq) return;
  $('tos-text').value = text;
  state.example = { ...ex, text };
  showTip();
  updateCharCount();
  submit();
}

function onTextInput() {
  if (state.example && $('tos-text').value !== state.example.text) {
    state.example = null;
    showTip();
  }
  updateCharCount();
}

function showTip() {
  const ex = state.example;
  for (const chip of document.querySelectorAll('.chip')) {
    chip.classList.toggle('active', !!ex && chip.dataset.slug === ex.slug);
  }
  $('tip').hidden = !ex || !ex.tip;
  if (!ex) return;
  $('tip-title').textContent = `What to look for in ${ex.name}'s terms:`;
  $('tip-text').textContent = ex.tip;
  $('tip-source').href = ex.source;
  $('tip-date').textContent = `, retrieved ${ex.retrieved}`;
}

function onUpload(e) {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  if (file.size > 1_000_000) return setError('That file is over 1MB. Try pasting just the terms.');
  const reader = new FileReader();
  reader.onload = () => {
    $('tos-text').value = reader.result;
    onTextInput();
    setError(null);
  };
  reader.readAsText(file);
}

function updateCharCount() {
  const n = $('tos-text').value.length;
  const max = state.config ? state.config.max_chars : Infinity;
  $('char-count').textContent = n ? `${n.toLocaleString()} characters` : '';
  const tooLong = n > max;
  if (tooLong) {
    setError(`That's ${n.toLocaleString()} characters; the limit is ${max.toLocaleString()}. ` +
             'Try pasting just the sections you care about.');
  } else if ($('input-error').dataset.kind === 'length') {
    setError(null);
  }
  $('input-error').dataset.kind = tooLong ? 'length' : '';
  $('analyze').disabled = !$('tos-text').value.trim() || tooLong;
}

function setError(msg) {
  $('input-error').hidden = !msg;
  $('input-error').textContent = msg || '';
}

// ---------------- Jobs ----------------

async function submit() {
  const text = $('tos-text').value;
  if (!text.trim()) return;
  setError(null);
  $('analyze').disabled = true;
  const seq = ++actionSeq;

  let res;
  try {
    res = await fetch('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, threshold: Number($('threshold').value) }),
    });
  } catch {
    $('analyze').disabled = false;
    return setError("Couldn't reach the server. Check your connection and try again.");
  }
  $('analyze').disabled = false;
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = typeof body.detail === 'string' ? body.detail : 'Something went wrong. Please try again.';
    return setError(detail);
  }
  const { job_id } = await res.json();
  if (seq !== actionSeq) return;
  resetResults();
  showStatus(state.example ? `Loading the ${state.example.name} results…` : 'Starting…');
  history.replaceState(null, '', `#job=${job_id}`);
  connect(job_id);
}

function connect(jobId) {
  if (state.es) state.es.close();
  if (state.jobId !== jobId) state.lastEventId = 0;
  state.jobId = jobId;

  const es = new EventSource(`/api/jobs/${jobId}/events?after=${state.lastEventId}`);
  state.es = es;
  const on = (type, fn) => es.addEventListener(type, (e) => {
    const id = Number(e.lastEventId);
    if (id && id <= state.lastEventId) return; // already handled before a reconnect
    state.lastEventId = id || state.lastEventId;
    fn(JSON.parse(e.data));
  });

  on('queued', ({ ahead }) =>
    showStatus(`Waiting for ${ahead} other document${ahead === 1 ? '' : 's'} to finish first…`));
  on('status', onStatus);
  on('clauses', onClauses);
  on('summaries', ({ items }) => items.forEach((it) => enqueue({ kind: 'summary', ...it })));
  on('done', () => {
    state.done = true;
    es.close();
    enqueue({ kind: 'done' });
  });
  on('failed', (data) => {
    state.done = true;
    es.close();
    hideStatus();
    setError(data.message);
  });

  es.onopen = () => { state.retries = 0; };
  es.onerror = () => {
    // EventSource retries dropped connections by itself; it only gives up (CLOSED)
    // on an HTTP error, e.g. the job expired, so check what happened
    if (es.readyState === EventSource.CLOSED && !state.done) onLost();
  };
}

async function onLost() {
  const res = await fetch(`/api/jobs/${state.jobId}`).catch(() => null);
  if (res && res.status === 404) {
    hideStatus();
    history.replaceState(null, '', location.pathname);
    return setError('That analysis has expired. Run it again to see the results.');
  }
  if (++state.retries <= 5) {
    setTimeout(() => connect(state.jobId), 1500 * state.retries);
  } else {
    showStatus('Lost the connection. It will pick back up when you return to this tab.');
  }
}

function onVisible() {
  // Phones often kill the connection while the screen is off. Reopen it, resuming
  // after the last event received
  if (document.visibilityState !== 'visible' || !state.jobId || state.done) return;
  if (!state.es || state.es.readyState === EventSource.CLOSED) {
    state.retries = 0;
    connect(state.jobId);
  }
}

function onStatus({ stage, sentences }) {
  const [text, step] = STAGES[stage] || ['Working…', null];
  showStatus(text.replace('{n}', sentences ?? ''), step);
  if (stage === 'summarizing') updateProgress();
}

function onClauses({ items, min_threshold, cached }) {
  state.jobThreshold = min_threshold;
  state.cached = cached;
  $('results-panel').hidden = false;
  // Pop cards in most-severe first
  const sorted = [...items].sort(compareItems);
  for (const item of sorted) enqueue({ kind: 'add', item });
}

// ---------------- Reveal queue ----------------

function enqueue(step) {
  state.queue.push(step);
  if (!state.revealing) revealNext();
}

function applyStep(step) {
  // Returns true if the step changed something on screen
  if (step.kind === 'add') {
    const item = { ...step.item, summary: null };
    item.el = makeCard(item);
    state.items.set(item.id, item);
    return isVisible(item);
  }
  if (step.kind === 'summary') {
    const item = state.items.get(step.id);
    if (!item || item.summary !== null) return false;
    item.summary = step.summary;
    state.summarized++;
    fillCard(item);
    updateProgress();
    return isVisible(item);
  }
  if (step.kind === 'done') finish();
  return true;
}

function revealNext() {
  // Apply steps until one changes something visible (clauses hidden by the slider
  // don't get an animation slot), then render and wait before the next
  let step;
  let shown = false;
  while (!shown && (step = state.queue.shift())) shown = applyStep(step);
  if (!step) {
    render();
    state.revealing = false;
    return;
  }
  state.revealing = true;
  render();

  let delay;
  if (reduceMotion) delay = 0;
  else if (state.cached) delay = step.kind === 'add' ? 18 : 28;  // saved results: quick
  else if (step.kind === 'add') delay = 60;
  else delay = Math.max(30, Math.min(220, 2400 / (state.queue.length + 1)));  // faster with a backlog
  setTimeout(revealNext, delay);
}

// ---------------- Rendering ----------------

function compareItems(a, b) {
  const pendingA = a.summary == null ? 1 : 0;
  const pendingB = b.summary == null ? 1 : 0;
  return (pendingA - pendingB)
    || (SEVERITY[a.label] - SEVERITY[b.label])
    || (b.unfair_prob - a.unfair_prob);
}

function passesThreshold(item) {
  return item.top_pick_unfair || item.unfair_prob >= Number($('threshold').value);
}

function isVisible(item) {
  return passesThreshold(item) && ($('show-potential').checked || item.label === 'clearly_unfair');
}

function makeCard(item) {
  const [emoji, name, cls] = LABELS[item.label];
  const li = document.createElement('li');
  li.className = `card ${cls} pending`;

  const head = document.createElement('div');
  head.className = 'card-head';
  const badge = document.createElement('span');
  badge.className = 'badge';
  badge.textContent = `${emoji} ${name}`;
  const prob = document.createElement('span');
  prob.className = 'prob';
  prob.textContent = `${Math.round(item.unfair_prob * 100)}% unfair`;
  head.append(badge, prob);

  const summary = document.createElement('div');
  summary.className = 'summary';
  summary.innerHTML = '<span class="skeleton"></span><span class="skeleton short"></span>';

  const details = document.createElement('details');
  details.className = 'clause-details';
  const label = document.createElement('summary');
  label.textContent = 'Original clause';
  const clause = document.createElement('p');
  clause.className = 'clause';
  clause.textContent = item.clause;
  details.append(label, clause);

  li.append(head, summary, details);
  return li;
}

function fillCard(item) {
  const summary = item.el.querySelector('.summary');
  summary.textContent = item.summary;
  item.el.classList.remove('pending');
  if (!reduceMotion) {
    item.el.classList.add('fresh');
    setTimeout(() => item.el.classList.remove('fresh'), 1700);
  }
}

function render() {
  const list = $('results');
  const all = [...state.items.values()];
  const visible = all.filter(isVisible).sort(compareItems);

  // First: where every card is now
  const before = new Map();
  for (const el of list.children) before.set(el, el.getBoundingClientRect().top);

  // Change the DOM to the new order
  const keep = new Set(visible.map((it) => it.el));
  for (const el of [...list.children]) if (!keep.has(el)) el.remove();
  visible.forEach((item, i) => {
    if (list.children[i] !== item.el) list.insertBefore(item.el, list.children[i] || null);
  });

  // Last + Invert + Play: slide moved cards from their old spot, fade new ones in
  if (!reduceMotion) {
    for (const item of visible) {
      const el = item.el;
      if (before.has(el)) {
        const dy = before.get(el) - el.getBoundingClientRect().top;
        if (Math.abs(dy) > 1) {
          el.animate([{ transform: `translateY(${dy}px)` }, { transform: 'none' }],
                     { duration: 700, easing: 'cubic-bezier(.2,.8,.2,1)' });
        }
      } else {
        el.animate([{ opacity: 0, transform: 'translateY(10px) scale(.97)' }, { opacity: 1, transform: 'none' }],
                   { duration: 420, easing: 'ease-out' });
      }
    }
  }

  const flagged = all.filter(passesThreshold);
  $('stat-flagged').textContent = flagged.length;
  $('stat-clear').textContent = flagged.filter((it) => it.label === 'clearly_unfair').length;
  $('stat-potential').textContent = flagged.filter((it) => it.label === 'potentially_unfair').length;
  $('empty').hidden = !(state.done && !state.queue.length && visible.length === 0);
  $('download').disabled = !flagged.some((it) => it.summary !== null);

  const below = !state.cached && Number($('threshold').value) < state.jobThreshold - 1e-9;
  $('threshold-note').hidden = !(state.done && below);
  $('job-threshold').textContent = state.jobThreshold.toFixed(2);
}

function resetResults() {
  if (state.es) state.es.close();
  Object.assign(state, {
    jobId: null, es: null, lastEventId: 0, retries: 0, done: false, cached: false,
    items: new Map(), summarized: 0, queue: [], revealing: false,
  });
  $('results').replaceChildren();
  $('results-panel').hidden = true;
  $('threshold-note').hidden = true;
  $('empty').hidden = true;
}

// ---------------- Status ----------------

function showStatus(text, step = null) {
  if ($('status').hidden) {
    // Bring the progress into view on phones, where it's below the input
    requestAnimationFrame(() => $('status').scrollIntoView({ behavior: 'smooth', block: 'nearest' }));
  }
  $('status').hidden = false;
  $('wait-link').hidden = state.cached || !!state.example || $('how').open;
  $('status-text').textContent = text;
  $('status').classList.toggle('indeterminate', step !== 2);
  document.querySelectorAll('.steps li').forEach((li, i) => li.classList.toggle('active', i === step));
  if (!state.timer) {
    state.startTime = Date.now();
    state.timer = setInterval(tick, 1000);
    tick();
  }
}

function updateProgress() {
  const total = state.items.size;
  if (!total) return;
  $('bar-fill').style.width = `${(100 * state.summarized) / total}%`;
  if (state.cached) {
    showStatus(`Loading saved results… ${state.summarized} / ${total}`, 2);
  } else {
    showStatus(`BART is rewriting flagged clauses in plain English… ${state.summarized} / ${total}`, 2);
  }
}

function tick() {
  const s = Math.round((Date.now() - state.startTime) / 1000);
  $('elapsed').textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function hideStatus() {
  $('status').hidden = true;
  $('bar-fill').style.width = '0';
  clearInterval(state.timer);
  state.timer = null;
  document.querySelectorAll('.steps li').forEach((li) => li.classList.remove('active'));
}

function finish() {
  hideStatus();
  $('results-panel').hidden = false;
}

// ---------------- CSV ----------------

function downloadCsv() {
  const rows = [...state.items.values()].filter(passesThreshold).sort(compareItems);
  const esc = (v) => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const lines = [['label', 'unfair_probability', 'summary', 'original_clause'].join(',')];
  for (const it of rows) lines.push([it.label, it.unfair_prob.toFixed(3), esc(it.summary), esc(it.clause)].join(','));
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = state.example ? `${state.example.slug}_tos_analysis.csv` : 'tos_analysis.csv';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

init();
