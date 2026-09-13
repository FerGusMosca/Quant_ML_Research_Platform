// reports_runner.js — Reports Runner screen (Bias UI Dashboard)
//
// Replaces the PowerShell websocket script. The browser opens an EventSource
// against the dashboard, the dashboard holds the websocket against
// run_report_mcp_server, and every message lands in the console below.

const API = '/reports_runner';

let REPORTS = [];
let SELECTED = null;
let DEST_SUFFIX = '';
let RANK_SUFFIX = '';
let STREAM = null;
let CAL_ROWS = [];
let RUNS_TIMER = null;
let CAL_COLS = [];

// ── Clock ──
(function tick() {
  const now = new Date();
  const pad = v => String(v).padStart(2, '0');
  const el = document.getElementById('navClock');
  if (el) el.textContent = [now.getHours(), now.getMinutes(), now.getSeconds()].map(pad).join(':');
  setTimeout(tick, 1000);
})();

const $ = id => document.getElementById(id);

function queryString(params) {
  const parts = [];
  Object.keys(params).forEach(key => {
    const value = params[key];
    if (value !== null && value !== undefined && value !== '') {
      parts.push(`${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
    }
  });
  return parts.length ? `?${parts.join('&')}` : '';
}

// ── Bootstrap ──
document.addEventListener('DOMContentLoaded', loadReference);

async function loadReference() {
  try {
    const resp = await fetch(`${API}/reference`);
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error || 'Could not read the report list');

    REPORTS = data.reports || [];
    DEST_SUFFIX = data.dest_folder_suffix || '';
    RANK_SUFFIX = data.rank_folder_suffix || '';
    paintCards();
    paintPortfolios(data.portfolios || []);

    // Changing the portfolio has to re-derive the folder names, otherwise the
    // boxes keep showing the previous portfolio and the run writes to the wrong place.
    $('fPortfolio').addEventListener('change', fillFolderDefaults);

    if (REPORTS.length) selectReport(REPORTS[0].report);
  } catch (e) {
    writeLine(`✕ ${e.message}`, 'err');
  }
}

function paintCards() {
  const wrap = $('reportCards');
  wrap.innerHTML = '';

  REPORTS.forEach(report => {
    const card = document.createElement('button');
    card.className = 'rr-card';
    card.dataset.report = report.report;
    card.onclick = () => selectReport(report.report);

    const label = document.createElement('div');
    label.className = 'rr-card-label';
    label.textContent = report.label;

    const code = document.createElement('div');
    code.className = 'rr-card-code';
    code.textContent = report.report;

    const desc = document.createElement('div');
    desc.className = 'rr-card-desc';
    desc.textContent = report.description;

    card.appendChild(label);
    card.appendChild(code);
    card.appendChild(desc);
    wrap.appendChild(card);
  });
}

function paintPortfolios(portfolios) {
  const select = $('fPortfolio');
  select.innerHTML = '';

  if (!portfolios.length) {
    // Nothing to pick from means the tag catalogue could not be read. Saying so
    // in the control itself beats an empty list the user cannot explain.
    const empty = document.createElement('option');
    empty.value = '';
    empty.textContent = 'No portfolios found — check SEC Securities';
    select.appendChild(empty);
    return;
  }

  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = 'Pick a portfolio';
  select.appendChild(placeholder);

  portfolios.forEach(code => {
    const option = document.createElement('option');
    option.value = code;
    option.textContent = code;
    select.appendChild(option);
  });
}

function selectReport(report) {
  SELECTED = report;
  document.querySelectorAll('.rr-card').forEach(card => {
    card.classList.toggle('active', card.dataset.report === report);
  });
  paintFolderFields();
}

function currentReport() {
  return REPORTS.find(r => r.report === SELECTED) || null;
}

function needsFolders() {
  const report = currentReport();
  return !!(report && report.needs_folders);
}

function paintFolderFields() {
  const show = needsFolders();
  $('wrapDestFolder').hidden = !show;
  $('wrapRankFolder').hidden = !show;
  $('sentimentNote').hidden = !show;
  if (show) fillFolderDefaults();
}

function fillFolderDefaults() {
  if (!needsFolders()) return;

  const portfolio = ($('fPortfolio').value || '').trim();
  if (!portfolio) {
    $('fDestFolder').value = '';
    $('fRankFolder').value = '';
    return;
  }

  $('fDestFolder').value = `${portfolio}${DEST_SUFFIX}`;
  $('fRankFolder').value = `${portfolio}${RANK_SUFFIX}`;
}

// ── Tabs ──
function showTab(tab) {
  document.querySelectorAll('.rr-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tab);
  });
  $('paneRun').hidden = tab !== 'run';
  $('paneRuns').hidden = tab !== 'runs';
  $('paneCalendar').hidden = tab !== 'calendar';

  if (tab === 'runs') loadRuns();
}

// ── Runs (lo que quedo anotado en la base) ──
async function loadRuns() {
  const params = queryString({
    top: $('rTop').value || 50,
    status: $('rStatus').value || ''
  });

  try {
    const resp = await fetch(`${API}/runs${params}`);
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error || 'Could not read the runs');
    paintRuns(data.items || []);
  } catch (e) {
    paintRuns([]);
  }
}

function paintRuns(rows) {
  const body = $('runsBody');
  body.innerHTML = '';

  rows.forEach(row => {
    const tr = document.createElement('tr');

    const cells = [
      row.id,
      row.report_key,
      row.portfolio,
      row.year,
      null,                       // el estado va como pastilla
      shortStamp(row.start_time),
      shortStamp(row.end_time),
      elapsed(row.elapsed_seconds),
      row.last_error
    ];

    cells.forEach((value, index) => {
      const td = document.createElement('td');
      td.className = 'mono';

      if (index === 4) {
        const pill = document.createElement('span');
        pill.className = `rr-pill rr-pill-${row.status}`;
        pill.textContent = row.status;
        td.appendChild(pill);
      } else if (index === 8 && value) {
        td.className = 'mono rr-err-cell';
        td.title = value;
        td.textContent = value.length > 60 ? `${value.slice(0, 60)}…` : value;
      } else {
        td.textContent = value === null || value === undefined || value === '' ? '—' : String(value);
      }

      tr.appendChild(td);
    });

    const actions = document.createElement('td');

    if (row.status === 'started') {
      const abort = document.createElement('button');
      abort.className = 'btn danger';
      abort.textContent = 'Abort';
      abort.onclick = () => abortRun(row.id);
      actions.appendChild(abort);
    }

    const del = document.createElement('button');
    del.className = 'btn ghost';
    del.textContent = 'Delete';
    del.onclick = () => deleteRun(row.id);
    actions.appendChild(del);

    tr.appendChild(actions);
    body.appendChild(tr);
  });

  $('runsEmpty').hidden = rows.length > 0;
}

async function abortRun(id) {
  if (!confirm(`Run ${id} will be marked as aborted. Continue?`)) return;
  await postRuns('/runs/abort', { run_id: id });
}

async function abortAllStuck() {
  if (!confirm('Every run still sitting in "started" will be marked as aborted. Continue?')) return;
  await postRuns('/runs/abort', {});
}

async function deleteRun(id) {
  if (!confirm(`Run ${id} will be deleted for good. Continue?`)) return;
  await postRuns('/runs/delete', { run_id: id });
}

async function postRuns(path, params) {
  try {
    const resp = await fetch(`${API}${path}${queryString(params)}`, { method: 'POST' });
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error || 'The database refused the change');
    loadRuns();
  } catch (e) {
    alert(e.message);
  }
}

function toggleAutoRuns() {
  const btn = $('btnAuto');

  if (RUNS_TIMER) {
    clearInterval(RUNS_TIMER);
    RUNS_TIMER = null;
    btn.textContent = '⟳ Auto off';
    return;
  }

  RUNS_TIMER = setInterval(loadRuns, 5000);
  btn.textContent = '⟳ Auto on';
  loadRuns();
}

function shortStamp(value) {
  if (!value) return '—';
  return String(value).replace('T', ' ').slice(0, 19);
}

function elapsed(seconds) {
  if (seconds === null || seconds === undefined || seconds === '') return '—';
  const total = parseInt(seconds, 10);
  if (isNaN(total)) return '—';
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h ? `${h}h ${m}m` : (m ? `${m}m ${s}s` : `${s}s`);
}

// ── Console ──
function writeLine(text, kind) {
  const box = $('console');
  const line = document.createElement('div');
  line.className = 'rr-line' + (kind ? ` rr-${kind}` : '');
  const stamp = new Date().toTimeString().slice(0, 8);
  line.textContent = `${stamp}  ${text}`;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

function clearConsole() {
  $('console').textContent = '';
}

function setStatus(text, kind) {
  const el = $('runStatus');
  el.textContent = text;
  el.className = 'rr-status' + (kind ? ` rr-${kind}` : '');
}

// ── Run ──
function runReport() {
  if (STREAM) return;

  if (!SELECTED) {
    writeLine('✕ Pick a report first.', 'err');
    return;
  }

  const portfolio = ($('fPortfolio').value || '').trim();
  if (!portfolio) {
    writeLine('✕ Portfolio is required.', 'err');
    $('fPortfolio').focus();
    return;
  }

  const request = {
    report: SELECTED,
    portfolio: portfolio,
    year_from: $('fYearFrom').value,
    year_to: $('fYearTo').value || $('fYearFrom').value
  };

  if (needsFolders()) {
    request.dest_folder = ($('fDestFolder').value || '').trim();
    request.rank_folder = ($('fRankFolder').value || '').trim();
  }

  const params = queryString(request);

  clearConsole();
  setStatus('running', 'run');
  $('btnRun').disabled = true;
  $('btnStop').hidden = false;

  STREAM = new EventSource(`${API}/run${params}`);

  // Lo que vale es lo anotado en la base, asi que la pantalla se para ahi y se
  // refresca sola mientras la corrida avanza.
  showTab('runs');
  if (!RUNS_TIMER) toggleAutoRuns();

  STREAM.onmessage = event => {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch (e) {
      writeLine(event.data);
      return;
    }

    if (payload.event === 'started') {
      writeLine(`▶ ${payload.report} → ${JSON.stringify(payload.arguments)}`, 'ok');
      writeLine(`   server ${payload.uri}`, 'dim');
    } else if (payload.event === 'message') {
      writeLine(payload.raw);
    } else if (payload.event === 'error') {
      writeLine(`✕ ${payload.error}`, 'err');
    } else if (payload.event === 'done') {
      if (payload.ok) {
        writeLine(`✔ Completed: ${payload.report}`, 'ok');
        if (payload.summary) writeLine(`   ${JSON.stringify(payload.summary)}`, 'dim');
        setStatus('completed', 'ok');
      } else {
        writeLine(`✕ Failed: ${payload.error || 'no completion event'}`, 'err');
        setStatus('failed', 'err');
      }
      finishStream();
    }
  };

  STREAM.onerror = () => {
    writeLine('✕ Connection to the dashboard dropped.', 'err');
    setStatus('disconnected', 'err');
    finishStream();
  };
}

function stopReport() {
  writeLine('■ Stopped listening. The report keeps running on the server.', 'dim');
  setStatus('detached', 'dim');
  finishStream();
}

function finishStream() {
  if (STREAM) {
    STREAM.close();
    STREAM = null;
  }
  $('btnRun').disabled = false;
  $('btnStop').hidden = true;
}

// ── Calendar ──
async function loadCalendar() {
  const params = queryString({
    year_from: $('cYearFrom').value,
    year_to: $('cYearTo').value || $('cYearFrom').value,
    symbol: ($('cSymbol').value || '').trim()
  });

  try {
    const resp = await fetch(`${API}/calendar${params}`);
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error || 'Could not read the calendar');

    CAL_ROWS = data.items || [];
    CAL_COLS = CAL_ROWS.length ? Object.keys(CAL_ROWS[0]) : [];
    paintCalendar();
  } catch (e) {
    CAL_ROWS = [];
    CAL_COLS = [];
    paintCalendar();
    alert(e.message);
  }
}

function paintCalendar() {
  const head = $('calHead');
  const body = $('calBody');
  head.innerHTML = '';
  body.innerHTML = '';

  CAL_COLS.forEach(col => {
    const th = document.createElement('th');
    th.textContent = col.replace(/_/g, ' ');
    head.appendChild(th);
  });

  CAL_ROWS.forEach(row => {
    const tr = document.createElement('tr');
    CAL_COLS.forEach(col => {
      const td = document.createElement('td');
      td.className = 'mono';
      const value = row[col];
      td.textContent = value === null || value === undefined || value === '' ? '—' : String(value);
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });

  $('calEmpty').hidden = CAL_ROWS.length > 0;
  $('btnCalCsv').disabled = CAL_ROWS.length === 0;
}

function exportCalendar() {
  if (!CAL_ROWS.length) return;

  const lines = [CAL_COLS.join(',')];
  CAL_ROWS.forEach(row => {
    lines.push(CAL_COLS.map(col => {
      const value = row[col] === null || row[col] === undefined ? '' : String(row[col]);
      return `"${value.replace(/"/g, '""')}"`;
    }).join(','));
  });

  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `securities_calendar_${$('cYearFrom').value}_${$('cYearTo').value}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}
