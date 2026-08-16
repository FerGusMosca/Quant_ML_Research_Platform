// vectorizations.js — Vectorizations screen (Bias UI Dashboard)
//
// Scope model: the screen always shows one of three scopes — everything, one
// sector, or one security. Every tab reads from the same scope, so switching
// tabs never loses where you were.

const API = '/vectorizations';

let REFERENCE = { sectors: [], portfolios: [], embedding_models: [] };
let SCOPE     = { kind: 'all', value: null };
let MODEL     = '';
let LAST_ROWS = [];
let EDITING_RUN_ID = null;

// ── Clock (same behaviour as the rest of the dashboard) ──
(function tick() {
  const now = new Date();
  const pad = v => String(v).padStart(2, '0');
  const el = document.getElementById('navClock');
  if (el) el.textContent = [now.getHours(), now.getMinutes(), now.getSeconds()].map(pad).join(':');
  setTimeout(tick, 1000);
})();

// ── Helpers ──
const $ = id => document.getElementById(id);

function num(value) {
  if (value === null || value === undefined || value === '') return '—';
  return Number(value).toLocaleString('en-US');
}

function setSelectValue(id, value) {
  // A select silently drops a value that is not one of its options. When an
  // old run carries a portfolio or a model that is no longer in the
  // catalogue, the option is added on the spot so editing never loses it.
  const select = $(id);
  const wanted = value || '';

  if (wanted && !Array.from(select.options).some(option => option.value === wanted)) {
    const option = document.createElement('option');
    option.value = wanted;
    option.textContent = wanted;
    select.appendChild(option);
  }

  select.value = wanted;
}

function shortDate(value) {
  if (!value) return '—';
  return String(value).replace('T', ' ').slice(0, 16);
}

async function getJson(url) {
  const resp = await fetch(url);
  const data = await resp.json();
  if (!data.ok) throw new Error(data.error || 'Request failed');
  return data;
}

async function postJson(url, payload) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await resp.json();
  if (!data.ok) throw new Error(data.error || 'Request failed');
  return data;
}

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
document.addEventListener('DOMContentLoaded', reloadAll);

async function reloadAll() {
  try {
    const data = await getJson(`${API}/reference`);
    REFERENCE = data;
    paintTotals(data.totals);
    paintModels(data.embedding_models, data.model_options);
    paintSectorRail(data.sectors);
    paintPortfolios(data.portfolios);
    await loadSymbols();
    await refreshScope();
  } catch (e) {
    console.error(e);
    paintFatal(e.message);
  }
}

function paintFatal(message) {
  const bar = $('scopeBar');
  bar.classList.add('is-error');
  $('scopeKind').textContent = 'Cannot read the vector store';
  $('scopeVal').textContent = message;
}

function paintTotals(totals) {
  totals = totals || {};
  $('mtSecurities').textContent = num(totals.securities);
  $('mtDocuments').textContent  = num(totals.documents);
  $('mtChunks').textContent     = num(totals.chunks);
  $('mtSize').textContent       = totals.pretty_size || '—';
}

function paintModels(models, options) {
  // Top filter: only models that actually have chunks, because filtering by a
  // model with nothing stored would always come back empty.
  const filter = $('modelFilter');
  filter.innerHTML = '<option value="">All models</option>';
  (models || []).forEach(row => {
    const option = document.createElement('option');
    option.value = row.embedding_model;
    option.textContent = row.embedding_model;
    filter.appendChild(option);
  });

  // Modal combo: the full catalogue, since a past run may have used a model
  // whose chunks are not in this database.
  const modal = $('fModel');
  modal.innerHTML = '<option value="">—</option>';
  (options || []).forEach(name => {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    modal.appendChild(option);
  });
}

function paintSectorRail(sectors) {
  const rail = $('railSectors');
  rail.innerHTML = '';

  (sectors || []).forEach(sector => {
    const item = document.createElement('button');
    item.className = 'rail-item' + (sector.vectorized ? '' : ' is-empty');
    item.onclick = () => selectSector(sector.sector_code);

    const name = document.createElement('span');
    name.className = 'rail-item-name';
    name.textContent = sector.sector_code;

    const count = document.createElement('span');
    count.className = 'rail-item-count';
    count.textContent = sector.vectorized ? '●' : '';

    item.appendChild(name);
    item.appendChild(count);
    rail.appendChild(item);
  });

  // The sector select of the modal reads from the same catalogue
  const select = $('fSector');
  select.innerHTML = '<option value="">—</option>';
  (sectors || []).forEach(sector => {
    const option = document.createElement('option');
    option.value = sector.sector_code;
    option.textContent = sector.sector_code;
    select.appendChild(option);
  });
}

function paintPortfolios(portfolios) {
  const select = $('fPortfolio');
  select.innerHTML = '<option value="">—</option>';

  (portfolios || []).forEach(code => {
    const option = document.createElement('option');
    option.value = code;
    option.textContent = code;
    select.appendChild(option);
  });

  if (!(portfolios || []).length) {
    select.options[0].textContent = 'No portfolios found — check SEC Securities';
  }
}

async function loadSymbols() {
  try {
    const data = await getJson(`${API}/symbols?top=2000`);
    const list = $('symbolList');
    list.innerHTML = '';
    data.items.forEach(row => {
      const option = document.createElement('option');
      option.value = row.symbol;
      option.label = `${row.sector_code} · ${row.documents} filings`;
      list.appendChild(option);
    });
  } catch (e) {
    console.error(e);
  }
}

// ── Scope ──
function selectSector(code) {
  SCOPE = { kind: 'sector', value: code };
  refreshScope();
}

function selectSymbol() {
  const symbol = ($('symbolInput').value || '').trim().toUpperCase();
  if (!symbol) return;
  SCOPE = { kind: 'symbol', value: symbol };
  refreshScope();
}

function clearScope() {
  SCOPE = { kind: 'all', value: null };
  $('symbolInput').value = '';
  refreshScope();
}

function onModelChange() {
  MODEL = $('modelFilter').value;
  refreshScope();
}

function paintScopeBar() {
  const kind = $('scopeKind');
  const value = $('scopeVal');
  $('scopeBar').classList.remove('is-error');

  if (SCOPE.kind === 'sector') {
    kind.textContent = 'Sector';
    value.textContent = SCOPE.value;
  } else if (SCOPE.kind === 'symbol') {
    kind.textContent = 'Security';
    value.textContent = SCOPE.value;
  } else {
    kind.textContent = 'Everything';
    value.textContent = 'all sectors, all securities';
  }

  $('btnClearScope').hidden = SCOPE.kind === 'all';
}

async function refreshScope() {
  paintScopeBar();
  try {
    if (SCOPE.kind === 'symbol') {
      const data = await getJson(`${API}/symbol${queryString({
        symbol: SCOPE.value, embedding_model: MODEL })}`);
      paintFilings(data.documents);
      paintRuns(data.runs);
    } else if (SCOPE.kind === 'sector') {
      const data = await getJson(`${API}/sector${queryString({
        sector_code: SCOPE.value, embedding_model: MODEL })}`);
      paintFilings(data.documents);
      paintRuns(data.runs);
    } else {
      const filings = await getJson(`${API}/storage${queryString({
        embedding_model: MODEL, top: 1000 })}`);
      paintFilings(filings.items);
      const runs = await getJson(`${API}/runs${queryString({ top: 300 })}`);
      paintRuns(runs.items);
    }

    const overview = await getJson(`${API}/overview${queryString({ embedding_model: MODEL })}`);
    paintTotals(overview.totals);
    paintSectors(overview.by_sector);
  } catch (e) {
    console.error(e);
    paintFatal(e.message);
  }
}

// ── Tabs ──
function showTab(tab) {
  document.querySelectorAll('.vz-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tab);
  });
  $('paneFilings').hidden = tab !== 'filings';
  $('paneSectors').hidden = tab !== 'sectors';
  $('paneRuns').hidden    = tab !== 'runs';
}

// ── Painters ──
function paintFilings(rows) {
  rows = rows || [];
  LAST_ROWS = rows;
  const body = $('tblFilings');
  body.innerHTML = '';

  rows.forEach(row => {
    const tr = document.createElement('tr');
    appendCell(tr, row.symbol, 'mono strong');
    appendCell(tr, row.file_name, 'file');
    appendCell(tr, row.report_type, 'mono');
    appendCell(tr, row.fiscal_year, 'mono');
    appendCell(tr, row.quarter || '—', 'mono');
    appendCell(tr, row.sector_code || 'UNCLASSIFIED', 'mono dim');
    appendCell(tr, shortModel(row.embedding_model), 'mono dim');
    appendCell(tr, num(row.chunks), 'num mono');
    appendCell(tr, row.pretty_size, 'num mono strong');
    appendCell(tr, shortDate(row.last_chunk_at), 'mono dim');
    body.appendChild(tr);
  });

  $('emptyFilings').hidden = rows.length > 0;
  $('btnExport').disabled = rows.length === 0;
}

function paintSectors(rows) {
  rows = rows || [];
  const body = $('tblSectors');
  body.innerHTML = '';

  rows.forEach(row => {
    const tr = document.createElement('tr');
    tr.className = 'clickable';
    tr.onclick = () => { selectSector(row.sector_code); showTab('filings'); };

    appendCell(tr, row.sector_code, 'mono strong');
    appendCell(tr, shortModel(row.embedding_model), 'mono dim');
    appendCell(tr, num(row.securities), 'num mono');
    appendCell(tr, num(row.documents), 'num mono');
    appendCell(tr, num(row.chunks), 'num mono');
    appendCell(tr, row.pretty_size, 'num mono strong');
    appendCell(tr, shortDate(row.last_vectorized_at), 'mono dim');
    body.appendChild(tr);
  });

  $('emptySectors').hidden = rows.length > 0;
}

function paintRuns(rows) {
  rows = rows || [];
  const body = $('tblRuns');
  body.innerHTML = '';

  rows.forEach(row => {
    const tr = document.createElement('tr');
    appendCell(tr, row.run_id, 'mono');

    const source = document.createElement('td');
    const badge = document.createElement('span');
    badge.className = 'badge ' + (row.run_source === 'MANUAL' ? 'badge-manual' : 'badge-auto');
    badge.textContent = row.run_source || 'AUTO';
    source.appendChild(badge);
    tr.appendChild(source);

    appendCell(tr, row.sector_code || '—', 'mono');
    appendCell(tr, row.portfolio || '—', 'mono dim');
    appendCell(tr, row.report_type, 'mono');
    appendCell(tr, row.fiscal_year + (row.quarter ? ' ' + row.quarter : ''), 'mono');
    appendCell(tr, shortModel(row.embedding_model), 'mono dim');

    const status = document.createElement('td');
    const dot = document.createElement('span');
    dot.className = 'status status-' + (row.status || '').toLowerCase();
    dot.textContent = row.status || '—';
    status.appendChild(dot);
    tr.appendChild(status);

    appendCell(tr, num(row.files_processed), 'num mono');
    appendCell(tr, num(row.chunks_persisted), 'num mono');
    appendCell(tr, shortDate(row.started_at), 'mono dim');

    const actions = document.createElement('td');
    actions.className = 'row-actions';
    if (row.run_source === 'MANUAL') {
      const edit = document.createElement('button');
      edit.className = 'icon-btn';
      edit.title = 'Edit';
      edit.textContent = '✎';
      edit.onclick = () => openRunModal(row);
      actions.appendChild(edit);

      const remove = document.createElement('button');
      remove.className = 'icon-btn danger';
      remove.title = 'Delete';
      remove.textContent = '🗑';
      remove.onclick = () => deleteRun(row.run_id);
      actions.appendChild(remove);
    }
    tr.appendChild(actions);

    body.appendChild(tr);
  });

  $('emptyRuns').hidden = rows.length > 0;
}

function appendCell(tr, value, className) {
  const td = document.createElement('td');
  if (className) td.className = className;
  td.textContent = value === null || value === undefined ? '—' : String(value);
  tr.appendChild(td);
}

function shortModel(model) {
  if (!model) return '—';
  const parts = String(model).split('/');
  return parts[parts.length - 1];
}

// ── Manual run modal ──
function openRunModal(row) {
  EDITING_RUN_ID = row ? row.run_id : null;
  $('runModalTitle').textContent = row ? `Edit run #${row.run_id}` : 'Log a past run';
  $('runModalError').hidden = true;

  setSelectValue('fSector', row ? (row.sector_code || '')
                                 : (SCOPE.kind === 'sector' ? SCOPE.value : ''));
  setSelectValue('fPortfolio', row ? (row.portfolio || '') : '');
  setSelectValue('fReportType', row ? (row.report_type || 'K10') : 'K10');
  setSelectValue('fQuarter', row ? (row.quarter || '') : '');
  setSelectValue('fModel', row ? (row.embedding_model || '') : (MODEL || defaultModel()));
  setSelectValue('fStatus', row ? (row.status || 'FINISHED') : 'FINISHED');

  $('fYear').value      = row ? (row.fiscal_year || '') : new Date().getFullYear();
  $('fProcessed').value = row ? (row.files_processed || 0) : '';
  $('fStarted').value   = row ? toLocalInput(row.started_at) : '';
  $('fFinished').value  = row ? toLocalInput(row.finished_at) : '';
  $('fSymbols').value   = row ? (row.symbols_csv || '') : '';
  $('fNotes').value     = row ? (row.notes || '') : '';

  syncQuarter();
  $('runModal').hidden = false;
}

function syncQuarter() {
  // A K10 is an annual filing, so it has no quarter. Leaving the field open
  // invites a run saved as K10 Q3, which then matches nothing.
  const isAnnual = $('fReportType').value === 'K10';
  const quarter = $('fQuarter');

  if (isAnnual) quarter.value = '';
  quarter.disabled = isAnnual;
}

function closeRunModal() {
  $('runModal').hidden = true;
  EDITING_RUN_ID = null;
}

function defaultModel() {
  // The model the runs actually use. Falls back to whatever already has chunks,
  // and then to the first of the catalogue, so the field is never left empty.
  const preferred = 'sentence-transformers/all-mpnet-base-v2';
  const options = REFERENCE.model_options || [];

  if (options.includes(preferred)) return preferred;

  const models = REFERENCE.embedding_models || [];
  if (models.length) return models[0].embedding_model;

  return options.length ? options[0] : '';
}

function toLocalInput(value) {
  if (!value) return '';
  return String(value).slice(0, 16);
}

async function saveRun() {
  const payload = {
    run_id:          EDITING_RUN_ID,
    sector_code:     $('fSector').value,
    portfolio:       $('fPortfolio').value,
    report_type:     $('fReportType').value,
    fiscal_year:     $('fYear').value,
    quarter:         $('fQuarter').value,
    embedding_model: $('fModel').value,
    status:          $('fStatus').value,
    files_processed: $('fProcessed').value || 0,
    files_found:     $('fProcessed').value || 0,
    started_at:      $('fStarted').value || null,
    finished_at:     $('fFinished').value || null,
    symbols_csv:     $('fSymbols').value,
    notes:           $('fNotes').value
  };

  const button = $('btnSaveRun');
  button.disabled = true;

  try {
    await postJson(`${API}/runs`, payload);
    closeRunModal();

    // The run is saved against the sector of the form, which is not always the
    // sector the screen was filtered by. Without this the row would be saved
    // and then filtered out, which reads exactly like nothing happened.
    SCOPE = payload.sector_code
      ? { kind: 'sector', value: payload.sector_code }
      : { kind: 'all', value: null };

    showTab('runs');
    await refreshScope();
  } catch (e) {
    const box = $('runModalError');
    box.textContent = e.message;
    box.hidden = false;
  } finally {
    button.disabled = false;
  }
}

async function deleteRun(runId) {
  if (!confirm(`Delete run #${runId}? Only manually logged runs can be removed.`)) return;
  try {
    await postJson(`${API}/runs/delete`, { run_id: runId });
    await refreshScope();
  } catch (e) {
    alert(e.message);
  }
}

// ── Export ──
function exportCsv() {
  if (!LAST_ROWS.length) return;

  const columns = ['symbol', 'file_name', 'report_type', 'fiscal_year', 'quarter',
                   'sector_code', 'embedding_model', 'chunks', 'bytes', 'pretty_size',
                   'last_chunk_at'];

  const lines = [columns.join(',')];
  LAST_ROWS.forEach(row => {
    lines.push(columns.map(col => {
      const value = row[col] === null || row[col] === undefined ? '' : String(row[col]);
      return `"${value.replace(/"/g, '""')}"`;
    }).join(','));
  });

  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `vectorizations_${SCOPE.value || 'all'}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}
