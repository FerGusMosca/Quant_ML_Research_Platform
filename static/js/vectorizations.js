// vectorizations.js — Vectorizations screen (Bias UI Dashboard)
//
// Scope model: the screen always shows one of three scopes — everything, one
// sector, or one security. The scope is now just the sector and the symbol of
// the filter bar, so the rail, the filters and the tables can never disagree.
//
// Punto #1.a: cualquier corrida se puede borrar, de a una o varias.
// Punto #1.b: los totales salen del vector store, no de lo que reportaron las
//             corridas, y la pantalla avisa cuando la lista viene cortada.
// Punto #1.c: la solapa By Sector muestra fechas y archivos.
// Punto #1.d: la solapa de filings filtra por Type, Year y Sector.

const API = '/vectorizations';

const PAGE_SIZE = 1000;

let REFERENCE = { sectors: [], portfolios: [], embedding_models: [], report_types: [], years: [] };
let FILTERS   = { sector: '', report_type: '', fiscal_year: '', quarter: '', symbol: '', pending: false };
// La solapa By Sector tiene sus propios filtros de archivo, anio y quarter,
// porque ahi se mira el conjunto y no una lista de archivos.
let SEC_FILTERS = { report_type: '', fiscal_year: '', quarter: '' };
let MODEL     = '';
let LAST_ROWS = [];
let LAST_TOTAL = 0;
let COVERAGE  = {};
let SELECTED_RUNS = new Set();
let EDITING_RUN_ID = null;
let PENDING_DELETE = [];
let LIVE_TIMER = null;
let EVENTS_AVAILABLE = true;

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

function fillSelect(id, values, allLabel) {
  const select = $(id);
  const current = select.value;
  select.innerHTML = `<option value="">${allLabel}</option>`;

  (values || []).forEach(value => {
    const option = document.createElement('option');
    option.value = String(value);
    option.textContent = String(value);
    select.appendChild(option);
  });

  select.value = current;
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
    if (value !== null && value !== undefined && value !== '' && value !== false) {
      parts.push(`${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
    }
  });
  return parts.length ? `?${parts.join('&')}` : '';
}

function scopeKind() {
  if (FILTERS.symbol) return 'symbol';
  if (FILTERS.sector) return 'sector';
  return 'all';
}

function filterParams(extra) {
  return Object.assign({
    symbol:          FILTERS.symbol,
    sector_code:     FILTERS.sector,
    report_type:     FILTERS.report_type,
    fiscal_year:     FILTERS.fiscal_year,
    quarter:         FILTERS.quarter,
    embedding_model: FILTERS.pending ? '' : MODEL,
    include_pending: FILTERS.pending ? 'true' : ''
  }, extra || {});
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
    paintFilterOptions(data);
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
  $('mtFiles').textContent      = num(totals.documents_registered);
  $('mtDocuments').textContent  = num(totals.documents);
  $('mtPending').textContent    = num(totals.documents_pending);
  $('mtChunks').textContent     = num(totals.chunks);
  $('mtSize').textContent       = totals.pretty_size || '—';

  // The whole point of #1.b: the difference between the files the store knows
  // about and the files that actually have vectors, said out loud.
  const pending = Number(totals.documents_pending || 0);
  const note = $('countNote');

  if (pending > 0) {
    note.classList.add('is-warn');
    note.textContent = `${num(totals.documents_registered)} files on record, `
      + `${num(totals.documents)} with vectors, ${num(pending)} still without any. `
      + `These counts come from the vector store itself, not from what the runs reported. `
      + `Tick "Show files without vectors" to list the ones missing.`;
  } else {
    note.classList.remove('is-warn');
    note.textContent = 'Counted straight from the vector store, not from what the runs reported.';
  }
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
  filter.value = MODEL;

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

function paintFilterOptions(data) {
  // #1.d — Type, Year and Sector, all fed from what the store really holds.
  fillSelect('fltSector', (data.sectors || []).map(row => row.sector_code), 'All sectors');
  fillSelect('fltType',   (data.report_types || []).map(row => row.report_type), 'All types');
  fillSelect('fltYear',   (data.years || []).map(row => row.fiscal_year), 'All years');
  fillSelect('fltQuarter', (data.quarters || []).map(row => row.quarter), 'All quarters');

  fillSelect('secType',    (data.report_types || []).map(row => row.report_type), 'All types');
  fillSelect('secYear',    (data.years || []).map(row => row.fiscal_year), 'All years');
  fillSelect('secQuarter', (data.quarters || []).map(row => row.quarter), 'All quarters');

  // NONE es el quarter en blanco de los archivos anuales. Mostrarlo asi evita
  // una fila muda en el combo.
  [$('fltQuarter'), $('secQuarter')].forEach(select => {
    Array.from(select.options).forEach(option => {
      if (option.value === 'NONE') option.textContent = 'Sin quarter (K10)';
    });
  });

  $('fltSector').value  = FILTERS.sector;
  $('fltType').value    = FILTERS.report_type;
  $('fltYear').value    = FILTERS.fiscal_year;
  $('fltQuarter').value = FILTERS.quarter;
  $('fltSymbol').value  = FILTERS.symbol;

  $('secType').value    = SEC_FILTERS.report_type;
  $('secYear').value    = SEC_FILTERS.fiscal_year;
  $('secQuarter').value = SEC_FILTERS.quarter;
}

function paintSectorRail(sectors) {
  const rail = $('railSectors');
  rail.innerHTML = '';

  (sectors || []).forEach(sector => {
    const item = document.createElement('button');
    item.className = 'rail-item' + (sector.vectorized ? '' : ' is-empty')
                   + (FILTERS.sector === sector.sector_code ? ' is-active' : '');
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

// ── Scope & filters ──
function selectSector(code) {
  FILTERS.sector = FILTERS.sector === code ? '' : code;
  $('fltSector').value = FILTERS.sector;
  paintSectorRail(REFERENCE.sectors);
  refreshScope();
}

function selectSymbol() {
  const symbol = ($('symbolInput').value || '').trim().toUpperCase();
  if (!symbol) return;
  FILTERS.symbol = symbol;
  $('fltSymbol').value = symbol;
  refreshScope();
}

function onSymbolFilterChange() {
  FILTERS.symbol = ($('fltSymbol').value || '').trim().toUpperCase();
  $('symbolInput').value = FILTERS.symbol;
  refreshScope();
}

function onFiltersChange() {
  FILTERS.sector      = $('fltSector').value;
  FILTERS.report_type = $('fltType').value;
  FILTERS.fiscal_year = $('fltYear').value;
  FILTERS.quarter     = $('fltQuarter').value;
  FILTERS.pending     = $('fltPending').checked;

  // Pending files have no model of their own, so a model filter on top of them
  // would always come back empty. The combo is disabled instead of lying.
  $('modelFilter').disabled = FILTERS.pending;

  paintSectorRail(REFERENCE.sectors);
  refreshScope();
}

function onSectorFiltersChange() {
  SEC_FILTERS.report_type = $('secType').value;
  SEC_FILTERS.fiscal_year = $('secYear').value;
  SEC_FILTERS.quarter     = $('secQuarter').value;
  refreshSectors();
}

function resetSectorFilters() {
  SEC_FILTERS = { report_type: '', fiscal_year: '', quarter: '' };
  $('secType').value = '';
  $('secYear').value = '';
  $('secQuarter').value = '';
  refreshSectors();
}

function resetFilters() {
  FILTERS = { sector: '', report_type: '', fiscal_year: '', quarter: '', symbol: '', pending: false };
  $('fltSector').value = '';
  $('fltType').value = '';
  $('fltYear').value = '';
  $('fltQuarter').value = '';
  $('fltSymbol').value = '';
  $('fltPending').checked = false;
  $('symbolInput').value = '';
  $('modelFilter').disabled = false;
  paintSectorRail(REFERENCE.sectors);
  refreshScope();
}

function clearScope() {
  resetFilters();
}

function onModelChange() {
  MODEL = $('modelFilter').value;
  refreshScope();
}

function paintScopeBar() {
  const kind = $('scopeKind');
  const value = $('scopeVal');
  $('scopeBar').classList.remove('is-error');

  const extra = [];
  if (FILTERS.report_type) extra.push(FILTERS.report_type);
  if (FILTERS.fiscal_year) extra.push(FILTERS.fiscal_year);
  if (FILTERS.quarter) extra.push(FILTERS.quarter === 'NONE' ? 'sin quarter' : FILTERS.quarter);
  const suffix = extra.length ? ` · ${extra.join(' · ')}` : '';

  if (scopeKind() === 'symbol') {
    kind.textContent = 'Security';
    value.textContent = FILTERS.symbol + (FILTERS.sector ? ` · ${FILTERS.sector}` : '') + suffix;
  } else if (scopeKind() === 'sector') {
    kind.textContent = 'Sector';
    value.textContent = FILTERS.sector + suffix;
  } else {
    kind.textContent = 'Everything';
    value.textContent = 'all sectors, all securities' + suffix;
  }

  $('btnClearScope').hidden = scopeKind() === 'all'
    && !FILTERS.report_type && !FILTERS.fiscal_year && !FILTERS.quarter && !FILTERS.pending;
}

async function refreshScope() {
  paintScopeBar();
  try {
    const filings = await getJson(`${API}/storage${queryString(filterParams({ top: PAGE_SIZE }))}`);
    paintFilings(filings.items, filings.total);

    const runs = await getJson(`${API}/runs${queryString({
      symbol: FILTERS.symbol, sector_code: FILTERS.sector, top: 300 })}`);
    paintRuns(runs.items);
    await loadLive();

    const overview = await getJson(`${API}/overview${queryString({
      embedding_model: FILTERS.pending ? '' : MODEL,
      sector_code:     FILTERS.sector,
      symbol:          FILTERS.symbol,
      report_type:     FILTERS.report_type,
      fiscal_year:     FILTERS.fiscal_year,
      quarter:         FILTERS.quarter })}`);

    paintTotals(overview.totals);
    await refreshSectors();
  } catch (e) {
    console.error(e);
    paintFatal(e.message);
  }
}

async function refreshSectors() {
  // La solapa By Sector se pide aparte: sus filtros de archivo, anio y quarter
  // son propios y no tienen por que arrastrar los de la lista de archivos.
  try {
    const data = await getJson(`${API}/overview${queryString({
      embedding_model: MODEL,
      report_type:     SEC_FILTERS.report_type,
      fiscal_year:     SEC_FILTERS.fiscal_year,
      quarter:         SEC_FILTERS.quarter })}`);

    COVERAGE = {};
    (data.coverage || []).forEach(row => { COVERAGE[coverageKey(row)] = row; });
    paintSectors(data.by_sector, data.coverage);
  } catch (e) {
    console.error(e);
  }
}

function coverageKey(row) {
  // Una fila de By Sector es sector + archivo + anio + quarter. La cobertura
  // se busca con la misma llave para que Files y Pending sean de esa fila y no
  // del sector entero.
  return [row.sector_code, row.report_type, row.fiscal_year, row.quarter || ''].join('|');
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
function paintFilings(rows, total) {
  rows = rows || [];
  LAST_ROWS = rows;
  LAST_TOTAL = total === undefined || total === null ? rows.length : total;

  const body = $('tblFilings');
  body.innerHTML = '';

  rows.forEach(row => {
    const tr = document.createElement('tr');
    const pending = Number(row.chunks || 0) === 0;
    if (pending) tr.className = 'is-pending';

    appendCell(tr, row.symbol, 'mono strong');
    appendCell(tr, row.file_name, 'file');
    appendCell(tr, row.report_type, 'mono');
    appendCell(tr, row.fiscal_year, 'mono');
    appendCell(tr, row.quarter || '—', 'mono');
    appendCell(tr, row.sector_code || 'UNCLASSIFIED', 'mono dim');
    appendCell(tr, row.embedding_model ? shortModel(row.embedding_model)
                                       : (row.models ? `${row.models} model(s)` : '—'), 'mono dim');

    const state = document.createElement('td');
    const badge = document.createElement('span');
    badge.className = 'badge ' + (pending ? 'badge-pending' : 'badge-ok');
    badge.textContent = pending ? 'PENDING' : 'VECTORIZED';
    state.appendChild(badge);
    tr.appendChild(state);

    appendCell(tr, num(row.chunks), 'num mono');
    appendCell(tr, row.pretty_size, 'num mono strong');
    appendCell(tr, shortDate(row.last_chunk_at), 'mono dim');
    body.appendChild(tr);
  });

  $('emptyFilings').hidden = rows.length > 0;
  $('btnExport').disabled = rows.length === 0;
  paintRowCount(rows.length, LAST_TOTAL);
}

function paintRowCount(shown, total) {
  // #1.b — the screen used to cut the list at its own limit and say nothing,
  // which reads exactly like "it is not counting the files right".
  const box = $('rowCount');

  if (!total) {
    box.textContent = '';
    box.classList.remove('is-warn');
    return;
  }

  if (shown < total) {
    box.classList.add('is-warn');
    box.textContent = `Showing ${num(shown)} of ${num(total)} rows — narrow the filters `
                    + `by sector, type or year to see the rest.`;
  } else {
    box.classList.remove('is-warn');
    box.textContent = `${num(total)} row${total === 1 ? '' : 's'}`;
  }
}

function paintSectors(rows, coverage) {
  rows = rows || [];
  coverage = coverage || [];

  const body = $('tblSectors');
  body.innerHTML = '';

  // Los grupos que tienen archivos pero ningun vector no aparecen en by_sector.
  // Sin ellos la solapa mostraria una foto limpia de un trabajo incompleto.
  const painted = new Set(rows.map(row => coverageKey(row)));
  const orphans = coverage
    .filter(row => !painted.has(coverageKey(row)) && Number(row.documents_registered || 0) > 0)
    .map(row => ({
      sector_code: row.sector_code,
      report_type: row.report_type,
      fiscal_year: row.fiscal_year,
      quarter: row.quarter,
      embedding_model: null,
      securities: row.securities,
      documents: 0,
      chunks: 0,
      bytes: 0,
      pretty_size: '0 bytes',
      first_vectorized_at: null,
      last_vectorized_at: null,
      last_file_name: null,
      last_symbol: null
    }));

  const all = rows.concat(orphans);

  all.forEach((row, index) => {
    const cov = COVERAGE[coverageKey(row)] || {};
    const tr = document.createElement('tr');

    const expand = document.createElement('td');
    const caret = document.createElement('button');
    caret.className = 'icon-btn caret';
    caret.textContent = '▸';
    caret.title = 'Show the last files of this group';
    caret.onclick = event => { event.stopPropagation(); toggleSectorDetail(index, row, caret); };
    expand.appendChild(caret);
    tr.appendChild(expand);

    const open = () => {
      FILTERS.sector      = row.sector_code;
      FILTERS.report_type = row.report_type || '';
      FILTERS.fiscal_year = row.fiscal_year || '';
      FILTERS.quarter     = row.quarter ? row.quarter : 'NONE';
      FILTERS.symbol      = '';
      $('fltSector').value  = FILTERS.sector;
      $('fltType').value    = FILTERS.report_type;
      $('fltYear').value    = FILTERS.fiscal_year;
      $('fltQuarter').value = FILTERS.quarter;
      $('fltSymbol').value  = '';
      paintSectorRail(REFERENCE.sectors);
      showTab('filings');
      refreshScope();
    };

    appendCell(tr, row.sector_code, 'mono strong clickable-cell').onclick = open;
    appendCell(tr, row.report_type || '—', 'mono');
    appendCell(tr, row.fiscal_year || '—', 'mono');
    appendCell(tr, row.quarter || '—', 'mono dim');
    appendCell(tr, row.embedding_model ? shortModel(row.embedding_model) : '—', 'mono dim');
    appendCell(tr, num(row.securities), 'num mono');
    appendCell(tr, num(cov.documents_registered), 'num mono');
    appendCell(tr, num(row.documents), 'num mono');

    const pendingCell = appendCell(tr, num(cov.documents_pending), 'num mono');
    if (Number(cov.documents_pending || 0) > 0) pendingCell.classList.add('warn');

    appendCell(tr, num(row.chunks), 'num mono');
    appendCell(tr, row.pretty_size, 'num mono strong');
    appendCell(tr, shortDate(row.first_vectorized_at), 'mono dim');
    appendCell(tr, shortDate(row.last_vectorized_at), 'mono');
    appendCell(tr, row.last_file_name
      ? `${row.last_symbol} · ${row.last_file_name}` : '—', 'file dim');

    body.appendChild(tr);

    // Placeholder row for the expanded detail, so the caret has somewhere to
    // put the files without repainting the table.
    const detail = document.createElement('tr');
    detail.className = 'detail-row';
    detail.id = `sectorDetail_${index}`;
    detail.hidden = true;
    const cell = document.createElement('td');
    cell.colSpan = 15;
    cell.className = 'detail-cell';
    detail.appendChild(cell);
    body.appendChild(detail);
  });

  $('emptySectors').hidden = all.length > 0;
}

async function toggleSectorDetail(index, row, caret) {
  const detail = $(`sectorDetail_${index}`);
  const cell = detail.firstChild;

  if (!detail.hidden) {
    detail.hidden = true;
    caret.textContent = '▸';
    return;
  }

  detail.hidden = false;
  caret.textContent = '▾';
  cell.textContent = 'Loading…';

  try {
    const data = await getJson(`${API}/storage${queryString({
      sector_code:     row.sector_code,
      embedding_model: row.embedding_model || '',
      report_type:     row.report_type || '',
      fiscal_year:     row.fiscal_year || '',
      quarter:         row.quarter ? row.quarter : 'NONE',
      include_pending: row.embedding_model ? '' : 'true',
      top: 25 })}`);

    cell.innerHTML = '';

    if (!data.items.length) {
      cell.textContent = 'No files for this sector.';
      return;
    }

    const head = document.createElement('div');
    head.className = 'detail-head';
    head.textContent = `Last ${data.items.length} of ${num(data.total)} files `
                     + `— ${row.sector_code} · ${row.report_type || ''} `
                     + `${row.fiscal_year || ''} ${row.quarter || ''}`.trimEnd();
    cell.appendChild(head);

    const table = document.createElement('table');
    table.className = 'tbl inner';
    const tbody = document.createElement('tbody');

    data.items.forEach(item => {
      const tr = document.createElement('tr');
      appendCell(tr, item.symbol, 'mono strong');
      appendCell(tr, item.file_name, 'file');
      appendCell(tr, item.report_type, 'mono');
      appendCell(tr, item.fiscal_year, 'mono');
      appendCell(tr, item.quarter || '—', 'mono dim');
      appendCell(tr, num(item.chunks), 'num mono');
      appendCell(tr, item.pretty_size, 'num mono');
      appendCell(tr, shortDate(item.last_chunk_at), 'mono dim');
      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    cell.appendChild(table);
  } catch (e) {
    cell.textContent = e.message;
  }
}

function paintRuns(rows) {
  rows = rows || [];
  const body = $('tblRuns');
  body.innerHTML = '';
  SELECTED_RUNS.clear();
  $('runsSelectAll').checked = false;

  rows.forEach((row, index) => {
    const tr = document.createElement('tr');

    const pick = document.createElement('td');
    const check = document.createElement('input');
    check.type = 'checkbox';
    check.className = 'run-check';
    check.dataset.runId = row.run_id;
    check.onchange = () => toggleRunSelection(row.run_id, check.checked);
    pick.appendChild(check);
    tr.appendChild(pick);

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

    // Found / processed / skipped / failed, all four. Showing only "processed"
    // was half the reason the file counts looked wrong: a resumed run skips
    // everything it had already done, and those files are not missing.
    appendCell(tr, num(row.files_found), 'num mono');
    appendCell(tr, num(row.files_processed), 'num mono');
    appendCell(tr, num(row.files_skipped), 'num mono dim');
    const failed = appendCell(tr, num(row.files_failed), 'num mono');
    if (Number(row.files_failed || 0) > 0) failed.classList.add('warn');

    appendCell(tr, num(row.chunks_persisted), 'num mono');

    // #II.1 — por donde va, sacado del log round robin
    const progress = document.createElement('td');
    progress.className = 'mono dim';
    if (row.live_total) {
      const done = Number(row.live_position || 0);
      const total = Number(row.live_total || 0);
      progress.textContent = `${num(done)}/${num(total)}`
        + (row.live_symbol ? ` · ${row.live_symbol}` : '');
      progress.classList.add('clickable-cell');
      progress.title = row.live_file_name || '';
    } else {
      progress.textContent = '—';
    }
    tr.appendChild(progress);

    appendCell(tr, shortDate(row.started_at), 'mono dim');

    const actions = document.createElement('td');
    actions.className = 'row-actions';

    // Abre el detalle de esa corrida: los archivos que fue tocando.
    const detailBtn = document.createElement('button');
    detailBtn.className = 'icon-btn';
    detailBtn.title = 'Ver el detalle de esta corrida';
    detailBtn.textContent = '▤';
    detailBtn.onclick = () => toggleRunEvents(index, row, detailBtn);
    actions.appendChild(detailBtn);

    if (row.run_source === 'MANUAL') {
      const edit = document.createElement('button');
      edit.className = 'icon-btn';
      edit.title = 'Edit';
      edit.textContent = '✎';
      edit.onclick = () => openRunModal(row);
      actions.appendChild(edit);
    }

    // #1.a — any run can go, manual or written by the job.
    const remove = document.createElement('button');
    remove.className = 'icon-btn danger';
    remove.title = 'Delete this run';
    remove.textContent = '🗑';
    remove.onclick = () => openDeleteModal([row.run_id]);
    actions.appendChild(remove);

    tr.appendChild(actions);
    body.appendChild(tr);

    const detail = document.createElement('tr');
    detail.className = 'detail-row';
    detail.id = `runDetail_${index}`;
    detail.hidden = true;
    const cell = document.createElement('td');
    cell.colSpan = 17;
    cell.className = 'detail-cell';
    detail.appendChild(cell);
    body.appendChild(detail);
  });

  $('emptyRuns').hidden = rows.length > 0;
  paintBulkBar();
}

// ── Log round robin de la corrida (#II.1) ──
async function toggleRunEvents(index, row, button) {
  const detail = $(`runDetail_${index}`);
  const cell = detail.firstChild;

  if (!detail.hidden) {
    detail.hidden = true;
    button.classList.remove('is-on');
    return;
  }

  detail.hidden = false;
  button.classList.add('is-on');
  cell.textContent = 'Loading…';

  try {
    const data = await getJson(`${API}/events${queryString({ run_id: row.run_id, top: 200 })}`);

    if (!data.available) {
      cell.textContent = 'El registro de corridas todavia no existe: corré db/vectors/04_vectorization_events.sql';
      return;
    }

    if (!data.items.length) {
      cell.textContent = 'Esta corrida ya salio de la ventana del registro (se guardan los ultimos N del dia).';
      return;
    }

    cell.innerHTML = '';
    const head = document.createElement('div');
    head.className = 'detail-head';
    head.textContent = `${data.items.length} pasos — run #${row.run_id}`;
    cell.appendChild(head);
    cell.appendChild(buildEventsTable(data.items));
  } catch (e) {
    cell.textContent = e.message;
  }
}

function buildEventsTable(items) {
  const table = document.createElement('table');
  table.className = 'tbl inner';
  const tbody = document.createElement('tbody');

  items.forEach(item => {
    const tr = document.createElement('tr');

    const kind = document.createElement('td');
    const badge = document.createElement('span');
    badge.className = 'badge ' + eventBadge(item.event_type);
    badge.textContent = eventLabel(item.event_type);
    kind.appendChild(badge);
    tr.appendChild(kind);

    appendCell(tr, item.total ? `${item.position || '—'}/${item.total}` : '—', 'mono dim');
    appendCell(tr, item.symbol || '—', 'mono strong');
    appendCell(tr, item.file_name || item.message || '—', 'file');
    appendCell(tr, item.sector_code || '—', 'mono dim');
    appendCell(tr, item.chunks !== null && item.chunks !== undefined ? num(item.chunks) : '—', 'num mono');
    appendCell(tr, item.elapsed_sec ? `${item.elapsed_sec}s` : '—', 'num mono dim');
    appendCell(tr, shortDate(item.created_at), 'mono dim');
    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  return table;
}

function eventLabel(type) {
  return ({ RUN_START: 'INICIO', RUN_END: 'FIN', FILE_START: 'EMPEZO',
            FILE_DONE: 'LISTO', FILE_SKIP: 'SALTEADO', FILE_FAIL: 'FALLO' })[type] || type;
}

function eventBadge(type) {
  if (type === 'FILE_DONE' || type === 'RUN_END') return 'badge-ok';
  if (type === 'FILE_FAIL') return 'badge-fail';
  if (type === 'FILE_SKIP') return 'badge-pending';
  return 'badge-auto';
}

function appendCell(tr, value, className) {
  const td = document.createElement('td');
  if (className) td.className = className;
  td.textContent = value === null || value === undefined ? '—' : String(value);
  tr.appendChild(td);
  return td;
}

function shortModel(model) {
  if (!model) return '—';
  const parts = String(model).split('/');
  return parts[parts.length - 1];
}

// ── Run selection (#1.a) ──
function toggleRunSelection(runId, checked) {
  if (checked) SELECTED_RUNS.add(runId);
  else SELECTED_RUNS.delete(runId);
  paintBulkBar();
}

function toggleAllRuns() {
  const checked = $('runsSelectAll').checked;
  SELECTED_RUNS.clear();

  document.querySelectorAll('.run-check').forEach(check => {
    check.checked = checked;
    if (checked) SELECTED_RUNS.add(Number(check.dataset.runId));
  });

  paintBulkBar();
}

function paintBulkBar() {
  const count = SELECTED_RUNS.size;
  $('bulkCount').textContent = `${count} selected`;
  $('btnBulkDelete').disabled = count === 0;
}

function deleteSelectedRuns() {
  if (!SELECTED_RUNS.size) return;
  openDeleteModal(Array.from(SELECTED_RUNS));
}

function openDeleteModal(runIds) {
  PENDING_DELETE = runIds || [];
  $('deleteModalError').hidden = true;
  $('deleteModalTitle').textContent = PENDING_DELETE.length === 1
    ? `Delete run #${PENDING_DELETE[0]}`
    : `Delete ${PENDING_DELETE.length} runs`;

  $('deleteModalText').textContent = PENDING_DELETE.length === 1
    ? `Run #${PENDING_DELETE[0]} will be removed from the history.`
    : `Runs ${PENDING_DELETE.join(', ')} will be removed from the history.`;

  $('deleteModal').hidden = false;
}

function closeDeleteModal() {
  $('deleteModal').hidden = true;
  PENDING_DELETE = [];
}

async function confirmDelete() {
  if (!PENDING_DELETE.length) return;

  const button = $('btnConfirmDelete');
  button.disabled = true;

  try {
    await postJson(`${API}/runs/delete`, { run_ids: PENDING_DELETE });
    closeDeleteModal();
    await refreshScope();
    showTab('runs');
  } catch (e) {
    const box = $('deleteModalError');
    box.textContent = e.message;
    box.hidden = false;
  } finally {
    button.disabled = false;
  }
}

// ── Manual run modal ──
function openRunModal(row) {
  EDITING_RUN_ID = row ? row.run_id : null;
  $('runModalTitle').textContent = row ? `Edit run #${row.run_id}` : 'Log a past run';
  $('runModalError').hidden = true;

  setSelectValue('fSector', row ? (row.sector_code || '') : FILTERS.sector);
  setSelectValue('fPortfolio', row ? (row.portfolio || '') : '');
  setSelectValue('fReportType', row ? (row.report_type || 'K10') : (FILTERS.report_type || 'K10'));
  setSelectValue('fQuarter', row ? (row.quarter || '') : '');
  setSelectValue('fModel', row ? (row.embedding_model || '') : (MODEL || defaultModel()));
  setSelectValue('fStatus', row ? (row.status || 'FINISHED') : 'FINISHED');

  $('fYear').value      = row ? (row.fiscal_year || '')
                              : (FILTERS.fiscal_year || new Date().getFullYear());
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
    FILTERS.sector = payload.sector_code || '';
    FILTERS.symbol = '';
    $('fltSector').value = FILTERS.sector;
    $('fltSymbol').value = '';
    paintSectorRail(REFERENCE.sectors);

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

// ── Export ──
function exportCsv() {
  if (!LAST_ROWS.length) return;

  const columns = ['symbol', 'file_name', 'report_type', 'fiscal_year', 'quarter',
                   'sector_code', 'embedding_model', 'vector_status', 'chunks', 'bytes',
                   'pretty_size', 'last_chunk_at'];

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
  link.download = `vectorizations_${FILTERS.symbol || FILTERS.sector || 'all'}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}


// ── Panel live del log round robin (#II.1) ──
//
// Muestra los ultimos pasos de la vectorizacion para el sector elegido, para
// no tener que leer el log de Python para saber por que archivo va.
async function loadLive() {
  const list = $('liveList');
  if (!list) return;

  try {
    const data = await getJson(`${API}/events${queryString({
      sector_code: FILTERS.sector, symbol: FILTERS.symbol, top: 40 })}`);

    EVENTS_AVAILABLE = data.available;

    if (!data.available) {
      $('liveNow').textContent = 'sin registro';
      list.innerHTML = '<div class="vz-live-empty">Corré db/vectors/04_vectorization_events.sql '
                     + 'para empezar a registrar lo que hace cada corrida.</div>';
      paintLiveBar(null);
      $('liveDot').className = 'vz-live-dot';
      return;
    }

    if (!data.items.length) {
      $('liveNow').textContent = 'sin actividad hoy';
      list.innerHTML = '<div class="vz-live-empty">Todavia no corrio nada hoy para este alcance.</div>';
      paintLiveBar(null);
      $('liveDot').className = 'vz-live-dot';
      return;
    }

    const last = data.items[0];
    const running = last.event_type !== 'RUN_END';

    $('liveDot').className = 'vz-live-dot' + (running ? ' is-on' : '');
    $('liveNow').textContent = running
      ? `${last.position || '—'}/${last.total || '—'} · ${last.symbol || ''} `
        + `${last.file_name || last.message || ''}`.trim()
      : `terminada — ${last.message || ''}`;

    paintLiveBar(last);

    list.innerHTML = '';
    list.appendChild(buildEventsTable(data.items));
  } catch (e) {
    console.error(e);
  }
}

function paintLiveBar(last) {
  const fill = $('liveBarFill');
  if (!fill) return;

  if (!last || !last.total) {
    fill.style.width = '0%';
    return;
  }

  const pct = Math.min(100, Math.round((Number(last.position || 0) / Number(last.total)) * 100));
  fill.style.width = `${pct}%`;
}

function toggleLiveAuto() {
  // Refresco corto porque un documento tarda segundos: mas seguido no aporta
  // nada y son consultas al mismo Postgres que esta escribiendo.
  if ($('liveAuto').checked) {
    LIVE_TIMER = setInterval(loadLive, 5000);
  } else if (LIVE_TIMER) {
    clearInterval(LIVE_TIMER);
    LIVE_TIMER = null;
  }
}
