// sec_securities.js

// ── Clock ─────────────────────────────────────────────────────────────────
(function tick() {
  const el = document.getElementById('navClock');
  if (el) {
    const n = new Date(), p = v => String(v).padStart(2, '0');
    el.textContent = [n.getHours(), n.getMinutes(), n.getSeconds()].map(p).join(':');
  }
  setTimeout(tick, 1000);
})();

// ── Constants / state ─────────────────────────────────────────────────────
const BASE = '/sec_securities';
const POLL_MS = 1500;

// Same colors as the dbo.SEC_Tags seed
const SECTOR_COLORS = {
  ENERGY: '#D29922', MATERIALS: '#9A7B4F', INDUSTRIALS: '#6E7681',
  CONS_DISCR: '#D95F7A', CONS_STAPLES: '#3FB950', HEALTH_CARE: '#58A6FF',
  FINANCIALS: '#1F6FEB', INFO_TECH: '#7D5BD6', COMM_SVCS: '#C74A9E',
  UTILITIES: '#4FB3A5', REAL_ESTATE: '#B58A3F', GOVT: '#3D444D',
  UNKNOWN: '#3D444D'
};

let _pollTimer = null;
let _sectors   = [];
let _tags      = [];
let _items     = [];
let _rowSymbol = null;

const F = { sector: '', tag: '', text: '' };

// ── Boot ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  refreshStatus();
  loadTags();

  document.getElementById('filterText').addEventListener('keydown', e => {
    if (e.key === 'Enter') applyFilter();
  });
  document.getElementById('singleSymbol').addEventListener('keydown', e => {
    if (e.key === 'Enter') runSingle();
  });
  document.getElementById('rowTagCode').addEventListener('keydown', e => {
    if (e.key === 'Enter') applyRowTag();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') { closeTagModal(); closeRowTagModal(); }
  });
});

// ── Helpers ───────────────────────────────────────────────────────────────
function esc(v) {
  if (v === null || v === undefined) return '';
  return String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function color(sector) { return SECTOR_COLORS[sector] || '#3D444D'; }

let _toastTimer = null;
function toast(text, kind) {
  const el = document.getElementById('toast');
  el.textContent = text;
  el.className = 'toast' + (kind ? ' ' + kind : '');
  el.hidden = false;
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { el.hidden = true; }, 5000);
}

async function api(url, options) {
  const res = await fetch(url, options);
  let body = {};
  try { body = await res.json(); } catch (e) { body = {}; }
  if (!res.ok || body.ok === false) {
    throw new Error(body.error || body.detail || ('HTTP ' + res.status));
  }
  return body;
}

// ══════════════════════════════════════════════════ STATUS
async function refreshStatus() {
  try {
    const data = await api(BASE + '/status');
    paintCoverage(data.summary || {});
    paintRun(data.run || {});
  } catch (e) {
    toast('Could not read status: ' + e.message, 'err');
  }
}

function paintCoverage(summary) {
  const total   = summary.total || 0;
  const ok      = summary.ok_qty || 0;
  const pending = summary.pending_qty || 0;
  const failed  = (summary.error_qty || 0) + (summary.not_found_qty || 0);

  document.getElementById('covOk').textContent    = ok.toLocaleString('en-US');
  document.getElementById('covTotal').textContent = total.toLocaleString('en-US');
  document.getElementById('covOkTxt').textContent      = ok + ' classified';
  document.getElementById('covPendingTxt').textContent = pending + ' pending';
  document.getElementById('covFailedTxt').textContent  = failed + ' failed';

  _sectors = summary.sectors || [];
  paintSpectrum(total);
  paintRailSectors();
}

// The spectrum: each sector is a segment sized against the whole universe.
// Whatever is still unclassified stays as the hatched background, so the bar
// visibly fills up while the job runs.
function paintSpectrum(total) {
  const strip  = document.getElementById('spectrum');
  const legend = document.getElementById('spectrumLegend');

  if (!_sectors.length) {
    strip.innerHTML = '';
    legend.innerHTML = '';
    return;
  }

  const denom = total || _sectors.reduce((a, s) => a + s.qty, 0);

  strip.innerHTML = _sectors.map(s => {
    const pct = denom ? (s.qty / denom) * 100 : 0;
    const on  = F.sector === s.sector_code ? ' on' : '';
    return `<div class="seg${on}" style="width:${pct.toFixed(3)}%;background:${color(s.sector_code)}"`
         + ` role="button" tabindex="0" title="${esc(s.sector_name || s.sector_code)} — ${s.qty}"`
         + ` onclick="toggleSector('${esc(s.sector_code)}')"`
         + ` onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();toggleSector('${esc(s.sector_code)}')}"></div>`;
  }).join('');

  legend.innerHTML = _sectors.map(s => {
    const on = F.sector === s.sector_code ? ' on' : '';
    return `<span class="leg${on}" onclick="toggleSector('${esc(s.sector_code)}')">`
         + `<i style="background:${color(s.sector_code)}"></i>${esc(s.sector_code)} ${esc(s.qty)}</span>`;
  }).join('');
}

function paintRailSectors() {
  document.getElementById('railSectors').innerHTML = _sectors.map(s => {
    const on = F.sector === s.sector_code ? ' on' : '';
    return `<button class="rail-item${on}" onclick="toggleSector('${esc(s.sector_code)}')">`
         + `<i style="background:${color(s.sector_code)}"></i>`
         + `<span class="nm">${esc(s.sector_name || s.sector_code)}</span>`
         + `<span class="qt">${esc(s.qty)}</span></button>`;
  }).join('') || '<div class="empty" style="padding:14px">Nothing classified yet</div>';
}

function paintRun(run) {
  const box  = document.getElementById('runProgress');
  const fill = document.getElementById('trackFill');

  if (run.running) {
    box.hidden = false;
    document.getElementById('btnCancel').hidden = false;
    document.getElementById('btnRun').disabled = true;
    document.getElementById('btnRetry').disabled = true;

    const pct = run.total ? Math.round((run.done / run.total) * 100) : 0;
    fill.style.width = pct + '%';
    document.getElementById('runCurrent').textContent = run.current ? '→ ' + run.current : '';
    document.getElementById('runNums').textContent =
      `${run.done}/${run.total} · ${pct}% · ok ${run.ok} · fail ${run.failed}`;
    startPolling();
  } else {
    document.getElementById('btnCancel').hidden = true;
    document.getElementById('btnRun').disabled = false;
    document.getElementById('btnRetry').disabled = false;
    stopPolling();

    if (run.finished_at) {
      box.hidden = false;
      fill.style.width = '100%';
      document.getElementById('runCurrent').textContent = 'Finished';
      document.getElementById('runNums').textContent =
        `${run.ok} ok · ${run.failed} failed · ${run.total} processed`;
    } else {
      box.hidden = true;
    }
  }
}

function startPolling() { if (!_pollTimer) _pollTimer = setInterval(refreshStatus, POLL_MS); }
function stopPolling()  { if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; } }

// ══════════════════════════════════════════════════ ACTIONS
async function runAll() {
  try {
    await api(BASE + '/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ include_errors: false })
    });
    toast('Run started: about 7 requests per second against the SEC.', 'ok');
    startPolling();
    refreshStatus();
  } catch (e) { toast(e.message, 'warn'); }
}

async function resetErrors() {
  try {
    const data = await api(BASE + '/reset_errors', { method: 'POST' });
    toast(data.reset + ' securities back in the queue.', 'ok');
    refreshStatus();
  } catch (e) { toast(e.message, 'err'); }
}

async function cancelRun() {
  try {
    await api(BASE + '/cancel', { method: 'POST' });
    toast('Stop requested: it finishes the current security and halts.', 'warn');
  } catch (e) { toast(e.message, 'err'); }
}

async function runSingle() {
  const input = document.getElementById('singleSymbol');
  const symbol = input.value.trim();
  if (!symbol) { toast('Type a symbol.', 'warn'); return; }

  const btn = document.getElementById('btnSingle');
  btn.disabled = true;
  try {
    const data = await api(BASE + '/run_single', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol: symbol })
    });
    toast(data.result.symbol + ' updated.', 'ok');
    input.value = '';
    refreshStatus();
    if (_items.length) applyFilter();
  } catch (e) {
    toast(symbol + ': ' + e.message, 'err');
  } finally { btn.disabled = false; }
}

async function refreshRow(symbol) {
  try {
    await api(BASE + '/run_single', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol: symbol })
    });
    toast(symbol + ' updated.', 'ok');
    refreshStatus();
    applyFilter();
  } catch (e) { toast(symbol + ': ' + e.message, 'err'); }
}

// ══════════════════════════════════════════════════ FILTERS
function toggleSector(code) {
  F.sector = (F.sector === code) ? '' : code;
  paintSpectrum(_sectors.reduce((a, s) => a + s.qty, 0));
  paintRailSectors();
  applyFilter();
}

function toggleTag(code) {
  F.tag = (F.tag === code) ? '' : code;
  paintRailTags();
  applyFilter();
}

function clearFilter() {
  F.sector = ''; F.tag = ''; F.text = '';
  document.getElementById('filterText').value = '';
  paintSpectrum(_sectors.reduce((a, s) => a + s.qty, 0));
  paintRailSectors();
  paintRailTags();
  _items = [];
  document.getElementById('tblBody').innerHTML = '';
  document.getElementById('resultsCount').hidden = true;
  document.getElementById('activeFilters').hidden = true;
  document.getElementById('emptyState').style.display = '';
  document.getElementById('btnExport').disabled = true;
}

function paintActiveFilters() {
  const box = document.getElementById('activeFilters');
  const chips = [];
  if (F.sector) chips.push(['Sector: ' + F.sector, "F.sector=''"]);
  if (F.tag)    chips.push(['Tag: ' + F.tag,       "F.tag=''"]);
  if (F.text)   chips.push(['"' + F.text + '"',    "F.text='';document.getElementById('filterText').value=''"]);

  if (!chips.length) { box.hidden = true; return; }
  box.hidden = false;
  box.innerHTML = chips.map(c =>
    `<span class="fchip">${esc(c[0])}<button onclick="${c[1]};paintRailSectors();paintRailTags();applyFilter()">×</button></span>`
  ).join('');
}

async function applyFilter() {
  F.text = document.getElementById('filterText').value.trim();

  if (!F.sector && !F.tag && !F.text) { clearFilter(); return; }

  const params = new URLSearchParams();
  if (F.sector) params.set('sector_code', F.sector);
  if (F.tag)    params.set('tag_code', F.tag);
  if (F.text)   params.set('text', F.text);

  try {
    const data = await api(BASE + '/securities?' + params.toString());
    _items = data.items || [];
    paintTable();
    paintActiveFilters();
  } catch (e) {
    toast('Could not filter: ' + e.message, 'err');
  }
}

function paintTable() {
  const body = document.getElementById('tblBody');

  body.innerHTML = _items.map(r => {
    const sym  = r.symbol || r.ticker || '';
    const tags = (r.tags || '').split(',').filter(Boolean).map(t =>
      `<span class="tag">${esc(t)}<button title="Remove tag"`
      + ` onclick="removeTag('${esc(t)}',${r.id})">×</button></span>`).join('');

    const sector = r.sector_code
      ? `<span class="sector-pill"><i style="background:${color(r.sector_code)}"></i>${esc(r.sector_code)}</span>`
      : '<span class="mono">—</span>';

    return '<tr>'
      + `<td class="sym">${esc(sym)}</td>`
      + `<td class="nm-cell" title="${esc(r.name)}">${esc(r.name)}</td>`
      + `<td class="mono">${esc(r.sic) || '—'}</td>`
      + `<td class="nm-cell" title="${esc(r.sic_description)}">${esc(r.sic_description) || '—'}</td>`
      + `<td>${sector}</td>`
      + `<td class="mono">${esc(r.industry_code) || '—'}</td>`
      + `<td class="mono">${esc(r.exchange) || '—'}</td>`
      + `<td>${tags}</td>`
      + `<td><button class="row-act" title="Re-download metadata" onclick="refreshRow('${esc(sym)}')">↻</button>`
      + `<button class="row-act" title="Add tag" onclick="openRowTagModal('${esc(sym)}')">＃</button></td>`
      + '</tr>';
  }).join('');

  document.getElementById('emptyState').style.display = _items.length ? 'none' : '';
  const count = document.getElementById('resultsCount');
  count.hidden = false;
  count.textContent = _items.length + ' securities'
    + (_items.length >= 500 ? ' — capped at 500, narrow the filter' : '');
  document.getElementById('btnExport').disabled = !_items.length;
}

function exportCsv() {
  if (!_items.length) return;
  const cols = ['symbol','ticker','cik','name','sic','sic_description','exchange',
                'entity_type','sector_code','industry_code','tags'];
  const cell = v => '"' + String(v === null || v === undefined ? '' : v).replace(/"/g, '""') + '"';
  const csv = [cols.join(',')]
    .concat(_items.map(r => cols.map(c => cell(r[c])).join(',')))
    .join('\n');

  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8;' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'sec_securities' + (F.sector ? '_' + F.sector : '') + (F.tag ? '_' + F.tag : '') + '.csv';
  a.click();
  URL.revokeObjectURL(url);
}

// ══════════════════════════════════════════════════ TAGS
async function loadTags() {
  try {
    const data = await api(BASE + '/tags');
    _tags = data.items || [];
    paintRailTags();
    document.getElementById('tagOptions').innerHTML =
      _tags.map(t => `<option value="${esc(t.tag_code)}">`).join('');
  } catch (e) { /* the tag list is not critical for the rest of the screen */ }
}

function paintRailTags() {
  const usable = _tags.filter(t => t.tag_group !== 'SECTOR' || t.qty > 0);
  document.getElementById('railTags').innerHTML = usable.map(t => {
    const on = F.tag === t.tag_code ? ' on' : '';
    return `<button class="rail-item${on}" onclick="toggleTag('${esc(t.tag_code)}')">`
         + `<span class="nm">${esc(t.tag_code)}</span>`
         + `<span class="qt">${esc(t.qty)}</span></button>`;
  }).join('') || '<div class="empty" style="padding:14px">No tags yet</div>';
}

async function removeTag(tagCode, securityId) {
  try {
    await api(BASE + '/tags/remove', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tag_code: tagCode, security_id: securityId })
    });
    toast('Tag ' + tagCode + ' removed.', 'ok');
    loadTags();
    applyFilter();
  } catch (e) { toast(e.message, 'err'); }
}

// ── CSV modal ──
function openTagModal() {
  document.getElementById('tagResult').hidden = true;
  document.getElementById('tagModal').hidden = false;
  document.getElementById('tagCode').focus();
}
function closeTagModal() { document.getElementById('tagModal').hidden = true; }

async function applyTagFromCsv() {
  const code = document.getElementById('tagCode').value.trim();
  const fileInput = document.getElementById('tagFile');
  const file = fileInput.files && fileInput.files[0];
  const out = document.getElementById('tagResult');

  if (!code) { out.hidden = false; out.textContent = 'Type the tag.'; return; }
  if (!file) { out.hidden = false; out.textContent = 'Pick a CSV file.'; return; }

  const form = new FormData();
  form.append('tag_code', code);
  form.append('tag_group', document.getElementById('tagGroup').value);
  form.append('file', file);

  const btn = document.getElementById('btnApplyTag');
  btn.disabled = true;
  try {
    const data = await api(BASE + '/tags/apply_csv', { method: 'POST', body: form });
    const r = data.result;
    let html = `<b>${esc(r.tag_code)}</b>: ${esc(r.tagged)} securities tagged out of `
             + `${esc(r.read)} read from the file.`;
    if (r.not_found && r.not_found.length) {
      html += `<br><br>No match in SEC_Securities (${r.not_found.length}): `
            + esc(r.not_found.slice(0, 60).join(', '));
    }
    out.hidden = false;
    out.innerHTML = html;
    loadTags();
  } catch (e) {
    out.hidden = false;
    out.textContent = 'Failed: ' + e.message;
  } finally { btn.disabled = false; }
}

// ── Single-row tag modal ──
function openRowTagModal(symbol) {
  _rowSymbol = symbol;
  document.getElementById('rowTagTitle').textContent = 'Tag ' + symbol;
  document.getElementById('rowTagCode').value = '';
  document.getElementById('rowTagModal').hidden = false;
  document.getElementById('rowTagCode').focus();
}
function closeRowTagModal() { document.getElementById('rowTagModal').hidden = true; }

async function applyRowTag() {
  const code = document.getElementById('rowTagCode').value.trim();
  if (!code || !_rowSymbol) { toast('Type the tag.', 'warn'); return; }

  try {
    const data = await api(BASE + '/tags/apply_symbols', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tag_code: code, symbols: [_rowSymbol] })
    });
    closeRowTagModal();
    if (data.result.tagged) {
      toast(data.result.tag_code + ' applied to ' + _rowSymbol + '.', 'ok');
    } else {
      toast(_rowSymbol + ' had no match in SEC_Securities.', 'warn');
    }
    loadTags();
    applyFilter();
  } catch (e) { toast(e.message, 'err'); }
}
