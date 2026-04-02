// argy_bonds_lecap.js
// ===========================================================
// All logic for the LECAPs & BONCAPs tab.
// Loaded after argy_bonds.js via extra_js block.
// ===========================================================

// ── State ────────────────────────────────────────────────────────
let _lcData       = [];   // full enriched list from /lecap/live
let _lcFilter     = 'ALL';
let _csvParsed    = [];   // rows parsed from CSV, pending import

// ── Boot ─────────────────────────────────────────────────────────
// Hook into the existing DOMContentLoaded flow.
// argy_bonds.js already fires loadBonds() on DOMContentLoaded,
// so we just attach our own loader here.
document.addEventListener('DOMContentLoaded', () => {
  loadLecapData();
});

// ── Override refreshAll from argy_bonds.js ───────────────────────
// Store reference to original, then extend.
const _origRefreshAll = typeof refreshAll === 'function' ? refreshAll : () => {};
// eslint-disable-next-line no-global-assign
refreshAll = async function () {
  await _origRefreshAll();
  await loadLecapData();
};

// ════════════════════════════════════════════════════
// DATA
// ════════════════════════════════════════════════════

async function loadLecapData() {
  try {
    const res = await apiFetch('/lecap/live');
    _lcData = res.data || [];
    renderLecapTable();
    updateLcCardSub();
  } catch (e) {
    document.getElementById('lecapTbody').innerHTML =
      `<tr><td colspan="12" class="ab-empty-cell" style="color:var(--red)">❌ ${escHtml(e.message)}</td></tr>`;
  }
}

function updateLcCardSub() {
  const active  = _lcData.filter(r => !r.is_expired).length;
  const expired = _lcData.filter(r =>  r.is_expired).length;
  document.getElementById('lcCardSub').textContent =
    `${active} activos · ${expired} vencidos · ARS`;
}

// ════════════════════════════════════════════════════
// RENDER
// ════════════════════════════════════════════════════

function renderLecapTable() {
  const showExpired = document.getElementById('showExpiredChk').checked;
  const tbody = document.getElementById('lecapTbody');
  tbody.innerHTML = '';

  let rows = _lcData;
  if (_lcFilter !== 'ALL')  rows = rows.filter(r => r.security_type === _lcFilter);
  if (!showExpired)          rows = rows.filter(r => !r.is_expired);

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="12" class="ab-empty-cell">Sin datos.</td></tr>';
    return;
  }

  rows.forEach(r => {
    const tr = document.createElement('tr');
    if (r.is_expired) tr.classList.add('lc-row-expired');

    const typeCls  = r.security_type.toLowerCase(); // 'lecap' | 'boncap'
    const expBadge = r.is_expired
      ? `<span class="lc-expired-badge">VENCIDO</span>` : '';

    const daysCls  = (!r.is_expired && r.days_to_maturity <= 15)
      ? 'lc-days-warn' : 'lc-days-ok';
    const daysStr  = r.is_expired
      ? `<span style="color:var(--red)">${r.days_to_maturity}d</span>`
      : `<span class="${daysCls}">${r.days_to_maturity}d</span>`;

    const noPrice  = !r.price || r.price <= 0;
    const priceStr = noPrice ? '<span class="lc-no-price">—</span>' : `$${fmt2(r.price)}`;
    const bidStr   = r.bid   ? `$${fmt2(r.bid)}`  : '—';
    const askStr   = r.ask   ? `$${fmt2(r.ask)}`  : '—';

    const tnaStr   = r.tna != null ? `<span class="lc-tna">${pct2(r.tna)}</span>` : '<span class="lc-no-price">—</span>';
    const temStr   = r.tem != null ? `<span class="lc-tem">${pct2(r.tem)}</span>` : '<span class="lc-no-price">—</span>';
    const tirStr   = r.tir != null ? `<span class="lc-tir">${pct2(r.tir)}</span>` : '<span class="lc-no-price">—</span>';

    tr.innerHTML = `
      <td>
        <span class="ab-ticker">${escHtml(r.symbol)}</span>
        ${expBadge}
      </td>
      <td><span class="lc-badge-type ${typeCls}">${escHtml(r.security_type)}</span></td>
      <td class="ab-price">${priceStr}</td>
      <td class="ab-price-dim">${bidStr}</td>
      <td class="ab-price-dim">${askStr}</td>
      <td class="ab-price">${r.final_payment ? '$' + fmt3(r.final_payment) : '—'}</td>
      <td>${daysStr}</td>
      <td class="ab-date">${escHtml(r.maturity_date || '—')}</td>
      <td>${tnaStr}</td>
      <td>${temStr}</td>
      <td>${tirStr}</td>
      <td>
        <button class="lc-del-btn"
          onclick="event.stopPropagation();confirmDeleteSecurity('${escHtml(r.symbol)}')"
          title="Delete security">🗑</button>
      </td>`;
    tbody.appendChild(tr);
  });
}

// ── Filter ────────────────────────────────────────────────────────
function setLecapFilter(filter, btn) {
  _lcFilter = filter;
  document.querySelectorAll('.lc-filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderLecapTable();
}

// ════════════════════════════════════════════════════
// ADD / EDIT SECURITY MODAL
// ════════════════════════════════════════════════════

function openAddSecurityModal(prefill = null) {
  document.getElementById('secModalTitle').textContent = prefill ? 'Edit Security' : 'Add Security';
  document.getElementById('secModalSub').textContent   = prefill ? prefill.symbol : '';
  document.getElementById('secFormError').textContent  = '';

  // Reset / prefill fields
  document.getElementById('secSymbol').value        = prefill?.symbol        || '';
  document.getElementById('secType').value          = prefill?.security_type || 'LECAP';
  document.getElementById('secDescription').value   = prefill?.description   || '';
  document.getElementById('secMaturity').value      = prefill?.maturity_date || '';
  document.getElementById('secFinalPayment').value  = prefill?.final_payment || '';
  document.getElementById('secCurrency').value      = prefill?.currency      || 'ARS';

  if (prefill) document.getElementById('secSymbol').readOnly = true;
  else         document.getElementById('secSymbol').readOnly = false;

  openModal('secModal', 'secBackdrop');
}

function closeAddSecurityModal() {
  closeModal('secModal', 'secBackdrop');
}

async function saveSecurityForm() {
  const symbol      = document.getElementById('secSymbol').value.trim().toUpperCase();
  const type        = document.getElementById('secType').value;
  const desc        = document.getElementById('secDescription').value.trim();
  const maturity    = document.getElementById('secMaturity').value;
  const finalPmt    = parseFloat(document.getElementById('secFinalPayment').value);
  const currency    = document.getElementById('secCurrency').value;
  const errEl       = document.getElementById('secFormError');

  errEl.textContent = '';

  if (!symbol)             { errEl.textContent = 'Symbol is required.';         return; }
  if (!maturity)           { errEl.textContent = 'Maturity date is required.';  return; }
  if (!finalPmt || finalPmt <= 0) { errEl.textContent = 'Final payment must be > 0.'; return; }

  const btn = document.querySelector('#secModal .lc-save-btn');
  btn.disabled = true;
  btn.textContent = 'Saving…';

  try {
    await apiFetch('/lecap/securities', {
      method: 'POST',
      body: JSON.stringify({ symbol, security_type: type, description: desc,
                             maturity_date: maturity, final_payment: finalPmt, currency }),
    });
    closeAddSecurityModal();
    showFlash('success', `✅ ${symbol} saved.`);
    await loadLecapData();
  } catch (e) {
    errEl.textContent = `Error: ${e.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Save Security';
  }
}

// ════════════════════════════════════════════════════
// DELETE SECURITY
// ════════════════════════════════════════════════════

async function confirmDeleteSecurity(symbol) {
  if (!confirm(`Delete ${symbol}? (soft-delete — recoverable from DB)`)) return;
  try {
    await apiFetch(`/lecap/securities/${encodeURIComponent(symbol)}`, { method: 'DELETE' });
    showFlash('success', `🗑 ${symbol} removed.`);
    await loadLecapData();
  } catch (e) {
    showFlash('error', `❌ ${e.message}`);
  }
}

// ════════════════════════════════════════════════════
// CSV IMPORT
// ════════════════════════════════════════════════════

// Expected CSV columns (header required):
// symbol, security_type, description, maturity_date, final_payment, currency

const CSV_COLUMNS = ['symbol', 'security_type', 'description', 'maturity_date', 'final_payment', 'currency'];

function onCsvFileSelected(event) {
  const file = event.target.files[0];
  if (!file) return;
  // Reset input so same file can be re-selected
  event.target.value = '';

  const reader = new FileReader();
  reader.onload = e => parseCsvAndPreview(e.target.result);
  reader.readAsText(file);
}

function parseCsvAndPreview(text) {
  const lines = text.split(/\r?\n/).filter(l => l.trim());
  if (lines.length < 2) {
    showFlash('error', '❌ CSV must have a header row and at least one data row.');
    return;
  }

  // Parse header
  const header = lines[0].split(',').map(h => h.trim().toLowerCase());
  const colIdx = {};
  CSV_COLUMNS.forEach(col => {
    const idx = header.indexOf(col);
    colIdx[col] = idx;  // -1 if missing
  });

  const missingCols = CSV_COLUMNS.filter(c => colIdx[c] === -1);
  if (missingCols.length) {
    showFlash('error', `❌ Missing CSV columns: ${missingCols.join(', ')}`);
    return;
  }

  // Parse rows
  _csvParsed = [];
  const tbody = document.getElementById('csvPreviewTbody');
  tbody.innerHTML = '';
  let validCount = 0;

  lines.slice(1).forEach((line, i) => {
    const cols    = splitCsvLine(line);
    const symbol  = (cols[colIdx['symbol']] || '').trim().toUpperCase();
    const type    = (cols[colIdx['security_type']] || '').trim().toUpperCase();
    const desc    = (cols[colIdx['description']] || '').trim();
    const matDate = (cols[colIdx['maturity_date']] || '').trim();
    const finalPm = parseFloat(cols[colIdx['final_payment']] || '0');
    const curr    = (cols[colIdx['currency']] || 'ARS').trim().toUpperCase();

    // Validate row
    const errors = [];
    if (!symbol)                                  errors.push('no symbol');
    if (!['LECAP','BONCAP','SOVEREIGN'].includes(type)) errors.push('invalid type');
    if (!/^\d{4}-\d{2}-\d{2}$/.test(matDate))    errors.push('bad date');
    if (isNaN(finalPm) || finalPm <= 0)           errors.push('bad final_payment');

    const isOk  = errors.length === 0;
    const rowObj = { symbol, security_type: type, description: desc,
                     maturity_date: matDate, final_payment: finalPm, currency: curr };
    if (isOk) { _csvParsed.push(rowObj); validCount++; }

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="ab-calc-td-m">${i + 1}</td>
      <td><strong>${escHtml(symbol)}</strong></td>
      <td>${escHtml(type)}</td>
      <td>${escHtml(desc)}</td>
      <td>${escHtml(matDate)}</td>
      <td>${isNaN(finalPm) ? '?' : fmt3(finalPm)}</td>
      <td>${escHtml(curr)}</td>
      <td>${isOk
        ? '<span class="lc-csv-row-ok">✓ OK</span>'
        : `<span class="lc-csv-row-err">✗ ${escHtml(errors.join(', '))}</span>`
      }</td>`;
    tbody.appendChild(tr);
  });

  document.getElementById('csvPreviewHdr').textContent =
    `Preview — ${validCount} valid rows (${lines.length - 1 - validCount} errors)`;
  document.getElementById('csvImportBtn').disabled = validCount === 0;
  document.getElementById('csvError').textContent  = '';

  openModal('csvModal', 'csvBackdrop');
}

function closeCsvModal() {
  closeModal('csvModal', 'csvBackdrop');
  _csvParsed = [];
}

async function confirmCsvImport() {
  if (!_csvParsed.length) return;

  const btn = document.getElementById('csvImportBtn');
  btn.disabled    = true;
  btn.textContent = 'Importing…';

  try {
    const res = await apiFetch('/lecap/securities/bulk', {
      method: 'POST',
      body: JSON.stringify({ securities: _csvParsed }),
    });
    closeCsvModal();
    showFlash('success', `✅ ${res.rows_affected} securities imported.`);
    await loadLecapData();
  } catch (e) {
    document.getElementById('csvError').textContent = `Error: ${e.message}`;
    btn.disabled    = false;
    btn.textContent = 'Import All';
  }
}

// ── CSV helpers ───────────────────────────────────────────────────

/**
 * Split a CSV line respecting quoted fields.
 * e.g.  'S15Y6,LECAP,"LECAP 15-May-2026",2026-05-15,105.178,ARS'
 */
function splitCsvLine(line) {
  const result = [];
  let current  = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === ',' && !inQuotes) {
      result.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  result.push(current);
  return result;
}

// ── Formatters (local; fmt2 / fmt3 / escHtml come from argy_bonds.js) ──
function pct2(v) {
  // Render a 0-1 ratio as percentage with 2 decimals, e.g. 0.2876 → "28.76%"
  return v != null ? (v * 100).toFixed(2) + '%' : '—';
}