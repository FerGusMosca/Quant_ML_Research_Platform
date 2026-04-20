// argy_bonds_cer.js
// ===========================================================
// CER-indexed sovereign bonds tab.
// Loaded after argy_bonds.js / argy_bonds_lecap.js / argy_bonds_ons.js.
// Stage 1: table + filters. No calculator, no chart modal.
// ===========================================================

// ── State ────────────────────────────────────────────────────────
let _cerBonds       = [];             // enriched list from /cer_bonds/live
let _cerTypeFilter  = 'ALL';          // 'ALL' | 'ZERO' | 'CUPON'
let _cerSortKey     = 'duration';     // 'ticker' | 'price' | 'duration' | 'maturity' | 'tir'
let _cerSortAsc     = true;
let _cerUi          = { latestCer: null, laggedCer: null };

// ── Boot ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadCerBonds();
});

// Chain into refreshAll (may already be extended by lecap/ons)
const _origRefreshAllCer = typeof refreshAll === 'function' ? refreshAll : () => {};
// eslint-disable-next-line no-global-assign
refreshAll = async function () {
  await _origRefreshAllCer();
  await loadCerBonds();
};

// ════════════════════════════════════════════════════
// DATA
// ════════════════════════════════════════════════════

async function loadCerBonds() {
  try {
    const data  = await apiFetch('/cer_bonds/live');
    _cerBonds   = data.bonds || [];
    _cerUi.latestCer = data.cer?.latest || null;
    _cerUi.laggedCer = data.cer?.lagged || null;
    renderCerTable();
    updateCerCardSub();
  } catch (e) {
    const tbody = document.getElementById('cerTbody');
    if (tbody) tbody.innerHTML =
      `<tr><td colspan="6" class="ab-empty-cell" style="color:var(--red)">❌ ${escHtml(e.message)}</td></tr>`;
  }
}

function updateCerCardSub() {
  const el = document.getElementById('cerCardSub');
  if (!el) return;
  const withTir = _cerBonds.filter(b => b.tir_real != null).length;
  const cerLatest = _cerUi.latestCer;
  const cerLagged = _cerUi.laggedCer;
  const cerStr = cerLatest && cerLagged
    ? `CER ${fmt2(cerLatest.valor)} (${escHtml(cerLatest.fecha)}) · T-10: ${fmt2(cerLagged.valor)} (${escHtml(cerLagged.fecha)})`
    : 'CER no disponible';
  el.textContent = `${_cerBonds.length} bonos · ${withTir} con TIR Real · ${cerStr}`;
}

// ════════════════════════════════════════════════════
// CLASSIFICATION — zero-coupon vs con cupones
// ════════════════════════════════════════════════════
// A CER bond is classified as zero-coupon when all of its principal amortizes
// on a single date AND it has no intermediate interest payments. Anything else
// is "con cupones" (semi-annual coupon bond like TX26/TX28/TX31/DICP/PARP).

function _classifyCerBond(b) {
  const ticker = b.symbol || '';
  // Simple heuristic consistent with the JSON: LeCer (Xnnn) + TZX* + TZXO/TZXD/TZXM
  // are zero coupon. TX*/DICP/PARP have coupons.
  if (/^(X\d|TZX)/i.test(ticker)) return 'ZERO';
  return 'CUPON';
}

// ════════════════════════════════════════════════════
// SORT / FILTER
// ════════════════════════════════════════════════════

function _cerSortedRows() {
  let rows = _cerBonds.slice();
  if (_cerTypeFilter !== 'ALL') {
    rows = rows.filter(b => _classifyCerBond(b) === _cerTypeFilter);
  }
  const key = _cerSortKey;
  const dir = _cerSortAsc ? 1 : -1;
  rows.sort((a, b) => {
    const va = _cerSortVal(a, key);
    const vb = _cerSortVal(b, key);
    if (va == null && vb == null) return 0;
    if (va == null) return 1;   // nulls last
    if (vb == null) return -1;
    if (typeof va === 'string') return va.localeCompare(vb) * dir;
    return (va - vb) * dir;
  });
  return rows;
}

function _cerSortVal(b, key) {
  if (key === 'ticker')   return b.symbol || '';
  if (key === 'price')    return b.price_ars;
  if (key === 'duration') return b.duration;
  if (key === 'maturity') return b.maturity || '';
  if (key === 'tir')      return b.tir_real;
  return null;
}

function setCerTypeFilter(type, btn) {
  _cerTypeFilter = type;
  document.querySelectorAll('.cer-filter-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  renderCerTable();
}

function setCerSort(key) {
  if (_cerSortKey === key) {
    _cerSortAsc = !_cerSortAsc;
  } else {
    _cerSortKey = key;
    _cerSortAsc = (key === 'duration' || key === 'maturity' || key === 'ticker' || key === 'price');
  }
  renderCerTable();
}

// ════════════════════════════════════════════════════
// RENDER
// ════════════════════════════════════════════════════

function renderCerTable() {
  const tbody = document.getElementById('cerTbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  const rows = _cerSortedRows();

  // Update sort arrows in header
  document.querySelectorAll('#cerTable thead th.sortable').forEach(th => {
    const key = th.getAttribute('data-sort-key');
    const arrow = th.querySelector('.cer-sort-arrow');
    if (!arrow) return;
    if (key === _cerSortKey) {
      arrow.textContent = _cerSortAsc ? '↑' : '↓';
      th.classList.add('sort-active');
    } else {
      arrow.textContent = '↕';
      th.classList.remove('sort-active');
    }
  });

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="ab-empty-cell">Sin resultados.</td></tr>';
    return;
  }

  rows.forEach(b => {
    const cls   = _classifyCerBond(b);
    const typeLabel = cls === 'ZERO' ? 'Zero' : 'Cupón';
    const typeCss   = cls === 'ZERO' ? 'cer-badge-zero' : 'cer-badge-coupon';

    const priceStr = b.price_ars > 0 ? '$' + fmt2(b.price_ars) : '—';
    const durStr   = b.duration != null ? b.duration.toFixed(1) : '—';
    const tirCls   = b.tir_real == null ? 'cer-tir-na'
                   : b.tir_real < 0     ? 'cer-tir-neg'
                   :                       'cer-tir-pos';
    const tirStr   = b.tir_real != null ? (b.tir_real * 100).toFixed(2) + '%' : '—';

    const chgCls   = b.pct_change > 0 ? 'ab-chg-pos' : b.pct_change < 0 ? 'ab-chg-neg' : 'ab-chg-neu';
    const chgSign  = b.pct_change > 0 ? '+' : '';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>
        <span class="ab-ticker">${escHtml(b.symbol)}</span>
        <span class="cer-type-badge ${typeCss}">${typeLabel}</span>
      </td>
      <td class="ab-price">${priceStr}</td>
      <td class="ab-dur">${durStr}</td>
      <td class="ab-date">${escHtml(b.maturity || '—')}</td>
      <td class="${tirCls}">${tirStr}</td>
      <td class="${chgCls}">${chgSign}${fmt2(b.pct_change)}%</td>`;
    tbody.appendChild(tr);
  });
}
