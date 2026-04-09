// argy_bonds_ons.js
// ===========================================================
// All logic for the ONs (Obligaciones Negociables) tab.
// Loaded after argy_bonds.js and argy_bonds_lecap.js.
// ===========================================================

// ── State ────────────────────────────────────────────────────────
let _onData       = [];   // enriched list from /ons/live
let _onSector     = 'ALL';
let _onDurBucket  = 'ALL';
let _onSearch     = '';
let _onCalcSymbol = null;

// ── Boot ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadOnData();
});

// ── Extend refreshAll ────────────────────────────────────────────
// Chain onto whatever refreshAll is at this point (may have been
// extended already by argy_bonds_lecap.js).
const _origRefreshAllOns = typeof refreshAll === 'function' ? refreshAll : () => {};
// eslint-disable-next-line no-global-assign
refreshAll = async function () {
  await _origRefreshAllOns();
  await loadOnData();
};

// ════════════════════════════════════════════════════
// DATA
// ════════════════════════════════════════════════════

async function loadOnData() {
  try {
    const res = await apiFetch('/ons/live');
    _onData = res.bonds || [];
    renderOnTable();
    updateOnCardSub();
  } catch (e) {
    const tbody = document.getElementById('onTbody');
    if (tbody) tbody.innerHTML =
      `<tr><td colspan="10" class="ab-empty-cell" style="color:var(--red)">❌ ${escHtml(e.message)}</td></tr>`;
  }
}

function updateOnCardSub() {
  const withPrice = _onData.filter(b => b.price_usd > 0).length;
  const withTir   = _onData.filter(b => b.tir != null).length;
  const el = document.getElementById('onCardSub');
  if (el) el.textContent = `${withPrice} con precio · ${withTir} con TIR · USD`;
}

// ════════════════════════════════════════════════════
// RENDER
// ════════════════════════════════════════════════════

function renderOnTable() {
  const tbody = document.getElementById('onTbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  let rows = _onData;

  // Sector filter
  if (_onSector !== 'ALL')
    rows = rows.filter(b => b.sector === _onSector);

  // Duration bucket filter
  if (_onDurBucket !== 'ALL') {
    const [lo, hi] = _onDurBucket === 'SHORT' ? [0, 1]
                   : _onDurBucket === 'MID'   ? [1, 3]
                   :                            [3, 999];
    rows = rows.filter(b => b.duration != null && b.duration >= lo && b.duration < hi);
  }

  // Search filter
  if (_onSearch) {
    const q = _onSearch.toLowerCase();
    rows = rows.filter(b =>
      b.symbol.toLowerCase().includes(q) ||
      (b.issuer || '').toLowerCase().includes(q)
    );
  }

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="ab-empty-cell">Sin resultados.</td></tr>';
    return;
  }

  rows.forEach(b => {
    const tr = document.createElement('tr');

    // Price
    const noPrice  = !b.price_usd || b.price_usd <= 0;
    const priceStr = noPrice ? '<span class="on-no-price">—</span>' : `US$${fmt2(b.price_usd)}`;
    const bidStr   = b.bid > 0 ? `US$${fmt2(b.bid)}` : '—';
    const askStr   = b.ask > 0 ? `US$${fmt2(b.ask)}` : '—';

    // Duration
    const durCls = b.duration == null ? 'on-no-price'
                 : b.duration < 1    ? 'on-dur-short'
                 : b.duration < 3    ? 'on-dur-mid'
                 :                     'on-dur-long';
    const durStr = b.duration != null ? b.duration.toFixed(2) : '—';

    // TIR
    let tirCls, tirStr;
    if (b.tir == null) {
      tirCls = 'on-tir-na'; tirStr = '—';
    } else {
      tirStr = (b.tir * 100).toFixed(2) + '%';
      tirCls = b.tir < 0    ? 'on-tir-neg'
             : b.tir < 0.03 ? 'on-tir-low'
             : b.tir < 0.06 ? 'on-tir-mid'
             :                 'on-tir-high';
    }

    // % change
    const chgCls  = b.pct_change > 0 ? 'ab-chg-pos' : b.pct_change < 0 ? 'ab-chg-neg' : 'ab-chg-neu';
    const chgSign = b.pct_change > 0 ? '+' : '';

    // Sector badge CSS class
    const sectorCss = 'on-sector-' +
      (b.sector || 'other').toLowerCase().replace(/\s+/g, '-').replace(/[^a-z-]/g, '');

    tr.innerHTML = `
      <td>
        <span class="ab-ticker">${escHtml(b.symbol)}</span>
        <span class="on-sector-badge ${sectorCss}">${escHtml(b.sector || '—')}</span>
      </td>
      <td class="on-issuer">${escHtml(b.issuer || '—')}</td>
      <td class="ab-price">${priceStr}</td>
      <td class="ab-price-dim">${bidStr}</td>
      <td class="ab-price-dim">${askStr}</td>
      <td class="${durCls}">${durStr}</td>
      <td class="ab-date">${escHtml(b.maturity || '—')}</td>
      <td class="${tirCls}">${tirStr}</td>
      <td class="${chgCls}">${chgSign}${fmt2(b.pct_change)}%</td>
      <td style="display:flex;gap:4px">
        <button class="ab-chart-btn"
          onclick="event.stopPropagation();openOnChartModal('${escHtml(b.symbol)}')"
          title="Ver gráfico">📈</button>
        <button class="ab-calc-btn"
          onclick="event.stopPropagation();openOnCalcModal('${escHtml(b.symbol)}')"
          title="Calculadora">🧮</button>
      </td>`;

    tr.style.cursor = 'pointer';
    tr.onclick = () => openOnChartModal(b.symbol);
    tbody.appendChild(tr);
  });
}

// ── Filters ───────────────────────────────────────────────────────

function setOnSectorFilter(sector, btn) {
  _onSector = sector;
  document.querySelectorAll('.on-filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderOnTable();
}

function setOnDurFilter(bucket, btn) {
  _onDurBucket = bucket;
  document.querySelectorAll('.on-dur-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderOnTable();
}

function onOnSearch(val) {
  _onSearch = val.trim();
  renderOnTable();
}

// ════════════════════════════════════════════════════
// CHART MODAL — reuses the existing chart infrastructure
// from argy_bonds.js (openModal / renderChart / etc.)
// ════════════════════════════════════════════════════

async function openOnChartModal(symbol) {
  const bond = _onData.find(b => b.symbol === symbol) || {};

  _currentSymbol = symbol;
  _adjEnabled    = false;
  document.getElementById('adjToggle').checked = false;

  document.getElementById('chartModalTitle').textContent = symbol;
  document.getElementById('chartModalSub').textContent   =
    `${bond.issuer || ''}  ·  Vto: ${bond.maturity || '?'}  ·  ${bond.sector || ''}`;

  document.getElementById('chartStats').innerHTML   = _buildOnStatsHtml(bond);
  document.getElementById('couponList').innerHTML   = '';
  document.getElementById('couponSub').textContent  =
    'Cupones: cargá flujos en dbo.bond_coupons para visualizarlos';

  document.getElementById('chartContainer').innerHTML =
    '<div class="ab-chart-loading">⏳ Descargando barras desde TradingView…</div>';

  openModal('chartModal', 'chartBackdrop');

  try {
    const url  = `/ons/ohlcv?symbol=${encodeURIComponent(symbol)}&exchange=BYMA`;
    const data = await apiFetch(url);
    if (!data.ok) throw new Error(data.error || 'Sin datos');
    renderChart(data.bars || []);
  } catch (e) {
    document.getElementById('chartContainer').innerHTML =
      `<div class="ab-chart-error">❌ ${e.message}</div>`;
  }
}

function _buildOnStatsHtml(bond) {
  const chgCls  = (bond.pct_change || 0) >= 0 ? 'pos' : 'neg';
  const chgSign = (bond.pct_change || 0) >= 0 ? '+' : '';
  const tirStr  = bond.tir      != null ? (bond.tir * 100).toFixed(2) + '%'  : '—';
  const durStr  = bond.duration != null ? bond.duration.toFixed(2) + ' años' : '—';
  return `
    <div class="ab-stat"><span class="ab-stat-lbl">Precio</span><span class="ab-stat-val">US$${fmt2(bond.price_usd)}</span></div>
    <div class="ab-stat"><span class="ab-stat-lbl">Bid</span><span class="ab-stat-val">${bond.bid > 0 ? 'US$' + fmt2(bond.bid) : '—'}</span></div>
    <div class="ab-stat"><span class="ab-stat-lbl">Ask</span><span class="ab-stat-val">${bond.ask > 0 ? 'US$' + fmt2(bond.ask) : '—'}</span></div>
    <div class="ab-stat"><span class="ab-stat-lbl">% Chg</span><span class="ab-stat-val ${chgCls}">${chgSign}${fmt2(bond.pct_change)}%</span></div>
    <div class="ab-stat"><span class="ab-stat-lbl">TIR</span><span class="ab-stat-val">${tirStr}</span></div>
    <div class="ab-stat"><span class="ab-stat-lbl">Duration</span><span class="ab-stat-val">${durStr}</span></div>
    <div class="ab-stat"><span class="ab-stat-lbl">Emisor</span><span class="ab-stat-val">${escHtml(bond.issuer || '—')}</span></div>
    <div class="ab-stat"><span class="ab-stat-lbl">Vencimiento</span><span class="ab-stat-val">${bond.maturity || '—'}</span></div>`;
}

// ════════════════════════════════════════════════════
// CALCULATOR MODAL
// ════════════════════════════════════════════════════

function openOnCalcModal(symbol) {
  const bond = _onData.find(b => b.symbol === symbol);
  if (!bond) return;
  _onCalcSymbol = symbol;

  document.getElementById('onCalcTitle').textContent = symbol + ' — Calculadora';
  document.getElementById('onCalcSub').textContent   =
    `${bond.issuer}  ·  Vto: ${bond.maturity}`;

  document.getElementById('onCalcPrice').value     = bond.price_usd > 0 ? fmt2(bond.price_usd) : '';
  document.getElementById('onCalcMonto').value     = '10000';
  document.getElementById('onCalcArancel').value   = '0.45';
  document.getElementById('onCalcImpuestos').value = '0.01';

  // Reset outputs
  ['onCalcTir','onCalcDur','onCalcVn','onCalcGain'].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.textContent = '—'; el.style.color = ''; }
  });
  document.getElementById('onCalcFlows').innerHTML  =
    '<tr><td colspan="3" class="ab-calc-empty">Ingresá precio y monto</td></tr>';
  document.getElementById('onCalcTfoot').innerHTML  = '';
  document.getElementById('onCalcNote').textContent = '';

  openModal('onCalcModal', 'onCalcBackdrop');
  recalcOnModal();
}

function closeOnCalcModal() {
  closeModal('onCalcModal', 'onCalcBackdrop');
  _onCalcSymbol = null;
}

async function recalcOnModal() {
  if (!_onCalcSymbol) return;

  const price    = parseFloat(document.getElementById('onCalcPrice').value);
  const monto    = parseFloat(document.getElementById('onCalcMonto').value)    || 0;
  const arancel  = parseFloat(document.getElementById('onCalcArancel').value)  || 0;
  const impuesto = parseFloat(document.getElementById('onCalcImpuestos').value)|| 0;

  if (!price || price <= 0 || !monto) {
    ['onCalcTir','onCalcDur','onCalcVn','onCalcGain'].forEach(id => {
      const el = document.getElementById(id);
      if (el) { el.textContent = '—'; el.style.color = ''; }
    });
    document.getElementById('onCalcFlows').innerHTML =
      '<tr><td colspan="3" class="ab-calc-empty">Ingresá precio y monto</td></tr>';
    document.getElementById('onCalcTfoot').innerHTML  = '';
    document.getElementById('onCalcNote').textContent = '';
    return;
  }

  try {
    const res = await apiFetch('/ons/calc', {
      method: 'POST',
      body: JSON.stringify({
        symbol: _onCalcSymbol, price, monto, arancel, impuesto,
      }),
    });

    document.getElementById('onCalcTir').textContent =
      res.tir != null ? (res.tir * 100).toFixed(2) + '%' : '—';
    document.getElementById('onCalcDur').textContent =
      res.duration != null ? res.duration.toFixed(2) + ' años' : '—';
    document.getElementById('onCalcVn').textContent =
      res.vn_bought.toLocaleString('es-AR') + ' VN';

    const gainColor = res.ganancia >= 0 ? 'var(--green)' : 'var(--red)';
    const gainSign  = res.ganancia >= 0 ? '+' : '';
    const gainEl    = document.getElementById('onCalcGain');
    gainEl.textContent = gainSign + 'US$' + fmt2(res.ganancia);
    gainEl.style.color = gainColor;

    // Cash flow rows
    const tbody = document.getElementById('onCalcFlows');
    if (!res.flows || !res.flows.length) {
      tbody.innerHTML =
        '<tr><td colspan="3" class="ab-calc-empty">Sin flujos futuros en BD · cargalos vía dbo.bond_coupons</td></tr>';
    } else {
      tbody.innerHTML = res.flows.map(f => `
        <tr>
          <td class="ab-calc-td-m">${fmtDate(f.date)}</td>
          <td class="ab-calc-td-r">US$${fmt3(f.per100)}</td>
          <td class="ab-calc-td-r">US$${fmt2(f.cobro)}</td>
        </tr>`).join('');
    }

    const cellBase = 'padding:8px 12px;font-family:var(--mono);font-size:12px;border-top:1px solid rgba(22,27,34,.7)';
    document.getElementById('onCalcTfoot').innerHTML = `
      <tr>
        <td style="${cellBase};color:var(--dim)">Total cobros</td>
        <td></td>
        <td style="${cellBase};text-align:right;color:#E6EDF3;font-weight:600">US$${fmt2(res.total_cobro)}</td>
      </tr>
      <tr style="border-top:2px solid var(--border)">
        <td style="${cellBase};color:${gainColor};font-size:13px;font-weight:700">Ganancia</td>
        <td></td>
        <td style="${cellBase};text-align:right;color:${gainColor};font-size:15px;font-weight:700">${gainSign}US$${fmt2(res.ganancia)}</td>
      </tr>`;

    document.getElementById('onCalcNote').textContent =
      `Precio efectivo c/costos: US$${fmt2(res.effective_price)}/100 VN  ·  ` +
      `Comprás ${res.vn_bought.toLocaleString('es-AR')} VN`;

  } catch (e) {
    showFlash('error', `❌ ${e.message}`);
  }
}