// argy_bonds.js

// ── Clock ─────────────────────────────────────────────────────────────────
(function tick() {
  const el = document.getElementById('navClock');
  if (el) {
    const n = new Date(), p = v => String(v).padStart(2, '0');
    el.textContent = [n.getHours(), n.getMinutes(), n.getSeconds()].map(p).join(':');
  }
  setTimeout(tick, 1000);
})();

// ── Bond metadata — loaded from /static/config/bonds_config.json ──────────
// Flows are exact values from official prospectuses.
// BOND_META is populated at boot via loadBondConfig().

let BOND_META = {};   // populated by loadBondConfig()

async function loadBondConfig() {
  try {
    const res = await fetch('/static/config/bonds_config.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const cfg = await res.json();

    // Future cash flows (for TIR / Duration / calculator)
    const sob = cfg.soberanos || {};
    for (const [symbol, bond] of Object.entries(sob)) {
      BOND_META[symbol] = {
        law:      bond.ley === 'NY' ? 'NY' : 'Local',
        maturity: bond.vencimiento,
        coupons:  bond.flujos.map(f => ({
          date:             f.fecha,
          amount_per_100vn: f.monto,
        })),
      };
    }
  } catch(e) {
    console.error('[ArgyBonds] Failed to load bonds_config.json:', e);
  }
}

// ── State ─────────────────────────────────────────────────────────────────
let _bonds         = [];   // live prices from API
let _enriched      = [];   // merged with meta
let _currentSymbol = null;
let _currentBars   = [];   // raw bars (last fetch — kept for re-render on toggle)
let _lwChart       = null;
let _adjEnabled    = false;

// ── Boot ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await loadBondConfig();
  loadBonds();
});

// ── Tab switching ─────────────────────────────────────────────────────────
function switchTab(tab, btn) {
  document.querySelectorAll('.ab-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.ab-tab-content').forEach(el => el.style.display = 'none');
  const content = document.getElementById('tab-' + tab);
  if (content) content.style.display = '';
}

// ════════════════════════════════════════════════════
// DATA — Bond prices + enrichment
// ════════════════════════════════════════════════════

async function loadBonds() {
  try {
    const data = await apiFetch('/argy_bonds/live');
    _bonds    = data.bonds || [];
    _enriched = _bonds.map(b => enrichBond(b));
    renderBondTable(_enriched);
    document.getElementById('lastUpdate').textContent =
      'Última actualización: ' + new Date().toLocaleTimeString('es-AR');
  } catch(e) {
    document.getElementById('bondTbody').innerHTML =
      `<tr><td colspan="11" class="ab-empty-cell" style="color:var(--red)">❌ ${e.message}</td></tr>`;
  }
}

function enrichBond(b) {
  const meta  = BOND_META[b.symbol] || {};
  const price = b.price_usd;
  const { tir, duration } = calcTirDuration(b.symbol, price, meta.coupons || []);
  return { ...b, ...meta, tir, duration };
}

async function refreshAll() {
  document.getElementById('lastUpdate').textContent = 'Actualizando…';
  await loadBonds();
}

// ── YTM / Duration calc (Newton-Raphson) ─────────────────────────────────
function calcTirDuration(symbol, price, coupons) {
  if (!price || !coupons || !coupons.length) return { tir: null, duration: null };
  const today  = new Date();
  const future = coupons
    .filter(c => new Date(c.date) > today)
    .map(c => ({
      t:  (new Date(c.date) - today) / (365.25 * 24 * 3600 * 1000),
      cf: c.amount_per_100vn,
    }));
  if (!future.length) return { tir: null, duration: null };

  let ytm = 0.09;
  for (let i = 0; i < 100; i++) {
    let pv = 0, dpv = 0;
    for (const { t, cf } of future) {
      const d  = Math.pow(1 + ytm, t);
      pv  += cf / d;
      dpv -= t * cf / (d * (1 + ytm));
    }
    const f = pv - price;
    if (Math.abs(f) < 1e-8) break;
    ytm -= f / dpv;
  }

  let weightedT = 0, sumPv = 0;
  for (const { t, cf } of future) {
    const pv = cf / Math.pow(1 + ytm, t);
    weightedT += t * pv;
    sumPv     += pv;
  }
  const duration = sumPv > 0 ? weightedT / sumPv : null;

  return {
    tir:      isFinite(ytm)      ? ytm      : null,
    duration: isFinite(duration) ? duration : null,
  };
}

// ════════════════════════════════════════════════════
// RENDER — Bond table
// ════════════════════════════════════════════════════

function renderBondTable(bonds) {
  const tbody = document.getElementById('bondTbody');
  tbody.innerHTML = '';
  if (!bonds.length) {
    tbody.innerHTML = '<tr><td colspan="11" class="ab-empty-cell">Sin datos.</td></tr>';
    return;
  }
  bonds.forEach(b => {
    const tr       = document.createElement('tr');
    const chgCls   = b.pct_change > 0 ? 'ab-chg-pos' : b.pct_change < 0 ? 'ab-chg-neg' : 'ab-chg-neu';
    const chgSign  = b.pct_change > 0 ? '+' : '';
    const lawCls   = (b.law || '').toLowerCase() === 'ny' ? 'ny' : 'local';
    const lawLabel = lawCls === 'ny' ? 'NY' : 'Local';
    const tirStr   = b.tir      != null ? (b.tir * 100).toFixed(2) + '%' : '—';
    const durStr   = b.duration != null ? b.duration.toFixed(1)          : '—';
    tr.innerHTML = `
      <td>
        <span class="ab-ticker">${escHtml(b.symbol)}</span>
        <span class="ab-badge-law ${lawCls}">${lawLabel}</span>
      </td>
      <td class="ab-dur">${escHtml(b.law || '—')}</td>
      <td class="ab-price">US$${fmt2(b.price_usd)}</td>
      <td class="ab-price-dim">${b.bid ? 'US$' + fmt2(b.bid) : '—'}</td>
      <td class="ab-price-dim">${b.ask ? 'US$' + fmt2(b.ask) : '—'}</td>
      <td class="ab-tir ${b.tir != null && b.tir > 0.10 ? 'neg' : 'pos'}">${tirStr}</td>
      <td class="ab-dur">${durStr}</td>
      <td class="ab-date">${escHtml(b.maturity || '—')}</td>
      <td class="${chgCls}">${chgSign}${fmt2(b.pct_change)}%</td>
      <td class="ab-vol">${fmtVol(b.volume)}</td>
      <td>
        <button class="ab-chart-btn"
          onclick="event.stopPropagation();openChartModal('${escHtml(b.symbol)}')"
          title="Ver gráfico">📈</button>
        <button class="ab-calc-btn"
          onclick="event.stopPropagation();openCalcModal('${escHtml(b.symbol)}')"
          title="Calculadora">🧮</button>
      </td>`;
    tr.onclick = () => openChartModal(b.symbol);
    tbody.appendChild(tr);
  });
}

// ════════════════════════════════════════════════════
// CHART MODAL
// ════════════════════════════════════════════════════

async function openChartModal(symbol) {
  _currentSymbol = symbol;
  _adjEnabled    = document.getElementById('adjToggle').checked;
  const bond = _enriched.find(b => b.symbol === symbol) || {};

  document.getElementById('chartModalTitle').textContent = symbol + 'D';
  document.getElementById('chartModalSub').textContent   =
    `${bond.law || ''}  ·  Vto: ${bond.maturity || '?'}`;

  document.getElementById('chartStats').innerHTML = buildStatsHtml(bond);
  loadCouponList(symbol);

  document.getElementById('chartContainer').innerHTML =
    '<div class="ab-chart-loading">⏳ Descargando barras diarias desde TradingView…</div>';

  openModal('chartModal', 'chartBackdrop');
  await fetchAndRenderChart(symbol, _adjEnabled);
}

async function fetchAndRenderChart(symbol, adjusted) {
  try {
    const url  = `/argy_bonds/ohlcv?symbol=${encodeURIComponent(symbol + 'D')}`
               + `&exchange=BYMA&adjusted=${adjusted}`;
    const data = await apiFetch(url);
    if (!data.ok) throw new Error(data.error || 'Sin datos');
    // Keep raw bars for re-fetching on toggle (we always re-fetch — backend is fast)
    _currentBars = data.bars || [];
    renderChart(_currentBars);
  } catch(e) {
    document.getElementById('chartContainer').innerHTML =
      `<div class="ab-chart-error">❌ ${e.message}</div>`;
  }
}

function closeChartModal() {
  closeModal('chartModal', 'chartBackdrop');
  _currentBars   = [];
  _currentSymbol = null;
  if (_lwChart) { _lwChart = null; }
  document.getElementById('chartContainer').innerHTML = '';
}

// Re-fetch when user toggles adjusted price — backend applies the math
async function onAdjToggle() {
  _adjEnabled = document.getElementById('adjToggle').checked;
  if (!_currentSymbol) return;
  document.getElementById('chartContainer').innerHTML =
    '<div class="ab-chart-loading">⏳ Aplicando ajuste…</div>';
  await fetchAndRenderChart(_currentSymbol, _adjEnabled);
}

// ── Core chart renderer (receives already-adjusted bars from backend) ─────
function renderChart(bars) {
  const container = document.getElementById('chartContainer');
  container.innerHTML = '';

  _lwChart = LightweightCharts.createChart(container, {
    width:  container.clientWidth,
    height: container.clientHeight || 480,
    layout: {
      background: { color: '#0d1117' },
      textColor:  '#8b949e',
    },
    grid: {
      vertLines: { color: '#1c2330' },
      horzLines: { color: '#1c2330' },
    },
    crosshair:       { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#2d333b' },
    timeScale: {
      borderColor:    '#2d333b',
      timeVisible:    false,
      secondsVisible: false,
    },
  });

  const candleSeries = _lwChart.addCandlestickSeries({
    upColor:         '#3fb950',
    downColor:       '#f85149',
    borderUpColor:   '#3fb950',
    borderDownColor: '#f85149',
    wickUpColor:     '#3fb950',
    wickDownColor:   '#f85149',
  });
  candleSeries.setData(bars);

  // Coupon markers — overlay on chart using date from BOND_META
  const meta   = BOND_META[_currentSymbol] || {};
  const paid   = getPaidCouponsInRange(meta.coupons || [], bars);
  if (paid.length) {
    const markers = paid
      .map(c => ({
        time:     dateToTimestamp(c.date),
        position: 'aboveBar',
        color:    '#e3b341',
        shape:    'circle',
        text:     '¢' + c.amount_per_100vn.toFixed(3),
        size:     1,
      }))
      .filter(m => m.time)
      .sort((a, b) => a.time - b.time);
    if (markers.length) candleSeries.setMarkers(markers);
  }

  new ResizeObserver(() => {
    if (_lwChart) _lwChart.resize(container.clientWidth, container.clientHeight);
  }).observe(container);
}

// ── Paid coupons within visible bar range ────────────────────────────────
function getPaidCouponsInRange(coupons, bars) {
  if (!bars.length || !coupons.length) return [];
  const today  = new Date().toISOString().slice(0, 10);
  const minTs  = bars[0].time;
  const maxTs  = bars[bars.length - 1].time;
  return coupons.filter(c => {
    if (c.date > today) return false;     // only paid coupons
    const ts = dateToTimestamp(c.date);
    return ts >= minTs && ts <= maxTs;
  });
}

// ── Date helpers ──────────────────────────────────────────────────────────
function dateToTimestamp(dateStr) {
  if (!dateStr) return null;
  return Math.floor(new Date(dateStr + 'T12:00:00Z').getTime() / 1000);
}

function localDateStr(date) {
  const y   = date.getFullYear();
  const m   = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

// ── Stats bar HTML ────────────────────────────────────────────────────────
function buildStatsHtml(bond) {
  const chgCls  = (bond.pct_change || 0) >= 0 ? 'pos' : 'neg';
  const chgSign = (bond.pct_change || 0) >= 0 ? '+' : '';
  const tirStr  = bond.tir      != null ? (bond.tir * 100).toFixed(2) + '%'  : '—';
  const durStr  = bond.duration != null ? bond.duration.toFixed(2) + ' años' : '—';
  return `
    <div class="ab-stat"><span class="ab-stat-lbl">Precio</span><span class="ab-stat-val">US$${fmt2(bond.price_usd)}</span></div>
    <div class="ab-stat"><span class="ab-stat-lbl">Bid</span><span class="ab-stat-val">${bond.bid ? 'US$' + fmt2(bond.bid) : '—'}</span></div>
    <div class="ab-stat"><span class="ab-stat-lbl">Ask</span><span class="ab-stat-val">${bond.ask ? 'US$' + fmt2(bond.ask) : '—'}</span></div>
    <div class="ab-stat"><span class="ab-stat-lbl">% Chg</span><span class="ab-stat-val ${chgCls}">${chgSign}${fmt2(bond.pct_change)}%</span></div>
    <div class="ab-stat"><span class="ab-stat-lbl">TIR (YTM)</span><span class="ab-stat-val">${tirStr}</span></div>
    <div class="ab-stat"><span class="ab-stat-lbl">Duration</span><span class="ab-stat-val">${durStr}</span></div>
    <div class="ab-stat"><span class="ab-stat-lbl">Ley</span><span class="ab-stat-val">${bond.law || '—'}</span></div>
    <div class="ab-stat"><span class="ab-stat-lbl">Vencimiento</span><span class="ab-stat-val">${bond.maturity || '—'}</span></div>
  `;
}

// ── Coupon schedule below the chart ─────────────────────────────────────────
//
// Read from dbo.bond_coupons, NOT from bonds_config.json. The price adjustment
// reads is_paid from that table, so showing anything else here would let the
// screen claim a coupon was paid while the chart still carries its step.

let _coupons = [];

async function loadCouponList(symbol) {
  const sub  = document.getElementById('couponSub');
  const list = document.getElementById('couponList');
  const warn = document.getElementById('couponWarn');

  sub.textContent = 'Cargando…';
  list.innerHTML  = '';
  warn.hidden     = true;

  try {
    const res  = await fetch(`/argy_bonds/coupons?symbol=${encodeURIComponent(symbol)}`);
    const data = await res.json();

    if (!data.ok) {
      sub.textContent = '';
      warn.hidden = false;
      warn.textContent = data.error || 'No se pudo leer el cronograma de cupones.';
      return;
    }

    _coupons = data.coupons || [];
    renderCouponList(symbol, data);

  } catch (e) {
    sub.textContent = '';
    warn.hidden = false;
    warn.textContent = `No se pudo consultar el servidor: ${e.message}`;
  }
}

function renderCouponList(symbol, data) {
  const sub  = document.getElementById('couponSub');
  const list = document.getElementById('couponList');
  const warn = document.getElementById('couponWarn');

  const coupons = data.coupons || [];
  const paid    = coupons.filter(c => c.is_paid);
  const overdue = coupons.filter(c => c.overdue);

  sub.textContent = paid.length
    ? `${paid.length} pagados · total ${fmt2(paid.reduce((s, c) => s + c.amount, 0))} por 100 VN`
    : 'Sin cupones marcados como pagados';

  // The whole point of the panel: say out loud which payment is missing
  if (overdue.length) {
    warn.hidden = false;
    warn.textContent =
      `${overdue.length} cupón${overdue.length > 1 ? 'es' : ''} con fecha vencida sin marcar `
      + `(${overdue.map(c => c.payment_date).join(', ')}). `
      + 'Mientras siga así, el gráfico ajustado mantiene el escalón.';
  } else {
    warn.hidden = true;
  }

  list.innerHTML = '';

  coupons.forEach(c => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'ab-coupon-chip'
      + (c.is_paid ? ' is-paid' : '')
      + (c.overdue ? ' is-overdue' : '');
    chip.title = c.is_paid ? 'Click para desmarcar' : 'Click para marcar como pagado';
    chip.onclick = () => toggleCoupon(symbol, c.payment_date, !c.is_paid);

    chip.innerHTML = `
      <span class="ab-cc-mark">${c.is_paid ? '✓' : ''}</span>
      <span class="ab-cc-date">${c.payment_date}</span>
      <span class="ab-cc-amt">${fmt3(c.amount)}</span>`;

    list.appendChild(chip);
  });
}

async function toggleCoupon(symbol, paymentDate, isPaid) {
  const warn = document.getElementById('couponWarn');

  try {
    const res = await fetch('/argy_bonds/coupons/set_paid', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ symbol, payment_date: paymentDate, is_paid: isPaid }),
    });
    const data = await res.json();

    if (!data.ok) {
      warn.hidden = false;
      warn.textContent = data.error || 'No se pudo actualizar el cupón.';
      return;
    }

    await loadCouponList(symbol);

    // The adjusted series changes the moment a coupon is flagged, so the chart
    // is redrawn instead of leaving a stale picture next to the new state.
    if (document.getElementById('adjToggle')?.checked) onAdjToggle();

  } catch (e) {
    warn.hidden = false;
    warn.textContent = `No se pudo consultar el servidor: ${e.message}`;
  }
}

async function markDueCoupons() {
  const btn  = document.getElementById('markDueBtn');
  const warn = document.getElementById('couponWarn');

  if (!_currentSymbol) return;

  btn.disabled = true;
  btn.textContent = 'Marcando…';

  try {
    const res = await fetch('/argy_bonds/coupons/mark_due', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ symbol: _currentSymbol }),
    });
    const data = await res.json();

    if (!data.ok) {
      warn.hidden = false;
      warn.textContent = data.error || 'No se pudieron marcar los cupones vencidos.';
      return;
    }

    await loadCouponList(_currentSymbol);
    if (document.getElementById('adjToggle')?.checked) onAdjToggle();

  } catch (e) {
    warn.hidden = false;
    warn.textContent = `No se pudo consultar el servidor: ${e.message}`;

  } finally {
    btn.disabled = false;
    btn.textContent = 'Marcar vencidos como pagados';
  }
}

// ════════════════════════════════════════════════════

function openModal(id, bid) {
  document.getElementById(bid).classList.add('open');
  document.getElementById(id).classList.add('open');
}
function closeModal(id, bid) {
  document.getElementById(bid).classList.remove('open');
  document.getElementById(id).classList.remove('open');
}
function escHtml(s) {
  return String(s || '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function fmt2(v)  { return v != null ? Number(v).toFixed(2) : '—'; }
function fmt3(v)  { return v != null ? Number(v).toFixed(3) : '—'; }
function fmtVol(v) {
  if (!v) return '—';
  if (v >= 1_000_000) return (v / 1_000_000).toFixed(1) + 'M';
  if (v >= 1_000)     return (v / 1_000).toFixed(0) + 'K';
  return String(v);
}
async function apiFetch(url, opts = {}) {
  const r = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
function showFlash(type, msg) {
  document.querySelector('.ab-flash')?.remove();
  const el = Object.assign(document.createElement('div'), {
    className: `ab-flash ${type}`, textContent: msg,
  });
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

// ════════════════════════════════════════════════════
// CALCULATOR MODAL
// ════════════════════════════════════════════════════

let _calcSymbol = null;

function openCalcModal(symbol) {
  _calcSymbol = symbol;
  const bond  = _enriched.find(b => b.symbol === symbol) || {};

  document.getElementById('calcModalTitle').textContent = symbol + ' — Calculadora';
  document.getElementById('calcModalSub').textContent   =
    `${bond.law || ''}  ·  Vencimiento: ${bond.maturity || '?'}`;

  const price = bond.price_usd || '';
  document.getElementById('calcPrice').value = price ? Number(price).toFixed(2) : '';

  openModal('calcModal', 'calcBackdrop');
  recalc();
}

function closeCalcModal() {
  closeModal('calcModal', 'calcBackdrop');
  _calcSymbol = null;
}

function recalc() {
  if (!_calcSymbol) return;

  const meta     = BOND_META[_calcSymbol] || {};
  const coupons  = meta.coupons || [];
  const price    = parseFloat(document.getElementById('calcPrice').value);
  const monto    = parseFloat(document.getElementById('calcMonto').value)    || 0;
  const arancel  = parseFloat(document.getElementById('calcArancel').value)  || 0;
  const impuesto = parseFloat(document.getElementById('calcImpuestos').value)|| 0;

  const { tir, duration } = calcTirDuration(_calcSymbol, price || 0, coupons);
  document.getElementById('calcTir').textContent =
    tir != null ? (tir * 100).toFixed(2) + '%' : '—';
  document.getElementById('calcDur').textContent =
    duration != null ? duration.toFixed(2) + ' años' : '—';

  if (!price || price <= 0 || !monto) {
    document.getElementById('calcTbody').innerHTML =
      '<tr><td colspan="3" class="ab-calc-empty">Ingresá precio y monto</td></tr>';
    document.getElementById('calcTfoot').innerHTML = '';
    document.getElementById('calcNote').textContent = '';
    return;
  }

  const pricePerVN  = price / 100;
  const vnCompradas = Math.floor(monto / pricePerVN);
  const montoReal   = vnCompradas * pricePerVN;

  const todayStr      = localDateStr(new Date());
  const futureCoupons = coupons
    .filter(c => c.date > todayStr)
    .sort((a, b) => a.date.localeCompare(b.date));

  const tbody = document.getElementById('calcTbody');
  tbody.innerHTML = '';
  let totalPer100 = 0, totalInversion = 0;

  futureCoupons.forEach(c => {
    const per100       = c.amount_per_100vn;
    const investorFlow = (vnCompradas / 100) * per100;
    totalPer100    += per100;
    totalInversion += investorFlow;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="ab-calc-td-m">${fmtDate(c.date)}</td>
      <td class="ab-calc-td-r">$${fmt2(per100)}</td>
      <td class="ab-calc-td-r">$${fmt2(investorFlow)}</td>`;
    tbody.appendChild(tr);
  });

  const ganancia = totalInversion - montoReal;
  const gainCls  = ganancia >= 0 ? 'pos' : 'neg';
  const gainSign = ganancia >= 0 ? '+' : '';

  document.getElementById('calcTfoot').innerHTML = `
    <tr>
      <td class="ab-calc-total-lbl">Total cobros</td>
      <td class="ab-calc-total-val">$${fmt2(totalPer100)}</td>
      <td class="ab-calc-total-val">$${fmt2(totalInversion)}</td>
    </tr>
    <tr>
      <td class="ab-calc-gain-lbl">Ganancia</td>
      <td></td>
      <td class="ab-calc-gain-val ${gainCls}">${gainSign}$${fmt2(ganancia)}</td>
    </tr>`;

  document.getElementById('calcNote').textContent =
    `Comprás ${vnCompradas.toLocaleString('es-AR')} VN a US$${pricePerVN.toFixed(4)}/VN`;
}

function fmtDate(dateStr) {
  if (!dateStr) return '—';
  const [y, m, d] = dateStr.split('-');
  return `${parseInt(m)}/${parseInt(d)}/${y}`;
}