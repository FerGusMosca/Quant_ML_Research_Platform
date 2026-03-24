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

// ── Bond metadata — loaded from /static/config/bonds_config.json ─────────────
// Flows are exact values from official prospectuses (same source as rendimientosar.com).
// BOND_META is populated at boot via loadBondConfig(). Do NOT hardcode flows here.

let BOND_META = {};  // populated by loadBondConfig()

let PAID_COUPONS_META = {};  // historical paid coupons for chart adjustment

async function loadBondConfig() {
  try {
    const res = await fetch('/static/config/bonds_config.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const cfg = await res.json();

    // Future cash flows (for TIR/Duration/calculator)
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

    // Historical paid coupons (for chart adjustment)
    const paid = cfg.paid_coupons || {};
    for (const [symbol, flows] of Object.entries(paid)) {
      PAID_COUPONS_META[symbol] = flows.map(f => ({
        date:             f.fecha,
        amount_per_100vn: f.monto,
      }));
    }
  } catch(e) {
    console.error('[ArgyBonds] Failed to load bonds_config.json:', e);
  }
}

// ── State ─────────────────────────────────────────────────────────────────
let _bonds         = [];   // live prices from API
let _enriched      = [];   // merged with meta
let _currentSymbol = null;
let _currentBars   = [];   // raw OHLCV
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
    _bonds = data.bonds || [];
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
  const meta = BOND_META[b.symbol] || {};
  const price = b.price_usd;
  const { tir, duration } = calcTirDuration(b.symbol, price, meta.coupons || []);
  return { ...b, ...meta, tir, duration };
}

async function refreshAll() {
  document.getElementById('lastUpdate').textContent = 'Actualizando…';
  await loadBonds();
}

// ── YTM / Duration calc (Newton-Raphson) ──────────────────────────────────
function calcTirDuration(symbol, price, coupons) {
  if (!price || !coupons || !coupons.length) return { tir: null, duration: null };
  const today = new Date();
  // Filter future coupons
  const future = coupons
    .filter(c => new Date(c.date) > today)
    .map(c => ({
      t: (new Date(c.date) - today) / (365.25 * 24 * 3600 * 1000), // years
      cf: c.amount_per_100vn,
    }));
  if (!future.length) return { tir: null, duration: null };

  // Newton-Raphson YTM
  let ytm = 0.09;
  for (let i = 0; i < 100; i++) {
    let pv = 0, dpv = 0;
    for (const { t, cf } of future) {
      const d = Math.pow(1 + ytm, t);
      pv  += cf / d;
      dpv -= t * cf / (d * (1 + ytm));
    }
    const f = pv - price;
    if (Math.abs(f) < 1e-8) break;
    ytm -= f / dpv;
  }

  // Macaulay Duration
  let weightedT = 0, sumPv = 0;
  for (const { t, cf } of future) {
    const pv = cf / Math.pow(1 + ytm, t);
    weightedT += t * pv;
    sumPv += pv;
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
    const tr = document.createElement('tr');
    const chgCls = b.pct_change > 0 ? 'ab-chg-pos' : b.pct_change < 0 ? 'ab-chg-neg' : 'ab-chg-neu';
    const chgSign = b.pct_change > 0 ? '+' : '';
    const lawCls  = (b.law || '').toLowerCase() === 'ny' ? 'ny' : 'local';
    const lawLabel = lawCls === 'ny' ? 'NY' : 'Local';
    const tirStr  = b.tir      != null ? (b.tir * 100).toFixed(2) + '%' : '—';
    const durStr  = b.duration != null ? b.duration.toFixed(1)          : '—';
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
        <button class="ab-chart-btn" onclick="event.stopPropagation();openChartModal('${escHtml(b.symbol)}')" title="Ver gráfico">📈</button>
        <button class="ab-calc-btn"  onclick="event.stopPropagation();openCalcModal('${escHtml(b.symbol)}')"  title="Calculadora">🧮</button>
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
  const meta = BOND_META[symbol] || {};

  document.getElementById('chartModalTitle').textContent = symbol + 'D';
  document.getElementById('chartModalSub').textContent   =
    `${bond.law || ''}  ·  Vto: ${bond.maturity || '?'}`;

  // Stats
  document.getElementById('chartStats').innerHTML = buildStatsHtml(bond);

  // Coupons
  renderCouponList(symbol, meta.coupons || []);

  // Chart loading state
  document.getElementById('chartContainer').innerHTML =
    '<div class="ab-chart-loading">⏳ Descargando barras diarias desde TradingView…</div>';

  openModal('chartModal', 'chartBackdrop');

  await fetchAndRenderChart(symbol);
}

async function fetchAndRenderChart(symbol) {
  try {
    const data = await apiFetch(
      `/argy_bonds/ohlcv?symbol=${encodeURIComponent(symbol + 'D')}&exchange=BYMA`
    );
    if (!data.ok) throw new Error(data.error || 'Sin datos');
    _currentBars = data.bars || [];
    renderChart(_currentBars, symbol, _adjEnabled);
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

// Called when user toggles adjusted price
async function onAdjToggle() {
  _adjEnabled = document.getElementById('adjToggle').checked;
  if (_currentBars.length && _currentSymbol) {
    renderChart(_currentBars, _currentSymbol, _adjEnabled);
  }
}

// ── Core chart renderer ───────────────────────────────────────────────────
function renderChart(rawBars, symbol, adjusted) {
  const container = document.getElementById('chartContainer');
  container.innerHTML = '';

  // Build adjusted bars if needed
  const meta    = BOND_META[symbol] || {};
  const coupons = meta.coupons || [];
  const bars    = adjusted ? applyAdjustment(rawBars, coupons, symbol) : rawBars;

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
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#2d333b' },
    timeScale: {
      borderColor:    '#2d333b',
      timeVisible:    false,
      secondsVisible: false,
    },
  });

  // Candlestick series
  const candleSeries = _lwChart.addCandlestickSeries({
    upColor:         '#3fb950',
    downColor:       '#f85149',
    borderUpColor:   '#3fb950',
    borderDownColor: '#f85149',
    wickUpColor:     '#3fb950',
    wickDownColor:   '#f85149',
  });
  candleSeries.setData(bars);

  // Coupon markers on the chart
  const paidCoupons = getPaidCoupons(coupons, bars);
  if (paidCoupons.length) {
    const markers = paidCoupons.map(c => ({
      time:     dateToTimestamp(c.date),
      position: 'aboveBar',
      color:    '#e3b341',
      shape:    'circle',
      text:     '¢' + c.amount_per_100vn.toFixed(3),
      size:     1,
    })).filter(m => m.time);
    if (markers.length) {
      markers.sort((a, b) => a.time - b.time);
      candleSeries.setMarkers(markers);
    }
  }

  // Responsive resize
  new ResizeObserver(() => {
    if (_lwChart) _lwChart.resize(container.clientWidth, container.clientHeight);
  }).observe(container);
}

// ── Adjusted price: sum PAID coupons per 100VN to each bar ───────────────
// Uses PAID_COUPONS_META which includes historical flows before the config's future flows.
function applyAdjustment(bars, coupons, symbol) {
  // Merge: historical paid coupons + future coupons that are now past
  const todayStr = localDateStr(new Date());

  // Get historical paid coupons for this symbol
  const historicalPaid = (PAID_COUPONS_META[symbol] || []);

  // Also include any config coupons that are already past
  const futurePaid = coupons.filter(c => c.date < todayStr);

  // Merge and deduplicate by date, sort ascending
  const allPaid = [...historicalPaid, ...futurePaid];
  const seen = new Set();
  const paidCoupons = allPaid
    .filter(c => { if (seen.has(c.date)) return false; seen.add(c.date); return true; })
    .sort((a, b) => a.date.localeCompare(b.date));

  if (!paidCoupons.length) return bars; // nothing to adjust

  let cumSum = 0;
  let ci = 0;

  // Ensure bars are sorted ascending (oldest first) — required for cumulative sum to work
  const sortedBars = [...bars].sort((a, b) => a.time - b.time);

  if (sortedBars.length > 0) {
    const firstDate = timestampToDate(sortedBars[0].time);
    const lastDate  = timestampToDate(sortedBars[sortedBars.length-1].time);
    console.log(`[ArgyBonds] ${symbol}: ${paidCoupons.length} paid coupons, bars ${firstDate}→${lastDate}, first coupon: ${paidCoupons[0]?.date}`);
  }

  const adjustedBars = sortedBars.map(bar => {
    // barDate: the trading date of this candle as YYYY-MM-DD
    const barDate = timestampToDate(bar.time);

    // Accumulate every paid coupon whose ex-date is STRICTLY before this bar date
    // (ex-dividend effect: price drops on the ex-date, so we add the coupon from that date onward)
    while (ci < paidCoupons.length && paidCoupons[ci].date <= barDate) {
      cumSum += paidCoupons[ci].amount_per_100vn;
      ci++;
    }

    return {
      time:  bar.time,
      open:  +(bar.open  + cumSum).toFixed(4),
      high:  +(bar.high  + cumSum).toFixed(4),
      low:   +(bar.low   + cumSum).toFixed(4),
      close: +(bar.close + cumSum).toFixed(4),
    };
  });

  return adjustedBars;
}

// ── Paid coupons: those with a date within the bar range ─────────────────
function getPaidCoupons(coupons, bars) {
  if (!bars.length) return [];
  const minTs = bars[0].time;
  const maxTs = bars[bars.length - 1].time;
  return coupons.filter(c => {
    const ts = dateToTimestamp(c.date);
    return ts >= minTs && ts <= maxTs;
  });
}

// ── Date helpers ──────────────────────────────────────────────────────────
function dateToTimestamp(dateStr) {
  // Use noon UTC so timezone rounding never shifts the date
  if (!dateStr) return null;
  return Math.floor(new Date(dateStr + 'T12:00:00Z').getTime() / 1000);
}

function timestampToDate(ts) {
  // tvdatafeed daily bars come as unix timestamps at midnight UTC.
  // Always read as UTC so dates match coupon strings (YYYY-MM-DD).
  const d = new Date(ts * 1000);
  const y   = d.getUTCFullYear();
  const m   = String(d.getUTCMonth() + 1).padStart(2, '0');
  const day = String(d.getUTCDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function localDateStr(date) {
  // Today as YYYY-MM-DD in the browser's local timezone
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

// ── Coupon list ───────────────────────────────────────────────────────────
function renderCouponList(symbol, coupons) {
  const today = new Date().toISOString().slice(0, 10);
  const paid  = coupons.filter(c => c.date <= today);
  const sub   = document.getElementById('couponSub');
  const list  = document.getElementById('couponList');

  sub.textContent = paid.length
    ? `${paid.length} pagados · total ${fmt2(paid.reduce((s, c) => s + c.amount_per_100vn, 0))} por 100 VN`
    : 'Sin cupones pagados en el historial';

  list.innerHTML = '';
  paid.forEach(c => {
    const chip = document.createElement('div');
    chip.className = 'ab-coupon-chip';
    chip.innerHTML = `
      <span class="ab-cc-date">${c.date}</span>
      <span class="ab-cc-amt">+${fmt3(c.amount_per_100vn)}</span>
    `;
    list.appendChild(chip);
  });
}

// ════════════════════════════════════════════════════
// UTILS
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
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function fmt2(v)  { return v != null ? Number(v).toFixed(2)  : '—'; }
function fmt3(v)  { return v != null ? Number(v).toFixed(3)  : '—'; }
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
    className: `ab-flash ${type}`, textContent: msg
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
  const bond = _enriched.find(b => b.symbol === symbol) || {};
  const meta = BOND_META[symbol] || {};

  document.getElementById('calcModalTitle').textContent = symbol + ' — Calculadora';
  document.getElementById('calcModalSub').textContent =
    `${bond.law || ''}  ·  Vencimiento: ${bond.maturity || '?'}`;

  // Pre-fill price from live data
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
  const monto    = parseFloat(document.getElementById('calcMonto').value) || 0;
  const arancel  = parseFloat(document.getElementById('calcArancel').value)   || 0;
  const impuesto = parseFloat(document.getElementById('calcImpuestos').value) || 0;

  // ── TIR / Duration ────────────────────────────────────────────────
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

  // ── Position sizing ───────────────────────────────────────────────
  // price is in USD per 100 VN face value → price per 1 VN = price / 100
  const costoTotal  = monto * (1 + (arancel + impuesto) / 100);
  const pricePerVN  = price / 100;
  const vnCompradas = Math.floor(monto / pricePerVN);  // integer VN
  const montoReal   = vnCompradas * pricePerVN;        // actual USD spent

  // ── Future cash flows ─────────────────────────────────────────────
  const today = new Date();
  const todayStr = localDateStr(today);
  const futureCoupons = coupons
    .filter(c => c.date > todayStr)
    .sort((a, b) => a.date.localeCompare(b.date));

  const tbody = document.getElementById('calcTbody');
  tbody.innerHTML = '';

  let totalPer100 = 0;
  let totalInversion = 0;

  futureCoupons.forEach(c => {
    // Per 100 VN: this is the raw coupon amount
    const per100 = c.amount_per_100vn;
    // For the investor: (vnCompradas / 100) * per100
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

  // ── Tfoot: totals + ganancia ──────────────────────────────────────
  const ganancia   = totalInversion - montoReal;
  const gainCls    = ganancia >= 0 ? 'pos' : 'neg';
  const gainSign   = ganancia >= 0 ? '+' : '';

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

  // ── Note ──────────────────────────────────────────────────────────
  document.getElementById('calcNote').textContent =
    `Comprás ${vnCompradas.toLocaleString('es-AR')} VN a US$${(pricePerVN).toFixed(4)}/VN`;
}

// ── Date formatter for cash flow table ───────────────────────────────────
function fmtDate(dateStr) {
  // YYYY-MM-DD → M/D/YYYY (same style as the screenshot)
  if (!dateStr) return '—';
  const [y, m, d] = dateStr.split('-');
  return `${parseInt(m)}/${parseInt(d)}/${y}`;
}