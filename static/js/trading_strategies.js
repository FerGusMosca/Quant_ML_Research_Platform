// trading_strategies.js

// ── Clock ─────────────────────────────────────────────────────────────────
(function tick() {
  const el = document.getElementById('navClock');
  if (el) {
    const n = new Date(), p = v => String(v).padStart(2, '0');
    el.textContent = [n.getHours(), n.getMinutes(), n.getSeconds()].map(p).join(':');
  }
  setTimeout(tick, 1000);
})();

// ── State ─────────────────────────────────────────────────────────────────
let _strategies  = [];
let _selStrategy = null;
let _stratDbs    = [];
let _selDbName   = null;
let _selSymbol   = null;
let _selExchange = null;   // resolved exchange for current symbol
let _trades      = [];

// ── Boot ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', loadStrategies);

// ════════════════════════════════════════════════════
// PANEL 1 — STRATEGIES
// ════════════════════════════════════════════════════

async function loadStrategies() {
  const list = document.getElementById('strategyList');
  list.innerHTML = '<div class="ts-loading">Loading…</div>';
  try {
    _strategies = await apiFetch('/trading_strategies/strategies');
    renderStrategies();
  } catch(e) {
    list.innerHTML = `<div class="ts-error">❌ ${e.message}</div>`;
  }
}

function renderStrategies() {
  const list = document.getElementById('strategyList');
  list.innerHTML = '';
  if (!_strategies.length) {
    list.innerHTML = '<div class="ts-empty">No strategies. Click + New to add.</div>';
    return;
  }
  _strategies.forEach(s => {
    const el = document.createElement('div');
    el.className = 'strategy-item' + (s.is_active ? '' : ' inactive');
    if (_selStrategy?.strategy_id === s.strategy_id) el.classList.add('selected');
    el.innerHTML = `
      <div class="si-main" onclick="selectStrategy(${s.strategy_id})">
        <div class="si-name" title="${escHtml(s.strategy_name)}">${escHtml(s.strategy_name)}</div>
      </div>
      <div class="si-actions">
        ${s.is_active ? '<span class="si-badge active">Active</span>' : '<span class="si-badge inactive">Off</span>'}
        <button class="btn-icon edit"
          onclick="event.stopPropagation();openStrategyModal(${s.strategy_id})"
          title="Edit">✎</button>
      </div>`;
    list.appendChild(el);
  });
}

async function selectStrategy(strategyId) {
  _selStrategy = _strategies.find(s => s.strategy_id === strategyId);
  if (!_selStrategy) return;
  document.querySelectorAll('.strategy-item').forEach(el =>
    el.classList.toggle('selected',
      el.querySelector('.si-name')?.getAttribute('title') === _selStrategy.strategy_name)
  );
  showPanel('panelSecurities');
  document.getElementById('secPanelTitle').textContent = _selStrategy.strategy_name;
  await loadStrategyDatabases(strategyId);
}

function backToStrategies() {
  _selStrategy = null; _selSymbol = null; _selDbName = null; _stratDbs = [];
  showPanel('panelStrategies');
}

// ════════════════════════════════════════════════════
// DATABASE SELECTOR
// ════════════════════════════════════════════════════

async function loadStrategyDatabases(strategyId) {
  const tabsEl = document.getElementById('dbTabs');
  tabsEl.innerHTML = '<span class="ts-loading" style="font-size:10px">…</span>';
  try {
    _stratDbs = await apiFetch(`/trading_strategies/strategy_databases?strategy_id=${strategyId}`);
    renderDbTabs();
    const def = _stratDbs.find(d => d.is_default) || _stratDbs[0];
    if (def) await selectDatabase(def.database_name);
    else {
      document.getElementById('securityList').innerHTML =
        '<div class="ts-empty">No databases configured. Click + to add one.</div>';
      document.getElementById('todaySignalsWrap').classList.add('hidden');
    }
  } catch(e) {
    tabsEl.innerHTML = `<span style="color:var(--red);font-size:10px">❌ ${e.message}</span>`;
  }
}

function renderDbTabs() {
  const tabsEl = document.getElementById('dbTabs');
  tabsEl.innerHTML = '';
  _stratDbs.forEach(db => {
    const btn = document.createElement('button');
    btn.className = 'db-tab' + (db.database_name === _selDbName ? ' active' : '');
    btn.title     = db.database_name;
    btn.innerHTML = `
      ${escHtml(db.label || db.database_name)}
      ${db.is_default ? '<span class="db-default-dot"></span>' : ''}
      <span class="db-tab-edit" onclick="event.stopPropagation();openDbModal(${db.db_id})" title="Edit">✎</span>`;
    btn.onclick = () => selectDatabase(db.database_name);
    tabsEl.appendChild(btn);
  });
}

async function selectDatabase(dbName) {
  _selDbName = dbName; _selSymbol = null;
  renderDbTabs();
  await Promise.all([loadSecurities(), loadTodaySignals()]);
}

// ════════════════════════════════════════════════════
// PANEL 2 — SECURITIES
// ════════════════════════════════════════════════════

async function loadSecurities() {
  const list = document.getElementById('securityList');
  list.innerHTML = '<div class="ts-loading">Loading securities…</div>';
  try {
    const data = await apiFetch(
      `/trading_strategies/securities?strategy_id=${_selStrategy.strategy_id}&database_name=${encodeURIComponent(_selDbName)}`
    );
    renderSecurities(data.securities);
  } catch(e) {
    list.innerHTML = `<div class="ts-error">❌ ${e.message}</div>`;
  }
}

function renderSecurities(securities) {
  const list = document.getElementById('securityList');
  list.innerHTML = '';
  if (!securities.length) {
    list.innerHTML = '<div class="ts-empty">No securities found.</div>';
    return;
  }
  securities.forEach(s => {
    const el = document.createElement('div');
    el.className = 'security-item';
    const pnlCls  = s.total_profit >= 0 ? 'pos' : 'neg';
    const pnlSign = s.total_profit >= 0 ? '+' : '';
    const lastDate = s.last_trade ? s.last_trade.slice(0, 10) : '—';
    el.innerHTML = `
      <div class="sec-left" onclick="selectSecurity('${escHtml(s.symbol)}')">
        <div class="sec-symbol">${escHtml(s.symbol)}</div>
        <div class="sec-meta">${s.trade_count} trades · last ${lastDate}</div>
      </div>
      <div class="sec-right">
        <div class="sec-pnl ${pnlCls}">${pnlSign}${fmt2(s.total_profit)}</div>
        <div class="sec-closed">${s.closed_trades}c / ${s.open_trades}o</div>
      </div>`;
    list.appendChild(el);
  });
}

async function loadTodaySignals() {
  const wrap = document.getElementById('todaySignalsWrap');
  wrap.classList.add('hidden');
  try {
    const data    = await apiFetch(
      `/trading_strategies/today_signals?strategy_id=${_selStrategy.strategy_id}&database_name=${encodeURIComponent(_selDbName)}`
    );
    const signals = data.signals || [];
    document.getElementById('signalDate').textContent = data.date || '';
    if (!signals.length) return;
    wrap.classList.remove('hidden');
    const list = document.getElementById('signalList');
    list.innerHTML = '';
    signals.forEach(t => {
      const el = document.createElement('div');
      el.className = 'signal-item';
      const dirCls = (t.trade_direction || '').toUpperCase() === 'LONG' ? 'long' : 'short';
      el.innerHTML = `
        <span class="sig-dir ${dirCls}">${t.trade_direction || '?'}</span>
        <span class="sig-sym">${t.symbol}</span>
        <span class="sig-price">@ ${fmt4(t.opening_price)}</span>
        <span class="sig-qty">qty ${t.qty}</span>
        <span class="sig-status">${t.is_closed ? '✅' : '⏳'}</span>
        <button class="btn-icon view" onclick='openChartModal(${JSON.stringify(t)})' title="Chart">📈</button>`;
      list.appendChild(el);
    });
  } catch(e) { /* silent */ }
}

async function selectSecurity(symbol) {
  _selSymbol = symbol;
  showPanel('panelTrades');
  document.getElementById('tradePanelTitle').textContent = symbol;
  await reloadTrades();
}

function backToSecurities() {
  _selSymbol = null;
  showPanel('panelSecurities');
}

// ════════════════════════════════════════════════════
// PANEL 3 — TRADES
// ════════════════════════════════════════════════════

async function reloadTrades() {
  if (!_selStrategy || !_selSymbol || !_selDbName) return;
  const tbody = document.getElementById('tradesTbody');
  const badge = document.getElementById('tradeCountBadge');
  tbody.innerHTML = '<tr><td colspan="10" class="empty-cell loading-cell">Loading…</td></tr>';
  badge.textContent = '';

  const df  = document.getElementById('tradeFrom').value;
  const dt  = document.getElementById('tradeTo').value;
  let url   = `/trading_strategies/trades?strategy_id=${_selStrategy.strategy_id}`
            + `&database_name=${encodeURIComponent(_selDbName)}`
            + `&symbol=${encodeURIComponent(_selSymbol)}`;
  if (df) url += `&date_from=${df}`;
  if (dt) url += `&date_to=${dt}`;

  try {
    const data   = await apiFetch(url);
    _trades      = data.trades || [];
    _selExchange = data.exchange || _selStrategy.exchange || 'NYSE';
    renderTrades(_trades);
    badge.textContent = `${_trades.length} trade${_trades.length !== 1 ? 's' : ''}`;
  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="10" class="empty-cell" style="color:var(--red)">❌ ${e.message}</td></tr>`;
  }
}

function renderTrades(trades) {
  const tbody = document.getElementById('tradesTbody');
  tbody.innerHTML = '';
  if (!trades.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="empty-cell">No trades found.</td></tr>';
    return;
  }
  trades.forEach((t, i) => {
    const tr      = document.createElement('tr');
    const dirCls  = (t.trade_direction || '').toUpperCase() === 'LONG' ? 'long' : 'short';
    const pnlCls  = (t.profit || 0) >= 0 ? 'pos' : 'neg';
    const pnlSign = (t.profit || 0) >= 0 ? '+' : '';
    const openDt  = t.opening_date ? t.opening_date.slice(0,16).replace('T',' ') : '—';
    const closeDt = t.closing_date ? t.closing_date.slice(0,16).replace('T',' ') : '⏳';
    tr.innerHTML = `
      <td class="td-idx">${i+1}</td>
      <td class="td-ts">${openDt}</td>
      <td class="td-ts ${t.is_closed ? '' : 'dim'}">${closeDt}</td>
      <td><span class="dir-badge ${dirCls}">${t.trade_direction || '?'}</span></td>
      <td class="td-mono">${t.qty}</td>
      <td class="td-price">${fmt4(t.opening_price)}</td>
      <td class="td-price ${t.is_closed ? '' : 'dim'}">${t.closing_price ? fmt4(t.closing_price) : '—'}</td>
      <td class="td-pnl ${pnlCls}">${t.profit != null ? pnlSign+fmt2(t.profit) : '—'}</td>
      <td class="td-fee dim">${t.total_fee != null ? fmt2(t.total_fee) : '—'}</td>
      <td><button class="btn-icon view" onclick='openChartModal(${JSON.stringify(t)})' title="Chart">📈</button></td>`;
    tbody.appendChild(tr);
  });
}

function clearTradeDates() {
  document.getElementById('tradeFrom').value = '';
  document.getElementById('tradeTo').value   = '';
  reloadTrades();
}

// ════════════════════════════════════════════════════
// CHART MODAL — Lightweight Charts + OHLCV from TV
// ════════════════════════════════════════════════════

async function openChartModal(trade) {
  const exchange = _selExchange || _selStrategy?.exchange || 'NYSE';
  const symbol   = trade.symbol.toUpperCase();
  const date     = trade.opening_date.slice(0, 10);  // YYYY-MM-DD

  document.getElementById('chartModalTitle').textContent =
    `${exchange}:${symbol} · ${trade.trade_direction} · ${date}`;
  document.getElementById('chartContainer').innerHTML =
    '<div class="chart-loading">⏳ Downloading 1-min bars from TradingView…</div>';
  document.getElementById('chartStats').innerHTML = buildStatsHtml(trade, exchange);

  openModal('chartModal', 'chartBackdrop');

  try {
    const data = await apiFetch(
      `/trading_strategies/ohlcv?symbol=${encodeURIComponent(symbol)}&exchange=${encodeURIComponent(exchange)}&date=${date}`
    );
    if (!data.ok) throw new Error(data.error);
    renderLightweightChart(data.bars, trade);
  } catch(e) {
    document.getElementById('chartContainer').innerHTML =
      `<div class="chart-error">❌ ${e.message}</div>`;
  }
}

function closeChartModal() {
  closeModal('chartModal', 'chartBackdrop');
  document.getElementById('chartContainer').innerHTML = '';
}

// Bars and trade dates are in the same timezone — no conversion needed.
function toUnixUTC3(isoStr) {
  if (!isoStr) return null;
  const normalized = isoStr.length === 16 ? isoStr + ':00' : isoStr;
  return Math.floor(new Date(normalized + 'Z').getTime() / 1000);
}

function renderLightweightChart(bars, trade) {
  const container = document.getElementById('chartContainer');
  container.innerHTML = '';

  const isLong    = (trade.trade_direction || '').toUpperCase() === 'LONG';
  const lineColor = isLong ? '#3fb950' : '#f85149';
  const zoneColor = isLong ? 'rgba(63,185,80,0.10)' : 'rgba(248,81,73,0.10)';

  const chart = LightweightCharts.createChart(container, {
    width:  container.clientWidth,
    height: container.clientHeight || 500,
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
      timeVisible:    true,
      secondsVisible: false,
    },
  });

  // ── Candlestick series ────────────────────────────────────────────────────
  const candleSeries = chart.addCandlestickSeries({
    upColor:         '#3fb950',
    downColor:       '#f85149',
    borderUpColor:   '#3fb950',
    borderDownColor: '#f85149',
    wickUpColor:     '#3fb950',
    wickDownColor:   '#f85149',
  });
  candleSeries.setData(bars);

  if (!trade.opening_date) return;

  const openTs  = toUnixUTC3(trade.opening_date);
  const closeTs = trade.closing_date ? toUnixUTC3(trade.closing_date) : openTs + 3600;

  // ── Shaded zone ───────────────────────────────────────────────────────────
  const shadeSeries = chart.addHistogramSeries({
    color:        zoneColor,
    priceFormat:  { type: 'volume' },
    priceScaleId: '',
    scaleMargins: { top: 0, bottom: 0 },
  });
  shadeSeries.setData(
    bars.filter(b => b.time >= openTs && b.time <= closeTs)
        .map(b => ({ time: b.time, value: 1 }))
  );

  // ── Entry line ────────────────────────────────────────────────────────────
  if (trade.opening_price != null) {
    const entryLine = chart.addLineSeries({
      color:            lineColor,
      lineWidth:        1,
      lineStyle:        LightweightCharts.LineStyle.Dashed,
      priceLineVisible: false,
      lastValueVisible: true,
      title:            (isLong ? '▲ LONG' : '▼ SHORT') + ' Entry ' + fmt4(trade.opening_price),
    });
    entryLine.setData(
      bars.filter(b => b.time >= openTs && b.time <= closeTs)
          .map(b => ({ time: b.time, value: trade.opening_price }))
    );
  }

  // ── Exit line ─────────────────────────────────────────────────────────────
  if (trade.closing_price != null && trade.closing_date) {
    const exitLine = chart.addLineSeries({
      color:            '#e3b341',
      lineWidth:        1,
      lineStyle:        LightweightCharts.LineStyle.Dashed,
      priceLineVisible: false,
      lastValueVisible: true,
      title:            'Exit ' + fmt4(trade.closing_price),
    });
    exitLine.setData(
      bars.filter(b => b.time >= openTs && b.time <= closeTs)
          .map(b => ({ time: b.time, value: trade.closing_price }))
    );
  }

  // ── Markers: arrows at exact entry/exit candles ───────────────────────────
  const markers = [];
  const entryBar = bars.reduce((p, c) => Math.abs(c.time - openTs) < Math.abs(p.time - openTs) ? c : p);
  markers.push({
    time:     entryBar.time,
    position: isLong ? 'belowBar' : 'aboveBar',
    color:    lineColor,
    shape:    isLong ? 'arrowUp' : 'arrowDown',
    text:     (isLong ? 'BUY' : 'SELL') + ' @ ' + fmt4(trade.opening_price),
    size:     2,
  });

  if (trade.closing_date && trade.closing_price != null) {
    const exitBar = bars.reduce((p, c) => Math.abs(c.time - closeTs) < Math.abs(p.time - closeTs) ? c : p);
    markers.push({
      time:     exitBar.time,
      position: isLong ? 'aboveBar' : 'belowBar',
      color:    '#e3b341',
      shape:    isLong ? 'arrowDown' : 'arrowUp',
      text:     (isLong ? 'SELL' : 'COVER') + ' @ ' + fmt4(trade.closing_price),
      size:     2,
    });
  }

  markers.sort((a, b) => a.time - b.time);
  candleSeries.setMarkers(markers);

  // ── Zoom to trade range ───────────────────────────────────────────────────
  const PAD = 30 * 60;
  chart.timeScale().setVisibleRange({ from: openTs - PAD, to: closeTs + PAD });

  // ── Responsive resize ─────────────────────────────────────────────────────
  new ResizeObserver(() => {
    chart.resize(container.clientWidth, container.clientHeight);
  }).observe(container);
}

function buildStatsHtml(trade, exchange) {
  const pnlCls = (trade.profit || 0) >= 0 ? 'pos' : 'neg';
  return `
    <div class="tvs-item"><span class="tvs-lbl">Exchange</span><span class="tvs-val">${exchange}</span></div>
    <div class="tvs-item"><span class="tvs-lbl">Open</span><span class="tvs-val">${trade.opening_date?.slice(0,16).replace('T',' ') ?? '—'}</span></div>
    <div class="tvs-item"><span class="tvs-lbl">Close</span><span class="tvs-val">${trade.closing_date?.slice(0,16).replace('T',' ') ?? '⏳'}</span></div>
    <div class="tvs-item"><span class="tvs-lbl">Open $</span><span class="tvs-val">${fmt4(trade.opening_price)}</span></div>
    <div class="tvs-item"><span class="tvs-lbl">Close $</span><span class="tvs-val">${trade.closing_price ? fmt4(trade.closing_price) : '—'}</span></div>
    <div class="tvs-item"><span class="tvs-lbl">Qty</span><span class="tvs-val">${trade.qty}</span></div>
    <div class="tvs-item"><span class="tvs-lbl">Duration</span><span class="tvs-val">${trade.duration_minutes ? trade.duration_minutes + ' min' : '—'}</span></div>
    <div class="tvs-item"><span class="tvs-lbl">Profit</span><span class="tvs-val ${pnlCls}">${trade.profit != null ? (trade.profit>=0?'+':'')+fmt2(trade.profit) : '—'}</span></div>
    <div class="tvs-item"><span class="tvs-lbl">Nominal</span><span class="tvs-val ${pnlCls}">${trade.nominal_profit != null ? (trade.nominal_profit>=0?'+':'')+fmt2(trade.nominal_profit) : '—'}</span></div>
    <div class="tvs-item"><span class="tvs-lbl">Fee</span><span class="tvs-val dim">${trade.total_fee != null ? fmt2(trade.total_fee) : '—'}</span></div>
    <div class="tvs-item"><span class="tvs-lbl">Init Cap</span><span class="tvs-val">${trade.initial_cap != null ? fmt2(trade.initial_cap) : '—'}</span></div>
    <div class="tvs-item"><span class="tvs-lbl">Final Cap</span><span class="tvs-val">${trade.final_cap != null ? fmt2(trade.final_cap) : '—'}</span></div>
  `;
}

// ════════════════════════════════════════════════════
// STRATEGY MODAL
// ════════════════════════════════════════════════════

function openStrategyModal(strategyId) {
  const s = strategyId ? _strategies.find(x => x.strategy_id === strategyId) : null;
  document.getElementById('stratModalTitle').textContent = s ? 'Edit Strategy' : 'New Strategy';
  document.getElementById('smId').value       = s?.strategy_id ?? '';
  document.getElementById('smName').value     = s?.strategy_name ?? '';
  document.getElementById('smExchange').value = s?.exchange ?? 'NYSE';
  document.getElementById('smDesc').value     = s?.description ?? '';
  document.getElementById('smActive').checked = s ? s.is_active : true;
  const del = document.getElementById('btnDeleteStrategy');
  s ? (del.classList.remove('hidden'), del.dataset.id = strategyId) : del.classList.add('hidden');
  openModal('stratModal', 'stratModalBackdrop');
}
function closeStrategyModal() { closeModal('stratModal', 'stratModalBackdrop'); }

async function saveStrategy() {
  const name = document.getElementById('smName').value.trim();
  if (!name) { document.getElementById('smName').focus(); return; }
  try {
    const id   = document.getElementById('smId').value;
    const data = await apiFetch('/trading_strategies/strategies', {
      method: 'POST',
      body: JSON.stringify({
        strategy_id:   id ? parseInt(id) : null,
        strategy_name: name,
        exchange:      document.getElementById('smExchange').value.trim() || 'NYSE',
        description:   document.getElementById('smDesc').value.trim() || null,
        is_active:     document.getElementById('smActive').checked,
      }),
    });
    if (!data.ok) { showFlash('error', data.error); return; }
    showFlash('success', 'Strategy saved');
    closeStrategyModal();
    await loadStrategies();
  } catch(e) { showFlash('error', e.message); }
}

async function deleteStrategy() {
  const id = document.getElementById('btnDeleteStrategy').dataset.id;
  if (!confirm('Delete this strategy?')) return;
  try {
    const data = await apiFetch(`/trading_strategies/strategies/${id}`, { method: 'DELETE' });
    if (!data.ok) { showFlash('error', data.error); return; }
    showFlash('success', 'Strategy deleted');
    closeStrategyModal();
    backToStrategies();
    await loadStrategies();
  } catch(e) { showFlash('error', e.message); }
}

// ════════════════════════════════════════════════════
// DATABASE MODAL
// ════════════════════════════════════════════════════

function openDbModal(dbId) {
  const db = dbId ? _stratDbs.find(d => d.db_id === dbId) : null;
  document.getElementById('dbModalTitle').textContent = db ? 'Edit Database' : 'Add Database';
  document.getElementById('dbmId').value      = db?.db_id ?? '';
  document.getElementById('dbmName').value    = db?.database_name ?? '';
  document.getElementById('dbmLabel').value   = db?.label ?? '';
  document.getElementById('dbmDefault').checked = db ? db.is_default : false;
  const del = document.getElementById('btnDeleteDb');
  db ? (del.classList.remove('hidden'), del.dataset.id = dbId) : del.classList.add('hidden');
  openModal('dbModal', 'dbModalBackdrop');
}
function closeDbModal() { closeModal('dbModal', 'dbModalBackdrop'); }

async function saveStrategyDb() {
  const name = document.getElementById('dbmName').value.trim();
  if (!name) { document.getElementById('dbmName').focus(); return; }
  const id = document.getElementById('dbmId').value;
  try {
    const data = await apiFetch('/trading_strategies/strategy_databases', {
      method: 'POST',
      body: JSON.stringify({
        db_id:         id ? parseInt(id) : null,
        strategy_id:   _selStrategy.strategy_id,
        database_name: name,
        label:         document.getElementById('dbmLabel').value.trim() || null,
        is_default:    document.getElementById('dbmDefault').checked,
      }),
    });
    if (!data.ok) { showFlash('error', data.error); return; }
    showFlash('success', 'Database saved');
    closeDbModal();
    await loadStrategyDatabases(_selStrategy.strategy_id);
  } catch(e) { showFlash('error', e.message); }
}

async function deleteStrategyDb() {
  const id = document.getElementById('btnDeleteDb').dataset.id;
  if (!confirm('Remove this database?')) return;
  try {
    const data = await apiFetch(`/trading_strategies/strategy_databases/${id}`, { method: 'DELETE' });
    if (!data.ok) { showFlash('error', data.error); return; }
    showFlash('success', 'Database removed');
    closeDbModal();
    await loadStrategyDatabases(_selStrategy.strategy_id);
  } catch(e) { showFlash('error', e.message); }
}

// ════════════════════════════════════════════════════
// PERFORMANCE MODAL
// ════════════════════════════════════════════════════

async function openPerfModal() {
  if (!_selStrategy || !_selSymbol || !_selDbName) return;
  document.getElementById('perfModalTitle').textContent = `Performance · ${_selSymbol}`;
  document.getElementById('perfBody').innerHTML = '<div class="ts-loading">Loading…</div>';
  openModal('perfModal', 'perfBackdrop');
  try {
    const url  = `/trading_strategies/monthly_performance?strategy_id=${_selStrategy.strategy_id}`
               + `&database_name=${encodeURIComponent(_selDbName)}&symbol=${encodeURIComponent(_selSymbol)}`;
    const rows = await apiFetch(url);
    renderPerfModal(rows);
  } catch(e) {
    document.getElementById('perfBody').innerHTML = `<div class="ts-error">❌ ${e.message}</div>`;
  }
}
function closePerfModal() { closeModal('perfModal', 'perfBackdrop'); }

function renderPerfModal(rows) {
  const body = document.getElementById('perfBody');
  if (!rows.length) { body.innerHTML = '<div class="ts-empty">No closed trades found.</div>'; return; }

  const totalPnl    = rows.reduce((s,r) => s + r.total_profit, 0);
  const totalTrades = rows.reduce((s,r) => s + r.trade_count,  0);
  const totalWins   = rows.reduce((s,r) => s + r.winning_trades, 0);
  const overallWR   = totalTrades ? Math.round(totalWins / totalTrades * 100) : 0;
  const maxAbs      = Math.max(...rows.map(r => Math.abs(r.total_profit)), 1);

  let html = `
    <div class="perf-summary">
      <div class="perf-sum-item"><span class="perf-sum-lbl">Total P&L</span>
        <span class="perf-sum-val ${totalPnl>=0?'pos':'neg'}">${totalPnl>=0?'+':''}${fmt2(totalPnl)}</span></div>
      <div class="perf-sum-item"><span class="perf-sum-lbl">Closed Trades</span>
        <span class="perf-sum-val">${totalTrades}</span></div>
      <div class="perf-sum-item"><span class="perf-sum-lbl">Win Rate</span>
        <span class="perf-sum-val ${overallWR>=50?'pos':'neg'}">${overallWR}%</span></div>
    </div><div class="perf-bars">`;

  rows.forEach(r => {
    const barPct = Math.round(Math.abs(r.total_profit) / maxAbs * 100);
    const cls    = r.total_profit >= 0 ? 'pos' : 'neg';
    const sign   = r.total_profit >= 0 ? '+' : '';
    html += `
      <div class="perf-row">
        <div class="perf-month">${r.month_label}</div>
        <div class="perf-bar-wrap"><div class="perf-bar ${cls}" style="width:${barPct}%"></div></div>
        <div class="perf-pnl ${cls}">${sign}${fmt2(r.total_profit)}</div>
        <div class="perf-wr">${r.win_rate}%</div>
        <div class="perf-trades dim">${r.trade_count}t</div>
      </div>`;
  });
  html += '</div>';
  body.innerHTML = html;
}

// ════════════════════════════════════════════════════
// PANEL NAVIGATION
// ════════════════════════════════════════════════════

function showPanel(panelId) {
  const panels = ['panelStrategies', 'panelSecurities', 'panelTrades'];
  const idx    = panels.indexOf(panelId);
  panels.forEach((id, i) => {
    const el = document.getElementById(id);
    el.classList.toggle('hidden',    i > idx);
    el.classList.toggle('collapsed', i < idx);
  });
}

// ════════════════════════════════════════════════════
// UTILS
// ════════════════════════════════════════════════════

function openModal(id, bid)  {
  document.getElementById(bid).classList.add('open');
  document.getElementById(id).classList.add('open');
}
function closeModal(id, bid) {
  document.getElementById(bid).classList.remove('open');
  document.getElementById(id).classList.remove('open');
}
function escHtml(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function fmt2(v)  { return v != null ? Number(v).toFixed(2) : '—'; }
function fmt4(v)  { return v != null ? Number(v).toFixed(4) : '—'; }
async function apiFetch(url, opts={}) {
  const r = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
function showFlash(type, msg) {
  document.querySelector('.flash')?.remove();
  const el = Object.assign(document.createElement('div'), { className: `flash ${type}`, textContent: msg });
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}