/**
 * simulate_model.js
 * Model Runner page — CRUD for model configs, XGBoost execution, Plotly chart.
 */

'use strict';

/* ── State ─────────────────────────────────────────────────── */
let models        = [];      // RunningModelConfigDTO[]
let activeModelId = null;    // currently selected model id
let lastResult    = null;    // last run result from /run_model
const BASE        = '/simulate_model';

/* ── Boot ───────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  loadModels();
});

/* ── Load model list ─────────────────────────────────────────── */
async function loadModels() {
  try {
    const res  = await fetch(`${BASE}/models`);
    const data = await res.json();
    models = data;
    renderModelList();
  } catch (e) {
    showFlash('Failed to load models: ' + e.message, 'error');
  }
}

function renderModelList() {
  const el = document.getElementById('modelList');
  if (!models.length) {
    el.innerHTML = '<div class="model-list-empty">No models configured</div>';
    return;
  }
  el.innerHTML = models.map(m => `
    <div class="model-card ${m.model_id === activeModelId ? 'active' : ''}"
         onclick="selectModel(${m.model_id})">
      <div class="model-card-name">${m.model_name}</div>
      <div class="model-card-meta">${m.symbol} · ${m.algo_type}</div>
    </div>
  `).join('');
}

/* ── Select model ────────────────────────────────────────────── */
function selectModel(modelId) {
  activeModelId = modelId;
  const m = models.find(x => x.model_id === modelId);
  if (!m) return;

  renderModelList();  // re-render to show active state

  // Show active panel, hide empty state
  document.getElementById('emptyState').style.display  = 'none';
  document.getElementById('activePanel').style.display = 'block';

  // Header
  document.getElementById('hdrModelName').textContent = m.model_name;
  document.getElementById('hdrModelMeta').textContent =
    `${m.symbol} · ${m.algo_type} · ${m.d_from} → ${m.d_to}`;

  // Pre-fill run bar
  document.getElementById('runDFrom').value  = m.d_from;
  document.getElementById('runDTo').value    = m.d_to;
  document.getElementById('runNFlip').value  = m.n_flip;
  const biasEl = document.getElementById('runBias');
  biasEl.value = m.bias;

  // Reset result areas
  resetResults();
}

function resetResults() {
  lastResult = null;
  // Chart
  document.getElementById('chartPlaceholder').style.display = 'block';
  document.getElementById('priceChart').style.display       = 'none';
  // Summary
  document.getElementById('summaryPlaceholder').style.display = 'block';
  document.getElementById('summaryContent').style.display     = 'none';
  // Positions
  document.getElementById('positionsPlaceholder').style.display = 'block';
  document.getElementById('positionsContent').style.display     = 'none';
  // Signal
  document.getElementById('signalNoData').style.display  = 'block';
  document.getElementById('signalContent').style.display = 'none';
}

/* ── Tab switching ───────────────────────────────────────────── */
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.tab === name);
  });
  document.querySelectorAll('.tab-pane').forEach(p => {
    p.classList.toggle('active', p.id === `tab-${name}`);
  });
}

/* ── Run model ───────────────────────────────────────────────── */
async function runModel() {
  if (!activeModelId) return;

  const btn     = document.getElementById('btnRun');
  const spinner = document.getElementById('runSpinner');
  btn.disabled  = true;
  btn.classList.add('running');
  spinner.style.display = 'inline-block';

  const payload = {
    model_id: activeModelId,
    d_from:   document.getElementById('runDFrom').value,
    d_to:     document.getElementById('runDTo').value,
    n_flip:   parseInt(document.getElementById('runNFlip').value),
    bias:     document.getElementById('runBias').value,
  };

  try {
    const res  = await fetch(`${BASE}/run_model`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();

    if (!data.ok) {
      showFlash('Run failed: ' + (data.error || 'unknown error'), 'error');
      return;
    }

    lastResult = data;
    const m    = models.find(x => x.model_id === activeModelId);

    // Render summary/positions/signal tabs
    renderSummary(data.summary);
    renderPositions(data.summary.positions);
    renderSignal(data.summary);

    // Switch to chart tab FIRST so the div is visible before Plotly renders
    switchTab('chart');

    // Fetch prices and render chart
    await renderChart(data.symbol, data.d_from, data.d_to, data.summary.positions);
    showFlash(`${m?.model_name} completed ✓`, 'success');

  } catch (e) {
    showFlash('Error: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.classList.remove('running');
    spinner.style.display = 'none';
  }
}

/* ── Chart ───────────────────────────────────────────────────────────── */
async function renderChart(symbol, dFrom, dTo, positions) {
  console.log('[renderChart] called', { symbol, dFrom, dTo, positions });
  try {
    const res  = await fetch(`${BASE}/prices?symbol=${symbol}&d_from=${dFrom}&d_to=${dTo}`);
    console.log('[renderChart] fetch status', res.status);
    const data = await res.json();
    console.log('[renderChart] data.ok=', data.ok, 'prices count=', data.prices?.length);

    if (!data.ok || !data.prices?.length) {
      showFlash('Could not load price data for chart', 'error');
      return;
    }

    const prices = data.prices.filter(p => p.close !== null && p.close !== undefined);
    console.log('[renderChart] valid prices=', prices.length);
    if (!prices.length) { showFlash('No valid price data', 'error'); return; }

    const dates  = prices.map(p => p.date);
    const closes = prices.map(p => p.close);

    // ── Build in-position set ──
    const inPosition = new Set();
    (positions || []).forEach(pos => {
      if (!pos.date_open || !pos.date_close) return;
      dates.forEach(d => {
        if (d >= pos.date_open && d <= pos.date_close) inPosition.add(d);
      });
    });
    console.log('[renderChart] inPosition size=', inPosition.size);

    // ── Split into contiguous segments (in / out) ──
    const traces = [];

    const pushSegment = (segDates, segCloses, isIn) => {
      if (!segDates.length) return;
      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: [...segDates],
        y: [...segCloses],
        line: {
          color: isIn ? '#3FB950' : '#4A7FBF',
          width: isIn ? 2.5 : 1.5,
        },
        hovertemplate: '%{x}<br>$%{y:.2f}<extra></extra>',
        showlegend: false,
      });
    };

    let segDates  = [];
    let segCloses = [];
    let segIn     = inPosition.has(dates[0]);

    dates.forEach((d, i) => {
      const isIn = inPosition.has(d);
      if (isIn !== segIn) {
        segDates.push(d);
        segCloses.push(closes[i]);
        pushSegment(segDates, segCloses, segIn);
        segDates  = [d];
        segCloses = [closes[i]];
        segIn     = isIn;
      } else {
        segDates.push(d);
        segCloses.push(closes[i]);
      }
    });
    pushSegment(segDates, segCloses, segIn);

    // ── Entry / exit markers ──
    (positions || []).forEach((pos, i) => {
      if (!pos.date_open) return;
      const isLong   = (pos.side || '').toUpperCase() === 'LONG';
      const isProfit = (pos.nom_profit || 0) >= 0;

      traces.push({
        type: 'scatter', mode: 'markers',
        x: [pos.date_open], y: [pos.price_open],
        marker: {
          symbol: isLong ? 'triangle-up' : 'triangle-down',
          size: 11,
          color: '#3FB950',
          line: { width: 1, color: '#238636' },
        },
        hovertemplate: `▶ Open #${i+1} ${pos.side || ''}<br>${pos.date_open}<br>$%{y:.2f}<extra></extra>`,
        showlegend: false,
      });

      if (pos.date_close) {
        traces.push({
          type: 'scatter', mode: 'markers',
          x: [pos.date_close], y: [pos.price_close],
          marker: {
            symbol: 'x',
            size: 11,
            color: isProfit ? '#3FB950' : '#F85149',
            line: { width: 2, color: isProfit ? '#3FB950' : '#F85149' },
          },
          hovertemplate: `✕ Close #${i+1}<br>${pos.date_close}<br>$%{y:.2f}<br>${isProfit?'+':''}${(pos.pct_profit||0).toFixed(2)}%<extra></extra>`,
          showlegend: false,
        });
      }
    });

    console.log('[renderChart] traces=', traces.length, 'Plotly=', typeof Plotly);

    if (typeof Plotly === 'undefined') {
      showFlash('Plotly not loaded', 'error');
      return;
    }

    const layout = {
      paper_bgcolor: 'transparent',
      plot_bgcolor:  '#060A0E',
      font: { family: "'IBM Plex Mono', monospace", color: '#6E7681', size: 10 },
      margin: { t: 20, r: 20, b: 40, l: 65 },
      xaxis: {
        gridcolor:  '#161B22',
        linecolor:  '#21262D',
        tickformat: '%b %Y',
        tickfont:   { size: 9 },
      },
      yaxis: {
        gridcolor:  '#161B22',
        linecolor:  '#21262D',
        tickprefix: '$',
        tickfont:   { size: 9 },
      },
      hovermode: 'x unified',
      hoverlabel: {
        bgcolor:     '#161B22',
        bordercolor: '#30363D',
        font:        { family: "'IBM Plex Mono', monospace", size: 10 },
      },
    };

    document.getElementById('chartPlaceholder').style.display = 'none';
    document.getElementById('priceChart').style.display       = 'block';

    Plotly.react('priceChart', traces, layout, {
      responsive:             true,
      displayModeBar:         true,
      modeBarButtonsToRemove: ['select2d', 'lasso2d', 'resetScale2d'],
      displaylogo:            false,
    });

    console.log('[renderChart] done ✓');

  } catch (e) {
    console.error('[renderChart] ERROR', e);
    showFlash('Chart error: ' + e.message, 'error');
  }
}

/* ── Summary ─────────────────────────────────────────────────── */
function renderSummary(s) {
  const fmtMoney = v => v != null ? '$' + Number(v).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}) : '—';
  const fmtPct   = v => v != null ? (v >= 0 ? '+' : '') + Number(v).toFixed(2) + '%' : '—';

  document.getElementById('kpiInit').textContent     = fmtMoney(s.portf_init);
  document.getElementById('kpiFinal').textContent    = fmtMoney(s.portf_final);
  document.getElementById('kpiProfit').textContent   = fmtPct(s.profit_pct);
  // cagr and max_drawdown arrive as "X.XX%" strings from controller
  const parsePct = v => { if (v==null) return null; if (typeof v==='string') return parseFloat(v.replace('%','')); return v; };
  document.getElementById('kpiCagr').textContent     = fmtPct(parsePct(s.cagr));
  document.getElementById('kpiDrawdown').textContent = fmtPct(parsePct(s.max_drawdown));
  document.getElementById('kpiPositions').textContent = (s.positions || []).length;

  // Colour profit card
  const profitCard = document.getElementById('kpiProfit').closest('.kpi-card');
  profitCard.classList.toggle('neg', s.profit_pct < 0);

  // Equity curve
  if (s.daily_profits?.length) {
    const eqTrace = [{
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      name: 'Portfolio MTM',
      y: s.daily_profits,
      line: { color: '#1F6FEB', width: 1.5 },
      fillcolor: 'rgba(31,111,235,.08)',
      hovertemplate: 'MTM: $%{y:,.2f}<extra></extra>',
    }];
    const eqLayout = {
      paper_bgcolor: 'transparent',
      plot_bgcolor:  '#060A0E',
      margin: { t: 10, r: 20, b: 30, l: 70 },
      font: { family: "'IBM Plex Mono', monospace", color: '#6E7681', size: 9 },
      xaxis: { gridcolor: '#161B22', showticklabels: false },
      yaxis: { gridcolor: '#161B22', tickprefix: '$', tickformat: ',.0f', tickfont: {size:9} },
      showlegend: false,
    };
    Plotly.react('equityChart', eqTrace, eqLayout, { responsive: true, displayModeBar: false });
  }

  document.getElementById('summaryPlaceholder').style.display = 'none';
  document.getElementById('summaryContent').style.display     = 'block';
}

/* ── Positions table ─────────────────────────────────────────── */
function renderPositions(positions) {
  const tbody = document.getElementById('positionsTbody');
  if (!positions?.length) {
    tbody.innerHTML = '<tr><td colspan="9" style="padding:24px;text-align:center;color:var(--faint);font-family:var(--mono);font-size:10px">No positions</td></tr>';
  } else {
    tbody.innerHTML = positions.map((p, i) => {
      const isLong   = (p.side || '').toUpperCase() === 'LONG';
      const isProfit = (p.nom_profit || 0) >= 0;
      const dd       = typeof p.max_drawdown === 'number' ? p.max_drawdown : 0;
      return `
        <tr>
          <td class="td-pos-num">#${i + 1}</td>
          <td class="${isLong ? 'td-side-long' : 'td-side-short'}">${p.side || '—'}</td>
          <td>${p.date_open  || '—'}</td>
          <td>${p.date_close || '—'}</td>
          <td>$${fmtNum(p.price_open)}</td>
          <td>$${fmtNum(p.price_close)}</td>
          <td class="${isProfit ? 'td-profit-pos' : 'td-profit-neg'}">
            ${isProfit ? '+' : ''}$${fmtNum(p.nom_profit)}
          </td>
          <td class="${isProfit ? 'td-profit-pos' : 'td-profit-neg'}">
            ${isProfit ? '+' : ''}${(p.pct_profit || 0).toFixed(2)}%
          </td>
          <td class="td-dd">${(dd * 100).toFixed(2)}%</td>
        </tr>
      `;
    }).join('');
  }
  document.getElementById('positionsPlaceholder').style.display = 'none';
  document.getElementById('positionsContent').style.display     = 'block';
}

/* ── Last signal ─────────────────────────────────────────────── */
function renderSignal(s) {
  const signalContent = document.getElementById('signalContent');
  const signalNoData  = document.getElementById('signalNoData');
  const signalRaw     = s.last_signal;   // e.g. "LONG → LONG → LONG" or null

  if (!signalRaw) {
    // Try to reconstruct from last 3 positions if signal not in params
    const pos    = s.positions || [];
    const latest = pos[pos.length - 1];

    if (!latest) {
      signalNoData.style.display  = 'block';
      signalContent.style.display = 'none';
      return;
    }

    // Last known date is close of last position (or ongoing)
    document.getElementById('signalDate').textContent = latest.date_close || latest.date_open || '—';

    // Build arrow sequence from last 3 positions
    const last3 = pos.slice(-3);
    renderArrows(document.getElementById('signalArrows'), last3.map(p => p.side || 'FLAT'));
  } else {
    // Parse the signal string "LONG → LONG → SHORT"
    const parts = String(signalRaw).split(/\s*(?:→|->|-|>)\s*/);
    const date = s.last_signal_date
          || s.positions?.[s.positions.length - 1]?.date_close
          || '—';
    document.getElementById('signalDate').textContent = date;
    renderArrows(document.getElementById('signalArrows'), parts);
  }

  signalNoData.style.display  = 'none';
  signalContent.style.display = 'block';
}

function renderArrows(container, signals) {
  container.innerHTML = signals.map((sig, i) => {
    const cls  = sig.toUpperCase() === 'LONG'  ? 'pill-long'
               : sig.toUpperCase() === 'SHORT' ? 'pill-short'
               : 'pill-flat';
    const arrow = i < signals.length - 1 ? '<span class="signal-arrow">→</span>' : '';
    return `<div class="signal-step"><span class="signal-pill ${cls}">${sig.toUpperCase()}</span>${arrow}</div>`;
  }).join('');
}

function safeCopy(text) {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text);
  } else {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);
    return Promise.resolve();
  }
}

function copySignal() {
  if (!lastResult) return;
  const s      = lastResult.summary;
  const pos    = s.positions || [];
  const latest = pos[pos.length - 1];
  const date   = latest?.date_close || latest?.date_open || 'N/A';
  const arrows = document.getElementById('signalArrows');
  const pills  = [...arrows.querySelectorAll('.signal-pill')].map(el => el.textContent.trim());
  const text   = pills.join(' → ');

  safeCopy(text).then(() => {
    const copied = document.getElementById('signalCopied');
    copied.style.display = 'inline';
    setTimeout(() => { copied.style.display = 'none'; }, 2000);
  });
}

/* ── CRUD MODAL ──────────────────────────────────────────────── */
function openAddModal() {
  document.getElementById('mModelId').value       = '';
  document.getElementById('mModelName').value     = '';
  document.getElementById('mAlgoType').value      = 'XGBOOST';
  document.getElementById('mModelPath').value     = '';
  document.getElementById('mSymbol').value        = '';
  document.getElementById('mBias').value          = 'LONG';
  document.getElementById('mDFrom').value         = '';
  document.getElementById('mDTo').value           = '';
  document.getElementById('mLowerPct').value      = '0.3';
  document.getElementById('mNFlip').value         = '3';
  document.getElementById('mInitPortf').value     = '100000';
  document.getElementById('mTradeComm').value     = '0';
  document.getElementById('mMakeStationary').checked = true;
  document.getElementById('mSeriesCsv').value     = '';
  document.getElementById('mError').textContent   = '';
  document.getElementById('modalTitle').textContent = 'Add Model';
  document.getElementById('mSaveBtn').textContent  = 'Save';
  openModal();
}

function openEditModal() {
  const m = models.find(x => x.model_id === activeModelId);
  if (!m) return;

  document.getElementById('mModelId').value       = m.model_id;
  document.getElementById('mModelName').value     = m.model_name;
  document.getElementById('mAlgoType').value      = m.algo_type;
  document.getElementById('mModelPath').value     = m.model_path;
  document.getElementById('mSymbol').value        = m.symbol;
  document.getElementById('mBias').value          = m.bias;
  document.getElementById('mDFrom').value         = m.d_from;
  document.getElementById('mDTo').value           = m.d_to;
  document.getElementById('mLowerPct').value      = m.lower_percentile_limit;
  document.getElementById('mNFlip').value         = m.n_flip;
  document.getElementById('mInitPortf').value     = m.init_portf_size;
  document.getElementById('mTradeComm').value     = m.trade_comm;
  document.getElementById('mMakeStationary').checked = m.make_stationary;
  document.getElementById('mSeriesCsv').value     = m.series_csv;
  document.getElementById('mError').textContent   = '';
  document.getElementById('modalTitle').textContent = 'Edit Model';
  document.getElementById('mSaveBtn').textContent  = 'Update';
  openModal();
}

async function saveModel() {
  const modelId = document.getElementById('mModelId').value;
  const isEdit  = !!modelId;

  const payload = {
    model_name:             document.getElementById('mModelName').value.trim(),
    algo_type:              document.getElementById('mAlgoType').value,
    model_path:             document.getElementById('mModelPath').value.trim(),
    symbol:                 document.getElementById('mSymbol').value.trim().toUpperCase(),
    bias:                   document.getElementById('mBias').value,
    d_from:                 document.getElementById('mDFrom').value,
    d_to:                   document.getElementById('mDTo').value,
    lower_percentile_limit: parseFloat(document.getElementById('mLowerPct').value) || 0.3,
    n_flip:                 parseInt(document.getElementById('mNFlip').value)  || 3,
    init_portf_size:        parseFloat(document.getElementById('mInitPortf').value) || 100000,
    trade_comm:             parseFloat(document.getElementById('mTradeComm').value) || 0,
    make_stationary:        document.getElementById('mMakeStationary').checked,
    series_csv:             document.getElementById('mSeriesCsv').value.trim(),
  };

  const errEl = document.getElementById('mError');
  if (!payload.model_name) { errEl.textContent = 'Model name is required'; return; }
  if (!payload.model_path) { errEl.textContent = 'Model path is required';  return; }
  if (!payload.symbol)     { errEl.textContent = 'Symbol is required';      return; }
  errEl.textContent = '';

  const saveBtn = document.getElementById('mSaveBtn');
  saveBtn.disabled = true;

  if (isEdit) payload.model_id = parseInt(modelId);

  try {
    const url = isEdit ? `${BASE}/edit_model` : `${BASE}/add_model`;
    const res  = await fetch(url, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();

    if (!data.ok) {
      errEl.textContent = data.error || 'Save failed';
      return;
    }

    closeModal();
    await loadModels();
    showFlash(isEdit ? 'Model updated ✓' : 'Model added ✓', 'success');

    // Re-select the model after edit
    if (isEdit) selectModel(parseInt(modelId));
    else if (data.model_id) selectModel(data.model_id);

  } catch (e) {
    errEl.textContent = e.message;
  } finally {
    saveBtn.disabled = false;
  }
}

async function deleteModel() {
  if (!activeModelId) return;
  const m = models.find(x => x.model_id === activeModelId);
  if (!confirm(`Delete model "${m?.model_name}"?`)) return;

  try {
    const res  = await fetch(`${BASE}/delete_model`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ model_id: activeModelId }),
    });
    const data = await res.json();
    if (!data.ok) { showFlash('Delete failed: ' + data.error, 'error'); return; }

    activeModelId = null;
    document.getElementById('emptyState').style.display  = 'flex';
    document.getElementById('activePanel').style.display = 'none';
    await loadModels();
    showFlash('Model deleted', 'success');
  } catch (e) {
    showFlash('Error: ' + e.message, 'error');
  }
}

/* ── Modal helpers ───────────────────────────────────────────── */
function openModal() {
  document.getElementById('modalBackdrop').classList.add('open');
  document.getElementById('modelModal').classList.add('open');
}
function closeModal() {
  document.getElementById('modalBackdrop').classList.remove('open');
  document.getElementById('modelModal').classList.remove('open');
}

/* ── Utilities ───────────────────────────────────────────────── */
function fmtNum(v) {
  if (v == null || isNaN(v)) return '—';
  return Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function showFlash(msg, type = 'success') {
  const el      = document.createElement('div');
  el.className  = `flash ${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}