/**
 * simulate_model.js
 * Model Runner — XGBoost backtesting + PCA indicator tool.
 */

'use strict';

/* ── State ──────────────────────────────────────────────────── */
let models        = [];
let activeModelId = null;
let lastResult    = null;

let pcaModels   = [];
let activePcaId = null;

const BASE = '/simulate_model';

/* ── Boot ───────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  loadModels();
  loadPcaModels();
});

/* ══════════════════════════════════════════════════════════════
   LEFT PANEL TAB SWITCHER
   ══════════════════════════════════════════════════════════════ */

function switchLeftTab(which) {
  // which: 'xgb' | 'pca'
  document.getElementById('ltabXgb').classList.toggle('active', which === 'xgb');
  document.getElementById('ltabPca').classList.toggle('active', which === 'pca');
  document.getElementById('leftXgbPane').style.display = which === 'xgb' ? 'flex' : 'none';
  document.getElementById('leftPcaPane').style.display = which === 'pca' ? 'flex' : 'none';
}

/* ══════════════════════════════════════════════════════════════
   RIGHT PANEL SWITCHER
   ══════════════════════════════════════════════════════════════ */

function showPanel(which) {
  // which: 'empty' | 'xgb' | 'pca'
  document.getElementById('emptyState').style.display     = which === 'empty' ? 'flex'  : 'none';
  document.getElementById('activePanel').style.display    = which === 'xgb'   ? 'block' : 'none';
  document.getElementById('pcaActivePanel').style.display = which === 'pca'   ? 'block' : 'none';
}

/* ══════════════════════════════════════════════════════════════
   XGBOOST MODELS
   ══════════════════════════════════════════════════════════════ */

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

function selectModel(modelId) {
  activeModelId = modelId;
  activePcaId   = null;

  const m = models.find(x => x.model_id === modelId);
  if (!m) return;

  renderModelList();
  renderPcaModelList();
  showPanel('xgb');

  document.getElementById('hdrModelName').textContent = m.model_name;
  document.getElementById('hdrModelMeta').textContent =
    `${m.symbol} · ${m.algo_type} · ${m.d_from} → ${m.d_to}`;

  document.getElementById('runDFrom').value = m.d_from;
  document.getElementById('runDTo').value   = m.d_to;
  document.getElementById('runNFlip').value = m.n_flip;
  document.getElementById('runBias').value  = m.bias;

  resetXgbResults();
  switchTab('chart');
}

function resetXgbResults() {
  lastResult = null;
  document.getElementById('chartPlaceholder').style.display    = 'block';
  document.getElementById('priceChart').style.display          = 'none';
  document.getElementById('summaryPlaceholder').style.display  = 'block';
  document.getElementById('summaryContent').style.display      = 'none';
  document.getElementById('positionsPlaceholder').style.display = 'block';
  document.getElementById('positionsContent').style.display    = 'none';
  document.getElementById('signalNoData').style.display        = 'block';
  document.getElementById('signalContent').style.display       = 'none';
}

/* ── XGBoost tab switching ───────────────────────────────────── */
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.tab === name);
  });
  document.querySelectorAll('.tab-pane').forEach(p => {
    p.classList.toggle('active', p.id === `tab-${name}`);
  });
}

/* ── Run XGBoost ─────────────────────────────────────────────── */
async function runModel() {
  if (!activeModelId) return;

  const btn    = document.getElementById('btnRun');
  btn.disabled = true;
  btn.classList.add('running');

  const payload = {
    model_id: activeModelId,
    d_from:   document.getElementById('runDFrom').value,
    d_to:     document.getElementById('runDTo').value,
    n_flip:   parseInt(document.getElementById('runNFlip').value),
    bias:     document.getElementById('runBias').value,
  };

  try {
    const res  = await fetch(`${BASE}/run_model`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.ok) { showFlash('Run failed: ' + (data.error || 'unknown'), 'error'); return; }

    lastResult = data;
    const m    = models.find(x => x.model_id === activeModelId);
    renderSummary(data.summary);
    renderPositions(data.summary.positions);
    renderSignal(data.summary);
    switchTab('chart');
    await renderChart(data.symbol, data.d_from, data.d_to, data.summary.positions);
    showFlash(`${m?.model_name} completed ✓`, 'success');
  } catch (e) {
    showFlash('Error: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.classList.remove('running');
  }
}

/* ── XGBoost Chart ───────────────────────────────────────────── */
async function renderChart(symbol, dFrom, dTo, positions) {
  try {
    const res  = await fetch(`${BASE}/prices?symbol=${symbol}&d_from=${dFrom}&d_to=${dTo}`);
    const data = await res.json();
    if (!data.ok || !data.prices?.length) { showFlash('Could not load price data', 'error'); return; }

    const prices = data.prices.filter(p => p.close != null);
    if (!prices.length) { showFlash('No valid price data', 'error'); return; }

    const dates  = prices.map(p => p.date);
    const closes = prices.map(p => p.close);

    const inPos = new Set();
    (positions || []).forEach(pos => {
      if (!pos.date_open || !pos.date_close) return;
      dates.forEach(d => { if (d >= pos.date_open && d <= pos.date_close) inPos.add(d); });
    });

    const traces = [];
    const pushSeg = (sd, sc, isIn) => {
      if (!sd.length) return;
      traces.push({
        type: 'scatter', mode: 'lines', x: [...sd], y: [...sc],
        line: { color: isIn ? '#3FB950' : '#4A7FBF', width: isIn ? 2.5 : 1.5 },
        hovertemplate: '%{x}<br>$%{y:.2f}<extra></extra>', showlegend: false,
      });
    };

    let sd = [], sc = [], sIn = inPos.has(dates[0]);
    dates.forEach((d, i) => {
      const isIn = inPos.has(d);
      if (isIn !== sIn) {
        sd.push(d); sc.push(closes[i]);
        pushSeg(sd, sc, sIn);
        sd = [d]; sc = [closes[i]]; sIn = isIn;
      } else { sd.push(d); sc.push(closes[i]); }
    });
    pushSeg(sd, sc, sIn);

    (positions || []).forEach((pos, i) => {
      if (!pos.date_open) return;
      const isLong = (pos.side || '').toUpperCase() === 'LONG';
      const isProf = (pos.nom_profit || 0) >= 0;
      traces.push({
        type: 'scatter', mode: 'markers', x: [pos.date_open], y: [pos.price_open],
        marker: { symbol: isLong ? 'triangle-up' : 'triangle-down', size: 11, color: '#3FB950', line: { width: 1, color: '#238636' } },
        hovertemplate: `▶ Open #${i+1} ${pos.side||''}<br>${pos.date_open}<br>$%{y:.2f}<extra></extra>`,
        showlegend: false,
      });
      if (pos.date_close) {
        traces.push({
          type: 'scatter', mode: 'markers', x: [pos.date_close], y: [pos.price_close],
          marker: { symbol: 'x', size: 11, color: isProf ? '#3FB950' : '#F85149', line: { width: 2, color: isProf ? '#3FB950' : '#F85149' } },
          hovertemplate: `✕ Close #${i+1}<br>${pos.date_close}<br>$%{y:.2f}<br>${isProf?'+':''}${(pos.pct_profit||0).toFixed(2)}%<extra></extra>`,
          showlegend: false,
        });
      }
    });

    const layout = {
      paper_bgcolor: 'transparent', plot_bgcolor: '#060A0E',
      font: { family: "'IBM Plex Mono', monospace", color: '#6E7681', size: 10 },
      margin: { t: 20, r: 20, b: 40, l: 65 },
      xaxis: { gridcolor: '#161B22', linecolor: '#21262D', tickformat: '%b %Y', tickfont: { size: 9 } },
      yaxis: { gridcolor: '#161B22', linecolor: '#21262D', tickprefix: '$', tickfont: { size: 9 } },
      hovermode: 'x unified',
      hoverlabel: { bgcolor: '#161B22', bordercolor: '#30363D', font: { family: "'IBM Plex Mono', monospace", size: 10 } },
    };

    document.getElementById('chartPlaceholder').style.display = 'none';
    document.getElementById('priceChart').style.display       = 'block';
    Plotly.react('priceChart', traces, layout, {
      responsive: true, displayModeBar: true,
      modeBarButtonsToRemove: ['select2d', 'lasso2d', 'resetScale2d'], displaylogo: false,
    });
  } catch (e) {
    showFlash('Chart error: ' + e.message, 'error');
  }
}

/* ── Summary ─────────────────────────────────────────────────── */
function renderSummary(s) {
  const fmtMoney = v => v != null ? '$' + Number(v).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}) : '—';
  const fmtPct   = v => v != null ? (v >= 0 ? '+' : '') + Number(v).toFixed(2) + '%' : '—';
  const parsePct = v => v == null ? null : typeof v === 'string' ? parseFloat(v.replace('%','')) : v;

  document.getElementById('kpiInit').textContent      = fmtMoney(s.portf_init);
  document.getElementById('kpiFinal').textContent     = fmtMoney(s.portf_final);
  document.getElementById('kpiProfit').textContent    = fmtPct(s.profit_pct);
  document.getElementById('kpiCagr').textContent      = fmtPct(parsePct(s.cagr));
  document.getElementById('kpiDrawdown').textContent  = fmtPct(parsePct(s.max_drawdown));
  document.getElementById('kpiPositions').textContent = (s.positions || []).length;
  document.getElementById('kpiProfit').closest('.kpi-card').classList.toggle('neg', s.profit_pct < 0);

  if (s.daily_profits?.length) {
    Plotly.react('equityChart', [{
      type: 'scatter', mode: 'lines', fill: 'tozeroy',
      y: s.daily_profits,
      line: { color: '#1F6FEB', width: 1.5 }, fillcolor: 'rgba(31,111,235,.08)',
      hovertemplate: 'MTM: $%{y:,.2f}<extra></extra>',
    }], {
      paper_bgcolor: 'transparent', plot_bgcolor: '#060A0E',
      margin: { t: 10, r: 20, b: 30, l: 70 },
      font: { family: "'IBM Plex Mono', monospace", color: '#6E7681', size: 9 },
      xaxis: { gridcolor: '#161B22', showticklabels: false },
      yaxis: { gridcolor: '#161B22', tickprefix: '$', tickformat: ',.0f', tickfont: {size:9} },
      showlegend: false,
    }, { responsive: true, displayModeBar: false });
  }

  document.getElementById('summaryPlaceholder').style.display = 'none';
  document.getElementById('summaryContent').style.display     = 'block';
}

/* ── Positions ───────────────────────────────────────────────── */
function renderPositions(positions) {
  const tbody = document.getElementById('positionsTbody');
  if (!positions?.length) {
    tbody.innerHTML = '<tr><td colspan="9" style="padding:24px;text-align:center;color:var(--faint);font-family:var(--mono);font-size:10px">No positions</td></tr>';
  } else {
    tbody.innerHTML = positions.map((p, i) => {
      const isLong = (p.side || '').toUpperCase() === 'LONG';
      const isProf = (p.nom_profit || 0) >= 0;
      const dd     = typeof p.max_drawdown === 'number' ? p.max_drawdown : 0;
      return `<tr>
        <td class="td-pos-num">#${i+1}</td>
        <td class="${isLong ? 'td-side-long' : 'td-side-short'}">${p.side || '—'}</td>
        <td>${p.date_open  || '—'}</td>
        <td>${p.date_close || '—'}</td>
        <td>$${fmtNum(p.price_open)}</td>
        <td>$${fmtNum(p.price_close)}</td>
        <td class="${isProf ? 'td-profit-pos' : 'td-profit-neg'}">${isProf?'+':''}$${fmtNum(p.nom_profit)}</td>
        <td class="${isProf ? 'td-profit-pos' : 'td-profit-neg'}">${isProf?'+':''}${(p.pct_profit||0).toFixed(2)}%</td>
        <td class="td-dd">${(dd*100).toFixed(2)}%</td>
      </tr>`;
    }).join('');
  }
  document.getElementById('positionsPlaceholder').style.display = 'none';
  document.getElementById('positionsContent').style.display     = 'block';
}

/* ── Signal ──────────────────────────────────────────────────── */
function renderSignal(s) {
  const noData  = document.getElementById('signalNoData');
  const content = document.getElementById('signalContent');
  const raw     = s.last_signal;

  if (!raw) {
    const pos = s.positions || [];
    const lat = pos[pos.length - 1];
    if (!lat) { noData.style.display = 'block'; content.style.display = 'none'; return; }
    document.getElementById('signalDate').textContent = lat.date_close || lat.date_open || '—';
    renderArrows(document.getElementById('signalArrows'), pos.slice(-3).map(p => p.side || 'FLAT'));
  } else {
    const parts = String(raw).split(/\s*(?:→|->|-|>)\s*/);
    const date  = s.last_signal_date || s.positions?.[s.positions.length-1]?.date_close || '—';
    document.getElementById('signalDate').textContent = date;
    renderArrows(document.getElementById('signalArrows'), parts);
  }
  noData.style.display  = 'none';
  content.style.display = 'block';
}

function renderArrows(container, signals) {
  container.innerHTML = signals.map((sig, i) => {
    const cls   = sig.toUpperCase() === 'LONG' ? 'pill-long' : sig.toUpperCase() === 'SHORT' ? 'pill-short' : 'pill-flat';
    const arrow = i < signals.length - 1 ? '<span class="signal-arrow">→</span>' : '';
    return `<div class="signal-step"><span class="signal-pill ${cls}">${sig.toUpperCase()}</span>${arrow}</div>`;
  }).join('');
}

function safeCopy(text) {
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
  const ta = document.createElement('textarea');
  ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
  document.body.appendChild(ta); ta.focus(); ta.select();
  document.execCommand('copy'); document.body.removeChild(ta);
  return Promise.resolve();
}

function copySignal() {
  if (!lastResult) return;
  const pills = [...document.getElementById('signalArrows').querySelectorAll('.signal-pill')].map(el => el.textContent.trim());
  safeCopy(pills.join(' → ')).then(() => {
    const el = document.getElementById('signalCopied');
    el.style.display = 'inline';
    setTimeout(() => { el.style.display = 'none'; }, 2000);
  });
}

/* ── XGBoost CRUD ────────────────────────────────────────────── */
function openAddModal() {
  document.getElementById('mModelId').value          = '';
  document.getElementById('mModelName').value        = '';
  document.getElementById('mAlgoType').value         = 'XGBOOST';
  document.getElementById('mModelPath').value        = '';
  document.getElementById('mSymbol').value           = '';
  document.getElementById('mBias').value             = 'LONG';
  document.getElementById('mDFrom').value            = '';
  document.getElementById('mDTo').value              = '';
  document.getElementById('mLowerPct').value         = '0.3';
  document.getElementById('mNFlip').value            = '3';
  document.getElementById('mInitPortf').value        = '100000';
  document.getElementById('mTradeComm').value        = '0';
  document.getElementById('mMakeStationary').checked = true;
  document.getElementById('mSeriesCsv').value        = '';
  document.getElementById('mError').textContent      = '';
  document.getElementById('modalTitle').textContent  = 'Add Model';
  document.getElementById('mSaveBtn').textContent    = 'Save';
  openModal();
}

function openEditModal() {
  const m = models.find(x => x.model_id === activeModelId);
  if (!m) return;
  document.getElementById('mModelId').value          = m.model_id;
  document.getElementById('mModelName').value        = m.model_name;
  document.getElementById('mAlgoType').value         = m.algo_type;
  document.getElementById('mModelPath').value        = m.model_path;
  document.getElementById('mSymbol').value           = m.symbol;
  document.getElementById('mBias').value             = m.bias;
  document.getElementById('mDFrom').value            = m.d_from;
  document.getElementById('mDTo').value              = m.d_to;
  document.getElementById('mLowerPct').value         = m.lower_percentile_limit;
  document.getElementById('mNFlip').value            = m.n_flip;
  document.getElementById('mInitPortf').value        = m.init_portf_size;
  document.getElementById('mTradeComm').value        = m.trade_comm;
  document.getElementById('mMakeStationary').checked = m.make_stationary;
  document.getElementById('mSeriesCsv').value        = m.series_csv;
  document.getElementById('mError').textContent      = '';
  document.getElementById('modalTitle').textContent  = 'Edit Model';
  document.getElementById('mSaveBtn').textContent    = 'Update';
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
    n_flip:                 parseInt(document.getElementById('mNFlip').value) || 3,
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
  if (isEdit) payload.model_id = parseInt(modelId);
  const saveBtn = document.getElementById('mSaveBtn');
  saveBtn.disabled = true;
  try {
    const url  = isEdit ? `${BASE}/edit_model` : `${BASE}/add_model`;
    const res  = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await res.json();
    if (!data.ok) { errEl.textContent = data.error || 'Save failed'; return; }
    closeModal();
    await loadModels();
    showFlash(isEdit ? 'Model updated ✓' : 'Model added ✓', 'success');
    if (isEdit) selectModel(parseInt(modelId));
    else if (data.model_id) selectModel(data.model_id);
  } catch (e) { errEl.textContent = e.message; }
  finally { saveBtn.disabled = false; }
}

async function deleteModel() {
  if (!activeModelId) return;
  const m = models.find(x => x.model_id === activeModelId);
  if (!confirm(`Delete model "${m?.model_name}"?`)) return;
  try {
    const res  = await fetch(`${BASE}/delete_model`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model_id: activeModelId }) });
    const data = await res.json();
    if (!data.ok) { showFlash('Delete failed: ' + data.error, 'error'); return; }
    activeModelId = null;
    showPanel('empty');
    await loadModels();
    showFlash('Model deleted', 'success');
  } catch (e) { showFlash('Error: ' + e.message, 'error'); }
}

function openModal()  { document.getElementById('modalBackdrop').classList.add('open'); document.getElementById('modelModal').classList.add('open'); }
function closeModal() { document.getElementById('modalBackdrop').classList.remove('open'); document.getElementById('modelModal').classList.remove('open'); }

function handleBackdropClick() {
  if (document.getElementById('modelModal').classList.contains('open')) closeModal();
  if (document.getElementById('pcaModal').classList.contains('open'))   closePcaModal();
}

/* ══════════════════════════════════════════════════════════════
   PCA MODELS
   ══════════════════════════════════════════════════════════════ */

async function loadPcaModels() {
  try {
    const res  = await fetch(`${BASE}/pca_models`);
    const data = await res.json();
    pcaModels  = data;
    renderPcaModelList();
  } catch (e) {
    console.warn('Failed to load PCA models:', e.message);
  }
}

function renderPcaModelList() {
  const el = document.getElementById('pcaModelList');
  if (!el) return;
  if (!pcaModels.length) {
    el.innerHTML = '<div class="model-list-empty">No PCA indicators yet</div>';
    return;
  }
  el.innerHTML = pcaModels.map(m => `
    <div class="pca-list-card ${m.model_id === activePcaId ? 'active' : ''}"
         onclick="selectPcaModel(${m.model_id})">
      <div class="pca-list-card-inner">
        <div class="pca-list-card-name">${m.model_name}</div>
        <div class="pca-list-card-meta">${m.symbol} · ${m.d_from?.slice(0,7)} → ${m.d_to?.slice(0,7)}</div>
      </div>
      <span class="pca-list-badge">PCA</span>
    </div>
  `).join('');
}

function selectPcaModel(modelId) {
  activePcaId   = modelId;
  activeModelId = null;

  const m = pcaModels.find(x => x.model_id === modelId);
  if (!m) return;

  renderModelList();
  renderPcaModelList();

  // Auto-switch left tab to PCA so the active card is visible
  switchLeftTab('pca');
  showPanel('pca');

  document.getElementById('pcaHdrName').textContent = m.model_name;
  document.getElementById('pcaHdrMeta').textContent =
    `Output: ${m.symbol}` +
    `  ·  ${m.d_from} → ${m.d_to}` +
    ((m.bias && m.bias !== 'NONE') ? `  ·  Benchmark: ${m.bias}` : '') +
    (m.series_csv ? `  ·  ${m.series_csv.split(',').length} indicators` : '');

  document.getElementById('pcaDFrom').value     = m.d_from;
  document.getElementById('pcaDTo').value       = m.d_to;
  document.getElementById('pcaBenchmark').value = (m.bias && m.bias !== 'NONE') ? m.bias : '';

  // Reset result area
  document.getElementById('pcaResultArea').style.display  = 'none';
  document.getElementById('pcaChartWrap').style.display   = 'none';
  document.getElementById('pcaPlaceholder').style.display = 'block';
}

/* ── Run PCA ─────────────────────────────────────────────────── */
async function runPcaModel() {
  if (!activePcaId) return;

  const btn    = document.getElementById('btnRunPca');
  btn.disabled = true;
  btn.classList.add('running');

  const payload = {
    model_id:  activePcaId,
    d_from:    document.getElementById('pcaDFrom').value,
    d_to:      document.getElementById('pcaDTo').value,
    benchmark: document.getElementById('pcaBenchmark').value.trim() || null,
  };

  try {
    const res  = await fetch(`${BASE}/run_pca`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!data.ok) {
      showFlash('PCA run failed: ' + (data.error || 'unknown'), 'error');
      return;
    }

    const m = pcaModels.find(x => x.model_id === activePcaId);

    // Status bar
    document.getElementById('pcaOutputSymbol').textContent = data.output_symbol || m?.symbol || '—';
    document.getElementById('pcaRowCount').textContent     = data.rows_persisted ?? '—';
    document.getElementById('pcaPeriod').textContent       = `${data.d_from} → ${data.d_to}`;
    const st = document.getElementById('pcaStatus');
    st.textContent = '✓ Persisted';
    st.className   = 'pca-status-value pca-ok';
    document.getElementById('pcaResultArea').style.display  = 'flex';
    document.getElementById('pcaPlaceholder').style.display = 'none';

    // Plotly chart
    if (data.pca_series?.length) {
      renderPcaChart(data.pca_series, data.benchmark_series, data.output_symbol, data.benchmark_symbol);
    }

    showFlash(`PCA "${m?.model_name}" computed & persisted ✓`, 'success');
  } catch (e) {
    showFlash('PCA error: ' + e.message, 'error');
    const st = document.getElementById('pcaStatus');
    st.textContent = '✕ Error';
    st.className   = 'pca-status-value pca-err';
    document.getElementById('pcaResultArea').style.display = 'flex';
  } finally {
    btn.disabled = false;
    btn.classList.remove('running');
  }
}

/* ── PCA Plotly chart ────────────────────────────────────────── */
function renderPcaChart(pcaSeries, benchmarkSeries, outputSymbol, benchmarkSymbol) {
  // pcaSeries: [{date, value}, ...]
  // benchmarkSeries: [{date, value}, ...] or null

  const traces = [];

  // PCA trace — primary y-axis
  traces.push({
    type: 'scatter',
    mode: 'lines',
    name: outputSymbol || 'PCA',
    x: pcaSeries.map(p => p.date),
    y: pcaSeries.map(p => p.value),
    line: { color: '#1F6FEB', width: 2 },
    yaxis: 'y',
    // %{x|%Y-%m-%d} → full date in tooltip (e.g. 2025-04-03, not "Apr 2025")
    hovertemplate: `%{x|%Y-%m-%d}<br>${outputSymbol}: %{y:.4f}<extra></extra>`,
  });

  // Benchmark trace — secondary y-axis (if present)
  if (benchmarkSeries?.length) {
    traces.push({
      type: 'scatter',
      mode: 'lines',
      name: benchmarkSymbol || 'Benchmark',
      x: benchmarkSeries.map(p => p.date),
      y: benchmarkSeries.map(p => p.value),
      line: { color: '#F0A500', width: 1.5, dash: 'dot' },
      yaxis: 'y2',
      hovertemplate: `%{x|%Y-%m-%d}<br>${benchmarkSymbol}: %{y:.4f}<extra></extra>`,
    });
  }

  // Compute a symmetric padding of 15% around the PCA range so the blue line
  // isn't cramped against the left axis wall.
  let yMin = null, yMax = null;
  pcaSeries.forEach(p => {
    if (p.value == null) return;
    if (yMin === null || p.value < yMin) yMin = p.value;
    if (yMax === null || p.value > yMax) yMax = p.value;
  });
  const yPad  = yMin !== null ? (yMax - yMin) * 0.15 : 0;
  const yRange = yMin !== null ? [yMin - yPad, yMax + yPad] : undefined;

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor:  '#060A0E',
    font: { family: "'IBM Plex Mono', monospace", color: '#6E7681', size: 10 },
    // r:90 gives breathing room next to the right-hand benchmark axis label
    margin: { t: 24, r: benchmarkSeries?.length ? 90 : 40, b: 40, l: 65 },
    legend: {
      orientation: 'h', x: 0, y: 1.08,
      font: { family: "'IBM Plex Mono', monospace", size: 9, color: '#C9D1D9' },
    },
    xaxis: {
      gridcolor: '#161B22', linecolor: '#21262D',
      tickformat: '%b %Y', tickfont: { size: 9 },
    },
    yaxis: {
      gridcolor: '#161B22', linecolor: '#21262D',
      title: { text: outputSymbol, font: { size: 9, color: '#1F6FEB' } },
      tickfont: { size: 9 },
      // explicit range with padding so the line has vertical breathing room
      ...(yRange ? { range: yRange } : {}),
    },
    hovermode: 'x unified',
    hoverlabel: { bgcolor: '#161B22', bordercolor: '#30363D', font: { family: "'IBM Plex Mono', monospace", size: 10 } },
  };

  // Add secondary axis only if benchmark exists
  if (benchmarkSeries?.length) {
    layout.yaxis2 = {
      overlaying: 'y',
      side: 'right',
      gridcolor: 'transparent',
      linecolor: '#21262D',
      title: { text: benchmarkSymbol, font: { size: 9, color: '#F0A500' } },
      tickfont: { size: 9, color: '#F0A500' },
    };
  }

  document.getElementById('pcaChartWrap').style.display = 'block';
  Plotly.react('pcaChart', traces, layout, {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['select2d', 'lasso2d', 'resetScale2d'],
    displaylogo: false,
  });
}

/* ── Delete PCA ──────────────────────────────────────────────── */
async function deletePcaModel() {
  if (!activePcaId) return;
  const m = pcaModels.find(x => x.model_id === activePcaId);
  if (!confirm(`Delete PCA model "${m?.model_name}"?`)) return;
  try {
    const res  = await fetch(`${BASE}/delete_pca_model`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: activePcaId }),
    });
    const data = await res.json();
    if (!data.ok) { showFlash('Delete failed: ' + data.error, 'error'); return; }
    activePcaId = null;
    showPanel('empty');
    await loadPcaModels();
    showFlash('PCA model deleted', 'success');
  } catch (e) { showFlash('Error: ' + e.message, 'error'); }
}

/* ── PCA CRUD Modal ──────────────────────────────────────────── */
function openAddPcaModal() {
  document.getElementById('pcaModelId').value     = '';
  document.getElementById('pcaModelName').value   = '';
  document.getElementById('pcaOutputSym').value   = '';
  document.getElementById('pcaMDFrom').value      = '';
  document.getElementById('pcaMDTo').value        = '';
  document.getElementById('pcaMBenchmark').value  = '';
  document.getElementById('pcaSeriesCsv').value   = '';
  document.getElementById('pcaError').textContent = '';
  document.getElementById('pcaModalTitle').textContent = 'Add PCA Indicator';
  document.getElementById('pcaSaveBtn').textContent    = 'Save';
  openPcaModal();
}

function openEditPcaModalActive() {
  if (!activePcaId) return;
  const m = pcaModels.find(x => x.model_id === activePcaId);
  if (!m) return;
  document.getElementById('pcaModelId').value     = m.model_id;
  document.getElementById('pcaModelName').value   = m.model_name;
  document.getElementById('pcaOutputSym').value   = m.symbol;
  document.getElementById('pcaMDFrom').value      = m.d_from;
  document.getElementById('pcaMDTo').value        = m.d_to;
  document.getElementById('pcaMBenchmark').value  = (m.bias && m.bias !== 'NONE') ? m.bias : '';
  document.getElementById('pcaSeriesCsv').value   = m.series_csv || '';
  document.getElementById('pcaError').textContent = '';
  document.getElementById('pcaModalTitle').textContent = 'Edit PCA Indicator';
  document.getElementById('pcaSaveBtn').textContent    = 'Update';
  openPcaModal();
}

async function savePcaModel() {
  const modelId   = document.getElementById('pcaModelId').value;
  const isEdit    = !!modelId;
  const name      = document.getElementById('pcaModelName').value.trim();
  const outSym    = document.getElementById('pcaOutputSym').value.trim().toUpperCase();
  const dFrom     = document.getElementById('pcaMDFrom').value;
  const dTo       = document.getElementById('pcaMDTo').value;
  const benchmark = document.getElementById('pcaMBenchmark').value.trim().toUpperCase();
  const seriesCsv = document.getElementById('pcaSeriesCsv').value.trim();
  const errEl     = document.getElementById('pcaError');

  if (!name)      { errEl.textContent = 'Model name is required';         return; }
  if (!outSym)    { errEl.textContent = 'Output symbol is required';      return; }
  if (!dFrom)     { errEl.textContent = 'Start date is required';         return; }
  if (!dTo)       { errEl.textContent = 'End date is required';           return; }
  if (!seriesCsv) { errEl.textContent = 'At least one indicator required'; return; }
  errEl.textContent = '';

  const payload = {
    model_name: name, algo_type: 'PCA', model_path: '',
    symbol: outSym, bias: benchmark || 'NONE',
    d_from: dFrom, d_to: dTo,
    lower_percentile_limit: 0, n_flip: 1,
    make_stationary: false, init_portf_size: 0, trade_comm: 0,
    series_csv: seriesCsv, display_order: 0,
  };
  if (isEdit) payload.model_id = parseInt(modelId);

  const saveBtn = document.getElementById('pcaSaveBtn');
  saveBtn.disabled = true;
  try {
    const url  = isEdit ? `${BASE}/edit_pca_model` : `${BASE}/add_pca_model`;
    const res  = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await res.json();
    if (!data.ok) { errEl.textContent = data.error || 'Save failed'; return; }
    closePcaModal();
    await loadPcaModels();
    showFlash(isEdit ? 'PCA indicator updated ✓' : 'PCA indicator added ✓', 'success');
    if (data.model_id) selectPcaModel(data.model_id);
    else if (isEdit)   selectPcaModel(parseInt(modelId));
  } catch (e) { errEl.textContent = e.message; }
  finally { saveBtn.disabled = false; }
}

function openPcaModal()  { document.getElementById('modalBackdrop').classList.add('open'); document.getElementById('pcaModal').classList.add('open'); }
function closePcaModal() { document.getElementById('modalBackdrop').classList.remove('open'); document.getElementById('pcaModal').classList.remove('open'); }

/* ══════════════════════════════════════════════════════════════
   UTILITIES
   ══════════════════════════════════════════════════════════════ */

function fmtNum(v) {
  if (v == null || isNaN(v)) return '—';
  return Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function showFlash(msg, type = 'success') {
  const el = document.createElement('div');
  el.className = `flash ${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}