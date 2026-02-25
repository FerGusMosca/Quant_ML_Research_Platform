// display_series.js — Seeking Bias · Series Analysis

// ── Live clock ──
(function tick() {
  const el = document.getElementById('navClock');
  if (el) {
    const n = new Date(), pad = v => String(v).padStart(2,'0');
    el.textContent = [n.getHours(), n.getMinutes(), n.getSeconds()].map(pad).join(':');
  }
  setTimeout(tick, 1000);
})();

// ── State ──
let currentDates  = [];
let currentValues = [];
let currentSeries = '';

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  const today    = new Date();
  const pastDate = new Date();
  pastDate.setFullYear(today.getFullYear() - 10);
  document.getElementById('start_date').value = pastDate.toISOString().split('T')[0];
  document.getElementById('end_date').value   = today.toISOString().split('T')[0];
  fetchChartData();
});

// ══ DISPLAY SERIES ══
function displaySeries() {
  const seriesKey    = document.getElementById('series_key').value.trim();
  const startDate    = document.getElementById('start_date').value;
  const endDate      = document.getElementById('end_date').value;
  const timeInterval = document.getElementById('time_interval').value;

  if (!seriesKey || !startDate || !endDate) {
    showFlash('error', 'Please enter a Series Key and select start/end dates.');
    return;
  }

  const btn = document.getElementById('displayButton');
  setBtnLoading(btn, true, 'Loading…');

  const formData = new FormData();
  formData.append('series_key', seriesKey);
  formData.append('start_date', startDate);
  formData.append('end_date', endDate);
  formData.append('time_interval', timeInterval);

  fetch('/display_series/do_display', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      currentSeries = seriesKey;
      document.getElementById('chartSeriesName').textContent = seriesKey;
      document.getElementById('chartSeriesMeta').textContent =
        `${timeInterval} · ${startDate} → ${endDate}`;
      showFlash('success', data.message || 'Series loaded.');
      fetchChartData();
    })
    .catch(() => showFlash('error', 'Error loading series.'))
    .finally(() => setBtnLoading(btn, false, '▶ Display'));
}

// ══ FETCH & DRAW CHART ══
function fetchChartData() {
  fetch('/display_series/get_chart_data')
    .then(r => r.json())
    .then(data => {
      if (!data.dates?.length || !data.values?.length) return;
      currentDates  = data.dates;
      currentValues = data.values;
      drawChart(data.dates, data.values);
      updateLastFive(data.dates, data.values);
      updateStats(data.values);
    })
    .catch(err => console.error('Error fetching chart data:', err));
}

function drawChart(dates, values) {
  const trace = {
    x: dates,
    y: values,
    mode: 'lines',
    line: { color: '#1F6FEB', width: 1.8, shape: 'spline' },
    fill: 'tozeroy',
    fillcolor: 'rgba(31,111,235,0.06)',
    hovertemplate: '<b>%{x}</b><br>Value: <b>%{y:.4f}</b><extra></extra>',
    name: currentSeries || 'Series',
  };

  const layout = {
    paper_bgcolor: 'rgba(8,12,16,0)',
    plot_bgcolor:  'rgba(8,12,16,0)',
    font: { family: "'IBM Plex Mono', monospace", color: '#6E7681', size: 10 },
    xaxis: {
      title: { text: '', standoff: 8 },
      gridcolor: 'rgba(31,111,235,0.06)', gridwidth: 0.5,
      zerolinecolor: 'rgba(31,111,235,0.1)', zerolinewidth: 0.5,
      tickfont: { size: 9, color: '#6E7681' },
      linecolor: '#161B22',
    },
    yaxis: {
      title: { text: '', standoff: 8 },
      gridcolor: 'rgba(31,111,235,0.06)', gridwidth: 0.5,
      zerolinecolor: 'rgba(31,111,235,0.1)', zerolinewidth: 0.5,
      tickfont: { size: 9, color: '#6E7681' },
      linecolor: '#161B22', autorange: true,
    },
    dragmode: 'zoom',
    margin: { t: 12, b: 36, l: 52, r: 20 },
    hoverlabel: {
      bgcolor: '#0F1923', bordercolor: '#21262D',
      font: { family: "'IBM Plex Mono', monospace", color: '#C9D1D9', size: 11 },
    },
    shapes: [],
    selections: [],
  };

  const config = {
    responsive: true, displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
    toImageButtonOptions: { format: 'png', filename: currentSeries },
  };

  Plotly.newPlot('chartContainer', [trace], layout, config);

  // Y-axis auto-range on zoom
  document.getElementById('chartContainer').on('plotly_relayout', eventData => {
    const range = eventData['xaxis.range'] || eventData['xaxis.range[0]'] && [eventData['xaxis.range[0]'], eventData['xaxis.range[1]']];
    if (range) {
      const filtered = values.filter((_, i) => dates[i] >= range[0] && dates[i] <= range[1]);
      if (filtered.length > 0) {
        Plotly.relayout('chartContainer', {
          'yaxis.range': [Math.min(...filtered) * 0.998, Math.max(...filtered) * 1.002],
        });
      }
    }
  });

  // Click on point → edit tooltip
  document.getElementById('chartContainer').on('plotly_click', eventData => {
    const pt = eventData.points?.[0];
    if (!pt) return;
    showPointTooltip(pt.x, pt.y, pt.pointIndex);
  });
}

// ══ STATS ══
function updateStats(values) {
  if (!values.length) return;
  const last   = values[values.length - 1];
  const prev   = values[values.length - 2] ?? last;
  const change = last - prev;
  const pct    = prev !== 0 ? ((change / Math.abs(prev)) * 100) : 0;
  const min    = Math.min(...values);
  const max    = Math.max(...values);

  const fmtN = n => n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 4 });

  const lastEl  = document.getElementById('statLast');
  const chEl    = document.getElementById('statChange');
  const minEl   = document.getElementById('statMin');
  const maxEl   = document.getElementById('statMax');
  const cntEl   = document.getElementById('statCount');

  if (lastEl) lastEl.textContent = fmtN(last);
  if (chEl) {
    chEl.textContent = `${change >= 0 ? '+' : ''}${fmtN(change)} (${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%)`;
    chEl.className = 'stat-val ' + (change >= 0 ? 'up' : 'down');
  }
  if (minEl) minEl.textContent = fmtN(min);
  if (maxEl) maxEl.textContent = fmtN(max);
  if (cntEl) cntEl.textContent = values.length.toLocaleString();
}

// ══ LAST 5 DATA POINTS ══
function updateLastFive(dates, values) {
  const container = document.getElementById('lastFiveList');
  if (!container) return;
  container.innerHTML = '';

  const recent = dates.map((d, i) => ({ date: d, value: values[i] }))
                      .slice(-5).reverse();

  recent.forEach(item => {
    const row = document.createElement('div');
    row.className = 'data-point-row';

    const dateSpan = document.createElement('span');
    dateSpan.className = 'dp-date';
    dateSpan.textContent = item.date;

    const inp = document.createElement('input');
    inp.type  = 'number';
    inp.step  = '0.0001';
    inp.value = item.value;
    inp.className = 'dp-input';

    const saveBtn = document.createElement('button');
    saveBtn.textContent = 'Save';
    saveBtn.className   = 'dp-save';
    saveBtn.onclick     = () => saveDataPoint(item.date, inp.value, saveBtn);

    row.append(dateSpan, inp, saveBtn);
    container.appendChild(row);
  });
}

function saveDataPoint(date, value, btn) {
  const orig = btn.textContent;
  btn.textContent = '…';
  btn.disabled    = true;

  const fd = new FormData();
  fd.append('date', date);
  fd.append('value', value);

  fetch('/display_series/add_data', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      showFlash('success', data.message || 'Saved.');
      fetchChartData();
    })
    .catch(() => showFlash('error', 'Error saving point.'))
    .finally(() => { btn.textContent = orig; btn.disabled = false; });
}

// ══ ADD DATA POINT ══
function addNewPoint() {
  const newDate  = document.getElementById('newDate').value;
  const newValue = document.getElementById('newValue').value;
  const btn      = document.getElementById('addBtn');

  if (!newDate || !newValue) { showFlash('error', 'Enter a valid date and value.'); return; }

  setBtnLoading(btn, true, '…');
  const fd = new FormData();
  fd.append('date', newDate);
  fd.append('value', newValue);

  fetch('/display_series/add_data', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      showFlash('success', data.message || 'Point added.');
      document.getElementById('newDate').value  = '';
      document.getElementById('newValue').value = '';
      fetchChartData();
    })
    .catch(() => showFlash('error', 'Error adding point.'))
    .finally(() => setBtnLoading(btn, false, 'Add'));
}

// ══ POINT TOOLTIP (click on chart) ══
function showPointTooltip(date, value, pointIndex) {
  const tt = document.getElementById('pointTooltip');
  if (!tt) return;

  document.getElementById('ttDate').textContent = date;
  const inp = document.getElementById('ttValue');
  inp.value = value;
  inp._date = date;
  inp._idx  = pointIndex;

  // Position near center of screen
  tt.classList.add('visible');
}

function closeTooltip() {
  document.getElementById('pointTooltip')?.classList.remove('visible');
}

function saveTooltipPoint() {
  const inp  = document.getElementById('ttValue');
  const date = inp._date;
  const val  = inp.value;
  if (!date || val === '') return;

  const saveBtn = document.getElementById('ttSaveBtn');
  saveBtn.textContent = '…';
  saveBtn.disabled    = true;

  const fd = new FormData();
  fd.append('date', date);
  fd.append('value', val);

  fetch('/display_series/add_data', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      showFlash('success', data.message || 'Point updated.');
      closeTooltip();
      fetchChartData();
    })
    .catch(() => showFlash('error', 'Error updating point.'))
    .finally(() => { saveBtn.textContent = 'Save'; saveBtn.disabled = false; });
}

// ══ CALCULATE SLOPE ══
function calculateSlope() {
  const slopeUnits = document.getElementById('slope_units').value;
  const newValue   = document.getElementById('new_value').value;
  const btn        = document.getElementById('calcSlopeBtn');

  if (!slopeUnits || slopeUnits < 1) { showFlash('error', 'Enter a valid integer for Slope.'); return; }

  setCalcLoading(btn, true);

  const fd = new FormData();
  fd.append('slope_units', slopeUnits);
  if (newValue) fd.append('new_value', newValue);

  fetch('/display_series/calculate_new_slope', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      renderSlopeResult(data.slope);
    })
    .catch(() => showFlash('error', 'Error calculating slope.'))
    .finally(() => setCalcLoading(btn, false));
}

function renderSlopeResult(slope) {
  const card = document.getElementById('slopeResultCard');
  const val  = document.getElementById('slopeResultValue');
  const fill = document.getElementById('slopeBarFill');
  const note = document.getElementById('slopeBarNote');

  card.classList.add('visible');

  const fmt = n => (n >= 0 ? '+' : '') + Number(n).toFixed(5);
  val.textContent = fmt(slope);

  const cls = slope > 0 ? 'positive' : slope < 0 ? 'negative' : 'neutral';
  val.className   = 'slope-result-value ' + cls;
  fill.className  = 'slope-bar-fill ' + (slope > 0 ? 'positive' : 'negative');

  // Bar width: clamp abs slope to 0-100%
  const maxAbsSlope = 10;
  const pct = Math.min(Math.abs(slope) / maxAbsSlope * 100, 100);
  fill.style.width = pct + '%';

  const interpretation = slope > 0.5  ? '↑ Strong uptrend'
                       : slope > 0.05 ? '↑ Mild uptrend'
                       : slope > 0    ? '→ Flat / slight up'
                       : slope > -0.05? '→ Flat / slight down'
                       : slope > -0.5 ? '↓ Mild downtrend'
                       :                '↓ Strong downtrend';
  note.textContent = interpretation;
}

// ══ TOGGLE SARIMA ══
function toggleSarima() {
  const useSarima = document.getElementById('use_sarima').checked;
  document.getElementById('arima_s').disabled = !useSarima;
  if (!useSarima) document.getElementById('arima_s').value = '';
}

// ══ CALCULATE ARIMA ══
function calculateArima() {
  const p  = document.getElementById('arima_p').value;
  const d  = document.getElementById('arima_d').value;
  const q  = document.getElementById('arima_q').value;
  const s  = document.getElementById('arima_s').value;
  const fp = document.getElementById('forecast_periods').value;
  const useSarima = document.getElementById('use_sarima').checked;
  const btn = document.getElementById('calcArimaBtn');

  if (!p || p < 0) { showFlash('error', 'p must be ≥ 0'); return; }
  if (!d || d < 0) { showFlash('error', 'd must be ≥ 0'); return; }
  if (!q || q < 0) { showFlash('error', 'q must be ≥ 0'); return; }
  if (useSarima && (!s || s < 1)) { showFlash('error', 's must be ≥ 1 for SARIMA'); return; }
  if (!fp || fp < 1) { showFlash('error', 'Forecast periods must be ≥ 1'); return; }

  setCalcLoading(btn, true);

  const fd = new FormData();
  fd.append('p', p); fd.append('d', d); fd.append('q', q);
  if (useSarima && s) fd.append('s', s);
  fd.append('forecast_periods', fp);

  fetch('/display_series/calculate_arima', { method: 'POST', body: fd })
    .then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); })
    .then(data => renderArimaResult(data.result))
    .catch(() => showFlash('error', 'Error calculating ARIMA.'))
    .finally(() => setCalcLoading(btn, false));
}

function renderArimaResult(result) {
  if (!result || !Array.isArray(result)) {
    showFlash('error', 'No predictions returned.'); return;
  }

  const preds   = result.map(v => parseFloat(v));
  const list    = document.getElementById('arimaPredictionsList');
  const verdict = document.getElementById('arimaVerdict');
  list.innerHTML = '';
  list.classList.add('visible');

  const maxAbs = Math.max(...preds.map(Math.abs), 0.0001);

  preds.forEach((v, i) => {
    const item = document.createElement('div');
    item.className = 'pred-item';

    const cls = v >= 0 ? 'pos' : 'neg';
    const pct = (Math.abs(v) / maxAbs * 100).toFixed(1);
    const fmt = v >= 0
      ? '+' + v.toExponential(3)
      : v.toExponential(3);

    item.innerHTML = `
      <span class="pred-idx">Pred ${i + 1}</span>
      <div class="pred-bar-wrap">
        <div class="pred-bar-track">
          <div class="pred-bar-fill ${cls}" style="width:${pct}%"></div>
        </div>
      </div>
      <span class="pred-val ${cls}">${fmt}</span>`;
    list.appendChild(item);
  });

  // Verdict
  verdict.classList.add('visible');
  const negIdx = preds.findIndex(v => v < 0);
  if (negIdx !== -1) {
    verdict.className  = 'arima-verdict visible inversion';
    verdict.textContent = `⚠ Possible trend inversion after Pred ${negIdx + 1}`;
  } else if (preds.every(v => v >= 0)) {
    verdict.className  = 'arima-verdict visible stable';
    verdict.textContent = '✓ No trend inversion detected — sustained uptrend';
  } else if (preds.every(v => v <= 0)) {
    verdict.className  = 'arima-verdict visible inversion';
    verdict.textContent = '↓ All predictions negative — sustained downtrend';
  } else {
    verdict.className  = 'arima-verdict visible inconclusive';
    verdict.textContent = '~ Trend analysis inconclusive';
  }
}

// ══ HELPERS ══
function setBtnLoading(btn, loading, label) {
  if (!btn) return;
  btn.disabled   = loading;
  btn.textContent = label;
}

function setCalcLoading(btn, loading) {
  if (!btn) return;
  btn.disabled = loading;
  btn.classList.toggle('running', loading);
}

function showFlash(type, msg) {
  const existing = document.querySelector('.flash');
  if (existing) existing.remove();

  const el = document.createElement('div');
  el.className = 'flash ' + type;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}