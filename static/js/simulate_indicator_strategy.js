// simulate_indicator_strategy.js — Seeking Bias · Simulate Strategy

// ── Live clock ──
(function tick() {
  const el = document.getElementById('navClock');
  if (el) {
    const n = new Date(), pad = v => String(v).padStart(2, '0');
    el.textContent = [n.getHours(), n.getMinutes(), n.getSeconds()].map(pad).join(':');
  }
  setTimeout(tick, 1000);
})();

// ── Default dates ──
document.addEventListener('DOMContentLoaded', () => {
  const today     = new Date();
  const tenYrsAgo = new Date();
  tenYrsAgo.setFullYear(today.getFullYear() - 10);
  document.getElementById('start_date').value = tenYrsAgo.toISOString().split('T')[0];
  document.getElementById('end_date').value   = today.toISOString().split('T')[0];
  fetchChartData();
});

// ── File drop zone ──
const dropZone  = document.getElementById('fileDropZone');
const fileInput = document.getElementById('custom_etf_file');
const fileLabel = document.getElementById('fileSelectedName');

dropZone?.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone?.addEventListener('dragleave', ()  => dropZone.classList.remove('dragover'));
dropZone?.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    showFileName(e.dataTransfer.files[0].name);
  }
});
fileInput?.addEventListener('change', () => {
  if (fileInput.files.length) showFileName(fileInput.files[0].name);
});
function showFileName(name) {
  if (fileLabel) { fileLabel.textContent = '📄 ' + name; fileLabel.style.display = 'block'; }
}

// ── Simulate ──
function simulateIndicator() {
  const file           = fileInput?.files[0];
  const indicator      = document.getElementById('indicator_name').value.trim();
  const d_from         = document.getElementById('start_date').value;
  const d_to           = document.getElementById('end_date').value;
  const trading_algo   = document.getElementById('trading_algo').value.trim();
  const portf_size     = document.getElementById('portf_size').value;
  const commissions    = document.getElementById('commissions').value;
  const min_units      = document.getElementById('min_units_to_pred').value;
  const slope_units    = document.getElementById('slope_units').value;

  if (!file)        { showFlash('error', 'Please select a CSV file.'); return; }
  if (!indicator)   { showFlash('error', 'Indicator Name is required.'); return; }
  if (!d_from || !d_to) { showFlash('error', 'Select start and end dates.'); return; }

  const fd = new FormData();
  fd.append('file',              file);
  fd.append('indicator',         indicator);
  fd.append('d_from',            d_from);
  fd.append('d_to',              d_to);
  fd.append('trading_algo',      trading_algo);
  fd.append('portf_size',        portf_size);
  fd.append('comm',              commissions);
  fd.append('min_units_to_pred', min_units);
  fd.append('slope_units',       slope_units);

  const btn    = document.getElementById('simulateButton');
  const result = document.getElementById('simulateResult');
  btn.classList.add('loading');
  btn.disabled     = true;
  result.className = 'sim-result';

  fetch('/simulate_indicator_strategy/simulate_indicator', { method: 'POST', body: fd })
    .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
    .then(data => {
      result.textContent = '✓ ' + data.message;
      result.className   = 'sim-result success';
      fetchChartData();
    })
    .catch(err => {
      result.textContent = '✕ Simulation failed: ' + err.message;
      result.className   = 'sim-result error';
    })
    .finally(() => { btn.classList.remove('loading'); btn.disabled = false; });
}

// ── Chart ──
function drawChart(dates, values) {
  document.getElementById('chartEmpty').style.display    = 'none';
  document.getElementById('chartContainer').style.display = 'block';

  // Colour the line green if final > initial, red if not
  const up = values[values.length - 1] >= values[0];

  const trace = {
    x: dates, y: values,
    mode: 'lines',
    name: 'Portfolio MTM',
    line: { color: up ? '#3FB950' : '#F85149', width: 2 },
    fill: 'tozeroy',
    fillcolor: up ? 'rgba(63,185,80,0.07)' : 'rgba(248,81,73,0.06)',
    hovertemplate: '<b>%{x}</b><br>MTM: <b>%{y:,.2f}</b><extra></extra>',
  };

  const layout = {
    paper_bgcolor: 'rgba(8,12,16,0)',
    plot_bgcolor:  'rgba(8,12,16,0)',
    font: { family: "'IBM Plex Mono', monospace", color: '#6E7681', size: 10 },
    xaxis: {
      gridcolor: 'rgba(31,111,235,0.06)', gridwidth: 0.5,
      zerolinecolor: 'rgba(31,111,235,0.1)',
      tickfont: { size: 9, color: '#6E7681' }, linecolor: '#161B22',
    },
    yaxis: {
      gridcolor: 'rgba(31,111,235,0.06)', gridwidth: 0.5,
      zerolinecolor: 'rgba(31,111,235,0.1)',
      tickfont: { size: 9, color: '#6E7681' }, linecolor: '#161B22', autorange: true,
      tickformat: ',.0f',
    },
    dragmode: 'zoom',
    margin: { t: 12, b: 36, l: 68, r: 20 },
    hoverlabel: {
      bgcolor: '#0F1923', bordercolor: '#21262D',
      font: { family: "'IBM Plex Mono', monospace", color: '#C9D1D9', size: 11 },
    },
  };

  const config = { responsive: true, displayModeBar: true, displaylogo: false };
  Plotly.newPlot('chartContainer', [trace], layout, config);

  // Auto Y-range on zoom
  document.getElementById('chartContainer').on('plotly_relayout', ev => {
    const r = ev['xaxis.range'] || (ev['xaxis.range[0]'] && [ev['xaxis.range[0]'], ev['xaxis.range[1]']]);
    if (r) {
      const filt = values.filter((_, i) => dates[i] >= r[0] && dates[i] <= r[1]);
      if (filt.length) Plotly.relayout('chartContainer', {
        'yaxis.range': [Math.min(...filt) * 0.998, Math.max(...filt) * 1.002],
      });
    }
  });

  updateStats(values, dates);
}

function updateStats(values, dates) {
  if (!values.length) return;
  const first  = values[0];
  const last   = values[values.length - 1];
  const change = last - first;
  const pct    = first !== 0 ? (change / Math.abs(first)) * 100 : 0;
  const maxVal = Math.max(...values);
  const minVal = Math.min(...values);
  const fmt    = n => n.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
  const fmtPct = n => (n >= 0 ? '+' : '') + n.toFixed(2) + '%';

  setStatEl('statFinal',  fmt(last));
  setStatEl('statReturn', fmtPct(pct), pct >= 0 ? 'up' : 'down');
  setStatEl('statMax',    fmt(maxVal));
  setStatEl('statMin',    fmt(minVal));
  setStatEl('statPts',    values.length.toString());
}

function setStatEl(id, text, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  if (cls) el.className = 'stat-val ' + cls;
}

function fetchChartData() {
  fetch('/simulate_indicator_strategy/get_chart_data')
    .then(r => r.json())
    .then(data => {
      if (!data.dates?.length || !data.values?.length) return;
      drawChart(data.dates, data.values);
    })
    .catch(err => console.error('Error fetching chart data:', err));
}

// ── Flash ──
function showFlash(type, msg) {
  document.querySelector('.flash')?.remove();
  const el = document.createElement('div');
  el.className = 'flash ' + type;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4500);
}