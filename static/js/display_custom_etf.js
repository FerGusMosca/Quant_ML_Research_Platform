// display_custom_etf.js — Seeking Bias · Custom ETF

// ── Live clock ──
(function tick() {
  const el = document.getElementById('navClock');
  if (el) {
    const n = new Date(), pad = v => String(v).padStart(2, '0');
    el.textContent = [n.getHours(), n.getMinutes(), n.getSeconds()].map(pad).join(':');
  }
  setTimeout(tick, 1000);
})();

// ── Default date range (today & 10 years ago) ──
document.addEventListener('DOMContentLoaded', () => {
  const today     = new Date();
  const tenYrsAgo = new Date();
  tenYrsAgo.setFullYear(today.getFullYear() - 10);
  document.getElementById('start_date').value = tenYrsAgo.toISOString().split('T')[0];
  document.getElementById('end_date').value   = today.toISOString().split('T')[0];
  fetchChartData();
});

// ── File drop zone ──
const dropZone = document.getElementById('fileDropZone');
const fileInput = document.getElementById('custom_etf_file');
const fileLabel = document.getElementById('fileSelectedName');

dropZone?.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone?.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
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
  if (fileLabel) {
    fileLabel.textContent = '📄 ' + name;
    fileLabel.style.display = 'block';
  }
}

// ── Moving average toggle ──
function toggleMovingAvg() {
  const on  = document.getElementById('enable_mavg').checked;
  const row = document.getElementById('mavgRow');
  const inp = document.getElementById('mavg_period');
  row.classList.toggle('visible', on);
  inp.disabled = !on;
  if (!on) { inp.value = ''; hideError('mavg_error'); }
}

// ── Save as symbol toggle ──
function toggleSaveAsSymbol() {
  const on  = document.getElementById('save_as_symbol').checked;
  document.getElementById('symbolRow').classList.toggle('visible', on);
  if (!on) hideError('save_symbol_error');
}

// ── Validation helpers ──
function showError(id, msg) {
  const el = document.getElementById(id);
  if (el) { el.textContent = msg; el.classList.add('visible'); }
}
function hideError(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove('visible');
}

// ── Upload ──
function uploadETF() {
  const fileInp = document.getElementById('custom_etf_file');
  if (!fileInp.files.length) { showFlash('error', 'Please select a CSV file.'); return; }

  const startDate = document.getElementById('start_date').value;
  const endDate   = document.getElementById('end_date').value;
  if (!startDate || !endDate) { showFlash('error', 'Select start and end dates.'); return; }

  const mavgEnabled = document.getElementById('enable_mavg').checked;
  let mavgVal = '';
  if (mavgEnabled) {
    mavgVal = document.getElementById('mavg_period').value;
    if (!mavgVal || parseInt(mavgVal) <= 0) {
      showError('mavg_error', '* Moving average period must be a positive integer.');
      return;
    }
    hideError('mavg_error');
  }

  const saveAs = document.getElementById('save_as_symbol').checked;
  let symbol = '', base = '1';
  if (saveAs) {
    symbol = (document.getElementById('symbol_input').value || '').trim();
    base   = document.getElementById('base_input').value;
    if (!symbol || !base || parseFloat(base) <= 0) {
      showError('save_symbol_error', '* Symbol is required and base must be > 0.');
      return;
    }
    hideError('save_symbol_error');
  }

  const fd = new FormData();
  fd.append('file',           fileInp.files[0]);
  fd.append('start_date',     startDate);
  fd.append('end_date',       endDate);
  fd.append('moving_avg',     mavgEnabled ? mavgVal : '');
  fd.append('save_as_symbol', saveAs ? 'true' : 'false');
  fd.append('symbol',         symbol);
  fd.append('base',           base);

  const btn    = document.getElementById('uploadButton');
  const result = document.getElementById('uploadResult');
  btn.classList.add('loading');
  btn.disabled    = true;
  result.className = 'upload-result';
  result.style.display = 'none';

  fetch('/display_custom_etf/upload_custom_etf', { method: 'POST', body: fd })
    .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
    .then(data => {
      result.textContent = '✓ ' + data.message;
      result.className = 'upload-result success';
      fetchChartData();
    })
    .catch(err => {
      result.textContent = '✕ Upload failed: ' + err.message;
      result.className = 'upload-result error';
    })
    .finally(() => {
      btn.classList.remove('loading');
      btn.disabled = false;
    });
}

// ── Chart ──
let chartInstance = null;

function drawChart(dates, values, movingAvg = []) {
  document.getElementById('chartEmpty').style.display   = 'none';
  document.getElementById('chartContainer').style.display = 'block';

  const traces = [{
    x: dates, y: values,
    mode: 'lines',
    name: 'ETF Value',
    line: { color: '#1F6FEB', width: 2 },
    fill: 'tozeroy',
    fillcolor: 'rgba(31,111,235,0.07)',
    hovertemplate: '<b>%{x}</b><br>Value: <b>%{y:.4f}</b><extra></extra>',
  }];

  if (movingAvg?.length) {
    traces.push({
      x: dates, y: movingAvg,
      mode: 'lines',
      name: 'Moving Avg',
      line: { color: '#D29922', width: 1.5, dash: 'dot' },
      hovertemplate: '<b>%{x}</b><br>MA: <b>%{y:.4f}</b><extra></extra>',
    });
  }

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
    },
    legend: {
      font: { family: "'IBM Plex Mono', monospace", size: 10, color: '#6E7681' },
      bgcolor: 'rgba(13,17,23,0.8)', bordercolor: '#21262D', borderwidth: 1,
    },
    dragmode: 'zoom',
    margin: { t: 12, b: 36, l: 52, r: 20 },
    hoverlabel: {
      bgcolor: '#0F1923', bordercolor: '#21262D',
      font: { family: "'IBM Plex Mono', monospace", color: '#C9D1D9', size: 11 },
    },
  };

  const config = { responsive: true, displayModeBar: true, displaylogo: false };

  Plotly.newPlot('chartContainer', traces, layout, config);

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

  // Update header meta
  const meta = document.getElementById('chartMeta');
  if (meta && dates.length) {
    meta.textContent = `${dates[0]} → ${dates[dates.length - 1]}  ·  ${dates.length} points`;
  }
}

function fetchChartData() {
  fetch('/display_custom_etf/get_chart_data')
    .then(r => r.json())
    .then(data => {
      if (!data.dates?.length || !data.values?.length) return;
      drawChart(data.dates, data.values, data.moving_avg);
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