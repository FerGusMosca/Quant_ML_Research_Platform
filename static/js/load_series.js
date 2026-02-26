// load_series.js — Seeking Bias · Load Series

// ── Live clock ──
(function tick() {
  const el = document.getElementById('navClock');
  if (el) {
    const n = new Date(), pad = v => String(v).padStart(2, '0');
    el.textContent = [n.getHours(), n.getMinutes(), n.getSeconds()].map(pad).join(':');
  }
  setTimeout(tick, 1000);
})();

// ── File drop zone ──
const dropZone = document.getElementById('fileDropZone');
const fileInput = document.getElementById('series_file');
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

// ── Add Days stepper ──
function stepDays(delta) {
  const inp = document.getElementById('add_days');
  const val = parseInt(inp.value) || 1;
  const next = Math.min(31, Math.max(1, val + delta));
  inp.value = next;
}

// ── Upload ──
function uploadSeries() {
  const file      = fileInput?.files[0];
  const seriesKey = document.getElementById('series_key').value.trim();
  const addDays   = parseInt(document.getElementById('add_days').value, 10);

  if (!file)            { showFlash('error', 'Please select a CSV file.'); return; }
  if (!seriesKey)       { showFlash('error', 'Please enter a Series Key.'); return; }
  if (isNaN(addDays) || addDays < 1 || addDays > 31) {
    showFlash('error', 'Add Days must be between 1 and 31.'); return;
  }

  const fd = new FormData();
  fd.append('file',       file);
  fd.append('series_key', seriesKey);
  fd.append('add_days',   addDays);

  const btn    = document.getElementById('uploadButton');
  const result = document.getElementById('uploadResult');
  btn.classList.add('loading');
  btn.disabled     = true;
  result.className = 'upload-result';

  fetch('/load_series/upload_series', { method: 'POST', body: fd })
    .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
    .then(data => {
      result.textContent = '✓ ' + data.message;
      result.className   = 'upload-result success';
      fetchChartData();
    })
    .catch(err => {
      result.textContent = '✕ Upload failed: ' + err.message;
      result.className   = 'upload-result error';
    })
    .finally(() => { btn.classList.remove('loading'); btn.disabled = false; });
}

// ── Chart ──
function drawChart(dates, values) {
  document.getElementById('chartEmpty').style.display    = 'none';
  document.getElementById('chartContainer').style.display = 'block';

  const trace = {
    x: dates, y: values,
    mode: 'lines',
    name: 'Series',
    line: { color: '#1F6FEB', width: 2 },
    fill: 'tozeroy',
    fillcolor: 'rgba(31,111,235,0.07)',
    hovertemplate: '<b>%{x}</b><br>Value: <b>%{y:.4f}</b><extra></extra>',
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
    },
    dragmode: 'zoom',
    margin: { t: 12, b: 36, l: 52, r: 20 },
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

  // Update chart meta
  const meta = document.getElementById('chartMeta');
  if (meta && dates.length) {
    meta.textContent = `${dates[0]} → ${dates[dates.length - 1]}  ·  ${dates.length} points`;
  }
}

function fetchChartData() {
  fetch('/load_series/get_chart_data')
    .then(r => r.json())
    .then(data => {
      if (!data.dates?.length || !data.values?.length) return;
      drawChart(data.dates, data.values);
    })
    .catch(err => console.error('Error fetching chart data:', err));
}

// ── Init ──
document.addEventListener('DOMContentLoaded', fetchChartData);

// ── Flash ──
function showFlash(type, msg) {
  document.querySelector('.flash')?.remove();
  const el = document.createElement('div');
  el.className = 'flash ' + type;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4500);
}