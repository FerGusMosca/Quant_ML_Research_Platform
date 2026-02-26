// global_m2_indicator.js — Seeking Bias · Global M2 Indicator

// ── Live clock ──
(function tick() {
  const el = document.getElementById('navClock');
  if (el) {
    const n = new Date(), pad = v => String(v).padStart(2, '0');
    el.textContent = [n.getHours(), n.getMinutes(), n.getSeconds()].map(pad).join(':');
  }
  setTimeout(tick, 1000);
})();

// ── Currency chip toggle ──
document.querySelectorAll('.currency-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    chip.classList.toggle('selected');
    syncHiddenSelect();
  });
});

function syncHiddenSelect() {
  const sel = document.getElementById('currencySelectHidden');
  if (!sel) return;
  Array.from(sel.options).forEach(opt => {
    const chip = document.querySelector(`.currency-chip[data-currency="${opt.value}"]`);
    opt.selected = chip ? chip.classList.contains('selected') : false;
  });
}

function selectAll() {
  document.querySelectorAll('.currency-chip').forEach(c => c.classList.add('selected'));
  syncHiddenSelect();
}

function selectNone() {
  document.querySelectorAll('.currency-chip').forEach(c => c.classList.remove('selected'));
  syncHiddenSelect();
}

// Set end_date to today by default
const todayStr = new Date().toISOString().split('T')[0];
const endDateEl = document.getElementById('end_date');
if (endDateEl) endDateEl.value = todayStr;

// Init: sync on load (all selected by default)
syncHiddenSelect();

// ── Chart instance ──
let chartInstance = null;

// ── Submit ──
document.getElementById('m2Form').addEventListener('submit', async function (e) {
  e.preventDefault();

  const selected = Array.from(document.querySelectorAll('.currency-chip.selected'))
                        .map(c => c.dataset.currency);
  if (selected.length === 0) {
    showFlash('error', 'Select at least one currency.');
    return;
  }

  const btn = document.getElementById('calcBtn');
  btn.classList.add('loading');
  btn.disabled = true;

  // Build FormData
  const fd = new FormData(this);
  // Remove stale currency entries and re-add from chips
  fd.delete('currencies');
  selected.forEach(v => fd.append('currencies', v));

  try {
    const res    = await fetch('/global_m2_indicator/calculate', { method: 'POST', body: fd });
    const result = await res.json();

    if (result.message) {
      showFlash('error', result.message);
      renderLogs(result.logs || []);
      return;
    }

    renderChart(result.dates, result.values);
    renderLogs(result.logs || []);

    // Update header meta
    const metaEl = document.getElementById('chartMeta');
    if (metaEl) {
      metaEl.textContent = `${result.dates[0]} → ${result.dates[result.dates.length - 1]}  ·  ${result.dates.length} points  ·  ${selected.length} currencies`;
    }

  } catch {
    showFlash('error', 'Error calculating indicator.');
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
});

// ── Render chart ──
function renderChart(dates, values) {
  // Hide empty state, show canvas
  document.getElementById('chartEmpty').style.display = 'none';
  document.getElementById('m2Chart').style.display    = 'block';

  if (chartInstance) chartInstance.destroy();

  const ctx = document.getElementById('m2Chart').getContext('2d');

  // Gradient fill
  const gradient = ctx.createLinearGradient(0, 0, 0, 380);
  gradient.addColorStop(0,   'rgba(31,111,235,0.22)');
  gradient.addColorStop(0.7, 'rgba(31,111,235,0.04)');
  gradient.addColorStop(1,   'rgba(31,111,235,0)');

  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [{
        label: 'Global M2 (USD)',
        data: values,
        borderColor:     '#1F6FEB',
        borderWidth:     2,
        pointRadius:     0,
        pointHoverRadius: 5,
        pointHoverBackgroundColor: '#58A6FF',
        pointHoverBorderColor:     '#E6EDF3',
        pointHoverBorderWidth:     1.5,
        fill:       true,
        backgroundColor: gradient,
        tension:    0.35,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: {
            color: '#6E7681',
            font: { family: "'IBM Plex Mono', monospace", size: 10 },
            boxWidth: 12,
          }
        },
        tooltip: {
          backgroundColor: '#0F1923',
          borderColor:     '#21262D',
          borderWidth:     1,
          titleColor:      '#C9D1D9',
          bodyColor:       '#58A6FF',
          titleFont:  { family: "'IBM Plex Mono', monospace", size: 10 },
          bodyFont:   { family: "'IBM Plex Mono', monospace", size: 12, weight: '600' },
          padding:    10,
          callbacks: {
            label: ctx => {
              const v = ctx.parsed.y;
              return '  ' + formatTrillions(v);
            }
          }
        },
      },
      scales: {
        x: {
          grid:  { color: 'rgba(31,111,235,0.06)', lineWidth: 0.5 },
          ticks: {
            color: '#6E7681', maxTicksLimit: 12,
            font: { family: "'IBM Plex Mono', monospace", size: 9 },
          },
          border: { color: '#161B22' },
        },
        y: {
          grid:  { color: 'rgba(31,111,235,0.06)', lineWidth: 0.5 },
          ticks: {
            color: '#6E7681',
            font: { family: "'IBM Plex Mono', monospace", size: 9 },
            callback: v => formatTrillions(v),
          },
          border: { color: '#161B22' },
        }
      }
    }
  });
}

function formatTrillions(v) {
  if (Math.abs(v) >= 1e12) return (v / 1e12).toFixed(2) + 'T';
  if (Math.abs(v) >= 1e9)  return (v / 1e9).toFixed(2) + 'B';
  if (Math.abs(v) >= 1e6)  return (v / 1e6).toFixed(2) + 'M';
  return v.toLocaleString('en-US', { maximumFractionDigits: 2 });
}

// ── Render logs ──
function renderLogs(logs) {
  const panel = document.getElementById('logPanel');
  const body  = document.getElementById('logBody');
  if (!panel || !body || !logs.length) return;

  panel.classList.add('visible');
  body.innerHTML = '';

  logs.forEach(line => {
    const div = document.createElement('div');
    div.className = 'log-line' +
      (line.includes('✅') ? ' ok' : line.includes('⚠') ? ' warn' : line.includes('❌') ? ' err' : '');
    div.textContent = line;
    body.appendChild(div);
  });
}

function toggleLogs() {
  const body   = document.getElementById('logBody');
  const toggle = document.getElementById('logToggle');
  const open   = body.classList.toggle('open');
  toggle.textContent = open ? '▲ hide' : '▼ show';
}

// ── Flash ──
function showFlash(type, msg) {
  document.querySelector('.flash')?.remove();
  const el = document.createElement('div');
  el.className   = 'flash ' + type;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}