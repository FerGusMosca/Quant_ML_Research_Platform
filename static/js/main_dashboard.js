// main_dashboard.js — Seeking Bias · Management Dashboard

// ── Live clock ──
(function tick() {
  const n = new Date();
  const pad = v => String(v).padStart(2, '0');
  const el = document.getElementById('liveClock');
  if (el) el.textContent = [n.getHours(), n.getMinutes(), n.getSeconds()].map(pad).join(':');
  setTimeout(tick, 1000);
})();

// ── Mark active sidebar link ──
document.querySelectorAll('.nav-item').forEach(a => {
  try {
    const p = new URL(a.href).pathname;
    if (p !== '/' && window.location.pathname.startsWith(p)) a.classList.add('active');
  } catch (e) {}
});

// ── FRED via local proxy (Flask) — evita CORS ──

// ── LEI Card definitions ──
const LEI_CARDS = [
  {
    title:    'G7 Composite Leading Indicator',
    source:   'OECD · Amplitude Adjusted · Monthly · FRED',
    fredId:   'G7LOLITOAASTSAM',
    metricValId:   'm-g7',
    metricDeltaId: 'd-g7',
    canvasId: 'chart-g7',
    status:   'slowing',
    oecdUrl:  'https://data-explorer.oecd.org/vis?lc=en&vw=tl&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_STES%40DF_CLI&df[ag]=OECD.SDD.STES&dq=G-7.M.LI...AA...H&to[TIME_PERIOD]=false&pd=2010-01%2C',
  },
  {
    title:    'United States CLI',
    source:   'OECD · Amplitude Adjusted · Monthly · FRED',
    fredId:   'USALOLITOAASTSAM',
    metricValId:   'm-us',
    metricDeltaId: 'd-us',
    canvasId: 'chart-us',
    status:   'slowing',
    oecdUrl:  'https://data-explorer.oecd.org/vis?lc=en&vw=tl&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_STES%40DF_CLI&df[ag]=OECD.SDD.STES&dq=USA.M.LI...AA...H&to[TIME_PERIOD]=false&pd=2010-01%2C',
  },
  {
    title:    'Germany CLI',
    source:   'OECD · Amplitude Adjusted · Monthly · FRED',
    fredId:   'DEULOLITOAASTSAM',
    metricValId:   'm-eu',
    metricDeltaId: 'd-eu',
    canvasId: 'chart-de',
    status:   'slowing',
    oecdUrl:  'https://data-explorer.oecd.org/vis?lc=en&vw=tl&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_STES%40DF_CLI&df[ag]=OECD.SDD.STES&dq=DEU.M.LI...AA...H&to[TIME_PERIOD]=false&pd=2010-01%2C',
  },
  {
    title:    'China CLI (Industrial Sector)',
    source:   'OECD · Amplitude Adjusted · Monthly · FRED',
    fredId:   'CHNLOLITOAASTSAM',
    metricValId:   'm-cn',
    metricDeltaId: 'd-cn',
    canvasId: 'chart-cn',
    status:   'slowing',
    oecdUrl:  'https://data-explorer.oecd.org/vis?lc=en&vw=tl&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_STES%40DF_CLI&df[ag]=OECD.SDD.STES&dq=CHN.M.LI...AA...H&to[TIME_PERIOD]=false&pd=2010-01%2C',
  },
  {
    title:    'Brazil CLI',
    source:   'OECD · Amplitude Adjusted · Monthly · FRED',
    fredId:   'BRALOLITOAASTSAM',
    metricValId:   null,
    metricDeltaId: null,
    canvasId: 'chart-br',
    status:   'slowing',
    oecdUrl:  'https://data-explorer.oecd.org/vis?lc=en&vw=tl&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_STES%40DF_CLI&df[ag]=OECD.SDD.STES&dq=BRA.M.LI...AA...H&to[TIME_PERIOD]=false&pd=2010-01%2C',
  },
  {
    title:    'Japan CLI',
    source:   'OECD · Amplitude Adjusted · Monthly · FRED',
    fredId:   'JPNLOLITOAASTSAM',
    metricValId:   null,
    metricDeltaId: null,
    canvasId: 'chart-jp',
    status:   'slowing',
    oecdUrl:  'https://data-explorer.oecd.org/vis?lc=en&vw=tl&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_STES%40DF_CLI&df[ag]=OECD.SDD.STES&dq=JPN.M.LI...AA...H&to[TIME_PERIOD]=false&pd=2010-01%2C',
  },
  {
    title:    'United Kingdom CLI',
    source:   'OECD · Amplitude Adjusted · Monthly · FRED',
    fredId:   'GBRLOLITOAASTSAM',
    metricValId:   null,
    metricDeltaId: null,
    canvasId: 'chart-uk',
    status:   'slowing',
    oecdUrl:  'https://data-explorer.oecd.org/vis?lc=en&vw=tl&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_STES%40DF_CLI&df[ag]=OECD.SDD.STES&dq=GBR.M.LI...AA...H&to[TIME_PERIOD]=false&pd=2010-01%2C',
  },
];

const BADGE_LABEL = { expanding: 'Expanding', slowing: 'Monitor', contracting: 'Contracting' };
const STATUS_COLOR = { expanding: '#3FB950', slowing: '#D29922', contracting: '#F85149' };

// ── Build LEI grid from JS (replaces static HTML cards) ──
function buildLeiGrid() {
  const grid = document.getElementById('leiGrid');
  if (!grid) return;
  grid.innerHTML = ''; // clear placeholders / static cards

  LEI_CARDS.forEach(card => {
    const el = document.createElement('div');
    el.className = 'lei-card';
    el.innerHTML = `
      <div class="lei-hdr">
        <div>
          <div class="lei-title">${esc(card.title)}</div>
          <div class="lei-source">${esc(card.source)}</div>
        </div>
        <div class="lei-hdr-right">
          <span class="lei-badge ${card.status}">${BADGE_LABEL[card.status]}</span>
          <a class="lei-oecd-btn" href="${card.oecdUrl}" target="_blank" rel="noopener">
            OECD ↗
          </a>
        </div>
      </div>
      <div class="lei-chart-wrap">
        <div class="lei-chart-meta">
          <span class="lei-chart-val" id="cv-${card.canvasId}">—</span>
          <span class="lei-chart-delta flat" id="cd-${card.canvasId}">Loading…</span>
        </div>
        <canvas class="lei-canvas" id="${card.canvasId}" height="220"></canvas>
        <div class="lei-chart-footer">
          <span class="lei-chart-src">Source: FRED · OECD</span>
          <span class="lei-chart-period" id="cp-${card.canvasId}"></span>
        </div>
      </div>`;
    grid.appendChild(el);
  });

  // Add placeholders
  for (let i = 0; i < 2; i++) {
    const ph = document.createElement('div');
    ph.className = 'lei-placeholder';
    ph.onclick = openModal;
    ph.innerHTML = `<div class="lei-plus">+</div><span class="lei-placeholder-label">Add LEI</span>`;
    grid.appendChild(ph);
  }
}

// ── Fetch FRED JSON API + draw chart ──
async function loadLeiCard(card) {
  try {
    // Llama al proxy Flask local — sin CORS
    const url = `/api/fred/${card.fredId}?start=2010-01-01`;
    const resp = await fetch(url);
    const json = await resp.json();
    const rows = (json.observations || [])
      .filter(o => o.value !== '.')
      .map(o => ({ date: o.date, val: parseFloat(o.value) }))
      .filter(r => !isNaN(r.val));

    if (rows.length < 2) return;

    // Show last 5 years (~60 months)
    const recent = rows.slice(-72);
    const val    = recent[recent.length - 1].val;
    const prev   = recent[recent.length - 2].val;
    const diff   = val - prev;

    // Update top metric cards
    const mVal = document.getElementById(card.metricValId);
    const mDelta = document.getElementById(card.metricDeltaId);
    if (mVal)   mVal.textContent   = val.toFixed(2);
    if (mDelta) {
      mDelta.textContent = `${diff >= 0 ? '▲' : '▼'} ${Math.abs(diff).toFixed(3)}`;
      mDelta.className   = `m-delta ${diff >= 0 ? 'up' : 'down'}`;
    }

    // Update in-card value
    const cvEl = document.getElementById(`cv-${card.canvasId}`);
    const cdEl = document.getElementById(`cd-${card.canvasId}`);
    const cpEl = document.getElementById(`cp-${card.canvasId}`);
    if (cvEl) cvEl.textContent = val.toFixed(2);
    if (cdEl) {
      cdEl.textContent = `${diff >= 0 ? '▲' : '▼'} ${Math.abs(diff).toFixed(3)} MoM`;
      cdEl.className   = `lei-chart-delta ${diff >= 0 ? 'up' : 'down'}`;
    }
    if (cpEl) {
      const first = recent[0].date.slice(0, 7);
      const last  = recent[recent.length - 1].date.slice(0, 7);
      cpEl.textContent = `${first} → ${last}`;
    }

    // Draw canvas chart
    drawChart(card.canvasId, recent, card.status);

  } catch (e) {
    const cdEl = document.getElementById(`cd-${card.canvasId}`);
    if (cdEl) { cdEl.textContent = 'Data unavailable'; cdEl.className = 'lei-chart-delta flat'; }
    const mDelta = document.getElementById(card.metricDeltaId);
    if (mDelta) mDelta.textContent = '—';
  }
}

// ── Canvas chart renderer ──
function drawChart(canvasId, rows, status) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  // Responsive width
  canvas.width = canvas.parentElement.offsetWidth || 600;
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const vals   = rows.map(r => r.val);
  const minVal = Math.min(...vals);
  const maxVal = Math.max(...vals);
  const range  = maxVal - minVal || 1;

  const PAD_L = 48, PAD_R = 16, PAD_T = 16, PAD_B = 28;
  const chartW = W - PAD_L - PAD_R;
  const chartH = H - PAD_T - PAD_B;

  const xOf = i => PAD_L + (i / (rows.length - 1)) * chartW;
  const yOf = v => PAD_T + chartH - ((v - minVal) / range) * chartH;

  const lineColor = STATUS_COLOR[status] || '#1F6FEB';

  // ── Grid lines ──
  ctx.strokeStyle = 'rgba(33,38,45,0.8)';
  ctx.lineWidth = 1;
  const gridSteps = 4;
  for (let i = 0; i <= gridSteps; i++) {
    const y = PAD_T + (chartH / gridSteps) * i;
    ctx.beginPath(); ctx.moveTo(PAD_L, y); ctx.lineTo(W - PAD_R, y); ctx.stroke();
    // Y labels
    const labelVal = maxVal - (range / gridSteps) * i;
    ctx.fillStyle = '#3D444D';
    ctx.font = '9px IBM Plex Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(labelVal.toFixed(1), PAD_L - 4, y + 3);
  }

  // 100 reference line
  if (minVal < 100 && maxVal > 100) {
    const y100 = yOf(100);
    ctx.strokeStyle = 'rgba(110,118,129,0.35)';
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD_L, y100); ctx.lineTo(W - PAD_R, y100); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#3D444D';
    ctx.font = '9px IBM Plex Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText('100', PAD_L - 4, y100 + 3);
  }

  // ── X-axis year labels ──
  ctx.fillStyle = '#3D444D';
  ctx.font = '9px IBM Plex Mono, monospace';
  ctx.textAlign = 'center';
  const years = [...new Set(rows.map(r => r.date.slice(0, 4)))];
  years.forEach(yr => {
    const idx = rows.findIndex(r => r.date.startsWith(yr));
    if (idx >= 0) {
      const x = xOf(idx);
      ctx.fillText(yr, x, H - 6);
    }
  });

  // ── Gradient fill ──
  const grad = ctx.createLinearGradient(0, PAD_T, 0, PAD_T + chartH);
  grad.addColorStop(0, hexAlpha(lineColor, 0.22));
  grad.addColorStop(1, hexAlpha(lineColor, 0.01));

  ctx.beginPath();
  rows.forEach((r, i) => {
    const x = xOf(i), y = yOf(r.val);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.lineTo(xOf(rows.length - 1), PAD_T + chartH);
  ctx.lineTo(xOf(0), PAD_T + chartH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // ── Main line ──
  ctx.beginPath();
  rows.forEach((r, i) => {
    const x = xOf(i), y = yOf(r.val);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = lineColor;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.stroke();

  // ── Last point dot ──
  const lx = xOf(rows.length - 1), ly = yOf(rows[rows.length - 1].val);
  ctx.beginPath();
  ctx.arc(lx, ly, 4, 0, Math.PI * 2);
  ctx.fillStyle = lineColor;
  ctx.fill();
  ctx.strokeStyle = '#0D1117';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function hexAlpha(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Modal ──
function openModal()  { document.getElementById('addModal').classList.add('open'); }
function closeModal() { document.getElementById('addModal').classList.remove('open'); }

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('addModal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeModal();
  });
});

// ── Add custom LEI card ──
function addCard() {
  const title  = document.getElementById('mTitle').value.trim();
  const source = document.getElementById('mSource').value.trim();
  const fredId = document.getElementById('mFredId').value.trim();
  const oecdUrl = document.getElementById('mOecdUrl').value.trim();
  const status = document.getElementById('mStatus').value;

  document.getElementById('mTitle').style.borderColor  = title  ? '' : '#F85149';
  document.getElementById('mFredId').style.borderColor = fredId ? '' : '#F85149';
  if (!title || !fredId) return;

  const canvasId = 'chart-custom-' + Date.now();
  const card = {
    title, source: source || 'Custom', fredId,
    metricValId: null, metricDeltaId: null,
    canvasId, status,
    oecdUrl: oecdUrl || '#',
  };

  const el = document.createElement('div');
  el.className = 'lei-card';
  el.style.animation = 'fade-up 0.3s ease both';
  el.innerHTML = `
    <div class="lei-hdr">
      <div>
        <div class="lei-title">${esc(title)}</div>
        <div class="lei-source">${esc(card.source)}</div>
      </div>
      <div class="lei-hdr-right">
        <span class="lei-badge ${status}">${BADGE_LABEL[status]}</span>
        ${oecdUrl ? `<a class="lei-oecd-btn" href="${esc(oecdUrl)}" target="_blank" rel="noopener">OECD ↗</a>` : ''}
      </div>
    </div>
    <div class="lei-chart-wrap">
      <div class="lei-chart-meta">
        <span class="lei-chart-val" id="cv-${canvasId}">—</span>
        <span class="lei-chart-delta flat" id="cd-${canvasId}">Loading…</span>
      </div>
      <canvas class="lei-canvas" id="${canvasId}" height="220"></canvas>
      <div class="lei-chart-footer">
        <span class="lei-chart-src">Source: FRED</span>
        <span class="lei-chart-period" id="cp-${canvasId}"></span>
      </div>
    </div>`;

  const grid = document.getElementById('leiGrid');
  const ph   = grid.querySelector('.lei-placeholder');
  if (ph) { grid.insertBefore(el, ph); ph.remove(); }
  else    { grid.appendChild(el); }

  loadLeiCard(card);
  closeModal();

  ['mTitle', 'mSource', 'mFredId', 'mOecdUrl'].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.value = ''; el.style.borderColor = ''; }
  });
}

function esc(t) {
  const d = document.createElement('div'); d.textContent = t; return d.innerHTML;
}

// ── Init ──
buildLeiGrid();
LEI_CARDS.forEach(loadLeiCard);

// Redraw charts on resize
window.addEventListener('resize', () => {
  LEI_CARDS.forEach(card => {
    const canvas = document.getElementById(card.canvasId);
    if (canvas && canvas._lastRows) drawChart(card.canvasId, canvas._lastRows, card.status);
  });
});