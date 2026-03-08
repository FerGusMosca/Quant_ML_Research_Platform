// chunk_management.js

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
let _currentCollection = 'zh_chunks';

// Page history: each entry is { from_order_value, label }
// Entry 0 = page 1 (no cursor), entry 1 = page 2, etc.
let _pageHistory   = [];   // stack of from_order_value cursors for pages 2, 3, ...
let _nextCursor    = null; // from_order_value to pass for the NEXT page
let _currentPage   = 1;

let _detailPointId = null;
let _allSources    = [];
let _filterTimer   = null;

// ── Boot ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  refreshAll();
});

function refreshAll() {
  loadCollectionStats(_currentCollection);
  resetPagination();
  reloadChunks();
}

// ── Collection switch ─────────────────────────────────────────────────────
function switchCollection(col) {
  _currentCollection = col;
  document.querySelectorAll('.coll-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.col === col)
  );
  loadCollectionStats(col);
  resetPagination();

  const activeTab = document.querySelector('.tab-btn.active')?.dataset.tab;
  if (activeTab === 'browse')  reloadChunks();
  if (activeTab === 'sources') loadSourceSummary();
  if (activeTab === 'runs')    loadIngestRuns();
}

// ── Tab switch ────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === name)
  );
  document.querySelectorAll('.tab-pane').forEach(p =>
    p.classList.toggle('active', p.id === 'tab-' + name)
  );
  if (name === 'sources') loadSourceSummary();
  if (name === 'runs')    loadIngestRuns();
}

// ── Collection stats ──────────────────────────────────────────────────────
async function loadCollectionStats(col) {
  document.getElementById('statPoints').textContent  = '…';
  document.getElementById('statVectors').textContent = '…';
  document.getElementById('statStatus').textContent  = '…';
  try {
    const data = await apiFetch(`/chunk_management/collection_info?collection=${col}`);
    document.getElementById('statPoints').textContent  = (data.points_count  || 0).toLocaleString();
    document.getElementById('statVectors').textContent = (data.indexed_vectors_count || 0).toLocaleString();
    const statusEl = document.getElementById('statStatus');
    statusEl.textContent = data.status || '—';
    statusEl.style.color = data.status === 'green' ? 'var(--green)' :
                           data.status === 'yellow' ? 'var(--orange)' : 'var(--text)';
  } catch(e) {
    document.getElementById('statPoints').textContent = 'ERR';
  }
}

// ════════════════════════════════════════════════════
// TAB 1 — BROWSE CHUNKS
// ════════════════════════════════════════════════════

function resetPagination() {
  _pageHistory = [];
  _nextCursor  = null;
  _currentPage = 1;
}

function onSourceFilter() {
  clearTimeout(_filterTimer);
  _filterTimer = setTimeout(() => { resetPagination(); reloadChunks(); }, 500);
}

function onDateFilter() {
  resetPagination();
  reloadChunks();
}

function clearDates() {
  document.getElementById('dateFrom').value = '';
  document.getElementById('dateTo').value   = '';
  resetPagination();
  reloadChunks();
}

// Main load function — from_order_value = cursor for the page we want
async function reloadChunks(from_order_value = null) {
  const tbody     = document.getElementById('chunksTbody');
  const limit     = parseInt(document.getElementById('limitSelect').value);
  const srcF      = document.getElementById('sourceFilter').value.trim();
  const dateFrom  = document.getElementById('dateFrom').value;
  const dateTo    = document.getElementById('dateTo').value;

  tbody.innerHTML = `<tr><td colspan="7" class="empty-cell loading-cell">Loading…</td></tr>`;

  let url = `/chunk_management/chunks?collection=${_currentCollection}&limit=${limit}`;
  if (from_order_value !== null) url += `&from_order_value=${from_order_value}`;
  if (srcF)     url += `&source_filter=${encodeURIComponent(srcF)}`;
  if (dateFrom) url += `&date_from=${dateFrom}`;
  if (dateTo)   url += `&date_to=${dateTo}`;

  try {
    const data = await apiFetch(url);
    _nextCursor = data.next_page_offset ? parseInt(data.next_page_offset) : null;
    renderChunks(data.points, limit);
    updatePagination();
  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-cell" style="color:var(--red)">❌ ${e.message}</td></tr>`;
  }
}

function nextPage() {
  if (_nextCursor === null) return;
  _pageHistory.push(_nextCursor);  // save cursor that leads to current next page
  _currentPage++;
  reloadChunks(_nextCursor);
}

function prevPage() {
  if (_currentPage <= 1) return;
  _pageHistory.pop();              // remove the cursor we just used
  _currentPage--;
  const prevCursor = _pageHistory.length > 0 ? _pageHistory[_pageHistory.length - 1] : null;
  reloadChunks(prevCursor);
}

function renderChunks(points, limit) {
  const tbody = document.getElementById('chunksTbody');
  tbody.innerHTML = '';

  if (!points || !points.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty-cell">No chunks found.</td></tr>';
    return;
  }

  points.forEach((p, i) => {
    const tr         = document.createElement('tr');
    const globalIdx  = (_currentPage - 1) * limit + i + 1;
    const preview    = (p.chunk_text || '').slice(0, 120).replace(/\n/g, ' ');
    const ts         = p.ingest_timestamp ? p.ingest_timestamp.slice(0, 16).replace('T', ' ') : '—';

    tr.innerHTML = `
      <td class="td-idx">${globalIdx}</td>
      <td class="td-source" title="${escHtml(p.source_pdf || '')}">${escHtml(shortName(p.source_pdf))}</td>
      <td class="td-mono">${p.chunk_index ?? '—'}</td>
      <td class="td-mono">${p.text_len ? p.text_len.toLocaleString() : '—'}</td>
      <td class="td-ts">${ts}</td>
      <td class="td-preview">${escHtml(preview)}${preview.length >= 120 ? '…' : ''}</td>
      <td>
        <div style="display:flex;gap:4px">
          <button class="btn-icon view"   title="View detail" onclick="openDetail('${p.id}')">🔍</button>
          <button class="btn-icon delete" title="Delete"      onclick="deleteChunk('${_currentCollection}','${p.id}',this)">🗑</button>
        </div>
      </td>`;
    tbody.appendChild(tr);
  });
}

function updatePagination() {
  document.getElementById('pageInfo').textContent = `Page ${_currentPage}`;
  document.getElementById('btnPrev').disabled = _currentPage <= 1;
  document.getElementById('btnNext').disabled = _nextCursor === null;
}

// ── Delete from table ─────────────────────────────────────────────────────
async function deleteChunk(collection, pointId, btn) {
  if (!confirm(`Delete point ${pointId}?\nThis cannot be undone.`)) return;
  btn.disabled = true;
  try {
    const data = await apiFetch('/chunk_management/delete_chunk', {
      method: 'POST',
      body: JSON.stringify({ collection, point_id: pointId })
    });
    if (!data.ok) { showFlash('error', data.error || 'Delete failed'); btn.disabled = false; return; }
    showFlash('success', `Point deleted`);
    btn.closest('tr').remove();
    loadCollectionStats(collection);
  } catch(e) {
    showFlash('error', e.message);
    btn.disabled = false;
  }
}

// ════════════════════════════════════════════════════
// CHUNK DETAIL MODAL
// ════════════════════════════════════════════════════

async function openDetail(pointId) {
  _detailPointId = pointId;
  document.getElementById('detailTitle').textContent = `Point: ${pointId.slice(0, 16)}…`;
  document.getElementById('detailBody').innerHTML    = '<div class="detail-loading">Loading…</div>';
  openModal('detailModal');

  try {
    const data = await apiFetch(
      `/chunk_management/chunk_detail?collection=${_currentCollection}&point_id=${pointId}`
    );
    if (!data.ok) {
      document.getElementById('detailBody').innerHTML = `<div style="color:var(--red)">${data.error}</div>`;
      return;
    }
    renderDetail(data.point);
  } catch(e) {
    document.getElementById('detailBody').innerHTML = `<div style="color:var(--red)">❌ ${e.message}</div>`;
  }
}

function renderDetail(p) {
  const rows = [
    ['ID',           p.id],
    ['Source PDF',   p.source_pdf],
    ['Chunk Index',  p.chunk_index],
    ['Chunk ID',     p.chunk_id],
    ['Text Length',  p.text_len],
    ['PDF Path',     p.pdf_path],
    ['Source Path',  p.source_path],
    ['Ingest Time',  p.ingest_timestamp],
    ['Ingest Run',   p.ingest_run_id],
  ];

  let html = '<div class="detail-grid">';
  rows.forEach(([label, val]) => {
    if (val == null) return;
    html += `<div class="detail-row">
      <span class="detail-label">${label}</span>
      <span class="detail-val">${escHtml(String(val))}</span>
    </div>`;
  });
  Object.entries(p.extra_payload || {}).forEach(([label, val]) => {
    html += `<div class="detail-row">
      <span class="detail-label extra-label">${label}</span>
      <span class="detail-val">${escHtml(JSON.stringify(val))}</span>
    </div>`;
  });
  html += '</div>';

  if (p.chunk_text) {
    html += `<div class="detail-text-section">
      <div class="detail-text-label">Full Chunk Text</div>
      <div class="detail-text-body">${escHtml(p.chunk_text)}</div>
    </div>`;
  }

  document.getElementById('detailBody').innerHTML = html;
}

async function deleteFromDetail() {
  const pointId = _detailPointId;
  if (!pointId) return;
  if (!confirm(`Delete point ${pointId}?\nThis cannot be undone.`)) return;
  try {
    const data = await apiFetch('/chunk_management/delete_chunk', {
      method: 'POST',
      body: JSON.stringify({ collection: _currentCollection, point_id: pointId })
    });
    if (!data.ok) { showFlash('error', data.error || 'Delete failed'); return; }
    showFlash('success', `Point deleted`);
    closeDetailModal();
    reloadChunks();
    loadCollectionStats(_currentCollection);
  } catch(e) { showFlash('error', e.message); }
}

// ════════════════════════════════════════════════════
// TAB 2 — SOURCES (full collection scan)
// ════════════════════════════════════════════════════

async function loadSourceSummary() {
  const tbody = document.getElementById('sourcesTbody');
  const note  = document.getElementById('sourcesNote');
  tbody.innerHTML = '<tr><td colspan="5" class="empty-cell loading-cell">Scanning full collection…</td></tr>';
  note.textContent = '';
  try {
    _allSources = await apiFetch(`/chunk_management/source_summary?collection=${_currentCollection}`);
    note.textContent = `${_allSources.length} sources`;
    renderSources(_allSources);
  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty-cell" style="color:var(--red)">❌ ${e.message}</td></tr>`;
  }
}

function filterSources() {
  const q = document.getElementById('srcSearch').value.toLowerCase();
  renderSources(_allSources.filter(s => s.source_pdf.toLowerCase().includes(q)));
}

function renderSources(sources) {
  const tbody = document.getElementById('sourcesTbody');
  tbody.innerHTML = '';
  if (!sources.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-cell">No sources found.</td></tr>';
    return;
  }
  sources.forEach((s, i) => {
    const tr = document.createElement('tr');
    const ts = s.last_ts ? s.last_ts.slice(0, 16).replace('T', ' ') : '—';
    tr.innerHTML = `
      <td class="td-idx">${i + 1}</td>
      <td class="td-source" title="${escHtml(s.source_pdf)}">${escHtml(shortName(s.source_pdf))}</td>
      <td class="td-mono" style="color:var(--blue-hi)">${s.count}</td>
      <td class="td-ts">${ts}</td>
      <td>
        <button class="btn-filter-src" onclick="filterBySource('${escAttr(s.source_pdf)}')">Browse →</button>
      </td>`;
    tbody.appendChild(tr);
  });
}

function filterBySource(sourcePdf) {
  switchTab('browse');
  document.getElementById('sourceFilter').value = sourcePdf;
  resetPagination();
  reloadChunks();
}

// ════════════════════════════════════════════════════
// TAB 3 — INGEST RUNS (full collection scan)
// ════════════════════════════════════════════════════

async function loadIngestRuns() {
  const tbody = document.getElementById('runsTbody');
  const note  = document.getElementById('runsNote');
  tbody.innerHTML = '<tr><td colspan="4" class="empty-cell loading-cell">Scanning full collection…</td></tr>';
  note.textContent = '';
  try {
    const runs = await apiFetch(`/chunk_management/ingest_runs?collection=${_currentCollection}`);
    note.textContent = `${runs.length} run${runs.length !== 1 ? 's' : ''} found`;
    renderRuns(runs);
  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="4" class="empty-cell" style="color:var(--red)">❌ ${e.message}</td></tr>`;
  }
}

function renderRuns(runs) {
  const tbody = document.getElementById('runsTbody');
  tbody.innerHTML = '';
  if (!runs.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty-cell">No runs found.</td></tr>';
    return;
  }
  runs.forEach((r, i) => {
    const tr  = document.createElement('tr');
    const ts  = r.last_ts ? r.last_ts.slice(0, 16).replace('T', ' ') : '—';
    const rid = r.run_id || '—';
    tr.innerHTML = `
      <td class="td-idx">${i + 1}</td>
      <td class="td-run-id" title="${escHtml(rid)}">${escHtml(rid.length > 70 ? '…' + rid.slice(-70) : rid)}</td>
      <td class="td-mono" style="color:var(--green)">${r.count}</td>
      <td class="td-ts">${ts}</td>`;
    tbody.appendChild(tr);
  });
}

// ════════════════════════════════════════════════════
// MODAL
// ════════════════════════════════════════════════════

function openModal(id) {
  document.getElementById('modalBackdrop').classList.add('open');
  document.getElementById(id).classList.add('open');
}

function closeDetailModal() {
  document.getElementById('modalBackdrop').classList.remove('open');
  document.getElementById('detailModal').classList.remove('open');
  _detailPointId = null;
}

// ════════════════════════════════════════════════════
// UTILS
// ════════════════════════════════════════════════════

function shortName(path) {
  if (!path) return '—';
  return path.split('/').pop().split('\\').pop();
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function escAttr(str) {
  return String(str).replace(/'/g, "\\'");
}

async function apiFetch(url, opts = {}) {
  const res = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function showFlash(type, msg) {
  document.querySelector('.flash')?.remove();
  const el = document.createElement('div');
  el.className   = `flash ${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}