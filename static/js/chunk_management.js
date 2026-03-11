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

// Browse chunks pagination
let _pageHistory = [];   // stack of cursors for pages already visited
let _nextCursor  = null; // cursor for the next page
let _currentPage = 1;

// Sources pagination (used when collection has no aggregation, e.g. zh_metadata)
let _sourcesPage        = 1;
let _sourcesNextCursor  = null;
let _sourcesPageHistory = [];

let _detailPointId = null;
let _allSources    = [];   // used only for zh_chunks aggregated sources
let _filterTimer   = null;

// Columns that are only meaningful for zh_chunks
const CHUNKS_ONLY_COLS = ['chunk', 'len', 'ingest_time', 'text_preview'];

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
  updateColumnVisibility(col);
  // Ingest Runs tab is only meaningful for zh_chunks
  document.querySelector('[data-tab="runs"]').style.display = col === 'zh_metadata' ? 'none' : '';
  // If runs tab was active and we switched to zh_metadata, fall back to browse
  if (col === 'zh_metadata' && document.querySelector('.tab-btn.active')?.dataset.tab === 'runs') {
    switchTab('browse');
  }

  const activeTab = document.querySelector('.tab-btn.active')?.dataset.tab;
  if (activeTab === 'browse')  reloadChunks();
  if (activeTab === 'sources') loadSourceSummary();
  if (activeTab === 'runs')    loadIngestRuns();
}

// ── Column visibility — hide chunk-specific columns for zh_metadata ───────
function updateColumnVisibility(col) {
  const isMetadata = col === 'zh_metadata';
  // Table headers: data-col attribute identifies each column
  document.querySelectorAll('[data-col]').forEach(el => {
    const hide = isMetadata && CHUNKS_ONLY_COLS.includes(el.dataset.col);
    el.style.display = hide ? 'none' : '';
  });
}

// ── Tab switch ────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === name)
  );
  document.querySelectorAll('.tab-pane').forEach(p =>
    p.classList.toggle('active', p.id === 'tab-' + name)
  );
  // Sources and runs require explicit user action (Load/Refresh button) to avoid
  // expensive full-collection scans on every tab switch
  // loadSourceSummary() and loadIngestRuns() are triggered by their own buttons
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

async function reloadChunks(from_order_value = null) {
  const tbody    = document.getElementById('chunksTbody');
  const limit    = parseInt(document.getElementById('limitSelect').value);
  const srcF     = document.getElementById('sourceFilter').value.trim();
  const dateFrom = document.getElementById('dateFrom').value;
  const dateTo   = document.getElementById('dateTo').value;

  tbody.innerHTML = `<tr><td colspan="7" class="empty-cell loading-cell">Loading…</td></tr>`;

  let url = `/chunk_management/chunks?collection=${_currentCollection}&limit=${limit}`;
  if (from_order_value !== null) url += `&from_order_value=${encodeURIComponent(from_order_value)}`;
  if (srcF)     url += `&source_filter=${encodeURIComponent(srcF)}`;
  if (dateFrom) url += `&date_from=${dateFrom}`;
  if (dateTo)   url += `&date_to=${dateTo}`;

  try {
    const data  = await apiFetch(url);
    // next_page_offset may be a UUID string or an epoch int — keep as-is
    _nextCursor = data.next_page_offset ?? null;
    renderChunks(data.points, limit);
    updatePagination();
  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-cell" style="color:var(--red)">❌ ${e.message}</td></tr>`;
  }
}

function nextPage() {
  if (_nextCursor === null) return;
  _pageHistory.push(_nextCursor);
  _currentPage++;
  reloadChunks(_nextCursor);
}

function prevPage() {
  if (_currentPage <= 1) return;
  _pageHistory.pop();
  _currentPage--;
  const prevCursor = _pageHistory.length > 0 ? _pageHistory[_pageHistory.length - 1] : null;
  reloadChunks(prevCursor);
}

function renderChunks(points, limit) {
  const tbody      = document.getElementById('chunksTbody');
  const isMetadata = _currentCollection === 'zh_metadata';
  tbody.innerHTML  = '';

  if (!points || !points.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty-cell">No chunks found.</td></tr>';
    return;
  }

  points.forEach((p, i) => {
    const tr        = document.createElement('tr');
    const globalIdx = (_currentPage - 1) * limit + i + 1;
    const preview   = (p.chunk_text || '').slice(0, 120).replace(/\n/g, ' ');
    const ts        = p.ingest_timestamp ? p.ingest_timestamp.slice(0, 16).replace('T', ' ') : '—';

    // Chunk-specific cells are hidden via CSS when in zh_metadata
    tr.innerHTML = `
      <td class="td-idx">${globalIdx}</td>
      <td class="td-source" title="${escHtml(p.source_pdf || '')}">${escHtml(shortName(p.source_pdf))}</td>
      <td class="td-mono" data-col="chunk">${p.chunk_index ?? '—'}</td>
      <td class="td-mono" data-col="len">${p.text_len ? p.text_len.toLocaleString() : '—'}</td>
      <td class="td-ts"   data-col="ingest_time">${ts}</td>
      <td class="td-preview" data-col="text_preview">${escHtml(preview)}${preview.length >= 120 ? '…' : ''}</td>
      <td>
        <div style="display:flex;gap:4px">
          <button class="btn-icon view"   title="View detail" onclick="openDetail('${p.id}')">🔍</button>
          <button class="btn-icon delete" title="Delete"      onclick="deleteChunk('${_currentCollection}','${p.id}',this)">🗑</button>
        </div>
      </td>`;
    tbody.appendChild(tr);
  });

  // Apply column visibility to newly rendered rows
  updateColumnVisibility(_currentCollection);
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
  // Extra fields (e.g. zh_metadata native fields: filename, path, status, sha256_*)
  Object.entries(p.extra_payload || {}).forEach(([label, val]) => {
    if (val == null) return;
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
// TAB 2 — SOURCES
// ════════════════════════════════════════════════════

function resetSourcesPagination() {
  _sourcesPage        = 1;
  _sourcesNextCursor  = null;
  _sourcesPageHistory = [];
}

async function loadSourceSummary() {
  // For zh_chunks, a single-day date filter is required to avoid scanning
  // tens of thousands of points. zh_metadata is paginated so no restriction needed.
  if (_currentCollection === 'zh_chunks') {
    const dateFrom = document.getElementById('srcDateFrom')?.value;
    const dateTo   = document.getElementById('srcDateTo')?.value;
    if (!dateFrom || !dateTo || dateFrom !== dateTo) {
      document.getElementById('sourcesTbody').innerHTML =
        `<tr><td colspan="5" class="empty-cell" style="color:var(--orange)">
          ⚠️ Select a single day using the date filter above before loading sources.<br>
          <span style="font-size:9px;color:var(--faint)">This collection has tens of thousands of chunks — scanning without a date filter would be too slow.</span>
        </td></tr>`;
      document.getElementById('sourcesNote').textContent = '';
      return;
    }
  }
  resetSourcesPagination();
  await _fetchSources(null);
}

async function _fetchSources(cursor) {
  const tbody = document.getElementById('sourcesTbody');
  const note  = document.getElementById('sourcesNote');
  tbody.innerHTML = '<tr><td colspan="5" class="empty-cell loading-cell">Loading…</td></tr>';
  note.textContent = '';

  try {
    if (_currentCollection === 'zh_metadata') {
      // Paginated browse — each point is already one source file
      let url = `/chunk_management/chunks?collection=zh_metadata&limit=20`;
      if (cursor !== null) url += `&from_order_value=${encodeURIComponent(cursor)}`;

      const data = await apiFetch(url);
      _sourcesNextCursor = data.next_page_offset ?? null;
      note.textContent   = `Page ${_sourcesPage}`;
      renderSourcesFromPoints(data.points);
      updateSourcesPagination();
    } else {
      // Full aggregation for zh_chunks — scoped to one day to avoid scanning entire collection
      const dateFrom = document.getElementById('srcDateFrom')?.value;
      const dateTo   = document.getElementById('srcDateTo')?.value;
      let srcUrl = `/chunk_management/source_summary?collection=${_currentCollection}`;
      if (dateFrom) srcUrl += `&date_from=${dateFrom}&date_to=${dateTo}`;
      _allSources = await apiFetch(srcUrl);
      note.textContent = `${_allSources.length} sources`;
      renderSources(_allSources);
      // Hide pagination bar for aggregated view
      document.getElementById('sourcesPaginationBar')?.style && (
        document.getElementById('sourcesPaginationBar').style.display = 'none'
      );
    }
  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty-cell" style="color:var(--red)">❌ ${e.message}</td></tr>`;
  }
}

function nextSourcesPage() {
  if (_sourcesNextCursor === null) return;
  _sourcesPageHistory.push(_sourcesNextCursor);
  _sourcesPage++;
  _fetchSources(_sourcesNextCursor);
}

function prevSourcesPage() {
  if (_sourcesPage <= 1) return;
  _sourcesPageHistory.pop();
  _sourcesPage--;
  const prev = _sourcesPageHistory.length > 0
    ? _sourcesPageHistory[_sourcesPageHistory.length - 1]
    : null;
  _fetchSources(prev);
}

function updateSourcesPagination() {
  const bar = document.getElementById('sourcesPaginationBar');
  if (!bar) return;
  bar.style.display = _currentCollection === 'zh_metadata' ? 'flex' : 'none';
  document.getElementById('sourcesPageInfo').textContent = `Page ${_sourcesPage}`;
  document.getElementById('btnSourcesPrev').disabled = _sourcesPage <= 1;
  document.getElementById('btnSourcesNext').disabled = _sourcesNextCursor === null;
}

// Render for zh_metadata: one row per point (each point = one file)
function renderSourcesFromPoints(points) {
  const tbody = document.getElementById('sourcesTbody');
  tbody.innerHTML = '';
  if (!points || !points.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-cell">No sources found.</td></tr>';
    return;
  }
  const offset = (_sourcesPage - 1) * 20;
  points.forEach((p, i) => {
    const tr       = document.createElement('tr');
    const extra    = p.extra_payload || {};
    const filename = p.source_pdf || extra.filename || '—';
    const status   = extra.status  || '—';
    tr.innerHTML = `
      <td class="td-idx">${offset + i + 1}</td>
      <td class="td-source" title="${escHtml(extra.path || filename)}">${escHtml(filename)}</td>
      <td class="td-mono" style="color:var(--dim)">${escHtml(status)}</td>
      <td class="td-ts">—</td>
      <td>
        <button class="btn-icon view" title="View detail" onclick="openDetail('${p.id}')">🔍</button>
      </td>`;
    tbody.appendChild(tr);
  });
}

// Render for zh_chunks: aggregated count per source
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

function filterSources() {
  const q = document.getElementById('srcSearch').value.toLowerCase();
  renderSources(_allSources.filter(s => s.source_pdf.toLowerCase().includes(q)));
}

function filterBySource(sourcePdf) {
  switchTab('browse');
  document.getElementById('sourceFilter').value = sourcePdf;
  resetPagination();
  reloadChunks();
}

// ════════════════════════════════════════════════════
// TAB 3 — INGEST RUNS
// ════════════════════════════════════════════════════

async function loadIngestRuns() {
  const tbody    = document.getElementById('runsTbody');
  const note     = document.getElementById('runsNote');
  const dateFrom = document.getElementById('runsDateFrom')?.value;
  const dateTo   = dateFrom; // single-day filter

  if (!dateFrom) {
    tbody.innerHTML = `<tr><td colspan="4" class="empty-cell" style="color:var(--orange)">
      ⚠️ Select a day above before loading ingest runs.<br>
      <span style="font-size:9px;color:var(--faint)">Scanning the full collection without a date filter would be too slow.</span>
    </td></tr>`;
    note.textContent = '';
    return;
  }

  tbody.innerHTML = '<tr><td colspan="4" class="empty-cell loading-cell">Scanning…</td></tr>';
  note.textContent = '';
  try {
    const runs = await apiFetch(`/chunk_management/ingest_runs?collection=${_currentCollection}&date_from=${dateFrom}&date_to=${dateTo}`);
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