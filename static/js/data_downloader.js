// data_downloader.js

// ── Clock ─────────────────────────────────────────────────────────────────
(function tick() {
  const el = document.getElementById('navClock');
  if (el) {
    const n = new Date(), p = v => String(v).padStart(2,'0');
    el.textContent = [n.getHours(), n.getMinutes(), n.getSeconds()].map(p).join(':');
  }
  setTimeout(tick, 1000);
})();

// ── Default date: 1st of this month - 15 days ─────────────────────────────
function calcDefaultFrom() {
  const now        = new Date();
  const firstOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
  firstOfMonth.setDate(firstOfMonth.getDate() - 15);
  return firstOfMonth.toISOString().slice(0,10);
}
function todayStr() { return new Date().toISOString().slice(0,10); }

// ── Tab switch ────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
  if (name === 'status') loadStatusTab();
  if (name === 'health') loadHealthTab();
}

// ── Boot ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Set global date defaults
  document.getElementById('globalDFrom').value = calcDefaultFrom();
  document.getElementById('globalDTo').value   = todayStr();
  loadGroups();
});

// ════════════════════════════════════════════════════════
// GLOBAL DATE BAR
// ════════════════════════════════════════════════════════

function toggleGlobalDTo() {
  const cb  = document.getElementById('globalDToEnable');
  const inp = document.getElementById('globalDTo');
  inp.disabled = !cb.checked;
}

function applyGlobalDates() {
  const dFrom = document.getElementById('globalDFrom').value;
  const dToEl = document.getElementById('globalDTo');
  const dTo   = dToEl.disabled ? null : dToEl.value || null;

  // Push to all rendered group date inputs
  document.querySelectorAll('.group-card').forEach(card => {
    const gid = card.dataset.groupId;
    if (!gid) return;
    const fromEl = document.getElementById(`gDFrom-${gid}`);
    const toEl   = document.getElementById(`gDTo-${gid}`);
    const cbEl   = document.getElementById(`gDToEnable-${gid}`);
    if (fromEl) fromEl.value = dFrom;
    if (toEl && cbEl) {
      if (dTo) {
        cbEl.checked  = true;
        toEl.disabled = false;
        toEl.value    = dTo;
      } else {
        cbEl.checked  = false;
        toEl.disabled = true;
      }
    }
  });
  showFlash('success', `Dates applied to all groups — From: ${dFrom}${dTo ? '  To: '+dTo : ''}`);
}

// ════════════════════════════════════════════════════════
// TAB 1 — EXECUTE
// ════════════════════════════════════════════════════════

async function loadGroups() {
  const container = document.getElementById('groupsContainer');
  container.innerHTML = '<div style="padding:20px;font-family:var(--mono);font-size:11px;color:var(--dim)">Loading…</div>';
  try {
    const groups = await apiFetch('/data_downloader/groups');
    container.innerHTML = '';
    groups.forEach(g => container.appendChild(buildGroupCard(g)));
  } catch (e) {
    container.innerHTML = `<div style="color:var(--red);padding:16px;font-family:var(--mono);font-size:11px">❌ ${e.message}</div>`;
  }
}

function buildGroupCard(group) {
  const card = document.createElement('div');
  card.className   = 'group-card';
  card.dataset.groupId = group.group_id;
  card.dataset.jobType = group.job_type;

  const badgeClass = group.job_type === 'SPREAD' ? 'badge-spread' : 'badge-download';
  const badgeLabel = group.job_type === 'SPREAD' ? '⇄ Spread' : '↓ Download';

  // Inherit from global date bar
  const dFrom = document.getElementById('globalDFrom')?.value || calcDefaultFrom();
  const today = todayStr();

  card.innerHTML = `
    <div class="group-hdr" onclick="toggleGroup(${group.group_id})">
      <span class="group-chevron">▶</span>
      <span class="group-name">${group.group_name}</span>
      <span class="group-badge ${badgeClass}">${badgeLabel}</span>
      <span class="group-count" id="groupCount-${group.group_id}">${group.job_count} jobs</span>
      <span class="group-dates" onclick="event.stopPropagation()">
        <span class="group-date-label">From</span>
        <input class="date-mini" type="date" id="gDFrom-${group.group_id}" value="${dFrom}">
        <span class="group-date-label" style="margin-left:4px">To</span>
        <input class="date-to-check" type="checkbox" id="gDToEnable-${group.group_id}"
          onchange="toggleGroupDTo(${group.group_id})">
        <input class="date-mini" type="date" id="gDTo-${group.group_id}" value="${today}" disabled>
      </span>
      <button class="btn-add-job"
        onclick="event.stopPropagation(); openAddJobModal(${group.group_id}, '${group.job_type}')">
        + Add Job
      </button>
      <button class="btn-run-all" id="runAll-${group.group_id}"
        onclick="event.stopPropagation(); runGroup(${group.group_id})">
        <div class="btn-spin"></div><span>Run All</span>
      </button>
    </div>
    <div class="group-body" id="groupBody-${group.group_id}">
      <div style="font-family:var(--mono);font-size:10px;color:var(--faint);padding:8px">Expand to load jobs…</div>
    </div>
    <div class="console-drawer" id="console-${group.group_id}">
      <div class="console-hdr">
        <span class="console-title">● Execution Log — ${group.group_name}</span>
        <button class="console-close" onclick="closeConsole(${group.group_id})">✕</button>
      </div>
      <div class="console-body" id="consoleBody-${group.group_id}"></div>
    </div>
  `;
  return card;
}

function toggleGroupDTo(groupId) {
  const cb  = document.getElementById(`gDToEnable-${groupId}`);
  const inp = document.getElementById(`gDTo-${groupId}`);
  inp.disabled = !cb.checked;
}

const _loadedGroups = new Set();

async function toggleGroup(groupId) {
  const card = document.querySelector(`.group-card[data-group-id="${groupId}"]`);
  if (card.classList.contains('open')) { card.classList.remove('open'); return; }
  card.classList.add('open');
  if (!_loadedGroups.has(groupId)) {
    await loadGroupJobs(groupId);
    _loadedGroups.add(groupId);
  }
}

async function loadGroupJobs(groupId) {
  const body = document.getElementById(`groupBody-${groupId}`);
  body.innerHTML = '<div style="font-family:var(--mono);font-size:10px;color:var(--dim);padding:8px">Loading…</div>';
  try {
    const jobs = await apiFetch(`/data_downloader/jobs_by_group?group_id=${groupId}`);
    renderGroupJobs(groupId, jobs);
  } catch (e) {
    body.innerHTML = `<div style="color:var(--red);font-family:var(--mono);font-size:10px;padding:8px">❌ ${e.message}</div>`;
  }
}

function renderGroupJobs(groupId, jobs) {
  const body = document.getElementById(`groupBody-${groupId}`);
  body.innerHTML = '';
  if (!jobs.length) {
    body.innerHTML = '<div style="font-family:var(--mono);font-size:10px;color:var(--faint);padding:8px">No jobs.</div>';
    return;
  }
  const table = document.createElement('table');
  table.className = 'job-table';
  table.innerHTML = `<thead><tr>
    <th>Symbol</th><th>Exchange</th><th>Output</th><th>Vendor</th>
    <th>Last Run</th><th>Status</th><th></th>
  </tr></thead><tbody></tbody>`;
  body.appendChild(table);
  jobs.forEach(j => table.querySelector('tbody').appendChild(buildJobRow(j)));
  const cnt = document.getElementById(`groupCount-${groupId}`);
  if (cnt) cnt.textContent = `${jobs.length} jobs`;
}

function buildJobRow(job) {
  const tr  = document.createElement('tr');
  tr.id = `jobRow-${job.job_id}`;
  const vClass  = vendorClass(job.vendor);
  const outCell = job.output_symbol
    ? `<span class="td-output">${job.output_symbol}</span>`
    : `<span style="color:var(--faint)">—</span>`;

  tr.innerHTML = `
    <td class="td-symbol">${job.symbol}</td>
    <td class="td-date">${job.exchange || '—'}</td>
    <td>${outCell}</td>
    <td class="${vClass}">${job.vendor}</td>
    <td class="td-date">${job.last_run_at ? job.last_run_at.slice(0,16) : '—'}</td>
    <td id="jobStatus-${job.job_id}">${mkStatusCell(job.last_status)}</td>
    <td>
      <div style="display:flex;gap:4px;align-items:center">
        <button class="btn-run" id="btnRun-${job.job_id}"
          onclick="runJob(${job.job_id}, ${job.group_id})">
          <div class="btn-spin"></div><span>Run</span>
        </button>
        <button class="btn-icon edit" title="Edit"
          onclick="openEditJobModal(${job.job_id}, ${job.group_id})">✏</button>
        <button class="btn-icon delete" title="Delete"
          onclick="deleteJob(${job.job_id}, ${job.group_id}, '${job.symbol}')">🗑</button>
      </div>
    </td>`;
  return tr;
}

function vendorClass(v) {
  if (v === 'FRED') return 'td-vendor-fred';
  if (v === 'TRADINGVIEW') return 'td-vendor-tv';
  if (v === 'SPREAD') return 'td-vendor-spr';
  if (v === 'MANUAL_VARIABLE') return 'td-vendor-man';
  return '';
}

function getGroupDates(groupId) {
  const dFrom = document.getElementById(`gDFrom-${groupId}`)?.value || null;
  const dToEl = document.getElementById(`gDTo-${groupId}`);
  const dTo   = (dToEl && !dToEl.disabled) ? dToEl.value || null : null;
  return { dFrom, dTo };
}

// ── Run single job ────────────────────────────────────────────────────────
async function runJob(jobId, groupId) {
  const btn   = document.getElementById(`btnRun-${jobId}`);
  const sc    = document.getElementById(`jobStatus-${jobId}`);
  const cbody = document.getElementById(`consoleBody-${groupId}`);
  const { dFrom, dTo } = getGroupDates(groupId);

  setSpinning(btn, true);
  if (sc) sc.innerHTML = mkStatusCell('RUNNING');
  openConsole(groupId);
  log(cbody, `▶ Job ${jobId} — from=${dFrom} to=${dTo||'today'} — ${new Date().toLocaleTimeString()}\n`, 'c-dim');

  try {
    const data = await apiFetch('/data_downloader/run_job', {
      method: 'POST',
      body: JSON.stringify({ job_id: jobId, group_id: groupId, d_from: dFrom, d_to: dTo })
    });
    log(cbody, data.log || '');
    if (data.ok) {
      log(cbody, `\n✅ Done\n`, 'c-ok');
      if (sc) sc.innerHTML = mkStatusCell('OK');
    } else {
      log(cbody, `\n❌ Failed\n${data.error || ''}\n`, 'c-err');
      if (sc) sc.innerHTML = mkStatusCell('ERROR');
    }
  } catch (e) {
    log(cbody, `\n❌ Network: ${e.message}\n`, 'c-err');
  } finally {
    setSpinning(btn, false);
  }
}

// ── Run group ─────────────────────────────────────────────────────────────
async function runGroup(groupId) {
  const btn   = document.getElementById(`runAll-${groupId}`);
  const cbody = document.getElementById(`consoleBody-${groupId}`);
  const { dFrom, dTo } = getGroupDates(groupId);

  setSpinning(btn, true);
  openConsole(groupId);
  log(cbody, `▶ Running ALL — from=${dFrom} to=${dTo||'today'} — ${new Date().toLocaleTimeString()}\n\n`, 'c-dim');

  try {
    const data = await apiFetch('/data_downloader/run_group', {
      method: 'POST',
      body: JSON.stringify({ group_id: groupId, d_from: dFrom, d_to: dTo })
    });
    (data.results || []).forEach(r => {
      log(cbody, `── ${r.symbol} ──────────────\n`, 'c-dim');
      log(cbody, r.log || '');
      log(cbody, r.ok ? `✅ OK\n\n` : `❌ FAILED\n${r.error||''}\n\n`, r.ok ? 'c-ok' : 'c-err');
      const sc = document.getElementById(`jobStatus-${r.job_id}`);
      if (sc) sc.innerHTML = mkStatusCell(r.ok ? 'OK' : 'ERROR');
    });
    log(cbody, data.ok ? '\n✅ Group complete\n' : '\n⚠ Finished with errors\n', data.ok ? 'c-ok' : 'c-warn');
    showFlash(data.ok ? 'success' : 'error',
      data.ok ? `Group complete (${(data.results||[]).length} jobs)` : 'Completed with errors');
  } catch (e) {
    log(cbody, `\n❌ Network: ${e.message}\n`, 'c-err');
    showFlash('error', e.message);
  } finally {
    setSpinning(btn, false);
  }
}

// ── Console ───────────────────────────────────────────────────────────────
function openConsole(gid)  { document.getElementById(`console-${gid}`)?.classList.add('open'); }
function closeConsole(gid) { document.getElementById(`console-${gid}`)?.classList.remove('open'); }
function log(el, text, cls) {
  if (!el || !text) return;
  const s = document.createElement('span');
  if (cls) s.className = cls;
  s.textContent = text;
  el.appendChild(s);
  el.scrollTop = el.scrollHeight;
}

function mkStatusCell(status) {
  if (!status) return `<span class="status-dot dot-none"></span><span style="color:var(--faint);font-size:9px;font-family:var(--mono)">NEVER</span>`;
  const cls   = status === 'OK' ? 'dot-ok' : status === 'ERROR' ? 'dot-error' : 'dot-running';
  const color = status === 'OK' ? 'var(--green)' : status === 'ERROR' ? 'var(--red)' : 'var(--orange)';
  const resetBtn = status === 'RUNNING'
    ? ` <button class="btn-reset" onclick="resetJob(event,this)">⟳ Reset</button>` : '';
  return `<span class="status-dot ${cls}"></span><span style="color:${color};font-size:9px;font-family:var(--mono)">${status}</span>${resetBtn}`;
}

// ════════════════════════════════════════════════════════
// CRUD  (#1)
// ════════════════════════════════════════════════════════

let _modalGroupId   = null;
let _modalJobType   = null;
let _modalEditJobId = null;

function openAddJobModal(groupId, jobType) {
  _modalGroupId   = groupId;
  _modalJobType   = jobType;
  _modalEditJobId = null;

  document.getElementById('modalTitle').textContent    = 'Add Job';
  document.getElementById('mJobId').value              = '';
  document.getElementById('mGroupId').value            = groupId;
  document.getElementById('mVendor').value             = '';
  document.getElementById('mSymbol').value             = '';
  document.getElementById('mSymbol').disabled          = false;
  document.getElementById('mExchange').value           = '';
  document.getElementById('mOutputSymbol').value       = '';
  document.getElementById('mDFrom').value              = calcDefaultFrom();
  document.getElementById('mDTo').value                = '';
  document.getElementById('mDToEnable').checked        = false;
  document.getElementById('mDTo').disabled             = true;
  document.getElementById('mError').textContent        = '';

  const isSpr = jobType === 'SPREAD';
  if (isSpr) {
    document.getElementById('mVendorSelectRow').style.display = 'none';
    document.getElementById('mVendorBadgeRow').style.display  = '';
    document.getElementById('mExchangeRow').style.display     = 'none';
    document.getElementById('mOutputRow').style.display       = '';
    setVendorBadge('SPREAD');
    document.getElementById('mVendor').value = 'SPREAD';
  } else {
    document.getElementById('mVendorSelectRow').style.display = '';
    document.getElementById('mVendorBadgeRow').style.display  = 'none';
    document.getElementById('mExchangeRow').style.display     = '';
    document.getElementById('mOutputRow').style.display       = 'none';
    document.querySelectorAll('.vendor-btn').forEach(b => b.classList.remove('active'));
  }
  openModal();
}

async function openEditJobModal(jobId, groupId) {
  const [groups, jobs] = await Promise.all([
    apiFetch('/data_downloader/groups'),
    apiFetch(`/data_downloader/jobs_by_group?group_id=${groupId}`)
  ]);
  const g   = groups.find(x => x.group_id === groupId);
  const job = jobs.find(j => j.job_id === jobId);
  if (!job) { showFlash('error', 'Job not found'); return; }

  _modalGroupId   = groupId;
  _modalJobType   = g?.job_type || 'DOWNLOAD';
  _modalEditJobId = jobId;

  document.getElementById('modalTitle').textContent    = `Edit — ${job.symbol}`;
  document.getElementById('mJobId').value              = jobId;
  document.getElementById('mGroupId').value            = groupId;
  document.getElementById('mVendor').value             = job.vendor;
  document.getElementById('mSymbol').value             = job.symbol;
  document.getElementById('mSymbol').disabled          = false;
  document.getElementById('mExchange').value           = job.exchange || '';
  document.getElementById('mOutputSymbol').value       = job.output_symbol || '';
  document.getElementById('mDFrom').value              = job.d_from || '';
  document.getElementById('mDTo').value                = job.d_to || '';
  document.getElementById('mDToEnable').checked        = !!job.d_to;
  document.getElementById('mDTo').disabled             = !job.d_to;
  document.getElementById('mError').textContent        = '';

  // Vendor locked on edit
  document.getElementById('mVendorSelectRow').style.display = 'none';
  document.getElementById('mVendorBadgeRow').style.display  = '';
  setVendorBadge(job.vendor);

  const isSpr = job.vendor === 'SPREAD' || _modalJobType === 'SPREAD';
  document.getElementById('mExchangeRow').style.display = isSpr ? 'none' : '';
  document.getElementById('mOutputRow').style.display   = isSpr ? '' : 'none';

  openModal();
}

function setAddVendor(v) {
  document.getElementById('mVendor').value = v;
  document.querySelectorAll('.vendor-btn').forEach(b => b.classList.toggle('active', b.dataset.v === v));
  document.getElementById('mExchangeRow').style.display = (v === 'FRED' || v === 'MANUAL_VARIABLE') ? 'none' : '';
}

function setVendorBadge(vendor) {
  const badge = document.getElementById('mVendorBadge');
  const MAP = {
    'FRED':            { label: 'FRED',           bg: 'rgba(210,153,34,.12)',  color: 'var(--orange)',  border: 'rgba(210,153,34,.3)' },
    'TRADINGVIEW':     { label: 'TRADINGVIEW',     bg: 'rgba(31,111,235,.12)', color: 'var(--blue-hi)', border: 'rgba(31,111,235,.3)' },
    'SPREAD':          { label: 'SPREAD',          bg: 'rgba(227,179,65,.1)',  color: 'var(--yellow)',  border: 'rgba(227,179,65,.25)' },
    'MANUAL_VARIABLE': { label: 'MANUAL VARIABLE', bg: 'rgba(163,113,247,.12)',color: '#a371f7',        border: 'rgba(163,113,247,.3)' },
  };
  const s = MAP[vendor] || { label: vendor||'—', bg:'transparent', color:'var(--dim)', border:'var(--border2)' };
  badge.style.cssText = `background:${s.bg};color:${s.color};border:1px solid ${s.border};font-family:var(--mono);font-size:10px;font-weight:600;padding:5px 12px;border-radius:6px;letter-spacing:.06em`;
  badge.textContent = s.label;
}

function toggleDTo() {
  const inp = document.getElementById('mDTo');
  inp.disabled = !document.getElementById('mDToEnable').checked;
  if (!inp.value) inp.value = todayStr();
}

async function saveJob() {
  const btn = document.getElementById('mSaveBtn');
  const err = document.getElementById('mError');
  err.textContent = '';

  const symbol   = document.getElementById('mSymbol').value.trim().toUpperCase();
  const exchange = document.getElementById('mExchange').value.trim().toUpperCase() || null;
  const output   = document.getElementById('mOutputSymbol').value.trim().toUpperCase() || null;
  const vendor   = document.getElementById('mVendor').value;
  const dFrom    = document.getElementById('mDFrom').value;
  const dToEl    = document.getElementById('mDTo');
  const dTo      = dToEl.disabled ? null : dToEl.value || null;
  const groupId  = parseInt(document.getElementById('mGroupId').value);
  const jobId    = document.getElementById('mJobId').value;

  if (!symbol) { err.textContent = 'Symbol is required';  return; }
  if (!dFrom)  { err.textContent = 'From date is required'; return; }
  if (!vendor) { err.textContent = 'Select a vendor'; return; }

  btn.disabled = true;
  try {
    const isEdit   = !!jobId;
    const endpoint = isEdit ? '/data_downloader/edit_job' : '/data_downloader/add_job';
    const payload  = isEdit
      ? { job_id: parseInt(jobId), symbol, exchange, output_symbol: output, d_from: dFrom, d_to: dTo }
      : { group_id: groupId, symbol, exchange, output_symbol: output, vendor, d_from: dFrom, d_to: dTo };

    const data = await apiFetch(endpoint, { method: 'POST', body: JSON.stringify(payload) });
    if (!data.ok) { err.textContent = data.error || 'Save failed'; return; }

    showFlash('success', isEdit ? 'Job updated' : 'Job added');
    closeModal();
    _loadedGroups.delete(groupId);
    await loadGroupJobs(groupId);
    _loadedGroups.add(groupId);
  } catch (e) {
    err.textContent = e.message;
  } finally {
    btn.disabled = false;
  }
}

async function deleteJob(jobId, groupId, symbol) {
  if (!confirm(`Delete "${symbol}"? This cannot be undone.`)) return;
  try {
    const data = await apiFetch('/data_downloader/delete_job', {
      method: 'POST', body: JSON.stringify({ job_id: jobId })
    });
    if (!data.ok) { showFlash('error', data.error || 'Delete failed'); return; }
    showFlash('success', `${symbol} deleted`);
    document.getElementById(`jobRow-${jobId}`)?.remove();
    const tbody = document.querySelector(`#groupBody-${groupId} tbody`);
    const rows  = tbody?.querySelectorAll('tr').length || 0;
    const cnt   = document.getElementById(`groupCount-${groupId}`);
    if (cnt) cnt.textContent = `${rows} jobs`;
  } catch(e) { showFlash('error', e.message); }
}

function openModal() {
  document.getElementById('modalBackdrop').classList.add('open');
  document.getElementById('jobModal').classList.add('open');
}
function closeModal() {
  document.getElementById('modalBackdrop').classList.remove('open');
  document.getElementById('jobModal').classList.remove('open');
  _modalEditJobId = null;
}

// ════════════════════════════════════════════════════════
// TAB 2 — STATUS  (with last_close)
// ════════════════════════════════════════════════════════

let _allJobs   = [];
let _candleMap = {};

async function loadStatusTab() {
  const tbody = document.getElementById('statusTbody');
  tbody.innerHTML = '<tr><td colspan="8" class="status-empty">Loading…</td></tr>';
  try {
    const [jobs, health] = await Promise.all([
      apiFetch('/data_downloader/jobs'),
      apiFetch('/data_downloader/last_values').catch(() => []),
    ]);
    _allJobs = jobs;
    _candleMap = {};
    health.forEach(h => { _candleMap[h.output_symbol || h.symbol] = h; });
    renderStatusTable(_allJobs);
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="8" class="status-empty" style="color:var(--red)">❌ ${e.message}</td></tr>`;
  }
}

function renderStatusTable(jobs) {
  const tbody = document.getElementById('statusTbody');
  tbody.innerHTML = '';
  if (!jobs.length) { tbody.innerHTML = '<tr><td colspan="8" class="status-empty">No jobs found.</td></tr>'; return; }
  jobs.forEach(j => {
    const tr       = document.createElement('tr');
    const vCl      = vendorClass(j.vendor);
    const candle   = _candleMap[j.output_symbol || j.symbol];
    const closeVal = candle?.last_close != null ? Number(candle.last_close).toFixed(4) : '—';
    tr.innerHTML = `
      <td style="color:var(--blue-hi)">${j.group_name}</td>
      <td style="color:#E6EDF3;font-weight:500">${j.symbol}</td>
      <td class="${vCl}">${j.vendor}</td>
      <td>${j.d_from || '—'}</td>
      <td>${j.last_run_at ? j.last_run_at.slice(0,16) : '—'}</td>
      <td style="color:#E6EDF3">${closeVal}</td>
      <td>${mkStatusCell(j.last_status)}</td>
      <td><button class="btn-run" onclick="rerunJob(${j.job_id}, this)">
        <div class="btn-spin"></div><span>Re-run</span>
      </button></td>`;
    tbody.appendChild(tr);
  });
}

function filterStatus() {
  const q = document.getElementById('statusSearch').value.toLowerCase();
  const v = document.getElementById('statusVendor').value;
  const s = document.getElementById('statusFilter').value;
  renderStatusTable(_allJobs.filter(j => {
    const mq = !q || j.symbol.toLowerCase().includes(q) || (j.group_name||'').toLowerCase().includes(q);
    const mv = !v || j.vendor === v;
    const ms = !s || j.last_status === s || (s === 'NEVER' && !j.last_status);
    return mq && mv && ms;
  }));
}

async function rerunJob(jobId, btn) {
  setSpinning(btn, true);
  try {
    const job  = _allJobs.find(j => String(j.job_id) === String(jobId));
    if (!job) { showFlash('error', `job_id ${jobId} not found`); return; }
    const data = await apiFetch('/data_downloader/run_job', {
      method: 'POST', body: JSON.stringify({ job_id: jobId, group_id: job.group_id })
    });
    showFlash(data.ok ? 'success' : 'error', data.ok ? `✅ ${job.symbol} — OK` : `❌ ${job.symbol} failed`);
    await loadStatusTab();
  } catch (e) { showFlash('error', e.message); }
  finally { setSpinning(btn, false); }
}

async function resetJob(event, btn) {
  event.stopPropagation();
  btn.disabled = true; btn.textContent = '…';
  try {
    const data = await apiFetch('/data_downloader/reset_job', {
      method: 'POST', body: JSON.stringify({ job_id: null })
    });
    showFlash('success', `Reset ${data.rows_reset} stuck job(s)`);
    const statusActive = document.getElementById('tab-status').classList.contains('active');
    if (statusActive) await loadStatusTab();
    else _loadedGroups.forEach(gid => loadGroupJobs(gid));
  } catch (e) {
    showFlash('error', `Reset failed: ${e.message}`);
    btn.disabled = false; btn.textContent = '⟳ Reset';
  }
}

async function resetAllStuck() {
  if (!confirm('Reset ALL stuck RUNNING jobs to ERROR?')) return;
  try {
    const data = await apiFetch('/data_downloader/reset_job', { method: 'POST', body: JSON.stringify({ job_id: null }) });
    showFlash('success', `Reset ${data.rows_reset} stuck job(s)`);
    await loadStatusTab();
  } catch(e) { showFlash('error', e.message); }
}

// ════════════════════════════════════════════════════════
// TAB 3 — DATA HEALTH
// ════════════════════════════════════════════════════════

let _healthData = [];

async function loadHealthTab() {
  const tbody = document.getElementById('healthTbody');
  tbody.innerHTML = '<tr><td colspan="7" class="status-empty">Loading…</td></tr>';
  try {
    _healthData = await apiFetch('/data_downloader/last_values');
    renderHealthTable(_healthData);
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="7" class="status-empty" style="color:var(--red)">❌ ${e.message}</td></tr>`;
  }
}

function renderHealthTable(data) {
  const tbody = document.getElementById('healthTbody');
  tbody.innerHTML = '';
  if (!data.length) { tbody.innerHTML = '<tr><td colspan="7" class="status-empty">No data.</td></tr>'; return; }
  data.forEach(row => {
    const tr       = document.createElement('tr');
    const vCl      = vendorClass(row.vendor);
    const symbol   = row.output_symbol || row.symbol;
    const closeVal = row.last_close != null ? Number(row.last_close).toFixed(4) : '—';
    const manualBtn = row.vendor === 'MANUAL_VARIABLE'
      ? `<button class="btn-manual" onclick="openManualModal('${symbol}')">✎ Add Value</button>` : '';
    tr.innerHTML = `
      <td style="color:var(--blue-hi)">${row.group_name}</td>
      <td style="color:#E6EDF3;font-weight:500">${symbol}</td>
      <td class="${vCl}">${row.vendor}</td>
      <td style="color:#C9D1D9">${row.last_date || '—'}</td>
      <td style="color:#E6EDF3">${closeVal}</td>
      <td>${makeAgeBadge(row.days_ago)}</td>
      <td>${manualBtn}</td>`;
    tbody.appendChild(tr);
  });
}

function makeAgeBadge(days) {
  if (days == null) return `<span class="age-null">NO DATA</span>`;
  if (days <= 3)    return `<span class="age-ok">${days}d ago</span>`;
  if (days <= 14)   return `<span class="age-warn">${days}d ago</span>`;
  return `<span class="age-crit">${days}d ago</span>`;
}

function filterHealth() {
  const q   = document.getElementById('healthSearch').value.toLowerCase();
  const v   = document.getElementById('healthVendor').value;
  const age = document.getElementById('healthAge').value;
  renderHealthTable(_healthData.filter(r => {
    const sym = (r.output_symbol || r.symbol).toLowerCase();
    const mq  = !q || sym.includes(q) || r.group_name.toLowerCase().includes(q);
    const mv  = !v || r.vendor === v;
    let ma = true;
    if (age === 'ok')   ma = r.days_ago != null && r.days_ago <= 3;
    if (age === 'warn') ma = r.days_ago != null && r.days_ago > 3 && r.days_ago <= 14;
    if (age === 'crit') ma = r.days_ago != null && r.days_ago > 14;
    if (age === 'null') ma = r.days_ago == null;
    return mq && mv && ma;
  }));
}

async function openManualModal(symbol) {
  document.getElementById('mmSymbol').value              = symbol;
  document.getElementById('manualModalTitle').textContent = `Manual — ${symbol}`;
  document.getElementById('mmDate').value                = todayStr();
  document.getElementById('mmValue').value               = '';
  document.getElementById('mmError').textContent         = '';

  const tbody = document.getElementById('mmTbody');
  tbody.innerHTML = '<tr><td colspan="2" style="color:var(--dim);font-family:var(--mono);font-size:10px;padding:8px">Loading…</td></tr>';
  document.getElementById('modalBackdrop').classList.add('open');
  document.getElementById('manualModal').classList.add('open');

  try {
    const candles = await apiFetch(`/data_downloader/manual_candles?symbol=${encodeURIComponent(symbol)}`);
    tbody.innerHTML = '';
    if (!candles.length) {
      tbody.innerHTML = '<tr><td colspan="2" style="color:var(--faint);font-family:var(--mono);font-size:10px;padding:8px">No entries yet.</td></tr>';
    } else {
      candles.forEach(c => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${c.date}</td><td style="color:var(--green)">${Number(c.value).toFixed(4)}</td>`;
        tbody.appendChild(tr);
      });
    }
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="2" style="color:var(--red);font-family:var(--mono);font-size:10px">❌ ${e.message}</td></tr>`;
  }
}

async function saveManualCandle() {
  const symbol = document.getElementById('mmSymbol').value;
  const d      = document.getElementById('mmDate').value;
  const val    = document.getElementById('mmValue').value;
  const err    = document.getElementById('mmError');
  if (!d || !val) { err.textContent = 'Date and value are required'; return; }
  try {
    const data = await apiFetch('/data_downloader/save_manual_candle', {
      method: 'POST', body: JSON.stringify({ symbol, date: d, value: parseFloat(val) })
    });
    if (!data.ok) { err.textContent = data.error || 'Save failed'; return; }
    showFlash('success', `${symbol} @ ${d} saved`);
    closeManualModal();
    await loadHealthTab();
  } catch(e) { err.textContent = e.message; }
}

function closeManualModal() {
  document.getElementById('modalBackdrop').classList.remove('open');
  document.getElementById('manualModal').classList.remove('open');
}

// ════════════════════════════════════════════════════════
// UTILS
// ════════════════════════════════════════════════════════

function setSpinning(btn, on) {
  if (!btn) return;
  btn.classList.toggle('spinning', on);
  btn.disabled = on;
}

async function apiFetch(url, opts = {}) {
  const res = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function showFlash(type, msg) {
  document.querySelector('.flash')?.remove();
  const el = document.createElement('div');
  el.className = `flash ${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}