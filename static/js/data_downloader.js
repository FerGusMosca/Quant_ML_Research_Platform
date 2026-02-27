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

// ── Tab switch ────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
  if (name === 'status') loadStatusTab();
}

// ── Boot ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadGroups();
  switchTab('execute');
});

// ════════════════════════════════════════
// TAB 1 — EXECUTE
// ════════════════════════════════════════

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
  card.className = 'group-card';
  card.dataset.groupId = group.group_id;

  const badgeClass = group.job_type === 'SPREAD' ? 'badge-spread' : 'badge-download';
  const badgeLabel = group.job_type === 'SPREAD' ? '⇄ Spread' : '↓ Download';

  card.innerHTML = `
    <div class="group-hdr" onclick="toggleGroup(${group.group_id})">
      <span class="group-chevron">▶</span>
      <span class="group-name">${group.group_name}</span>
      <span class="group-badge ${badgeClass}">${badgeLabel}</span>
      <span class="group-count">${group.job_count} jobs</span>
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

// lazy-load set
const _loadedGroups = new Set();

async function toggleGroup(groupId) {
  const card = document.querySelector(`.group-card[data-group-id="${groupId}"]`);
  if (card.classList.contains('open')) {
    card.classList.remove('open');
    return;
  }
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
    body.innerHTML = '';
    if (!jobs.length) {
      body.innerHTML = '<div style="font-family:var(--mono);font-size:10px;color:var(--faint);padding:8px">No jobs.</div>';
      return;
    }
    const table = document.createElement('table');
    table.className = 'job-table';
    table.innerHTML = `<thead><tr>
      <th>Symbol</th><th>Output</th><th>Vendor</th><th>From</th><th>Last Run</th><th>Status</th><th></th>
    </tr></thead><tbody id="jobRows-${groupId}"></tbody>`;
    body.appendChild(table);
    jobs.forEach(j => table.querySelector('tbody').appendChild(buildJobRow(j)));
  } catch (e) {
    body.innerHTML = `<div style="color:var(--red);font-family:var(--mono);font-size:10px;padding:8px">❌ ${e.message}</div>`;
  }
}

function buildJobRow(job) {
  const tr = document.createElement('tr');
  tr.id = `jobRow-${job.job_id}`;
  const vClass  = job.vendor === 'FRED' ? 'td-vendor-fred' : 'td-vendor-tv';
  const outCell = job.output_symbol
    ? `<span class="td-output">${job.output_symbol}</span>`
    : `<span style="color:var(--faint)">—</span>`;
  tr.innerHTML = `
    <td class="td-symbol">${job.symbol}</td>
    <td>${outCell}</td>
    <td class="${vClass}">${job.vendor}</td>
    <td class="td-date">${job.d_from || '—'}</td>
    <td class="td-date">${job.last_run_at ? job.last_run_at.slice(0,16) : '—'}</td>
    <td id="jobStatus-${job.job_id}">${mkStatusCell(job.last_status)}</td>
    <td>
      <button class="btn-run" id="btnRun-${job.job_id}"
        onclick="runJob(${job.job_id}, ${job.group_id})">
        <div class="btn-spin"></div><span>Run</span>
      </button>
    </td>`;
  return tr;
}

function mkStatusCell(status) {
  if (!status) return `<span class="status-dot dot-none"></span><span style="color:var(--faint);font-size:9px;font-family:var(--mono)">NEVER</span>`;
  const cls   = status === 'OK' ? 'dot-ok' : status === 'ERROR' ? 'dot-error' : 'dot-running';
  const color = status === 'OK' ? 'var(--green)' : status === 'ERROR' ? 'var(--red)' : 'var(--orange)';
  // Add reset button inline for stuck RUNNING jobs
  const resetBtn = status === 'RUNNING'
    ? ` <button class="btn-reset" onclick="resetJob(event, this)" title="Reset stuck job">⟳ Reset</button>`
    : '';
  return `<span class="status-dot ${cls}"></span><span style="color:${color};font-size:9px;font-family:var(--mono)">${status}</span>${resetBtn}`;
}

// ── Run single job ────────────────────────────────────────────────────────
async function runJob(jobId, groupId) {
  const btn    = document.getElementById(`btnRun-${jobId}`);
  const sc     = document.getElementById(`jobStatus-${jobId}`);
  const cbody  = document.getElementById(`consoleBody-${groupId}`);

  setSpinning(btn, true);
  if (sc) sc.innerHTML = mkStatusCell('RUNNING');
  openConsole(groupId);
  log(cbody, `▶ Job ${jobId} — ${new Date().toLocaleTimeString()}\n`, 'c-dim');

  try {
    const data = await apiFetch('/data_downloader/run_job', {
      method: 'POST', body: JSON.stringify({ job_id: jobId, group_id: groupId })
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

// ── Run entire group ──────────────────────────────────────────────────────
async function runGroup(groupId) {
  const btn   = document.getElementById(`runAll-${groupId}`);
  const cbody = document.getElementById(`consoleBody-${groupId}`);

  setSpinning(btn, true);
  openConsole(groupId);
  log(cbody, `▶ Running ALL — ${new Date().toLocaleTimeString()}\n\n`, 'c-dim');

  try {
    const data = await apiFetch('/data_downloader/run_group', {
      method: 'POST', body: JSON.stringify({ group_id: groupId })
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

// ── Console helpers ───────────────────────────────────────────────────────
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

// ════════════════════════════════════════
// TAB 2 — STATUS
// ════════════════════════════════════════

let _allJobs = [];

async function loadStatusTab() {
  const tbody = document.getElementById('statusTbody');
  tbody.innerHTML = '<tr><td colspan="7" class="status-empty">Loading…</td></tr>';
  try {
    _allJobs = await apiFetch('/data_downloader/jobs');
    renderStatusTable(_allJobs);
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="7" class="status-empty" style="color:var(--red)">❌ ${e.message}</td></tr>`;
  }
}

function renderStatusTable(jobs) {
  const tbody = document.getElementById('statusTbody');
  tbody.innerHTML = '';
  if (!jobs.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="status-empty">No jobs found.</td></tr>';
    return;
  }
  jobs.forEach(j => {
    const tr  = document.createElement('tr');
    const vCl = j.vendor === 'FRED' ? 'td-vendor-fred' : 'td-vendor-tv';
    tr.innerHTML = `
      <td style="font-family:var(--mono);font-size:10px;color:var(--blue-hi)">${j.group_name}</td>
      <td style="font-family:var(--mono);font-size:11px;color:#E6EDF3">${j.symbol}</td>
      <td style="font-family:var(--mono);font-size:10px" class="${vCl}">${j.vendor}</td>
      <td class="td-date">${j.d_from || '—'}</td>
      <td class="td-date">${j.last_run_at ? j.last_run_at.slice(0,16) : '—'}</td>
      <td>${mkStatusCell(j.last_status)}</td>
      <td>
        <button class="btn-run" onclick="rerunJob(${j.job_id}, this)">
          <div class="btn-spin"></div><span>Re-run</span>
        </button>
      </td>`;
    tbody.appendChild(tr);
  });
}

function filterStatus() {
  const q  = document.getElementById('statusSearch').value.toLowerCase();
  const v  = document.getElementById('statusVendor').value;
  const s  = document.getElementById('statusFilter').value;
  renderStatusTable(_allJobs.filter(j => {
    const mq = !q || j.symbol.toLowerCase().includes(q) || j.group_name.toLowerCase().includes(q);
    const mv = !v || j.vendor === v;
    const ms = !s || j.last_status === s || (s === 'NEVER' && !j.last_status);
    return mq && mv && ms;
  }));
}

async function rerunJob(jobId, btn) {
  setSpinning(btn, true);
  try {
    const job  = _allJobs.find(j => String(j.job_id) === String(jobId));
    if (!job)  { showFlash('error', `job_id ${jobId} not found`); return; }
    const data = await apiFetch('/data_downloader/run_job', {
      method: 'POST', body: JSON.stringify({ job_id: jobId, group_id: job.group_id })
    });
    showFlash(data.ok ? 'success' : 'error',
              data.ok ? `✅ ${job.symbol} — OK` : `❌ ${job.symbol} failed`);
    await loadStatusTab();
  } catch (e) {
    showFlash('error', e.message);
  } finally {
    setSpinning(btn, false);
  }
}

// ── Reset stuck jobs ──────────────────────────────────────────────────────

async function resetJob(event, btn) {
  // Can be called from status-cell inline button (has data-job-id)
  // or from the global "Reset All Stuck" button (no job_id)
  event.stopPropagation();
  const jobId = btn.dataset.jobId ? parseInt(btn.dataset.jobId) : null;
  btn.disabled = true;
  btn.textContent = '…';
  try {
    const data = await apiFetch('/data_downloader/reset_job', {
      method: 'POST',
      body: JSON.stringify({ job_id: jobId }),
    });
    showFlash('success', `Reset ${data.rows_reset} stuck job(s)`);
    // Refresh whichever tab is visible
    const statusActive = document.getElementById('tab-status').classList.contains('active');
    if (statusActive) await loadStatusTab();
    else {
      // reload all open groups
      _loadedGroups.forEach(gid => loadGroupJobs(gid));
    }
  } catch (e) {
    showFlash('error', `Reset failed: ${e.message}`);
    btn.disabled = false;
    btn.textContent = '⟳ Reset';
  }
}

async function resetAllStuck() {
  if (!confirm('Reset ALL stuck RUNNING jobs to ERROR?')) return;
  try {
    const data = await apiFetch('/data_downloader/reset_job', {
      method: 'POST',
      body: JSON.stringify({ job_id: null }),
    });
    showFlash('success', `Reset ${data.rows_reset} stuck job(s)`);
    await loadStatusTab();
  } catch (e) {
    showFlash('error', e.message);
  }
}
function setSpinning(btn, on) {
  btn.classList.toggle('spinning', on);
  btn.disabled = on;
}

async function apiFetch(url, opts = {}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
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