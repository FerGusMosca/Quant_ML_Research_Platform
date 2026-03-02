// manage_accounts.js — Seeking Bias · Account Management

// ── Live clock ──
(function tick() {
  const el = document.getElementById('navClock');
  if (el) {
    const n = new Date(), pad = v => String(v).padStart(2, '0');
    el.textContent = [n.getHours(), n.getMinutes(), n.getSeconds()].map(pad).join(':');
  }
  setTimeout(tick, 1000);
})();

// ── Search / filter table ──
document.getElementById('tableSearch')?.addEventListener('input', function () {
  const q = this.value.toLowerCase();
  document.querySelectorAll('#accountsTbody tr[data-row]').forEach(row => {
    const panelRow = document.getElementById('panel-' + row.dataset.accountId);
    const visible  = row.textContent.toLowerCase().includes(q);
    row.style.display = visible ? '' : 'none';
    if (panelRow) panelRow.style.display = visible && row.classList.contains('expanded') ? '' : 'none';
  });
});

// ══════════════════════════════════════════════════════════════
//  ACCORDION
// ══════════════════════════════════════════════════════════════

const _loadedPanels = new Set();

function toggleDataPanel(accountRow) {
  if (accountRow.classList.contains('editing')) return;

  const accountId  = accountRow.dataset.accountId;
  const panelRow   = document.getElementById('panel-' + accountId);
  const chevron    = accountRow.querySelector('.chevron');
  const isExpanded = accountRow.classList.contains('expanded');

  if (isExpanded) {
    accountRow.classList.remove('expanded');
    chevron.classList.remove('open');
    panelRow.style.display = 'none';
  } else {
    accountRow.classList.add('expanded');
    chevron.classList.add('open');
    panelRow.style.display = '';
    if (!_loadedPanels.has(accountId)) {
      loadDataPanel(accountId);
    }
  }
}

function loadDataPanel(accountId) {
  fetch('/accounts/' + accountId + '/data')
    .then(r => { if (!r.ok) throw new Error(); return r.json(); })
    .then(entries => { _loadedPanels.add(accountId); renderDataPanel(accountId, entries); })
    .catch(() => {
      const inner = document.getElementById('data-panel-inner-' + accountId);
      if (inner) inner.innerHTML = '<span style="color:var(--red);font-family:var(--mono);font-size:11px">Error loading data.</span>';
    });
}

function renderDataPanel(accountId, entries) {
  const inner = document.getElementById('data-panel-inner-' + accountId);
  if (!inner) return;

  const rows = entries.map(e => dataRowHTML(e.data_id, e.data_key, e.data_value)).join('');

  inner.innerHTML =
    '<table class="data-kv-table">' +
      '<thead><tr><th style="width:220px">Key</th><th>Value</th><th style="width:120px"></th></tr></thead>' +
      '<tbody id="kv-tbody-' + accountId + '">' +
        (rows || '<tr><td colspan="3"><span style="font-family:var(--mono);font-size:11px;color:var(--faint)">No data yet.</span></td></tr>') +
      '</tbody>' +
    '</table>' +
    '<div class="kv-add-row">' +
      '<input class="inline-input kv-add-key" id="kv-key-' + accountId + '" placeholder="key" type="text" autocomplete="off">' +
      '<input class="inline-input kv-add-val" id="kv-val-' + accountId + '" placeholder="value" type="text" autocomplete="off">' +
      '<button class="btn-save kv-add-btn" onclick="addDataEntry(' + accountId + ')">Add</button>' +
    '</div>';

  updateDataCountBadge(accountId);
}

function dataRowHTML(dataId, dataKey, dataValue) {
  const safeKey = esc(dataKey);
  const safeVal = esc(dataValue);
  const dispVal = dataValue ? dataValue : '<span style="color:var(--faint)">—</span>';
  return '<tr data-data-id="' + dataId + '">' +
    '<td class="kv-key-td" data-value="' + safeKey + '"><span class="kv-key">' + dataKey + '</span></td>' +
    '<td class="kv-val-td" data-value="' + safeVal + '"><span class="kv-val">' + dispVal + '</span></td>' +
    '<td class="kv-actions"><div class="actions-cell">' +
      '<button class="btn-edit" onclick="startDataEdit(' + dataId + ');event.stopPropagation()">Edit</button>' +
      '<button class="btn-delete" onclick="confirmDataDelete(' + dataId + ',\'' + safeKey + '\');event.stopPropagation()">Delete</button>' +
    '</div></td>' +
  '</tr>';
}

// ══════════════════════════════════════════════════════════════
//  DATA ENTRY — EDIT
// ══════════════════════════════════════════════════════════════

function startDataEdit(dataId) {
  const row = document.querySelector('tr[data-data-id="' + dataId + '"]');
  if (!row || row.classList.contains('editing')) return;
  row.classList.add('editing');

  const valCell = row.querySelector('.kv-val-td');
  const actCell = row.querySelector('.kv-actions');
  const curKey  = row.querySelector('.kv-key-td').dataset.value;
  const curVal  = valCell.dataset.value;

  valCell.innerHTML = '<input class="inline-input" id="kv-edit-' + dataId + '" value="' + esc(curVal) + '" type="text" autocomplete="off">';
  actCell.innerHTML = '<div class="actions-cell">' +
    '<button class="btn-save" onclick="saveDataEdit(' + dataId + ');event.stopPropagation()">Save</button>' +
    '<button class="btn-cancel" onclick="cancelDataEdit(' + dataId + ',\'' + esc(curKey) + '\',\'' + esc(curVal) + '\');event.stopPropagation()">Cancel</button>' +
    '</div>';

  document.getElementById('kv-edit-' + dataId)?.focus();
}

function cancelDataEdit(dataId, origKey, origVal) {
  const row = document.querySelector('tr[data-data-id="' + dataId + '"]');
  if (!row) return;
  row.classList.remove('editing');
  const dispVal = origVal ? origVal : '<span style="color:var(--faint)">—</span>';
  row.querySelector('.kv-val-td').innerHTML = '<span class="kv-val">' + dispVal + '</span>';
  row.querySelector('.kv-actions').innerHTML = '<div class="actions-cell">' +
    '<button class="btn-edit" onclick="startDataEdit(' + dataId + ');event.stopPropagation()">Edit</button>' +
    '<button class="btn-delete" onclick="confirmDataDelete(' + dataId + ',\'' + esc(origKey) + '\');event.stopPropagation()">Delete</button>' +
    '</div>';
}

function saveDataEdit(dataId) {
  const row      = document.querySelector('tr[data-data-id="' + dataId + '"]');
  const valInput = document.getElementById('kv-edit-' + dataId);
  if (!row || !valInput) return;

  const panelEl   = row.closest('[id^="data-panel-inner-"]');
  const accountId = panelEl ? panelEl.id.replace('data-panel-inner-', '') : null;
  const dataKey   = row.querySelector('.kv-key-td').dataset.value;
  const newVal    = valInput.value.trim();

  const saveBtn = row.querySelector('.btn-save');
  if (saveBtn) { saveBtn.textContent = '…'; saveBtn.disabled = true; }

  const fd = new FormData();
  fd.append('account_id', accountId);
  fd.append('data_key',   dataKey);
  fd.append('data_value', newVal);

  fetch('/accounts/data/save', { method: 'POST', body: fd })
    .then(r => { if (!r.ok) throw new Error(); return r.json(); })
    .then(() => {
      row.classList.remove('editing');
      const valCell = row.querySelector('.kv-val-td');
      valCell.dataset.value = newVal;
      const dispVal = newVal ? newVal : '<span style="color:var(--faint)">—</span>';
      valCell.innerHTML = '<span class="kv-val">' + dispVal + '</span>';
      row.querySelector('.kv-actions').innerHTML = '<div class="actions-cell">' +
        '<button class="btn-edit" onclick="startDataEdit(' + dataId + ');event.stopPropagation()">Edit</button>' +
        '<button class="btn-delete" onclick="confirmDataDelete(' + dataId + ',\'' + esc(dataKey) + '\');event.stopPropagation()">Delete</button>' +
        '</div>';
      showFlash('success', '"' + dataKey + '" updated.');
    })
    .catch(() => showFlash('error', 'Error saving entry.'));
}

// ══════════════════════════════════════════════════════════════
//  DATA ENTRY — QUICK ADD
// ══════════════════════════════════════════════════════════════

function addDataEntry(accountId) {
  const keyInput = document.getElementById('kv-key-' + accountId);
  const valInput = document.getElementById('kv-val-' + accountId);
  if (!keyInput || !valInput) return;

  const dataKey = keyInput.value.trim();
  const dataVal = valInput.value.trim();
  if (!dataKey) { showFlash('error', 'Key cannot be empty.'); keyInput.focus(); return; }

  const btn = document.querySelector('#data-panel-inner-' + accountId + ' .kv-add-btn');
  if (btn) { btn.textContent = '…'; btn.disabled = true; }

  const fd = new FormData();
  fd.append('account_id', accountId);
  fd.append('data_key',   dataKey);
  fd.append('data_value', dataVal);

  fetch('/accounts/data/save', { method: 'POST', body: fd })
    .then(r => { if (!r.ok) throw new Error(); return r.json(); })
    .then(data => {
      const tbody = document.getElementById('kv-tbody-' + accountId);
      const placeholder = tbody.querySelector('td[colspan]');
      if (placeholder) placeholder.closest('tr').remove();

      tbody.insertAdjacentHTML('beforeend', dataRowHTML(data.data_id, dataKey, dataVal));

      keyInput.value = '';
      valInput.value = '';
      keyInput.focus();
      updateDataCountBadge(accountId);
      showFlash('success', '"' + dataKey + '" added.');
    })
    .catch(() => showFlash('error', 'Error adding entry.'))
    .finally(() => { if (btn) { btn.textContent = 'Add'; btn.disabled = false; } });
}

// ══════════════════════════════════════════════════════════════
//  DATA ENTRY — DELETE
// ══════════════════════════════════════════════════════════════

let _pendingDataDeleteId = null;

function confirmDataDelete(dataId, dataKey) {
  _pendingDataDeleteId = dataId;
  document.getElementById('delDataKey').textContent = dataKey;
  document.getElementById('deleteDataModal').classList.add('open');
}

function closeDataDeleteModal() {
  _pendingDataDeleteId = null;
  document.getElementById('deleteDataModal').classList.remove('open');
}

document.getElementById('deleteDataModal')?.addEventListener('click', e => {
  if (e.target === document.getElementById('deleteDataModal')) closeDataDeleteModal();
});

function executeDataDelete() {
  if (!_pendingDataDeleteId) return;
  const dataId = _pendingDataDeleteId;
  _pendingDataDeleteId = null;
  closeDataDeleteModal();

  const fd = new FormData();
  fd.append('data_id', dataId);

  fetch('/accounts/data/delete', { method: 'POST', body: fd })
    .then(r => { if (!r.ok) throw new Error(); return r.json(); })
    .then(() => {
      const row = document.querySelector('tr[data-data-id="' + dataId + '"]');
      if (row) {
        const panelEl   = row.closest('[id^="data-panel-inner-"]');
        const accountId = panelEl ? panelEl.id.replace('data-panel-inner-', '') : null;
        row.style.transition = 'opacity 0.25s';
        row.style.opacity    = '0';
        setTimeout(() => { row.remove(); if (accountId) updateDataCountBadge(accountId); }, 260);
      }
      showFlash('success', 'Entry deleted.');
    })
    .catch(() => showFlash('error', 'Error deleting entry.'));
}

// ══════════════════════════════════════════════════════════════
//  ACCOUNT — INLINE EDIT
// ══════════════════════════════════════════════════════════════

function startEdit(accountNumber) {
  const row = document.querySelector('tr[data-account="' + accountNumber + '"]');
  if (!row || row.classList.contains('editing')) return;
  row.classList.add('editing');

  const nameCell    = row.querySelector('.cell-name-td');
  const brokerCell  = row.querySelector('.cell-broker-td');
  const actionsCell = row.querySelector('.cell-actions');
  const currentName   = nameCell.dataset.value;
  const currentBroker = brokerCell.dataset.value;

  nameCell.innerHTML = '<input class="inline-input" id="edit-name-' + accountNumber + '" value="' + currentName + '" type="text" autocomplete="off">';
  brokerCell.innerHTML =
    '<select class="inline-select" id="edit-broker-' + accountNumber + '">' +
    '<option value="IB_PROD"'   + (currentBroker === 'IB_PROD'   ? ' selected' : '') + '>IB_PROD</option>' +
    '<option value="IB_DEV"'    + (currentBroker === 'IB_DEV'    ? ' selected' : '') + '>IB_DEV</option>' +
    '<option value="BYMA_PROD"' + (currentBroker === 'BYMA_PROD' ? ' selected' : '') + '>BYMA_PROD</option>' +
    '</select>';
  actionsCell.innerHTML =
    '<div class="actions-cell">' +
    '<button class="btn-save" onclick="saveEdit(\'' + accountNumber + '\')">Save</button>' +
    '<button class="btn-cancel" onclick="cancelEdit(\'' + accountNumber + '\',\'' + currentName + '\',\'' + currentBroker + '\')">Cancel</button>' +
    '</div>';

  document.getElementById('edit-name-' + accountNumber)?.focus();
}

function cancelEdit(accountNumber, originalName, originalBroker) {
  const row = document.querySelector('tr[data-account="' + accountNumber + '"]');
  if (!row) return;
  row.classList.remove('editing');
  row.querySelector('.cell-name-td').innerHTML   = '<span class="cell-name">' + originalName + '</span>';
  row.querySelector('.cell-broker-td').innerHTML = brokerBadgeHTML(originalBroker);
  row.querySelector('.cell-actions').innerHTML   = defaultActionsHTML(accountNumber);
}

function saveEdit(accountNumber) {
  const nameInput   = document.getElementById('edit-name-' + accountNumber);
  const brokerInput = document.getElementById('edit-broker-' + accountNumber);
  if (!nameInput || !brokerInput) return;

  const newName   = nameInput.value.trim();
  const newBroker = brokerInput.value;
  if (!newName) { showFlash('error', 'Account name cannot be empty.'); return; }

  const saveBtn = document.querySelector('tr[data-account="' + accountNumber + '"] .btn-save');
  if (saveBtn) { saveBtn.textContent = '…'; saveBtn.disabled = true; }

  const fd = new FormData();
  fd.append('account_number', accountNumber);
  fd.append('account_name',   newName);
  fd.append('broker',         newBroker);

  fetch('/accounts/save', { method: 'POST', body: fd })
    .then(r => { if (!r.ok) throw new Error(); })
    .then(() => {
      const row = document.querySelector('tr[data-account="' + accountNumber + '"]');
      row.classList.remove('editing');
      const nameCell = row.querySelector('.cell-name-td');
      const brkCell  = row.querySelector('.cell-broker-td');
      nameCell.dataset.value = newName;
      brkCell.dataset.value  = newBroker;
      nameCell.innerHTML = '<span class="cell-name">' + newName + '</span>';
      brkCell.innerHTML  = brokerBadgeHTML(newBroker);
      row.querySelector('.cell-actions').innerHTML = defaultActionsHTML(accountNumber);
      showFlash('success', 'Account ' + accountNumber + ' updated.');
    })
    .catch(() => showFlash('error', 'Error saving account.'));
}

// ══════════════════════════════════════════════════════════════
//  ACCOUNT — DELETE
// ══════════════════════════════════════════════════════════════

let _pendingDeleteId = null;

function confirmDelete(accountNumber, accountName) {
  _pendingDeleteId = accountNumber;
  document.getElementById('delAccountId').textContent   = accountNumber;
  document.getElementById('delAccountName').textContent = accountName;
  document.getElementById('deleteModal').classList.add('open');
}

function closeDeleteModal() {
  _pendingDeleteId = null;
  document.getElementById('deleteModal').classList.remove('open');
}

document.getElementById('deleteModal')?.addEventListener('click', e => {
  if (e.target === document.getElementById('deleteModal')) closeDeleteModal();
});

function executeDelete() {
  if (!_pendingDeleteId) return;
  const accountNumber = _pendingDeleteId;
  _pendingDeleteId = null;
  closeDeleteModal();

  const fd = new FormData();
  fd.append('account_number', accountNumber);

  fetch('/accounts/delete', { method: 'POST', body: fd })
    .then(r => { if (!r.ok) throw new Error(); })
    .then(() => {
      const row = document.querySelector('tr[data-account="' + accountNumber + '"]');
      if (row) {
        const accountId = row.dataset.accountId;
        const panelRow  = document.getElementById('panel-' + accountId);
        row.style.transition = 'opacity 0.25s';
        row.style.opacity    = '0';
        if (panelRow) { panelRow.style.transition = 'opacity 0.25s'; panelRow.style.opacity = '0'; }
        setTimeout(() => { row.remove(); panelRow?.remove(); updateCount(); }, 260);
      }
      showFlash('success', 'Account ' + accountNumber + ' deleted.');
    })
    .catch(() => showFlash('error', 'Error deleting account.'));
}

// ══════════════════════════════════════════════════════════════
//  ACCOUNT — ADD FORM
// ══════════════════════════════════════════════════════════════

document.getElementById('addAccountForm')?.addEventListener('submit', function (e) {
  e.preventDefault();
  const btn = document.getElementById('saveAccountBtn');
  btn.classList.add('loading');
  btn.disabled = true;

  const fd = new FormData(this);

  fetch('/accounts/save', { method: 'POST', body: fd })
    .then(r => { if (!r.ok) throw new Error(); })
    .then(() => {
      // Reload to get the real account_id from the DB
      window.location.reload();
    })
    .catch(() => showFlash('error', 'Error saving account.'))
    .finally(() => { btn.classList.remove('loading'); btn.disabled = false; });
});

// ══════════════════════════════════════════════════════════════
//  HELPERS
// ══════════════════════════════════════════════════════════════

function brokerBadgeHTML(broker) {
  return '<span class="broker-badge broker-' + broker + '">' + broker + '</span>';
}

function defaultActionsHTML(accountNumber) {
  const name = document.querySelector('tr[data-account="' + accountNumber + '"] .cell-name-td')?.dataset.value ?? '';
  return '<div class="actions-cell">' +
    '<button class="btn-edit" onclick="startEdit(\'' + accountNumber + '\')">Edit</button>' +
    '<button class="btn-delete" onclick="confirmDelete(\'' + accountNumber + '\',\'' + name + '\')">Delete</button>' +
    '</div>';
}

function updateCount() {
  const count = document.querySelectorAll('#accountsTbody tr[data-row]').length;
  const badge = document.getElementById('accountCountBadge');
  if (badge) badge.textContent = count + ' account' + (count !== 1 ? 's' : '');
}

function updateDataCountBadge(accountId) {
  const tbody = document.getElementById('kv-tbody-' + accountId);
  const badge = document.querySelector('#datacount-' + accountId + ' .data-count-badge');
  if (!tbody || !badge) return;
  const n = tbody.querySelectorAll('tr[data-data-id]').length;
  badge.textContent = n > 0 ? n + ' key' + (n !== 1 ? 's' : '') : '0 keys';
}

function esc(s) {
  return String(s ?? '').replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

function showFlash(type, msg) {
  document.querySelector('.flash')?.remove();
  const el = document.createElement('div');
  el.className   = 'flash ' + type;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}