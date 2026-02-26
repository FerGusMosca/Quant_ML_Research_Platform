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
    const text = row.textContent.toLowerCase();
    row.style.display = text.includes(q) ? '' : 'none';
  });
});

// ══ INLINE EDIT ══
function startEdit(accountNumber) {
  const row = document.querySelector(`tr[data-account="${accountNumber}"]`);
  if (!row) return;

  // Already in edit mode
  if (row.classList.contains('editing')) return;
  row.classList.add('editing');

  const nameCell   = row.querySelector('.cell-name-td');
  const brokerCell = row.querySelector('.cell-broker-td');
  const actionsCell = row.querySelector('.cell-actions');

  const currentName   = nameCell.dataset.value;
  const currentBroker = brokerCell.dataset.value;

  // Replace cells with inputs
  nameCell.innerHTML = `<input class="inline-input" id="edit-name-${accountNumber}" value="${currentName}" type="text" autocomplete="off">`;

  brokerCell.innerHTML = `
    <select class="inline-select" id="edit-broker-${accountNumber}">
      <option value="IB_PROD"   ${currentBroker === 'IB_PROD'   ? 'selected' : ''}>IB_PROD</option>
      <option value="IB_DEV"    ${currentBroker === 'IB_DEV'    ? 'selected' : ''}>IB_DEV</option>
      <option value="BYMA_PROD" ${currentBroker === 'BYMA_PROD' ? 'selected' : ''}>BYMA_PROD</option>
    </select>`;

  actionsCell.innerHTML = `
    <div class="actions-cell">
      <button class="btn-save"   onclick="saveEdit('${accountNumber}')">Save</button>
      <button class="btn-cancel" onclick="cancelEdit('${accountNumber}', '${currentName}', '${currentBroker}')">Cancel</button>
    </div>`;

  // Focus the name input
  document.getElementById(`edit-name-${accountNumber}`)?.focus();
}

function cancelEdit(accountNumber, originalName, originalBroker) {
  const row = document.querySelector(`tr[data-account="${accountNumber}"]`);
  if (!row) return;
  row.classList.remove('editing');

  const nameCell    = row.querySelector('.cell-name-td');
  const brokerCell  = row.querySelector('.cell-broker-td');
  const actionsCell = row.querySelector('.cell-actions');

  nameCell.innerHTML   = `<span class="cell-name">${originalName}</span>`;
  brokerCell.innerHTML = brokerBadgeHTML(originalBroker);
  actionsCell.innerHTML = defaultActionsHTML(accountNumber);
}

function saveEdit(accountNumber) {
  const nameInput   = document.getElementById(`edit-name-${accountNumber}`);
  const brokerInput = document.getElementById(`edit-broker-${accountNumber}`);
  if (!nameInput || !brokerInput) return;

  const newName   = nameInput.value.trim();
  const newBroker = brokerInput.value;
  if (!newName) { showFlash('error', 'Account name cannot be empty.'); return; }

  const saveBtn = document.querySelector(`tr[data-account="${accountNumber}"] .btn-save`);
  if (saveBtn) { saveBtn.textContent = '…'; saveBtn.disabled = true; }

  const fd = new FormData();
  fd.append('account_number', accountNumber);
  fd.append('account_name',   newName);
  fd.append('broker',         newBroker);

  fetch('/accounts/save', { method: 'POST', body: fd })
    .then(r => {
      if (!r.ok) throw new Error('Save failed');
      // Update row visually without full reload
      const row = document.querySelector(`tr[data-account="${accountNumber}"]`);
      row.classList.remove('editing');

      const nameCell   = row.querySelector('.cell-name-td');
      const brokerCell = row.querySelector('.cell-broker-td');
      const actCell    = row.querySelector('.cell-actions');

      nameCell.dataset.value   = newName;
      brokerCell.dataset.value = newBroker;
      nameCell.innerHTML   = `<span class="cell-name">${newName}</span>`;
      brokerCell.innerHTML = brokerBadgeHTML(newBroker);
      actCell.innerHTML    = defaultActionsHTML(accountNumber);

      showFlash('success', `Account ${accountNumber} updated.`);
    })
    .catch(() => showFlash('error', 'Error saving account.'));
}

// ══ DELETE MODAL ══
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
    .then(r => {
      if (!r.ok) throw new Error('Delete failed');
      // Remove row from DOM
      const row = document.querySelector(`tr[data-account="${accountNumber}"]`);
      if (row) {
        row.style.transition = 'opacity 0.25s, transform 0.25s';
        row.style.opacity    = '0';
        row.style.transform  = 'translateX(8px)';
        setTimeout(() => {
          row.remove();
          updateCount();
        }, 260);
      }
      showFlash('success', `Account ${accountNumber} deleted.`);
    })
    .catch(() => showFlash('error', 'Error deleting account.'));
}

function updateCount() {
  const count = document.querySelectorAll('#accountsTbody tr[data-row]').length;
  const badge = document.getElementById('accountCountBadge');
  if (badge) badge.textContent = `${count} account${count !== 1 ? 's' : ''}`;
}

// ══ ADD ACCOUNT FORM ══
document.getElementById('addAccountForm')?.addEventListener('submit', function (e) {
  e.preventDefault();
  const btn = document.getElementById('saveAccountBtn');
  btn.classList.add('loading');
  btn.disabled = true;

  const fd = new FormData(this);

  fetch('/accounts/save', { method: 'POST', body: fd })
    .then(r => {
      if (!r.ok) throw new Error('Save failed');
      // Add row to table dynamically
      const acctNumber = fd.get('account_number').trim();
      const acctName   = fd.get('account_name').trim();
      const broker     = fd.get('broker');

      const tbody = document.getElementById('accountsTbody');
      const empty = document.getElementById('emptyState');
      if (empty) empty.remove();

      const tr = document.createElement('tr');
      tr.dataset.row     = '1';
      tr.dataset.account = acctNumber;
      tr.innerHTML = `
        <td class="cell-id">${acctNumber}</td>
        <td class="cell-name-td" data-value="${acctName}"><span class="cell-name">${acctName}</span></td>
        <td class="cell-broker-td" data-value="${broker}">${brokerBadgeHTML(broker)}</td>
        <td class="cell-actions">${defaultActionsHTML(acctNumber)}</td>`;
      tr.style.opacity = '0';
      tbody.appendChild(tr);
      requestAnimationFrame(() => {
        tr.style.transition = 'opacity 0.3s';
        tr.style.opacity    = '1';
      });

      this.reset();
      updateCount();
      showFlash('success', `Account ${acctNumber} saved.`);
    })
    .catch(() => showFlash('error', 'Error saving account.'))
    .finally(() => { btn.classList.remove('loading'); btn.disabled = false; });
});

// ══ HTML HELPERS ══
function brokerBadgeHTML(broker) {
  return `<span class="broker-badge broker-${broker}">${broker}</span>`;
}

function defaultActionsHTML(accountNumber) {
  const name = document.querySelector(`tr[data-account="${accountNumber}"] .cell-name-td`)?.dataset.value ?? '';
  return `<div class="actions-cell">
    <button class="btn-edit"   onclick="startEdit('${accountNumber}')">Edit</button>
    <button class="btn-delete" onclick="confirmDelete('${accountNumber}', '${name}')">Delete</button>
  </div>`;
}

// ══ FLASH ══
function showFlash(type, msg) {
  document.querySelector('.flash')?.remove();
  const el = document.createElement('div');
  el.className   = 'flash ' + type;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}