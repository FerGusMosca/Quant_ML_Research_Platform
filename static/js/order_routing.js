// order_routing.js — Seeking Bias · Order Routing Dashboard

// ── State ──
let currentSide     = 'Buy';
let currentQtyMode  = 'cash';
let allAccounts     = [];
let selectedAccountId   = '';
let selectedAccountName = '';
let _execReportCache = {};
// ── Live clock ──
(function tick() {
  const n = new Date(), pad = v => String(v).padStart(2, '0');
  const el = document.getElementById('liveClock');
  if (el) el.textContent = [n.getHours(), n.getMinutes(), n.getSeconds()].map(pad).join(':');
  setTimeout(tick, 1000);
})();

// ── Tabs ──
function switchTab(name, btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}

// ── Side selector ──
function setSide(side) {
  currentSide = side;
  document.getElementById('btn-buy').className  = 'side-btn' + (side === 'Buy'  ? ' active-buy'  : '');
  document.getElementById('btn-sell').className = 'side-btn' + (side === 'Sell' ? ' active-sell' : '');

  const sb = document.getElementById('submitBtn');
  sb.textContent = side === 'Buy' ? 'Submit Buy Order' : 'Submit Sell Order';
  sb.className   = 'submit-btn' + (side === 'Sell' ? ' sell-mode' : '');

  if (side === 'Sell') {
    setQtyMode('nom');
    document.getElementById('qty-mode-tabs').style.display = 'none';
  } else {
    document.getElementById('qty-mode-tabs').style.display = 'flex';
    setQtyMode(currentQtyMode === 'nom' ? 'nom' : 'cash');
  }
}

// ── Qty mode ──
function setQtyMode(mode) {
  currentQtyMode = mode;
  document.getElementById('qtab-cash').className = 'qty-tab' + (mode === 'cash' ? ' active' : '');
  document.getElementById('qtab-nom').className  = 'qty-tab' + (mode === 'nom'  ? ' active' : '');
  document.getElementById('cash_qty').style.display = mode === 'cash' ? 'block' : 'none';
  document.getElementById('nom_qty').style.display  = mode === 'nom'  ? 'block' : 'none';
}

// ── Currency format ──
function formatCurrency(input) {
  let raw = input.value.replace(/[^0-9.]/g, '');
  if (raw === '') { input.value = ''; return; }
  const parts = raw.split('.');
  parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  input.value = '$' + parts.join('.');
}

// ── Broker change ──
function onBrokerChange() {
  const broker = document.getElementById('broker').value;
  if (broker.startsWith('IB')) {
    document.getElementById('currency').value = 'USD';
    document.getElementById('exchange').value = 'SMART';
  } else if (broker === 'BYMA_PROD') {
    document.getElementById('currency').value = 'ARS';
    document.getElementById('exchange').value = 'BUE';
  }
  loadAccounts();
}

// ── Searchable account dropdown ──
function loadAccounts() {
  const broker = document.getElementById('broker').value;
  fetch(`/routing_dashboard/get_accounts?broker=${broker}`)
    .then(r => r.json())
    .then(accounts => {
      allAccounts         = accounts;
      selectedAccountId   = accounts.length ? accounts[0].account_id  : '';
      selectedAccountName = accounts.length ? accounts[0].client_name : '';
      document.getElementById('accountSearch').value = accounts.length
        ? `${accounts[0].account_id} — ${accounts[0].client_name}` : '';
      renderDropdown(accounts);
    })
    .catch(() => {});
}

function renderDropdown(accounts) {
  const dd = document.getElementById('accountDropdown');
  dd.innerHTML = '';
  accounts.forEach(acc => {
    const div = document.createElement('div');
    div.className = 'dropdown-option' + (acc.account_id === selectedAccountId ? ' selected' : '');
    div.innerHTML = `<span class="opt-id">${acc.account_id}</span> <span class="opt-name">— ${acc.client_name}</span>`;
    div.onmousedown = () => selectAccount(acc);
    dd.appendChild(div);
  });
}

function filterAccounts() {
  const q = document.getElementById('accountSearch').value.toLowerCase();
  const filtered = allAccounts.filter(a =>
    a.account_id.toLowerCase().includes(q) || a.client_name.toLowerCase().includes(q)
  );
  renderDropdown(filtered);
  openDropdown();
}

function openDropdown() {
  const q = document.getElementById('accountSearch').value.toLowerCase();
  renderDropdown(allAccounts.filter(a =>
    a.account_id.toLowerCase().includes(q) || a.client_name.toLowerCase().includes(q)
  ));
  document.getElementById('accountDropdown').classList.add('open');
}

function closeDropdownDelayed() {
  setTimeout(() => document.getElementById('accountDropdown').classList.remove('open'), 200);
}

function selectAccount(acc) {
  selectedAccountId   = acc.account_id;
  selectedAccountName = acc.client_name;
  document.getElementById('accountSearch').value = `${acc.account_id} — ${acc.client_name}`;
  document.getElementById('accountDropdown').classList.remove('open');
}

// ── Confirm order modal ──
function confirmOrderModal() {
  const symbol   = document.getElementById('symbol').value.trim().toUpperCase();
  const cashRaw  = document.getElementById('cash_qty').value.replace(/[^0-9.]/g, '');
  const nomQty   = document.getElementById('nom_qty').value;
  const broker   = document.getElementById('broker');
  const exchange = document.getElementById('exchange').value;

  if (!symbol)            { showFlash('error', 'Symbol is required.');        return; }
  if (!selectedAccountId) { showFlash('error', 'Please select an account.');  return; }

  let qtyDisplay = '';
  if (currentQtyMode === 'cash' && currentSide === 'Buy') {
    if (!cashRaw) { showFlash('error', 'Enter a cash amount.'); return; }
    qtyDisplay = '$' + parseFloat(cashRaw).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  } else {
    if (!nomQty) { showFlash('error', 'Enter a nominal quantity.'); return; }
    qtyDisplay = parseInt(nomQty).toLocaleString('en-US') + ' units';
  }

  document.getElementById('conf-symbol').textContent   = symbol;
  document.getElementById('conf-qty').textContent      = qtyDisplay;
  document.getElementById('conf-broker').textContent   = broker.selectedOptions[0].text;
  document.getElementById('conf-exchange').textContent = exchange;
  document.getElementById('conf-account').textContent  = `${selectedAccountId} — ${selectedAccountName}`;

  const sideEl = document.getElementById('conf-side');
  sideEl.textContent = currentSide;
  sideEl.className   = 'confirm-val ' + (currentSide === 'Buy' ? 'buy-val' : 'sell-val');

  const sendBtn = document.getElementById('confirmSendBtn');
  sendBtn.className   = 'btn ' + (currentSide === 'Buy' ? 'btn-confirm-buy' : 'btn-confirm-sell');
  sendBtn.textContent = currentSide === 'Buy' ? 'Confirm Buy ▲' : 'Confirm Sell ▼';

  openModal('confirmModal');
}

function sendOrder() {
  const symbol   = document.getElementById('symbol').value.trim().toUpperCase();
  const cashRaw  = document.getElementById('cash_qty').value.replace(/[^0-9.]/g, '');
  const nomQty   = document.getElementById('nom_qty').value;
  const broker   = document.getElementById('broker').value;
  const currency = document.getElementById('currency').value;
  const exchange = document.getElementById('exchange').value;

  const payload = {
    symbol,
    side:     currentSide,
    cash_qty: (currentQtyMode === 'cash' && currentSide === 'Buy' && cashRaw) ? parseFloat(cashRaw) : null,
    nom_qty:  nomQty ? parseInt(nomQty) : null,
    broker, currency, exchange,
    account: selectedAccountId,
  };

  closeModal('confirmModal');
  showSpinner('Sending order…');

  fetch('/routing_dashboard/submit_order', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  .then(r => r.json())
  .then(data => {
    hideSpinner();
    showFlash('success', data.message || 'Order sent successfully.');
    updateExecutionReports();
  })
  .catch(() => {
    hideSpinner();
    showFlash('error', 'Network error sending order.');
  });
}

// ── Portfolio ──
function loadPortfolio() {
  if (!selectedAccountId) { showFlash('error', 'Select an account first.'); return; }
  showSpinner('Loading portfolio…');

  fetch(`/routing_dashboard/get_portfolio?account_id=${selectedAccountId}`)
    .then(r => r.json())
    .then(data => {
      hideSpinner();
      if (data.error) { showFlash('error', data.error); return; }
      renderPortfolioModal(data);
      openModal('portfolioModal');
    })
    .catch(() => { hideSpinner(); showFlash('error', 'Error loading portfolio.'); });
}

function renderPortfolioModal(data) {
  const body = document.getElementById('portfolioModalBody');

  let html = `
    <div class="portfolio-account-hdr">
      <span class="portfolio-account-name">${selectedAccountName}</span>
      <span class="portfolio-account-id">${selectedAccountId}</span>
    </div>`;

  // Currencies chips
  html += `<div class="portfolio-section-title">Cash &amp; Currencies</div>
    <div class="currencies-row">`;
  (data.currencies || []).forEach(c => {
    const sym = c.currency === 'ARS' ? '$' : c.currency === 'USD' ? 'US$' : c.currency;
    const fmt = parseFloat(c.amount).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    html += `<div class="currency-chip">
      <span class="chip-amount">${sym} ${fmt}</span>
      <span class="chip-label">${c.currency}</span>
    </div>`;
  });
  html += `</div>`;

  // Securities table
  html += `<div class="portfolio-section-title" style="margin-top:8px">Securities</div>
    <table class="portfolio-table">
      <thead><tr>
        <th>Symbol</th><th>Type</th><th>Currency</th><th>Qty</th><th>Avg. Price</th><th>Action</th>
      </tr></thead><tbody>`;
  (data.securities || []).forEach(s => {
    const avgFmt = parseFloat(s.avg_px).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    html += `<tr>
      <td class="mono-cell" style="font-weight:600;color:#E6EDF3">${s.symbol}</td>
      <td class="dim-cell">${s.type}</td>
      <td class="dim-cell">${s.currency}</td>
      <td class="mono-cell">${parseInt(s.qty).toLocaleString('en-US')}</td>
      <td class="mono-cell">$ ${avgFmt}</td>
      <td>
        <button class="unwind-btn" onclick="unwindPosition('${s.symbol}','${s.qty}','${s.currency}')">
          Sell →
        </button>
      </td>
    </tr>`;
  });
  html += `</tbody></table>`;

  body.innerHTML = html;
}

function unwindPosition(symbol, qty, currency) {
  document.getElementById('symbol').value   = symbol.toUpperCase();
  document.getElementById('nom_qty').value  = parseInt(qty);
  document.getElementById('currency').value = currency;
  setSide('Sell');
  closeModal('portfolioModal');
}

// ── Execution reports ──
function updateExecutionReports() {
  fetch('/routing_dashboard/get_execution_reports')
    .then(r => r.json())
    .then(data => {
      const tbody = document.getElementById('execution-reports-table');
      const count = document.getElementById('exec-count');
      count.textContent = `${data.length} order${data.length !== 1 ? 's' : ''}`;
      tbody.innerHTML = '';

      _execReportCache = {};                              // reset cache each poll

      data.forEach(row => {
        _execReportCache[row.cl_ord_id] = row;           // ← store full row

        const status     = (row.ord_status || '').toLowerCase();
        const badgeClass = status.includes('fill')   ? 'filled'    :
                           status.includes('reject') ? 'rejected'  :
                           status.includes('cancel') ? 'cancelled' : 'pending';
        const tr = document.createElement('tr');
        tr.className = 'clickable-row';
        tr.onclick   = () => showExecReportDetail(row.cl_ord_id);   // ← click handler
        tr.innerHTML = `
          <td class="dim-cell">${row.short_cl_ord_id}</td>
          <td class="mono-cell" style="font-weight:600;color:#E6EDF3">${row.symbol}</td>
          <td class="${row.side === 'Buy' ? 'side-cell-buy' : 'side-cell-sell'}">${row.side}</td>
          <td class="mono-cell">${row.cum_qty}</td>
          <td><span class="status-badge ${badgeClass}">${row.ord_status}</span></td>
          <td class="dim-cell">${row.transact_time}</td>
          <td>
            <button class="cancel-btn"
              onclick="event.stopPropagation(); confirmCancel('${row.cl_ord_id}','${row.symbol}')"
              title="Cancel order">✕</button>
          </td>`;
        tbody.appendChild(tr);
      });
    })
    .catch(() => {});
}

// ── Cancel order modal ──
let _pendingCancelId = null;

function confirmCancel(clOrdId, symbol) {
  _pendingCancelId = clOrdId;
  // Trim the short display ID (last 8 chars after last dash)
  const shortId = clOrdId.length > 12 ? '…' + clOrdId.slice(-10) : clOrdId;
  document.getElementById('cancelOrderId').textContent = shortId;
  document.getElementById('cancelSymbol').textContent  = symbol;
  openModal('cancelModal');
}

function executeCancelOrder() {
  if (!_pendingCancelId) return;
  const clOrdId = _pendingCancelId;
  _pendingCancelId = null;
  closeModal('cancelModal');
  showSpinner('Sending cancellation…');

  fetch('/routing_dashboard/cancel_order', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cl_ord_id: clOrdId }),
  })
  .then(r => r.json())
  .then(data => {
    hideSpinner();
    showFlash('success', data.message || 'Cancellation request sent.');
    updateExecutionReports();
  })
  .catch(() => {
    hideSpinner();
    showFlash('error', 'Error sending cancellation.');
  });
}

// ── Market data ──
function updateMarketData() {
  fetch('/routing_dashboard/get_market_data')
    .then(r => r.json())
    .then(data => {
      const tbody = document.getElementById('market-data-table');
      const count = document.getElementById('mktdata-count');
      count.textContent = `${data.length} symbol${data.length !== 1 ? 's' : ''}`;
      tbody.innerHTML = '';
      data.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="mono-cell" style="font-weight:600;color:#E6EDF3">${row.symbol}</td>
          <td class="dim-cell">${row.exchange}</td>
          <td class="mono-cell">${fmt2(row.opening_price)}</td>
          <td class="mono-cell" style="color:var(--green)">${fmt2(row.high_price)}</td>
          <td class="mono-cell" style="color:var(--red)">${fmt2(row.low_price)}</td>
          <td class="mono-cell">${fmt2(row.closing_price)}</td>
          <td class="mono-cell" style="color:#E6EDF3;font-weight:600">${fmt2(row.last_trade_price)}</td>
          <td class="dim-cell">${row.timestamp}</td>`;
        tbody.appendChild(tr);
      });
    })
    .catch(() => {});
}

function fmt2(v) {
  const n = parseFloat(v);
  return isNaN(n) ? '—' : n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// ── Connection status ──
function updateConnectionStatus() {
  fetch('/routing_dashboard/get_connection_status')
    .then(r => r.json())
    .then(status => {
      ['IB_PROD', 'BYMA_PROD', 'IB_DEV'].forEach(k => {
        const dot = document.getElementById(`status-${k}`);
        if (dot) dot.className = 'conn-dot ' + (status[k] ? 'connected' : 'disconnected');
      });
    })
    .catch(() => {});
}

function retryConnection(broker) {
  const btn = document.getElementById(`retry-${broker}`);
  if (btn) btn.classList.add('spinning');
  fetch('/routing_dashboard/retry_connection', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ broker }),
  })
  .then(r => r.json())
  .then(() => { if (btn) btn.classList.remove('spinning'); updateConnectionStatus(); })
  .catch(() => { if (btn) btn.classList.remove('spinning'); });
}

// ── Modal helpers ──
function openModal(id)  { document.getElementById(id).classList.add('open'); }
function closeModal(id) { document.getElementById(id).classList.remove('open'); }

document.querySelectorAll('.modal-overlay').forEach(overlay => {
  overlay.addEventListener('click', e => {
    if (e.target === overlay) overlay.classList.remove('open');
  });
});

// ── Spinner ──
function showSpinner(msg) {
  document.getElementById('spinnerMsg').textContent = msg || 'Loading…';
  document.getElementById('loadingSpinner').classList.add('active');
}
function hideSpinner() {
  document.getElementById('loadingSpinner').classList.remove('active');
}

// ── Flash messages ──
function showFlash(type, msg) {
  const el = document.getElementById('formFlash');
  el.className      = 'flash ' + type;
  el.textContent    = msg;
  el.style.display  = 'block';
  setTimeout(() => { el.style.display = 'none'; }, 4000);
}

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  setSide('Buy');
  loadAccounts();
  updateExecutionReports();
  updateMarketData();
  updateConnectionStatus();

  setInterval(updateExecutionReports,   5000);
  setInterval(updateMarketData,         5000);
  setInterval(updateConnectionStatus,   5000);
});

function showExecReportDetail(clOrdId) {
  const r = _execReportCache[clOrdId];
  if (!r) return;

  const status     = (r.ord_status || '').toLowerCase();
  const badgeClass = status.includes('fill')   ? 'filled'    :
                     status.includes('reject') ? 'rejected'  :
                     status.includes('cancel') ? 'cancelled' : 'pending';

  const fmt2l = v => { const n = parseFloat(v); return isNaN(n) ? '—' : n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }); };

  // Text / rejection reason row — only shown when populated
  const textRow = r.text
    ? `<div class="er-detail-row er-detail-full">
         <span class="er-detail-lbl">Reason / Text</span>
         <span class="er-detail-val er-detail-reason">${r.text}</span>
       </div>`
    : '';

  document.getElementById('erDetailBody').innerHTML = `
    <div class="er-detail-grid">

      <div class="er-detail-row er-detail-full">
        <span class="er-detail-lbl">ClOrdID</span>
        <span class="er-detail-val er-detail-mono" style="font-size:11px;color:var(--dim)">${r.cl_ord_id}</span>
      </div>

      <div class="er-detail-row">
        <span class="er-detail-lbl">Symbol</span>
        <span class="er-detail-val er-detail-mono" style="font-weight:700;color:#E6EDF3;font-size:18px">${r.symbol}</span>
      </div>
      <div class="er-detail-row">
        <span class="er-detail-lbl">Side</span>
        <span class="er-detail-val ${r.side === 'Buy' ? 'side-cell-buy' : 'side-cell-sell'}" style="font-size:15px">${r.side}</span>
      </div>

      <div class="er-detail-row">
        <span class="er-detail-lbl">Status</span>
        <span class="er-detail-val"><span class="status-badge ${badgeClass}">${r.ord_status}</span></span>
      </div>
      <div class="er-detail-row">
        <span class="er-detail-lbl">Exec Type</span>
        <span class="er-detail-val er-detail-mono">${r.exec_type}</span>
      </div>

      <div class="er-detail-row">
        <span class="er-detail-lbl">Order Qty</span>
        <span class="er-detail-val er-detail-mono">${fmt2l(r.order_qty)}</span>
      </div>
      <div class="er-detail-row">
        <span class="er-detail-lbl">Cum Qty</span>
        <span class="er-detail-val er-detail-mono">${fmt2l(r.cum_qty)}</span>
      </div>
      <div class="er-detail-row">
        <span class="er-detail-lbl">Leaves Qty</span>
        <span class="er-detail-val er-detail-mono">${fmt2l(r.leaves_qty)}</span>
      </div>
      <div class="er-detail-row">
        <span class="er-detail-lbl">Last Px</span>
        <span class="er-detail-val er-detail-mono">${fmt2l(r.last_px)}</span>
      </div>
      <div class="er-detail-row">
        <span class="er-detail-lbl">Avg Px</span>
        <span class="er-detail-val er-detail-mono">${fmt2l(r.avg_px)}</span>
      </div>
      <div class="er-detail-row">
        <span class="er-detail-lbl">Price</span>
        <span class="er-detail-val er-detail-mono">${fmt2l(r.price)}</span>
      </div>

      <div class="er-detail-row">
        <span class="er-detail-lbl">Order ID</span>
        <span class="er-detail-val er-detail-mono" style="font-size:11px">${r.order_id}</span>
      </div>
      <div class="er-detail-row">
        <span class="er-detail-lbl">Orig ClOrdID</span>
        <span class="er-detail-val er-detail-mono" style="font-size:11px">${r.orig_cl_ord_id}</span>
      </div>

      <div class="er-detail-row">
        <span class="er-detail-lbl">Ord Type</span>
        <span class="er-detail-val er-detail-mono">${r.ord_type}</span>
      </div>
      <div class="er-detail-row">
        <span class="er-detail-lbl">Currency</span>
        <span class="er-detail-val er-detail-mono">${r.currency}</span>
      </div>
      <div class="er-detail-row">
        <span class="er-detail-lbl">Broker</span>
        <span class="er-detail-val er-detail-mono">${r.broker}</span>
      </div>
      <div class="er-detail-row">
        <span class="er-detail-lbl">Time</span>
        <span class="er-detail-val er-detail-mono">${r.transact_time}</span>
      </div>

      ${textRow}
    </div>`;

  openModal('erDetailModal');
}