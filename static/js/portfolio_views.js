// portfolio_views.js — Seeking Bias · Portfolio Views

// ── Live clock ──
(function tick() {
  const el = document.getElementById('navClock');
  if (el) {
    const n = new Date(), pad = v => String(v).padStart(2, '0');
    el.textContent = [n.getHours(), n.getMinutes(), n.getSeconds()].map(pad).join(':');
  }
  setTimeout(tick, 1000);
})();

// ── Account search filter ──
document.getElementById('accountSearch')?.addEventListener('input', function () {
  const q = this.value.toLowerCase();
  document.querySelectorAll('#accountsTbody tr').forEach(row => {
    row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
  });
});

// ── Holdings search filter ──
document.getElementById('holdingsSearch')?.addEventListener('input', function () {
  const q = this.value.toLowerCase();
  document.querySelectorAll('#holdingsTbody tr').forEach(row => {
    row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
  });
});

// ══════════════════════════════════════════════════════════════
//  STATE
// ══════════════════════════════════════════════════════════════

let _currentAccountId     = null;
let _currentAccountNumber = '';
let _currentAccountName   = '';
let _currentBroker        = '';

// ══════════════════════════════════════════════════════════════
//  LOAD PORTFOLIO
// ══════════════════════════════════════════════════════════════

function loadPortfolio(accountId, accountNumber, accountName, broker) {
  _currentAccountId     = accountId;
  _currentAccountNumber = accountNumber;
  _currentAccountName   = accountName;
  _currentBroker        = broker;

  // Show the panel
  const panel = document.getElementById('portfolioPanel');
  panel.style.display = 'flex';

  // Update header metadata
  document.getElementById('portfolioBrokerBadge').innerHTML =
    '<span class="broker-badge broker-' + broker + '">' + broker + '</span>';
  document.getElementById('portfolioTitle').textContent     = accountName;
  document.getElementById('portfolioSubtitle').textContent  = accountNumber;

  // Reset stats
  document.getElementById('statTotalValueVal').textContent = '—';
  document.getElementById('statPositionsVal').textContent  = '—';

  // Scroll to panel
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });

  _fetchHoldings(accountId);
}

function refreshPortfolio() {
  if (!_currentAccountId) return;
  _fetchHoldings(_currentAccountId);
}

function _fetchHoldings(accountId) {
  // Reset states
  _setView('loading');

  const refreshBtn = document.getElementById('refreshBtn');
  if (refreshBtn) { refreshBtn.disabled = true; }

  fetch('/portfolio_views/' + accountId + '/holdings')
    .then(r => r.json())
    .then(data => {
      if (!data.ok) {
        if (data.not_implemented) {
          document.getElementById('portfolioNotImplBroker').textContent =
            'Broker "' + data.broker + '" is not implemented yet.';
          _setView('not_implemented');
        } else {
          document.getElementById('portfolioErrorMsg').textContent = data.error || 'Unknown error.';
          _setView('error');
        }
        return;
      }

      renderHoldings(data.holdings);
      _setView('holdings');
    })
    .catch(err => {
      document.getElementById('portfolioErrorMsg').textContent =
        'Network error: ' + err.message;
      _setView('error');
    })
    .finally(() => {
      if (refreshBtn) { refreshBtn.disabled = false; }
    });
}

// ══════════════════════════════════════════════════════════════
//  RENDER
// ══════════════════════════════════════════════════════════════

function renderHoldings(holdings) {
  const tbody = document.getElementById('holdingsTbody');
  if (!tbody) return;

  if (!holdings || holdings.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5"><div class="empty-state">No holdings found.</div></td></tr>';
    document.getElementById('statPositionsVal').textContent  = '0';
    //document.getElementById('statTotalValueVal').textContent = fmtCurrency(0);
    return;
  }

  const totalAmount = holdings.reduce((sum, h) => sum + (h.purchase_amount || 0), 0);

  document.getElementById('statPositionsVal').textContent  = holdings.length;
  //document.getElementById('statTotalValueVal').textContent = fmtCurrency(totalAmount);

  tbody.innerHTML = holdings.map(h => `
    <tr>
      <td><span class="holding-symbol">${esc(h.symbol)}</span></td>
      <td><span class="holding-name">${esc(h.name || h.symbol)}</span></td>
      <td class="holding-qty">${fmtQty(h.qty)}</td>
      <td class="holding-price">${h.purchase_price != null ? fmtPrice(h.purchase_price) : '<span style="color:var(--faint)">—</span>'}</td>
      <td class="holding-amount">${h.purchase_amount != null ? fmtCurrency(h.purchase_amount) : '<span style="color:var(--faint)">—</span>'}</td>
    </tr>
  `).join('');
}

// ══════════════════════════════════════════════════════════════
//  VIEW STATE MANAGER
// ══════════════════════════════════════════════════════════════

function _setView(state) {
  document.getElementById('portfolioLoading').style.display  = state === 'loading'         ? 'flex'  : 'none';
  document.getElementById('portfolioError').style.display    = state === 'error'           ? 'flex'  : 'none';
  document.getElementById('portfolioNotImpl').style.display  = state === 'not_implemented' ? 'flex'  : 'none';
  document.getElementById('holdingsCard').style.display      = state === 'holdings'        ? 'block' : 'none';
}

// ══════════════════════════════════════════════════════════════
//  FORMATTERS
// ══════════════════════════════════════════════════════════════

function fmtCurrency(val) {
  if (val == null) return '—';
  return '$' + Number(val).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtPrice(val) {
  if (val == null) return '—';
  return '$' + Number(val).toLocaleString('en-US', { minimumFractionDigits: 4, maximumFractionDigits: 4 });
}

function fmtQty(val) {
  if (val == null) return '—';
  return Number(val).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 6 });
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = String(s ?? '');
  return d.innerHTML;
}

function showFlash(type, msg) {
  document.querySelector('.flash')?.remove();
  const el = document.createElement('div');
  el.className   = 'flash ' + type;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}