// login.js — Seeking Bias · Access Terminal
// Generates a live multi-panel trading terminal background on canvas

(function () {
  const canvas = document.getElementById('bgCanvas');
  const ctx    = canvas.getContext('2d');

  // ── Config ──
  const COLORS = {
    green:  '#3FB950',
    red:    '#F85149',
    blue:   '#1F6FEB',
    orange: '#D29922',
    cyan:   '#39C5CF',
    dim:    '#1A2535',
    grid:   'rgba(31,111,235,0.06)',
    text:   'rgba(88,166,255,0.45)',
  };

  const PANEL_COLS  = 3;
  const PANEL_ROWS  = 3;
  const SYMBOLS = ['SPY', 'QQQ', 'MELI', 'GGAL', 'XLV', 'GLD', 'BRK.B', 'AL30', 'BMA'];
  const TIMEFRAMES = ['1m', '5m', '15m', '1H', '4H'];

  // ── Resize ──
  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  // ── Price series generator ──
  function genSeries(len, startPrice, volatility) {
    const pts = [startPrice];
    for (let i = 1; i < len; i++) {
      const drift   = (Math.random() - 0.48) * volatility;
      const meanRev = (startPrice - pts[i - 1]) * 0.02;
      pts.push(Math.max(pts[i - 1] + drift + meanRev, startPrice * 0.5));
    }
    return pts;
  }

  // ── Panel definitions ──
  const panels = SYMBOLS.map((sym, i) => ({
    sym,
    tf:         TIMEFRAMES[i % TIMEFRAMES.length],
    series:     genSeries(120, 100 + Math.random() * 400, 2 + Math.random() * 4),
    color:      Math.random() > 0.45 ? COLORS.green : COLORS.red,
    scrollOff:  0,
    speed:      0.15 + Math.random() * 0.25,
    barSeries:  Array.from({ length: 120 }, () => ({
      open:  Math.random(),
      close: Math.random(),
      high:  Math.random(),
      low:   Math.random(),
    })),
  }));

  // ── Animate series slowly ──
  function tickSeries() {
    panels.forEach(p => {
      const last = p.series[p.series.length - 1];
      const next = last + (Math.random() - 0.485) * (3 + Math.random() * 2);
      p.series.push(Math.max(next, last * 0.7));
      if (p.series.length > 300) p.series.shift();
      p.scrollOff += p.speed;
    });
  }

  // ── Draw one chart panel ──
  function drawPanel(p, x, y, w, h, type) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();

    // Panel background
    const grad = ctx.createLinearGradient(x, y, x, y + h);
    grad.addColorStop(0, 'rgba(8,12,20,0.95)');
    grad.addColorStop(1, 'rgba(4,8,16,0.98)');
    ctx.fillStyle = grad;
    ctx.fillRect(x, y, w, h);

    // Grid lines
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 0.5;
    for (let gi = 1; gi < 5; gi++) {
      const gy = y + (h / 5) * gi;
      ctx.beginPath();
      ctx.moveTo(x, gy);
      ctx.lineTo(x + w, gy);
      ctx.stroke();
    }
    for (let gi = 1; gi < 8; gi++) {
      const gx = x + (w / 8) * gi;
      ctx.beginPath();
      ctx.moveTo(gx, y);
      ctx.lineTo(gx, y + h);
      ctx.stroke();
    }

    // Series slice for this panel
    const pts      = p.series;
    const viewLen  = Math.floor(w / 4);
    const start    = Math.max(0, pts.length - viewLen - Math.floor(p.scrollOff % viewLen));
    const slice    = pts.slice(start, start + viewLen);
    if (slice.length < 2) { ctx.restore(); return; }

    const minV = Math.min(...slice) * 0.998;
    const maxV = Math.max(...slice) * 1.002;
    const range = maxV - minV || 1;
    const padT = 22, padB = 20;
    const chartH = h - padT - padB;

    const toY = v => y + padT + (1 - (v - minV) / range) * chartH;
    const toX = (i) => x + (i / (slice.length - 1)) * w;

    if (type === 'line') {
      // Gradient fill
      const areaGrad = ctx.createLinearGradient(0, y + padT, 0, y + h - padB);
      areaGrad.addColorStop(0, p.color.replace(')', ',0.18)').replace('#', 'rgba(').replace('rgba(', 'rgba(').replace(/^rgba\(([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2}),/, (_, r, g, b) => `rgba(${parseInt(r,16)},${parseInt(g,16)},${parseInt(b,16)},`));
      areaGrad.addColorStop(1, 'rgba(0,0,0,0)');

      ctx.beginPath();
      ctx.moveTo(toX(0), toY(slice[0]));
      for (let i = 1; i < slice.length; i++) {
        const cp = toX(i - 0.5);
        ctx.bezierCurveTo(cp, toY(slice[i-1]), cp, toY(slice[i]), toX(i), toY(slice[i]));
      }
      ctx.lineTo(toX(slice.length - 1), y + h - padB);
      ctx.lineTo(toX(0), y + h - padB);
      ctx.closePath();
      ctx.fillStyle = colorWithAlpha(p.color, 0.12);
      ctx.fill();

      // Line
      ctx.beginPath();
      ctx.moveTo(toX(0), toY(slice[0]));
      for (let i = 1; i < slice.length; i++) {
        const cp = toX(i - 0.5);
        ctx.bezierCurveTo(cp, toY(slice[i-1]), cp, toY(slice[i]), toX(i), toY(slice[i]));
      }
      ctx.strokeStyle = p.color;
      ctx.lineWidth = 1.5;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 6;
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Last dot
      const lastX = toX(slice.length - 1);
      const lastY = toY(slice[slice.length - 1]);
      ctx.beginPath();
      ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 10;
      ctx.fill();
      ctx.shadowBlur = 0;

    } else {
      // Candlestick-style bars
      const barW  = Math.max(1, (w / slice.length) * 0.55);
      for (let i = 0; i < slice.length - 1; i++) {
        const o  = slice[i];
        const c  = slice[i + 1];
        const hi = Math.max(o, c) * 1.003;
        const lo = Math.min(o, c) * 0.997;
        const col = c >= o ? COLORS.green : COLORS.red;
        const bx  = toX(i);

        ctx.fillStyle   = col;
        ctx.strokeStyle = col;
        ctx.lineWidth   = 0.5;

        // Wick
        ctx.beginPath();
        ctx.moveTo(bx, toY(hi));
        ctx.lineTo(bx, toY(lo));
        ctx.stroke();

        // Body
        const bodyTop = toY(Math.max(o, c));
        const bodyH   = Math.max(1, Math.abs(toY(o) - toY(c)));
        ctx.fillRect(bx - barW / 2, bodyTop, barW, bodyH);
      }
    }

    // Symbol label
    ctx.font = `600 11px 'IBM Plex Mono', monospace`;
    ctx.fillStyle = COLORS.text;
    ctx.fillText(`${p.sym}  ${p.tf}`, x + 8, y + 14);

    // Price label
    const lastVal = slice[slice.length - 1];
    const priceTxt = lastVal.toFixed(2);
    ctx.font = `500 10px 'IBM Plex Mono', monospace`;
    ctx.fillStyle = p.color;
    ctx.fillText(priceTxt, x + w - ctx.measureText(priceTxt).width - 8, y + 14);

    // Border
    ctx.strokeStyle = 'rgba(31,111,235,0.08)';
    ctx.lineWidth   = 0.5;
    ctx.strokeRect(x, y, w, h);

    ctx.restore();
  }

  function colorWithAlpha(hex, alpha) {
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  // ── Draw all panels tiled ──
  function drawAllPanels() {
    const W = canvas.width, H = canvas.height;
    const pw = W / PANEL_COLS;
    const ph = H / PANEL_ROWS;

    panels.forEach((p, idx) => {
      const col  = idx % PANEL_COLS;
      const row  = Math.floor(idx / PANEL_COLS);
      const type = idx % 3 === 1 ? 'candles' : 'line';
      drawPanel(p, col * pw, row * ph, pw, ph, type);
    });

    // Overall dark overlay to keep panels as background
    const overlay = ctx.createLinearGradient(0, 0, W, H);
    overlay.addColorStop(0, 'rgba(4,8,16,0.45)');
    overlay.addColorStop(1, 'rgba(4,8,16,0.55)');
    ctx.fillStyle = overlay;
    ctx.fillRect(0, 0, W, H);
  }

  // ── Ticker tape at bottom ──
  let tickerOffset = 0;
  const tickerItems = [
    'SPY  +0.42%', 'QQQ  +1.18%', 'MELI  -0.33%', 'GGAL  +3.14%',
    'AL30  +0.88%', 'GLD  +0.06%', 'XLV  -0.21%', 'BRK.B  +0.55%',
    'BMA  +2.07%',  'GD  +1.44%',  'PAMP  +0.92%', 'ARS/USD  -0.13%',
  ];
  const tickerStr = tickerItems.join('   ·   ') + '   ·   ';

  function drawTicker() {
    const W = canvas.width, H = canvas.height;
    const tY = H - 28;
    const tH = 24;

    ctx.fillStyle = 'rgba(4,8,16,0.82)';
    ctx.fillRect(0, tY, W, tH);
    ctx.strokeStyle = 'rgba(31,111,235,0.15)';
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(0, tY); ctx.lineTo(W, tY); ctx.stroke();

    ctx.font = '10px "IBM Plex Mono", monospace';
    ctx.save();
    ctx.beginPath(); ctx.rect(0, tY, W, tH); ctx.clip();

    let x = -tickerOffset;
    const fullW = ctx.measureText(tickerStr).width;
    while (x < W) {
      tickerItems.forEach((item, i) => {
        const isUp   = item.includes('+');
        const isDash = item.includes('-0') || item.includes('-');
        ctx.fillStyle = isUp ? COLORS.green : (isDash ? COLORS.red : COLORS.text);
        ctx.fillText(item, x, tY + 16);
        x += ctx.measureText(item).width;
        ctx.fillStyle = 'rgba(88,166,255,0.3)';
        ctx.fillText('   ·   ', x, tY + 16);
        x += ctx.measureText('   ·   ').width;
      });
    }
    ctx.restore();
    tickerOffset += 0.6;
    if (tickerOffset > fullW) tickerOffset = 0;
  }

  // ── Clock live update ──
  function updateClock() {
    const el = document.getElementById('loginClock');
    if (!el) return;
    const n = new Date(), pad = v => String(v).padStart(2,'0');
    el.textContent = [n.getDate(), n.getMonth()+1, n.getFullYear()].map(pad).join('-')
      + '  ' + [n.getHours(), n.getMinutes(), n.getSeconds()].map(pad).join(':');
  }

  // ── Main loop ──
  let frame = 0;
  function loop() {
    if (frame % 2 === 0) tickSeries(); // slow down updates
    drawAllPanels();
    drawTicker();
    if (frame % 30 === 0) updateClock();
    frame++;
    requestAnimationFrame(loop);
  }

  // Wait for fonts
  document.fonts.ready.then(() => loop());
  setInterval(updateClock, 1000);

})();