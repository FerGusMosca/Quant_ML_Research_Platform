"""
argy_bonds_controller.py
========================
FastAPI controller for Argentine Sovereign Bonds module.

Endpoints
---------
GET  /argy_bonds/               → Render page (HTML)
GET  /argy_bonds/live           → Live bond prices from data912.com
GET  /argy_bonds/ohlcv          → Daily OHLCV bars from TradingView (via tvdatafeed or yfinance fallback)

Wiring in main_dashboard_controller.py
---------------------------------------
    from controllers.argy_bonds_controller import ArgyBondsController
    self.argy_bonds_ctrl = ArgyBondsController(config_settings, logger)
    self.app.include_router(self.argy_bonds_ctrl.router, prefix="/argy_bonds")

And in base.html sidebar (Analytics section):
    <a href="/argy_bonds" class="nav-item {% if active_page == 'argy_bonds' %}active{% endif %}">
      <span class="nav-icon">🇦🇷</span> Argy Bonds
    </a>
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# ── Constants ─────────────────────────────────────────────────────────────

DATA912_URL = "https://data912.com/live/arg_bonds"

# Tickers in data912 include the trailing "D" for USD price.
# We strip it for our internal representation.
SOVEREIGN_TICKERS_D = [
    "BPD7D", "AO27D", "AN29D", "AL29D", "AL30D",
    "AL35D", "AE38D", "AL41D",
    "GD29D", "GD30D", "GD35D", "GD38D", "GD41D",
]

# Simple in-memory cache to avoid hammering data912 on every request
_PRICE_CACHE: dict = {"ts": 0, "data": []}
CACHE_TTL_SECONDS = 30  # refresh prices at most every 30s


class ArgyBondsController:
    def __init__(self, config_settings: dict, logger):
        self.logger   = logger
        self.router   = APIRouter()
        self.config   = config_settings

        templates_path = Path(__file__).parent.parent / "templates"
        self.templates = Jinja2Templates(directory=str(templates_path))

        # ── Register routes ────────────────────────────────────────────
        self.router.get("/",                    response_class=HTMLResponse)(self.page)
        self.router.get("/live",                response_class=JSONResponse)(self.live_bonds)
        self.router.get("/ohlcv",               response_class=JSONResponse)(self.ohlcv)
        # The bonds_config.json is served as a static file via FastAPI's StaticFiles
        # mounted at /static in main_dashboard_controller.py — no extra route needed.
        # But we expose it explicitly here as a fallback:
        self.router.get("/config/bonds",        response_class=JSONResponse)(self.bonds_config)

    # ══════════════════════════════════════════════════════════════════
    # PAGE
    # ══════════════════════════════════════════════════════════════════

    async def page(self, request: Request):
        return self.templates.TemplateResponse(
            "argy_bonds.html", {"request": request}
        )

    # ══════════════════════════════════════════════════════════════════
    # CONFIG — Bond cash flow schedules
    # ══════════════════════════════════════════════════════════════════

    async def bonds_config(self, request: Request):
        """Serve bonds_config.json — fallback route if static files aren't mounted."""
        import json
        config_path = Path(__file__).parent.parent / "static" / "config" / "bonds_config.json"
        if not config_path.exists():
            return JSONResponse(status_code=404, content={"error": "bonds_config.json not found"})
        with open(config_path) as f:
            return JSONResponse(json.load(f))

    # ══════════════════════════════════════════════════════════════════
    # LIVE PRICES — data912.com
    # ══════════════════════════════════════════════════════════════════

    async def live_bonds(self, request: Request):
        """
        Fetch live sovereign bond prices from data912.com/live/arg_bonds.
        Returns JSON: { bonds: [...], source: 'data912', ts: ... }
        """
        global _PRICE_CACHE

        now = time.time()
        if now - _PRICE_CACHE["ts"] < CACHE_TTL_SECONDS and _PRICE_CACHE["data"]:
            return JSONResponse({"bonds": _PRICE_CACHE["data"], "source": "data912_cache"})

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(DATA912_URL)
                resp.raise_for_status()
                raw = resp.json()

            result = []
            for bond in raw:
                if bond.get("symbol") not in SOVEREIGN_TICKERS_D:
                    continue
                price_usd = float(bond.get("c") or 0)
                if price_usd <= 0:
                    continue
                base_symbol = bond["symbol"].rstrip("D")  # e.g. GD30D -> GD30
                result.append({
                    "symbol":     base_symbol,
                    "price_usd":  price_usd,
                    "bid":        float(bond.get("px_bid")    or 0),
                    "ask":        float(bond.get("px_ask")    or 0),
                    "volume":     bond.get("v")               or 0,
                    "pct_change": float(bond.get("pct_change") or 0),
                })

            _PRICE_CACHE = {"ts": now, "data": result}
            return JSONResponse({"bonds": result, "source": "data912", "ts": int(now)})

        except Exception as exc:
            self.logger and self.logger.do_log(
                f"[ArgyBonds] live_bonds error: {exc}", "ERROR"
            )
            # Return cache if available, else error
            if _PRICE_CACHE["data"]:
                return JSONResponse({"bonds": _PRICE_CACHE["data"], "source": "data912_stale"})
            return JSONResponse(
                status_code=502,
                content={"error": str(exc), "bonds": []}
            )

    # ══════════════════════════════════════════════════════════════════
    # OHLCV — Daily bars from TradingView via tvdatafeed (or yfinance)
    # ══════════════════════════════════════════════════════════════════

    async def ohlcv(
        self,
        request: Request,
        symbol:   str = "GD30D",
        exchange: str = "BYMA",
    ):
        """
        Download daily OHLCV bars for a bond symbol.
        Tries tvdatafeed first; falls back to yfinance (limited).

        Query params:
            symbol   - e.g. GD30D (with trailing D for USD price on BYMA)
            exchange - e.g. BYMA (default)

        Returns: { ok: bool, bars: [{time, open, high, low, close, volume}], symbol, exchange }
        """
        symbol   = symbol.upper().strip()
        exchange = exchange.upper().strip()

        bars = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_bars_sync, symbol, exchange
        )

        if bars is None:
            return JSONResponse(
                status_code=200,
                content={"ok": False, "error": f"No OHLCV data found for {symbol}", "bars": []}
            )

        return JSONResponse({"ok": True, "bars": bars, "symbol": symbol, "exchange": exchange})

    # ── Sync fetcher (runs in thread pool) ───────────────────────────

    def _fetch_bars_sync(self, symbol: str, exchange: str):
        """
        Try tvdatafeed first, then yfinance as fallback.
        Returns list of {time, open, high, low, close, volume} dicts (UNIX timestamps).
        """
        bars = self._try_tvdatafeed(symbol, exchange)
        if bars:
            return bars
        bars = self._try_yfinance(symbol)
        return bars  # may be None

    def _try_tvdatafeed(self, symbol: str, exchange: str):
        try:
            from tvDatafeed import TvDatafeed, Interval
            tv = TvDatafeed()  # anonymous session
            df = tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=Interval.in_daily,
                n_bars=1000,
            )
            if df is None or df.empty:
                return None
            df = df.reset_index()
            bars = []
            for _, row in df.iterrows():
                ts = int(row["datetime"].timestamp())
                bars.append({
                    "time":   ts,
                    "open":   round(float(row["open"]),  4),
                    "high":   round(float(row["high"]),  4),
                    "low":    round(float(row["low"]),   4),
                    "close":  round(float(row["close"]), 4),
                    "volume": int(row.get("volume", 0)),
                })
            bars.sort(key=lambda x: x["time"])
            return bars or None
        except Exception as exc:
            self.logger and self.logger.do_log(
                f"[ArgyBonds] tvdatafeed error for {symbol}: {exc}", "WARNING"
            )
            return None

    def _try_yfinance(self, symbol: str):
        """
        Fallback: yfinance. Argentine bonds are not always available here,
        but it's worth trying.  Ticker format varies (e.g. GD30.BA).
        """
        try:
            import yfinance as yf
            # Try BYMA suffix (.BA) and strip trailing D
            base = symbol.rstrip("D")
            for ticker_str in [f"{base}.BA", symbol, f"{symbol}.BA"]:
                tk = yf.Ticker(ticker_str)
                df = tk.history(period="5y", interval="1d", auto_adjust=True)
                if df is not None and not df.empty:
                    bars = []
                    for ts, row in df.iterrows():
                        bars.append({
                            "time":   int(ts.timestamp()),
                            "open":   round(float(row["Open"]),  4),
                            "high":   round(float(row["High"]),  4),
                            "low":    round(float(row["Low"]),   4),
                            "close":  round(float(row["Close"]), 4),
                            "volume": int(row.get("Volume", 0)),
                        })
                    bars.sort(key=lambda x: x["time"])
                    return bars or None
            return None
        except Exception as exc:
            self.logger and self.logger.do_log(
                f"[ArgyBonds] yfinance error for {symbol}: {exc}", "WARNING"
            )
            return None