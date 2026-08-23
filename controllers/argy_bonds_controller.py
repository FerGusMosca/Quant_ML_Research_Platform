"""
argy_bonds_controller.py
========================
FastAPI controller for Argentine Sovereign Bonds module.

Endpoints
---------
GET  /argy_bonds/               → Render page (HTML)
GET  /argy_bonds/live           → Live bond prices (data912.com via Data912Downloader)
GET  /argy_bonds/ohlcv          → Daily OHLCV bars; optional ?adjusted=true applies
                                  trailing coupon adjustment via BondPriceAdjuster
GET  /argy_bonds/config/bonds   → Serve bonds_config.json (fallback if StaticFiles not mounted)

Wiring in main_dashboard_controller.py
---------------------------------------
    from controllers.argy_bonds_controller import ArgyBondsController
    self.argy_bonds_ctrl = ArgyBondsController(config_settings, logger)
    self.app.include_router(self.argy_bonds_ctrl.router, prefix="/argy_bonds")
"""

import asyncio
import json
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from common.util.downloaders.data912_downloader import Data912Downloader
from common.util.financial_calculations.bond_price_adjuster import BondPriceAdjuster
from common.util.financial_calculations.bond_price_adjuster import _timestamp_to_date
from datetime import datetime
from data_access_layer.security_manager import SecurityManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sovereign tickers loaded from DB at runtime — no hardcoded list

_PRICE_CACHE: dict = {"ts": 0, "data": []}
CACHE_TTL_SECONDS  = 30

_dl       = Data912Downloader()
_adjuster = BondPriceAdjuster()


class ArgyBondsController:

    def __init__(self, config_settings: dict, logger):
        self.logger   = logger
        self.config   = config_settings
        self.router   = APIRouter()

        templates_path = Path(__file__).parent.parent / "templates"
        self.templates = Jinja2Templates(directory=str(templates_path))
        self.sec_mgr   = SecurityManager(config_settings["fund_mgmt_dashboard_cs"])

        # Register routes
        self.router.get("/",             response_class=HTMLResponse)(self.page)
        self.router.get("/live",         response_class=JSONResponse)(self.live_bonds)
        self.router.get("/ohlcv",        response_class=JSONResponse)(self.ohlcv)
        self.router.get("/config/bonds", response_class=JSONResponse)(self.bonds_config)

        # Coupon schedule maintenance: the price adjustment reads is_paid from
        # the table, so a payment that nobody flags leaves a step in the chart.
        self.router.get("/coupons",           response_class=JSONResponse)(self.list_coupons)
        self.router.post("/coupons/set_paid", response_class=JSONResponse)(self.set_coupon_paid)
        self.router.post("/coupons/mark_due", response_class=JSONResponse)(self.mark_due_coupons)

    # ======================================================================
    # PAGE
    # ======================================================================

    async def page(self, request: Request):
        return self.templates.TemplateResponse(
            "argy_bonds.html", {"request": request}
        )

    # ======================================================================
    # CONFIG — Bond cash-flow schedules
    # ======================================================================

    async def bonds_config(self, request: Request):
        """Serve bonds_config.json — fallback if StaticFiles is not mounted."""
        config_path = (
            Path(__file__).parent.parent / "static" / "config" / "bonds_config.json"
        )
        if not config_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": "bonds_config.json not found"},
            )
        with open(config_path) as f:
            return JSONResponse(json.load(f))

    # ======================================================================
    # COUPONS — schedule maintenance
    #
    # The chart used to read its coupon chips from bonds_config.json while the
    # price adjustment read is_paid from dbo.bond_coupons. Two sources meant the
    # screen could show a coupon as paid while the chart still had its step in
    # it. Everything here reads and writes the table, which is the one the
    # adjustment actually uses.
    # ======================================================================

    async def list_coupons(self, request: Request, symbol: str):
        """Full coupon schedule for a bond, paid and future alike."""
        base_symbol = symbol.rstrip("D").upper().strip()

        try:
            coupons = self.sec_mgr.get_bond_coupons(
                base_symbol, paid_filter=SecurityManager.COUPONS_ALL
            )
        except Exception as exc:
            self._log(f"list_coupons DB error for {base_symbol}: {exc}", "WARNING")
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            )

        today = datetime.now().strftime("%Y-%m-%d")

        rows = []
        for c in sorted(coupons, key=lambda x: x.payment_date):
            payment_date = str(c.payment_date)[:10]
            rows.append({
                "payment_date": payment_date,
                "amount":       c.amount,
                "is_paid":      bool(c.is_paid),
                # A coupon whose date has passed but is still flagged unpaid is
                # exactly what leaves an unexplained step in the chart.
                "overdue":      payment_date <= today and not c.is_paid,
            })

        return JSONResponse({
            "ok":       True,
            "symbol":   base_symbol,
            "today":    today,
            "coupons":  rows,
            "pending":  sum(1 for r in rows if r["overdue"]),
        })

    async def set_coupon_paid(self, request: Request):
        """Flips one coupon between paid and unpaid."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400,
                                content={"ok": False, "error": "Body must be JSON."})

        symbol       = str(body.get("symbol") or "").rstrip("D").upper().strip()
        payment_date = str(body.get("payment_date") or "").strip()
        is_paid      = bool(body.get("is_paid"))

        if not symbol or not payment_date:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "symbol and payment_date are required."})

        # The amount is never taken from the caller: it stays whatever the
        # schedule already holds, so a click in the UI cannot rewrite a cash flow.
        try:
            coupons = self.sec_mgr.get_bond_coupons(
                symbol, paid_filter=SecurityManager.COUPONS_ALL
            )
            match = next((c for c in coupons
                          if str(c.payment_date)[:10] == payment_date), None)

            if match is None:
                return JSONResponse(
                    status_code=404,
                    content={"ok": False,
                             "error": f"{symbol} has no coupon dated {payment_date}."})

            self.sec_mgr.persist_bond_coupon(
                symbol=symbol,
                payment_date=payment_date,
                amount=match.amount,
                is_paid=is_paid,
            )

        except Exception as exc:
            self._log(f"set_coupon_paid failed for {symbol} {payment_date}: {exc}",
                      "WARNING")
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": f"{type(exc).__name__}: {exc}"})

        self._log(f"coupon {symbol} {payment_date} marked "
                  f"{'paid' if is_paid else 'unpaid'}", "INFO")

        return JSONResponse({"ok": True, "symbol": symbol,
                             "payment_date": payment_date,
                             "amount": match.amount, "is_paid": is_paid})

    async def mark_due_coupons(self, request: Request):
        """Marks every coupon already past its date as paid, for one bond."""
        try:
            body = await request.json()
        except Exception:
            body = {}

        symbol = str(body.get("symbol") or "").rstrip("D").upper().strip() or None

        try:
            marked = self.sec_mgr.mark_coupons_paid(symbol)
        except Exception as exc:
            self._log(f"mark_due_coupons failed for {symbol}: {exc}", "WARNING")
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": f"{type(exc).__name__}: {exc}"})

        return JSONResponse({"ok": True, "symbol": symbol, "marked": marked})

    # ======================================================================
    # LIVE PRICES — data912.com (via Data912Downloader)
    # ======================================================================

    async def live_bonds(self, request: Request):
        """
        Fetch live sovereign bond prices from data912.com.
        Returns: { bonds: [...], source: str, ts: int }
        """
        global _PRICE_CACHE

        now = time.time()
        if now - _PRICE_CACHE["ts"] < CACHE_TTL_SECONDS and _PRICE_CACHE["data"]:
            return JSONResponse(
                {"bonds": _PRICE_CACHE["data"], "source": "data912_cache"}
            )

        try:
            raw = await _dl.fetch(Data912Downloader.ENDPOINT_ARG_BONDS)

            # Build allowed set from DB at request time
            sovereign_bonds = self.sec_mgr.get_bond_details()
            allowed_tickers = {b.symbol + "D" for b in sovereign_bonds}

            result = []
            for item in raw:
                if item.get("symbol") not in allowed_tickers:
                    continue
                parsed = _dl.parse_price_item(item)
                if parsed["price"] <= 0:
                    continue
                base_symbol = item["symbol"].rstrip("D")   # GD30D → GD30
                result.append({
                    "symbol":     base_symbol,
                    "price_usd":  parsed["price"],
                    "bid":        parsed["bid"],
                    "ask":        parsed["ask"],
                    "volume":     parsed["volume"],
                    "pct_change": parsed["pct_change"],
                })

            _PRICE_CACHE = {"ts": now, "data": result}
            return JSONResponse(
                {"bonds": result, "source": "data912", "ts": int(now)}
            )

        except Exception as exc:
            self._log(f"live_bonds error: {exc}", "ERROR")
            if _PRICE_CACHE["data"]:
                return JSONResponse(
                    {"bonds": _PRICE_CACHE["data"], "source": "data912_stale"}
                )
            return JSONResponse(
                status_code=502,
                content={"error": str(exc), "bonds": []},
            )

    # ======================================================================
    # OHLCV — Daily bars + optional server-side coupon adjustment
    # ======================================================================

    async def ohlcv(
        self,
        request:  Request,
        symbol:   str  = "GD30D",
        exchange: str  = "BYMA",
        adjusted: bool = False,
    ):
        """
        Download daily OHLCV bars.

        Query params
        ------------
        symbol   : e.g. GD30D  (trailing D = USD price on BYMA)
        exchange : e.g. BYMA (default)
        adjusted : if true, applies trailing coupon adjustment via BondPriceAdjuster
                   before returning bars.  The JS just plots — no math on the client.

        Returns
        -------
        {
          ok:       bool,
          bars:     [{time, open, high, low, close, volume}],
          adjusted: bool,   ← echoes back whether adjustment was applied
          symbol:   str,
          exchange: str,
        }
        """
        symbol   = symbol.upper().strip()
        exchange = exchange.upper().strip()

        raw_bars = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_bars_sync, symbol, exchange
        )

        if raw_bars is None:
            return JSONResponse(
                status_code=200,
                content={
                    "ok":       False,
                    "error":    f"No OHLCV data found for {symbol}",
                    "bars":     [],
                    "adjusted": False,
                },
            )

        bars = raw_bars

        if adjusted:
            # Strip trailing D to look up coupons: GD30D → GD30
            base_symbol  = symbol.rstrip("D")
            all_coupons  = self._load_paid_coupons(base_symbol)
            if all_coupons and raw_bars:
                # Only pass coupons that fall within the bar date range.
                # Coupons before the first bar would inflate every bar equally
                # and create a false level shift — we only want to neutralise
                # drops that are actually visible in the chart.
                first_bar_date = _timestamp_to_date(min(b["time"] for b in raw_bars))
                last_bar_date  = _timestamp_to_date(max(b["time"] for b in raw_bars))
                # Every PAID coupon whose date falls inside the visible range,
                # each with its own amount.
                #
                # This used to pass a single entry built from the last paid date
                # and the NEXT coupon's amount. Two things went wrong with that:
                # a range covering several payments only ever neutralised one of
                # them, and the amount subtracted was not the amount paid.
                #
                # Passing older coupons is harmless: the trailing adjustment adds
                # up the coupons that fall AFTER each bar, so anything before the
                # first bar contributes zero to every bar.
                in_range = [
                    c for c in all_coupons
                    if first_bar_date <= str(c["date"])[:10] <= last_bar_date
                ]
                if in_range:
                    bars = _adjuster.apply_trailing_adjustment(raw_bars, in_range)
                    self._log(
                        f"ohlcv adjustment: {symbol} — {len(in_range)}/{len(all_coupons)} "
                        f"coupons in bar range [{first_bar_date} → {last_bar_date}]",
                        "INFO",
                    )
                else:
                    self._log(
                        f"ohlcv adjustment: {symbol} — no coupons in bar range "
                        f"[{first_bar_date} → {last_bar_date}]",
                        "WARNING",
                    )
            else:
                self._log(
                    f"ohlcv adjustment requested for {symbol} but no paid coupons found",
                    "WARNING",
                )

        return JSONResponse({
            "ok":       True,
            "bars":     bars,
            "adjusted": adjusted and bool(bars),
            "symbol":   symbol,
            "exchange": exchange,
        })

    # ------------------------------------------------------------------
    # Load paid coupons from bonds_config.json
    # ------------------------------------------------------------------

    def _load_paid_coupons(self, base_symbol: str) -> list[dict]:
        """Returns paid coupon history from DB for a bond symbol."""
        try:
            coupons = self.sec_mgr.get_bond_coupons(
                base_symbol, paid_filter=SecurityManager.COUPONS_PAID
            )
            return [{"date": c.payment_date, "amount_per_100vn": c.amount} for c in coupons]
        except Exception as exc:
            self._log(f"_load_paid_coupons DB error for {base_symbol}: {exc}", "WARNING")
            return []

    def _load_future_coupons(self, base_symbol: str) -> list[dict]:
        # Build adjustment entry from DB:
        #   date   = last paid coupon date
        #   amount = next future coupon amount
        # This places the adjustment at the correct ex-date with the correct amount.
        try:
            paid   = self.sec_mgr.get_bond_coupons(base_symbol, paid_filter=SecurityManager.COUPONS_PAID)
            future = self.sec_mgr.get_bond_coupons(base_symbol, paid_filter=SecurityManager.COUPONS_FUTURE)
            if not paid or not future:
                return []
            last_paid_date = max(c.payment_date for c in paid)
            next_amount    = min(future, key=lambda c: c.payment_date).amount
            return [{"date": last_paid_date, "amount_per_100vn": next_amount}]
        except Exception as exc:
            self._log(f"_load_future_coupons DB error for {base_symbol}: {exc}", "WARNING")
            return []

        # ------------------------------------------------------------------
    # Sync bar fetchers (run in thread pool — no event loop blocking)
    # ------------------------------------------------------------------

    def _fetch_bars_sync(self, symbol: str, exchange: str):
        """Try tvdatafeed first, then yfinance as fallback."""
        bars = self._try_tvdatafeed(symbol, exchange)
        if bars:
            return bars
        return self._try_yfinance(symbol)

    def _try_tvdatafeed(self, symbol: str, exchange: str):
        try:
            from tvDatafeed import TvDatafeed, Interval
            tv = TvDatafeed()
            df = tv.get_hist(
                symbol   = symbol,
                exchange = exchange,
                interval = Interval.in_daily,
                n_bars   = 1000,
            )
            if df is None or df.empty:
                return None
            df   = df.reset_index()
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
            self._log(f"tvdatafeed error for {symbol}: {exc}", "WARNING")
            return None

    def _try_yfinance(self, symbol: str):
        try:
            import yfinance as yf
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
            self._log(f"yfinance error for {symbol}: {exc}", "WARNING")
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str, level: str = "INFO") -> None:
        if self.logger:
            self.logger.do_log(f"[ArgyBonds] {msg}", level)