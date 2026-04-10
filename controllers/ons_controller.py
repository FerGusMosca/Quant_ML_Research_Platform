"""
ons_controller.py
=================
FastAPI controller for Obligaciones Negociables (ONs / corporate bonds).

Endpoints
---------
GET  /ons/live   → Live ON prices enriched with DB metadata + TIR/Duration
GET  /ons/ohlcv  → Daily OHLCV bars (tvdatafeed → yfinance fallback)
POST /ons/calc   → Cash-flow calculator for a given symbol + price + monto

Wiring (in main_dashboard_controller.py)
-----------------------------------------
    from controllers.ons_controller import ONsController
    self.ons_ctrl = ONsController(config_settings, logger)
    self.app.include_router(self.ons_ctrl.router, prefix="/ons")
"""

import asyncio
import time
from datetime import date

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from common.util.downloaders.data912_downloader import Data912Downloader
from common.util.financial_calculations.bullet_bond_calculator import BulletBondCalculator
from data_access_layer.security_manager import SecurityManager

# ── Sector mapping ────────────────────────────────────────────────────────
_SECTOR_MAP: dict[str, str] = {
    "YPF":                  "Energy",
    "Vista Energy":         "Energy",
    "Tecpetrol":            "Energy",
    "Genneia":              "Energy",
    "Pampa Energía":        "Energy",
    "TGS":                  "Energy",
    "Pluspetrol":           "Energy",
    "Pan American Energy":  "Energy",
    "Oldelval":             "Energy",
    "CGC":                  "Energy",
    "Banco BBVA":           "Banking",
    "Banco Hipotecario":    "Banking",
    "Banco Galicia":        "Banking",
    "Banco Comafi":         "Banking",
    "Banco de Valores":     "Banking",
    "Banco Macro":          "Banking",
    "Tarjeta Naranja":      "Banking",
    "IRSA":                 "Real Estate",
    "AA2000":               "Infrastructure",
    "Arcor":                "Consumer",
    "John Deere":           "Industry",
    "Aluar":                "Industry",
    "Cresud":               "Agro",
    "Loma Negra":           "Construction",
    "Ledesma":              "Agro",
    "FyO":                  "Agro",
}

_ENDPOINT_CORP = "/live/arg_corp"
_PRICE_CACHE: dict = {"ts": 0, "data": []}
CACHE_TTL = 30

_dl   = Data912Downloader()
_calc = BulletBondCalculator()


class ONsController:

    def __init__(self, config_settings: dict, logger):
        self.logger  = logger
        self.config  = config_settings
        self.router  = APIRouter()
        self.sec_mgr = SecurityManager(config_settings["fund_mgmt_dashboard_cs"])

        if not hasattr(Data912Downloader, 'ENDPOINT_ARG_CORP'):
            self._log(
                "ENDPOINT_ARG_CORP missing from Data912Downloader — "
                "add `ENDPOINT_ARG_CORP = '/live/arg_corp'`. Using fallback.",
                "WARNING",
            )

        self.router.get("/live",  response_class=JSONResponse)(self.live_ons)
        self.router.get("/ohlcv", response_class=JSONResponse)(self.ohlcv)
        self.router.post("/calc", response_class=JSONResponse)(self.calc)

    # ======================================================================
    # LIVE
    # ======================================================================

    async def live_ons(self, request: Request):
        global _PRICE_CACHE
        now = time.time()

        if now - _PRICE_CACHE["ts"] < CACHE_TTL and _PRICE_CACHE["data"]:
            return JSONResponse({"bonds": _PRICE_CACHE["data"], "source": "cache"})

        try:
            endpoint = getattr(Data912Downloader, 'ENDPOINT_ARG_CORP', _ENDPOINT_CORP)
            try:
                raw = await _dl.fetch(endpoint)
            except Exception as fetch_exc:
                self._log(f"data912 fetch failed: {fetch_exc}", "ERROR")
                raw = []

            price_map = _dl.build_price_map(raw) if raw else {}
            self._log(f"live_ons: {len(raw)} items from data912, {len(price_map)} mapped", "INFO")

            try:
                details = {d.symbol: d for d in self.sec_mgr.get_bond_details()}
                on_secs = self.sec_mgr.get_securities(security_type="ON", include_expired=False)
            except Exception as db_exc:
                self._log(f"DB error loading ONs: {db_exc}", "ERROR")
                return JSONResponse(status_code=502, content={"error": str(db_exc), "bonds": []})

            self._log(f"live_ons: {len(on_secs)} ONs loaded from DB", "INFO")

            result = []
            for sec in on_secs:
                try:
                    raw_price = price_map.get(sec.symbol)
                    detail    = details.get(sec.symbol)
                    issuer    = (
                        (detail.issuer if detail and detail.issuer else None)
                        or sec.description or "—"
                    )
                    law    = detail.law if detail else "Local"
                    sector = _SECTOR_MAP.get(issuer, "Other")

                    try:
                        coupons     = self.sec_mgr.get_bond_coupons(
                            sec.symbol, paid_filter=SecurityManager.COUPONS_FUTURE
                        )
                        coupon_list = [{"date": c.payment_date, "amount": c.amount} for c in coupons]
                    except Exception as ce:
                        self._log(f"coupon error {sec.symbol}: {ce}", "WARNING")
                        coupon_list = []

                    price = raw_price["price"] if raw_price else 0.0

                    if coupon_list and price > 0:
                        res = _calc.calculate(price=price, coupons=coupon_list)
                        tir, dur = res.tir, res.duration
                    else:
                        tir, dur = None, None

                    result.append({
                        "symbol":     sec.symbol,
                        "issuer":     issuer,
                        "sector":     sector,
                        "law":        law,
                        "maturity":   sec.maturity_date,
                        "price_usd":  price,
                        "bid":        raw_price["bid"]        if raw_price else 0.0,
                        "ask":        raw_price["ask"]        if raw_price else 0.0,
                        "pct_change": raw_price["pct_change"] if raw_price else 0.0,
                        "volume":     raw_price["volume"]     if raw_price else 0,
                        "tir":        tir,
                        "duration":   dur,
                        "coupons":    coupon_list,
                    })
                except Exception as enrich_exc:
                    self._log(f"enrich error for {sec.symbol}: {enrich_exc}", "WARNING")
                    continue

            result.sort(key=lambda x: (x["duration"] is None, x["duration"] or 0, x["maturity"]))

            _PRICE_CACHE = {"ts": now, "data": result}
            self._log(f"live_ons: returning {len(result)} ONs", "INFO")
            return JSONResponse({"bonds": result, "source": "data912", "ts": int(now)})

        except Exception as exc:
            self._log(f"live_ons unhandled error: {exc}", "ERROR")
            if _PRICE_CACHE["data"]:
                return JSONResponse({"bonds": _PRICE_CACHE["data"], "source": "stale"})
            return JSONResponse(status_code=502, content={"error": str(exc), "bonds": []})

    # ======================================================================
    # OHLCV
    # ======================================================================

    async def ohlcv(self, request: Request, symbol: str = "YM34D", exchange: str = "BYMA"):
        symbol   = symbol.upper().strip()
        exchange = exchange.upper().strip()
        try:
            bars = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_bars_sync, symbol, exchange
            )
        except Exception as exc:
            self._log(f"ohlcv error {symbol}: {exc}", "ERROR")
            return JSONResponse({"ok": False, "error": str(exc), "bars": []})

        if bars is None:
            return JSONResponse({"ok": False, "error": f"No data for {symbol}", "bars": []})
        return JSONResponse({"ok": True, "bars": bars, "symbol": symbol, "exchange": exchange})

    # ======================================================================
    # CALC
    # ======================================================================

    async def calc(self, request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

        symbol   = str(body.get("symbol", "")).upper()
        price    = float(body.get("price",    0))
        monto    = float(body.get("monto",    0))
        arancel  = float(body.get("arancel",  0))
        impuesto = float(body.get("impuesto", 0))

        if not symbol or price <= 0 or monto <= 0:
            return JSONResponse(status_code=400,
                content={"error": "symbol, price (>0) and monto (>0) required"})

        try:
            coupons     = self.sec_mgr.get_bond_coupons(symbol, paid_filter=SecurityManager.COUPONS_FUTURE)
            coupon_list = [{"date": c.payment_date, "amount": c.amount} for c in coupons]
        except Exception as exc:
            return JSONResponse(status_code=502, content={"error": str(exc)})

        inv = _calc.calculate_investment(
            symbol=symbol,
            price=price,
            coupons=coupon_list,
            monto=monto,
            arancel_pct=arancel,
            impuesto_pct=impuesto,
        )

        return JSONResponse({
            "symbol":          inv.symbol,
            "effective_price": inv.effective_price,
            "vn_bought":       inv.vn_bought,
            "monto":           inv.monto_real,
            "flows":           inv.flows,
            "total_cobro":     inv.total_cobro,
            "ganancia":        inv.ganancia,
            "tir":             inv.tir,
            "duration":        inv.duration,
            "is_bullet":       inv.is_bullet,
        })

    # ── Helpers ───────────────────────────────────────────────────────────

    def _fetch_bars_sync(self, symbol, exchange):
        return self._try_tvdatafeed(symbol, exchange) or self._try_yfinance(symbol)

    def _try_tvdatafeed(self, symbol, exchange):
        try:
            from tvDatafeed import TvDatafeed, Interval
            tv = TvDatafeed()
            df = tv.get_hist(symbol=symbol, exchange=exchange,
                             interval=Interval.in_daily, n_bars=1000)
            if df is None or df.empty:
                return None
            df = df.reset_index()
            bars = [{"time": int(r["datetime"].timestamp()),
                     "open": round(float(r["open"]), 4), "high": round(float(r["high"]), 4),
                     "low":  round(float(r["low"]),  4), "close": round(float(r["close"]), 4),
                     "volume": int(r.get("volume", 0))} for _, r in df.iterrows()]
            return sorted(bars, key=lambda x: x["time"]) or None
        except Exception as exc:
            self._log(f"tvdatafeed error {symbol}: {exc}", "WARNING")
            return None

    def _try_yfinance(self, symbol):
        try:
            import yfinance as yf
            base = symbol.rstrip("D")
            for tkr in [f"{base}.BA", symbol, f"{symbol}.BA"]:
                df = yf.Ticker(tkr).history(period="5y", interval="1d", auto_adjust=True)
                if df is not None and not df.empty:
                    bars = [{"time": int(ts.timestamp()),
                             "open": round(float(r["Open"]), 4), "high": round(float(r["High"]), 4),
                             "low":  round(float(r["Low"]),  4), "close": round(float(r["Close"]), 4),
                             "volume": int(r.get("Volume", 0))} for ts, r in df.iterrows()]
                    return sorted(bars, key=lambda x: x["time"]) or None
        except Exception as exc:
            self._log(f"yfinance error {symbol}: {exc}", "WARNING")
        return None

    def _log(self, msg: str, level: str = "INFO") -> None:
        if self.logger:
            self.logger.do_log(f"[ONs] {msg}", level)