"""
cer_bonds_controller.py
=======================
FastAPI controller for CER-indexed Argentine sovereign bonds.

Endpoints
---------
GET  /cer_bonds/live          → Live prices + CER-adjusted TIR/Duration
GET  /cer_bonds/cer           → Latest CER + CER T-10 (used for computations)
GET  /cer_bonds/config/bonds  → Serve bonos_cer section of bonds_config.json

Wiring in main_dashboard_controller.py
---------------------------------------
    from controllers.cer_bonds_controller import CerBondsController
    self.cer_bonds_ctrl = CerBondsController(config_settings, logger)
    self.app.include_router(self.cer_bonds_ctrl.router, prefix="/cer_bonds")

Frontend
--------
The CER tab is served by the existing argy_bonds.html template; this
controller only exposes data endpoints. See static/js/argy_bonds_cer.js
for the tab's rendering logic.
"""

import json
import time
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from common.util.downloaders.bcra_downloader import BcraDownloader
from common.util.downloaders.data912_downloader import Data912Downloader
from common.util.financial_calculations.cer_bond_calculator import CerBondCalculator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bonds that live on the /live/arg_bonds endpoint (sovereign long CER series)
_BOND_TICKERS_ON_BONDS_ENDPOINT = {
    "TZX26", "TZXO6", "TX26", "TZXD6", "TZXM7", "TZX27",
    "TZXD7", "TZX28", "TX28", "TX31", "DICP", "PARP",
}

# Bonds (Lecer) that live on the /live/arg_notes endpoint
_BOND_TICKERS_ON_NOTES_ENDPOINT = {
    "X15Y6", "X29Y6", "X31L6", "X30S6", "X30N6",
}

# Lag used to index CER-bond flows. Market convention is T-10 business days.
_CER_LAG_BUSINESS_DAYS = 10

_LIVE_CACHE: dict = {"ts": 0, "data": []}
_CER_CACHE:  dict = {"ts": 0, "latest": None, "lagged": None}
_LIVE_CACHE_TTL_SECONDS = 30
_CER_CACHE_TTL_SECONDS  = 3600  # CER only publishes once/day; 1h is conservative

_dl_d912 = Data912Downloader()
_dl_bcra = BcraDownloader()
_calc    = CerBondCalculator()


class CerBondsController:

    def __init__(self, config_settings: dict, logger):
        self.logger = logger
        self.config = config_settings
        self.router = APIRouter()

        self.router.get("/live",         response_class=JSONResponse)(self.live)
        self.router.get("/cer",          response_class=JSONResponse)(self.cer_values)
        self.router.get("/config/bonds", response_class=JSONResponse)(self.bonds_config)

    # ======================================================================
    # CONFIG — CER bond cash-flow schedules (from bonds_config.json)
    # ======================================================================

    async def bonds_config(self, request: Request):
        """Serve just the bonos_cer section of bonds_config.json."""
        config_path = (
            Path(__file__).parent.parent / "static" / "config" / "bonds_config.json"
        )
        if not config_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": "bonds_config.json not found"},
            )
        with open(config_path) as f:
            full = json.load(f)
        return JSONResponse(full.get("bonos_cer", {}))

    # ======================================================================
    # CER values (latest + lagged T-10)
    # ======================================================================

    async def cer_values(self, request: Request):
        """
        Returns
        -------
        {
          latest: {fecha, valor},   # most recent CER published
          lagged: {fecha, valor},   # CER T-10 business days (used for flows)
          lag_business_days: 10,
        }
        """
        latest, lagged = await self._get_cer_values()
        return JSONResponse({
            "latest":            latest,
            "lagged":            lagged,
            "lag_business_days": _CER_LAG_BUSINESS_DAYS,
        })

    # ======================================================================
    # LIVE — market prices + CER-adjusted TIR/Duration
    # ======================================================================

    async def live(self, request: Request):
        """
        Merge live prices from data912 (arg_bonds + arg_notes) with the
        CER bond config, then compute TIR Real and Macaulay Duration on
        CER-adjusted flows.

        Returns
        -------
        {
          bonds: [
            {
              symbol, price_ars, bid, ask, volume, pct_change,
              maturity, cer_emision, tir_real, duration,
            }, ...
          ],
          cer:        {latest: {...}, lagged: {...}},
          source:     'data912',
          ts:         unix_ts,
        }
        """
        global _LIVE_CACHE

        now = time.time()
        if now - _LIVE_CACHE["ts"] < _LIVE_CACHE_TTL_SECONDS and _LIVE_CACHE["data"]:
            return JSONResponse({
                "bonds":  _LIVE_CACHE["data"],
                "cer":    _LIVE_CACHE.get("cer"),
                "source": "cache",
            })

        try:
            # Load CER bond configs from JSON (flows + cer_emision per bond)
            bonos_cer = self._load_bonos_cer_config()
            if not bonos_cer:
                return JSONResponse(
                    status_code=500,
                    content={"error": "bonos_cer config missing", "bonds": []},
                )

            # Fetch prices from both data912 endpoints concurrently
            bonds_raw, notes_raw = await _dl_d912.fetch_many(
                Data912Downloader.ENDPOINT_ARG_BONDS,
                Data912Downloader.ENDPOINT_ARG_NOTES,
            )
            price_map: dict[str, dict] = {}
            price_map.update(_dl_d912.build_price_map(
                bonds_raw, symbols=_BOND_TICKERS_ON_BONDS_ENDPOINT
            ))
            price_map.update(_dl_d912.build_price_map(
                notes_raw, symbols=_BOND_TICKERS_ON_NOTES_ENDPOINT
            ))

            # Fetch CER T-10 (for TIR) and latest CER (for UI display)
            latest, lagged = await self._get_cer_values()
            cer_for_calc = (lagged or latest or {}).get("valor", 0.0)

            today = date.today()
            result = []
            for symbol, cfg in bonos_cer.items():
                price_item = price_map.get(symbol)
                if not price_item or price_item["price"] <= 0:
                    # No live quote — skip. UI can render with precio=0 if desired;
                    # we skip to avoid polluting the curve with phantom zero-TIR points.
                    continue

                calc = _calc.calculate(
                    price_ars   = price_item["price"],
                    raw_flows   = cfg.get("flujos", []),
                    cer_emision = cfg.get("cer_emision", 1.0),
                    cer_current = cer_for_calc,
                    symbol      = symbol,
                    today       = today,
                )

                result.append({
                    "symbol":      symbol,
                    "price_ars":   price_item["price"],
                    "bid":         price_item["bid"],
                    "ask":         price_item["ask"],
                    "volume":      price_item["volume"],
                    "pct_change":  price_item["pct_change"],
                    "maturity":    cfg.get("vencimiento"),
                    "cer_emision": cfg.get("cer_emision"),
                    "tir_real":    calc.tir_real,
                    "duration":    calc.duration,
                })

            # Sort by duration asc, with None at the end
            result.sort(key=lambda x: (x["duration"] is None, x["duration"] or 0))

            cer_payload = {"latest": latest, "lagged": lagged, "lag_business_days": _CER_LAG_BUSINESS_DAYS}
            _LIVE_CACHE = {"ts": now, "data": result, "cer": cer_payload}
            return JSONResponse({
                "bonds":  result,
                "cer":    cer_payload,
                "source": "data912",
                "ts":     int(now),
            })

        except Exception as exc:
            self._log(f"live error: {exc}", "ERROR")
            if _LIVE_CACHE["data"]:
                return JSONResponse({
                    "bonds":  _LIVE_CACHE["data"],
                    "cer":    _LIVE_CACHE.get("cer"),
                    "source": "stale",
                })
            return JSONResponse(
                status_code=502,
                content={"error": str(exc), "bonds": []},
            )

    # ======================================================================
    # Internal helpers
    # ======================================================================

    def _load_bonos_cer_config(self) -> dict:
        config_path = (
            Path(__file__).parent.parent / "static" / "config" / "bonds_config.json"
        )
        if not config_path.exists():
            return {}
        try:
            with open(config_path) as f:
                return json.load(f).get("bonos_cer", {}) or {}
        except Exception as exc:
            self._log(f"_load_bonos_cer_config error: {exc}", "ERROR")
            return {}

    async def _get_cer_values(self) -> tuple[dict, dict]:
        """
        Returns (latest, lagged) tuple, cached for _CER_CACHE_TTL_SECONDS.
        Each element is {fecha, valor} or {}.
        """
        global _CER_CACHE
        now = time.time()
        if (
            now - _CER_CACHE["ts"] < _CER_CACHE_TTL_SECONDS
            and _CER_CACHE["latest"] is not None
        ):
            return _CER_CACHE["latest"] or {}, _CER_CACHE["lagged"] or {}

        latest = await _dl_bcra.get_variable_latest(BcraDownloader.CER)
        lagged = await _dl_bcra.get_variable_lagged(
            BcraDownloader.CER,
            lag_business_days=_CER_LAG_BUSINESS_DAYS,
        )
        _CER_CACHE = {"ts": now, "latest": latest or {}, "lagged": lagged or {}}
        return _CER_CACHE["latest"], _CER_CACHE["lagged"]

    def _log(self, msg: str, level: str = "INFO") -> None:
        if self.logger:
            self.logger.do_log(f"[CerBonds] {msg}", level)
