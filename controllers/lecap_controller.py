"""
lecap_controller.py
===================
FastAPI controller for LECAPs & BONCAPs module.

Endpoints
---------
GET  /lecap/live                   -> Live prices from data912.com + DB metadata merged
GET  /lecap/securities             -> All LECAPs & BONCAPs from DB (with expired flag)
POST /lecap/securities             -> Add / update a single security
DELETE /lecap/securities/{symbol}  -> Soft-delete a security
POST /lecap/securities/bulk        -> Bulk upsert from JSON body (CSV import flow)

Wiring in main_dashboard_controller.py
---------------------------------------
    from controllers.lecap_controller import LecapController
    self.lecap_ctrl = LecapController(config_settings, logger)
    self.app.include_router(self.lecap_ctrl.router, prefix="/lecap")
"""

import math
import time
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from business_entities.security import Security
from common.util.downloaders.data912_downloader import Data912Downloader
from data_access_layer.security_manager import SecurityManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PRICE_CACHE: dict = {"ts": 0, "data": []}
CACHE_TTL_SECONDS  = 30
_dl = Data912Downloader()


# ---------------------------------------------------------------------------
# Pydantic request schemas
# ---------------------------------------------------------------------------

class SecurityIn(BaseModel):
    symbol:        str
    security_type: str
    description:   Optional[str] = ""
    maturity_date: str                # 'YYYY-MM-DD'
    final_payment: float = 0.0
    currency:      str   = "ARS"

    @field_validator("security_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"LECAP", "BONCAP", "SOVEREIGN"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"security_type must be one of {allowed}")
        return v_upper

    @field_validator("maturity_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("maturity_date must be 'YYYY-MM-DD'")
        return v


class BulkUpsertIn(BaseModel):
    securities: list[SecurityIn]


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class LecapController:

    def __init__(self, config_settings: dict, logger):
        self.logger  = logger
        self.config  = config_settings
        self.router  = APIRouter()
        self.sec_mgr = SecurityManager(config_settings["fund_mgmt_dashboard_cs"])

        # Register routes
        self.router.get("/live",                   response_class=JSONResponse)(self.live)
        self.router.get("/securities",             response_class=JSONResponse)(self.get_securities)
        self.router.post("/securities",            response_class=JSONResponse)(self.persist_security)
        self.router.delete("/securities/{symbol}", response_class=JSONResponse)(self.delete_security)
        self.router.post("/securities/bulk",       response_class=JSONResponse)(self.bulk_upsert)

    # ======================================================================
    # LIVE PRICES  (data912 + DB metadata)
    # ======================================================================

    async def live(self, request: Request):
        """
        Merge live prices from data912 with securities metadata from DB.
        Returns enriched list ready for the frontend table.
        """
        global _PRICE_CACHE

        now = time.time()
        if now - _PRICE_CACHE["ts"] < CACHE_TTL_SECONDS and _PRICE_CACHE["data"]:
            return JSONResponse({"data": _PRICE_CACHE["data"], "source": "cache"})

        # Fetch metadata from DB (LECAP and BONCAP only)
        db_securities: list[Security] = []
        try:
            db_securities = self.sec_mgr.get_securities(include_expired=True)
            db_securities = [s for s in db_securities
                             if s.security_type in ("LECAP", "BONCAP")]
        except Exception as exc:
            self._log(f"DB read error: {exc}", "ERROR")

        lecap_symbols  = {s.symbol for s in db_securities if s.security_type == "LECAP"}
        boncap_symbols = {s.symbol for s in db_securities if s.security_type == "BONCAP"}

        # Fetch prices from data912 concurrently via shared downloader
        notes_raw, bonds_raw = await _dl.fetch_many(
            Data912Downloader.ENDPOINT_ARG_NOTES,
            Data912Downloader.ENDPOINT_ARG_BONDS,
        )

        # Build price map using downloader helpers
        price_map: dict[str, dict] = {}
        price_map.update(_dl.build_price_map(notes_raw, symbols=lecap_symbols))
        price_map.update(_dl.build_price_map(bonds_raw, symbols=boncap_symbols))

        # Merge DB metadata + live prices + computed yields
        today_str = date.today().isoformat()
        result = []
        for sec in db_securities:
            live_data = price_map.get(sec.symbol, {})
            price     = live_data.get("price", 0.0)
            tna, tem, tir = _calc_yields(price, sec.final_payment, sec.maturity_date, today_str)

            result.append({
                "symbol":           sec.symbol,
                "security_type":    sec.security_type,
                "description":      sec.description,
                "maturity_date":    sec.maturity_date,
                "final_payment":    sec.final_payment,
                "currency":         sec.currency,
                "is_expired":       sec.is_expired,
                "days_to_maturity": sec.days_to_maturity,
                "price":            price,
                "bid":              live_data.get("bid", 0.0),
                "ask":              live_data.get("ask", 0.0),
                "tna":              tna,
                "tem":              tem,
                "tir":              tir,
            })

        # Active first, then expired; within each group sort by maturity asc
        result.sort(key=lambda x: (x["is_expired"], x["maturity_date"]))

        _PRICE_CACHE = {"ts": now, "data": result}
        return JSONResponse({"data": result, "source": "data912", "ts": int(now)})

    # ======================================================================
    # SECURITIES CRUD
    # ======================================================================

    async def get_securities(
        self,
        request:         Request,
        security_type:   Optional[str] = None,
        include_expired: bool          = True,
    ):
        try:
            secs = self.sec_mgr.get_securities(
                security_type   = security_type,
                include_expired = include_expired,
            )
            return JSONResponse({"securities": [_sec_to_dict(s) for s in secs]})
        except Exception as exc:
            self._log(f"get_securities error: {exc}", "ERROR")
            raise HTTPException(status_code=500, detail=str(exc))

    async def persist_security(self, request: Request, body: SecurityIn):
        try:
            sec = Security(
                symbol        = body.symbol.upper(),
                security_type = body.security_type,
                description   = body.description or "",
                maturity_date = body.maturity_date,
                final_payment = body.final_payment,
                currency      = body.currency.upper(),
            )
            self.sec_mgr.persist_security(sec)
            _PRICE_CACHE["ts"] = 0
            return JSONResponse({"ok": True, "symbol": sec.symbol})
        except Exception as exc:
            self._log(f"persist_security error: {exc}", "ERROR")
            raise HTTPException(status_code=500, detail=str(exc))

    async def delete_security(self, request: Request, symbol: str):
        try:
            self.sec_mgr.delete_security(symbol.upper())
            _PRICE_CACHE["ts"] = 0
            return JSONResponse({"ok": True, "symbol": symbol.upper()})
        except Exception as exc:
            self._log(f"delete_security error: {exc}", "ERROR")
            raise HTTPException(status_code=500, detail=str(exc))

    async def bulk_upsert(self, request: Request, body: BulkUpsertIn):
        try:
            secs = [
                Security(
                    symbol        = s.symbol.upper(),
                    security_type = s.security_type,
                    description   = s.description or "",
                    maturity_date = s.maturity_date,
                    final_payment = s.final_payment,
                    currency      = s.currency.upper(),
                )
                for s in body.securities
            ]
            rows = self.sec_mgr.bulk_upsert_securities(secs)
            _PRICE_CACHE["ts"] = 0
            return JSONResponse({"ok": True, "rows_affected": rows})
        except Exception as exc:
            self._log(f"bulk_upsert error: {exc}", "ERROR")
            raise HTTPException(status_code=500, detail=str(exc))

    # ======================================================================
    # Internal helpers
    # ======================================================================

    def _log(self, msg: str, level: str = "INFO") -> None:
        if self.logger:
            self.logger.do_log(f"[LecapCtrl] {msg}", level)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _calc_yields(
    price:         float,
    final_payment: float,
    maturity_date: str,
    today_str:     str,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    TNA / TEM / TIR for a zero-coupon / capitalising instrument (LECAP / BONCAP).

    Formula:
        days = calendar days from T+1 settlement to maturity
        tir  = (final_payment / price) ^ (365 / days) - 1
        tem  = (1 + tir) ^ (1/12) - 1
        tna  = tem * 12

    Returns (tna, tem, tir) as [0-1] floats, or (None, None, None) on error.
    """
    try:
        if not price or price <= 0 or not final_payment or final_payment <= 0:
            return None, None, None

        today  = date.fromisoformat(today_str)
        mature = date.fromisoformat(maturity_date)
        days   = (mature - today).days - 1      # T+1 settlement
        if days <= 0:
            return None, None, None

        ratio = final_payment / price
        tir   = ratio ** (365 / days) - 1
        tem   = (1 + tir) ** (1 / 12) - 1
        tna   = tem * 12

        return (
            round(tna, 6) if math.isfinite(tna) else None,
            round(tem, 6) if math.isfinite(tem) else None,
            round(tir, 6) if math.isfinite(tir) else None,
        )
    except Exception:
        return None, None, None


def _sec_to_dict(s: Security) -> dict:
    return {
        "id":               s.id,
        "symbol":           s.symbol,
        "security_type":    s.security_type,
        "description":      s.description,
        "maturity_date":    s.maturity_date,
        "final_payment":    s.final_payment,
        "currency":         s.currency,
        "is_active":        s.is_active,
        "is_expired":       s.is_expired,
        "days_to_maturity": s.days_to_maturity,
        "created_at":       s.created_at,
        "updated_at":       s.updated_at,
    }