import traceback
from decimal import Decimal
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from controllers.base_controller import BaseController
from framework.common.logger.message_type import MessageType
from logic_layer.vectorization_history_logic import VectorizationHistoryLogic


class VectorizationsController(BaseController):
    """
    "Vectorizations" screen: what was vectorized, for which security and which
    sector, how much it weighs, and the run history — including the old runs
    that were never logged and have to be registered by hand.

    Routes (under the /vectorizations prefix):
        GET    /                     page
        GET    /reference            sectors, portfolios, models, totals
        GET    /overview             totals + breakdown by sector
        GET    /symbols              symbol search for the combo
        GET    /symbol               detail of one security
        GET    /sector               detail of one sector
        GET    /storage              the weight query, filterable
        GET    /runs                 run history
        POST   /runs                 registers or updates a manual run
        POST   /runs/delete          removes a manual run
    """

    def __init__(self, config_settings: dict, logger):
        super().__init__()
        self.config = config_settings
        self.logger = logger
        self.logic = VectorizationHistoryLogic(config_settings, logger)

        self.router = APIRouter()
        self.templates = Jinja2Templates(
            directory=str(Path(__file__).parent.parent / "templates")
        )

        # ── Page ──────────────────────────────────────────────────────────────
        self.router.get("/", response_class=HTMLResponse)(self.display_page)

        # ── Reads ─────────────────────────────────────────────────────────────
        self.router.get("/reference", response_class=JSONResponse)(self.api_reference)
        self.router.get("/overview",  response_class=JSONResponse)(self.api_overview)
        self.router.get("/symbols",   response_class=JSONResponse)(self.api_symbols)
        self.router.get("/symbol",    response_class=JSONResponse)(self.api_symbol_detail)
        self.router.get("/sector",    response_class=JSONResponse)(self.api_sector_detail)
        self.router.get("/storage",   response_class=JSONResponse)(self.api_storage)
        self.router.get("/runs",      response_class=JSONResponse)(self.api_runs)

        # ── Manual register ───────────────────────────────────────────────────
        self.router.post("/runs",        response_class=JSONResponse)(self.api_persist_run)
        self.router.post("/runs/delete", response_class=JSONResponse)(self.api_delete_run)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def __serialize__(rows):
        """
        JSONResponse cannot encode datetimes or Decimals, and Postgres returns
        both. Converting here keeps every endpoint free of the concern.
        """
        if isinstance(rows, dict):
            rows = [rows]
        for row in rows:
            for key, value in list(row.items()):
                if hasattr(value, "isoformat"):
                    row[key] = value.isoformat()
                elif isinstance(value, Decimal):
                    row[key] = int(value) if value == value.to_integral_value() else float(value)
        return rows

    def __fail__(self, where, error, status=500):
        print(traceback.format_exc())
        self.logger.do_log(f"{where}: ❌ {str(error)}", MessageType.ERROR)
        return JSONResponse({"ok": False, "error": str(error)}, status_code=status)

    # ── Page ──────────────────────────────────────────────────────────────────

    async def display_page(self, request: Request):
        return self.templates.TemplateResponse("vectorizations.html", {"request": request})

    # ── Reads ─────────────────────────────────────────────────────────────────

    async def api_reference(self):
        try:
            data = self.logic.get_reference_data()
            data["totals"] = self.__serialize__(data.get("totals") or {})[0] \
                if data.get("totals") else {}
            data["embedding_models"] = self.__serialize__(data.get("embedding_models") or [])
            return JSONResponse({"ok": True, **data})
        except Exception as e:
            return self.__fail__("api_reference", e)

    async def api_overview(self, embedding_model: str = None):
        try:
            data = self.logic.get_overview(embedding_model or None)
            return JSONResponse({"ok": True,
                                 "totals": data["totals"],
                                 "by_sector": self.__serialize__(data["by_sector"])})
        except Exception as e:
            return self.__fail__("api_overview", e)

    async def api_symbols(self, text: str = None, top: int = 500):
        try:
            rows = self.logic.search_symbols(text=text or None, top=top)
            return JSONResponse({"ok": True, "count": len(rows), "items": rows})
        except Exception as e:
            return self.__fail__("api_symbols", e)

    async def api_symbol_detail(self, symbol: str, embedding_model: str = None):
        try:
            data = self.logic.get_symbol_detail(symbol, embedding_model or None)
            return JSONResponse({"ok": True,
                                 "symbol": data["symbol"],
                                 "summary": self.__serialize__(data["summary"]),
                                 "documents": self.__serialize__(data["documents"]),
                                 "runs": self.__serialize__(data["runs"])})
        except Exception as e:
            return self.__fail__("api_symbol_detail", e, status=422)

    async def api_sector_detail(self, sector_code: str, embedding_model: str = None):
        try:
            data = self.logic.get_sector_detail(sector_code, embedding_model or None)
            return JSONResponse({"ok": True,
                                 "sector_code": data["sector_code"],
                                 "documents": self.__serialize__(data["documents"]),
                                 "runs": self.__serialize__(data["runs"])})
        except Exception as e:
            return self.__fail__("api_sector_detail", e, status=422)

    async def api_storage(self, symbol: str = None, sector_code: str = None,
                          embedding_model: str = None, report_type: str = None,
                          fiscal_year: int = None, top: int = 500):
        try:
            rows = self.logic.get_storage(symbol=symbol, sector_code=sector_code,
                                          embedding_model=embedding_model,
                                          report_type=report_type,
                                          fiscal_year=fiscal_year, top=top)
            return JSONResponse({"ok": True, "count": len(rows),
                                 "items": self.__serialize__(rows)})
        except Exception as e:
            return self.__fail__("api_storage", e)

    async def api_runs(self, symbol: str = None, sector_code: str = None,
                       portfolio: str = None, run_source: str = None, top: int = 300):
        try:
            rows = self.logic.get_runs(symbol=symbol, sector_code=sector_code,
                                       portfolio=portfolio, run_source=run_source, top=top)
            return JSONResponse({"ok": True, "count": len(rows),
                                 "items": self.__serialize__(rows)})
        except Exception as e:
            return self.__fail__("api_runs", e)

    # ── Manual register ───────────────────────────────────────────────────────

    async def api_persist_run(self, request: Request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        try:
            run_id = self.logic.persist_manual_run(payload)
            return JSONResponse({"ok": True, "run_id": run_id})
        except Exception as e:
            return self.__fail__("api_persist_run", e, status=422)

    async def api_delete_run(self, request: Request):
        try:
            payload = await request.json()
            run_id = payload.get("run_id")
            if not run_id:
                return JSONResponse({"ok": False, "error": "run_id is required"},
                                    status_code=400)

            deleted = self.logic.delete_manual_run(int(run_id))
            if not deleted:
                return JSONResponse({"ok": False,
                                     "error": "Only manually registered runs can be deleted"},
                                    status_code=422)

            return JSONResponse({"ok": True, "deleted": deleted})
        except Exception as e:
            return self.__fail__("api_delete_run", e)
