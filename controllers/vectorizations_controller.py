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
        GET    /reference            sectors, portfolios, models, types, years
        GET    /overview             totals + breakdown by sector + coverage
        GET    /symbols              symbol search for the combo
        GET    /symbol               detail of one security
        GET    /sector               detail of one sector
        GET    /storage              the weight query, filterable
        GET    /runs                 run history
        POST   /runs                 registers or updates a manual run
        POST   /runs/delete          removes runs (any source, one or many)
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
        self.router.post("/runs/delete", response_class=JSONResponse)(self.api_delete_runs)

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

    @staticmethod
    def __as_bool__(value):
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

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
            data["report_types"] = self.__serialize__(data.get("report_types") or [])
            data["years"] = self.__serialize__(data.get("years") or [])
            data["quarters"] = self.__serialize__(data.get("quarters") or [])
            return JSONResponse({"ok": True, **data})
        except Exception as e:
            return self.__fail__("api_reference", e)

    async def api_overview(self, embedding_model: str = None, sector_code: str = None,
                           symbol: str = None, report_type: str = None,
                           fiscal_year: str = None, quarter: str = None):
        try:
            data = self.logic.get_overview(embedding_model=embedding_model or None,
                                           sector_code=sector_code or None,
                                           symbol=symbol or None,
                                           report_type=report_type or None,
                                           fiscal_year=fiscal_year or None,
                                           quarter=quarter or None)
            return JSONResponse({"ok": True,
                                 "totals": self.__serialize__(data["totals"] or {})[0]
                                 if data["totals"] else {},
                                 "by_sector": self.__serialize__(data["by_sector"]),
                                 "coverage": self.__serialize__(data["coverage"])})
        except Exception as e:
            return self.__fail__("api_overview", e)

    async def api_symbols(self, text: str = None, top: int = 500):
        try:
            rows = self.logic.search_symbols(text=text or None, top=top)
            return JSONResponse({"ok": True, "count": len(rows), "items": rows})
        except Exception as e:
            return self.__fail__("api_symbols", e)

    async def api_symbol_detail(self, symbol: str, embedding_model: str = None,
                                report_type: str = None, fiscal_year: str = None,
                                quarter: str = None, include_pending: str = None,
                                top: int = 1000):
        try:
            data = self.logic.get_symbol_detail(
                symbol,
                embedding_model=embedding_model or None,
                report_type=report_type or None,
                fiscal_year=fiscal_year or None,
                quarter=quarter or None,
                include_pending=self.__as_bool__(include_pending),
                top=top)
            return JSONResponse({"ok": True,
                                 "symbol": data["symbol"],
                                 "total": data["total"],
                                 "summary": self.__serialize__(data["summary"]),
                                 "documents": self.__serialize__(data["documents"]),
                                 "runs": self.__serialize__(data["runs"])})
        except Exception as e:
            return self.__fail__("api_symbol_detail", e, status=422)

    async def api_sector_detail(self, sector_code: str, embedding_model: str = None,
                                report_type: str = None, fiscal_year: str = None,
                                quarter: str = None, include_pending: str = None,
                                top: int = 1000):
        try:
            data = self.logic.get_sector_detail(
                sector_code,
                embedding_model=embedding_model or None,
                report_type=report_type or None,
                fiscal_year=fiscal_year or None,
                quarter=quarter or None,
                include_pending=self.__as_bool__(include_pending),
                top=top)
            return JSONResponse({"ok": True,
                                 "sector_code": data["sector_code"],
                                 "total": data["total"],
                                 "coverage": self.__serialize__(data["coverage"]),
                                 "documents": self.__serialize__(data["documents"]),
                                 "runs": self.__serialize__(data["runs"])})
        except Exception as e:
            return self.__fail__("api_sector_detail", e, status=422)

    async def api_storage(self, symbol: str = None, sector_code: str = None,
                          embedding_model: str = None, report_type: str = None,
                          fiscal_year: str = None, quarter: str = None,
                          include_pending: str = None, top: int = 500):
        try:
            include_pending = self.__as_bool__(include_pending)

            rows = self.logic.get_storage(symbol=symbol, sector_code=sector_code,
                                          embedding_model=embedding_model,
                                          report_type=report_type,
                                          fiscal_year=fiscal_year,
                                          quarter=quarter,
                                          include_pending=include_pending,
                                          top=top)

            # The real number of matches, so the screen can tell the difference
            # between "this is everything" and "this is the first page".
            total = self.logic.count_storage(symbol=symbol, sector_code=sector_code,
                                             embedding_model=embedding_model,
                                             report_type=report_type,
                                             fiscal_year=fiscal_year,
                                             quarter=quarter,
                                             include_pending=include_pending)

            return JSONResponse({"ok": True, "count": len(rows), "total": total,
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

    async def api_delete_runs(self, request: Request):
        """
        Accepts run_id (one) or run_ids (a list). Any run can be deleted now,
        manual or written by the job: the screen has to be cleanable.
        """
        try:
            payload = await request.json()
            run_ids = payload.get("run_ids")
            if not run_ids:
                run_ids = payload.get("run_id")

            if not run_ids:
                return JSONResponse({"ok": False, "error": "run_id or run_ids is required"},
                                    status_code=400)

            deleted = self.logic.delete_runs(run_ids)
            return JSONResponse({"ok": True, "deleted": deleted})
        except Exception as e:
            return self.__fail__("api_delete_runs", e, status=422)
