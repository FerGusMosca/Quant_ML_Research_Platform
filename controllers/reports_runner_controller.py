import traceback
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from controllers.base_controller import BaseController
from data_access_layer.securities_calendar_manager import SecuritiesCalendarManager
from framework.common.logger.message_type import MessageType
from logic_layer.reports_runner_logic import ReportsRunnerLogic
from logic_layer.vectorization_history_logic import VectorizationHistoryLogic


class ReportsRunnerController(BaseController):
    """
    "Reports Runner" screen: launches the run_report_mcp_server reports from the
    browser instead of from a PowerShell script, and shows the securities
    calendar that says on which date each filing actually landed.

    Routes (under the /reports_runner prefix):
        GET    /                page
        GET    /reference       available reports + portfolio combo
        GET    /run             runs a report, streaming (Server-Sent Events)
        GET    /calendar        securities filing calendar by year range
    """

    def __init__(self, config_settings: dict, logger):
        super().__init__()
        self.config = config_settings
        self.logger = logger
        self.logic = ReportsRunnerLogic(config_settings, logger)

        self.ml_reports_conn_str = config_settings["ml_reports_conn_str"]
        self._calendar_mgr = None

        # The portfolio combo has to offer exactly what the other screen offers,
        # so the list is built in one place only and reused here.
        self.history_logic = VectorizationHistoryLogic(config_settings, logger)

        self.router = APIRouter()
        self.templates = Jinja2Templates(
            directory=str(Path(__file__).parent.parent / "templates")
        )

        # ── Page ──────────────────────────────────────────────────────────────
        self.router.get("/", response_class=HTMLResponse)(self.display_page)

        # ── API ───────────────────────────────────────────────────────────────
        self.router.get("/reference", response_class=JSONResponse)(self.api_reference)
        self.router.get("/run")(self.api_run)
        self.router.get("/calendar",  response_class=JSONResponse)(self.api_calendar)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def __calendar_mgr__(self):
        """Built once and reused: the manager reconnects on its own when idle."""
        if self._calendar_mgr is None:
            self._calendar_mgr = SecuritiesCalendarManager(self.ml_reports_conn_str)
        return self._calendar_mgr

    def __portfolio_options__(self):
        """The same folder-backed list the Vectorizations screen shows."""
        try:
            return self.history_logic.get_portfolio_options()
        except Exception as e:
            self.logger.do_log(f"[REPORTS_RUNNER] portfolio list unavailable: {e}",
                               MessageType.WARNING)
            return []

    def __fail__(self, where, error, status=500):
        print(traceback.format_exc())
        self.logger.do_log(f"{where}: ❌ {str(error)}", MessageType.ERROR)
        return JSONResponse({"ok": False, "error": str(error)}, status_code=status)

    # ── Page ──────────────────────────────────────────────────────────────────

    async def display_page(self, request: Request):
        return self.templates.TemplateResponse("reports_runner.html", {
            "request": request,
            "mcp_configured": self.logic.is_configured(),
        })

    # ── API ───────────────────────────────────────────────────────────────────

    async def api_reference(self):
        try:
            return JSONResponse({"ok": True,
                                 "configured": self.logic.is_configured(),
                                 "uri": self.logic.mcp_uri,
                                 "reports": self.logic.get_reports(),
                                 "portfolios": self.__portfolio_options__()})
        except Exception as e:
            return self.__fail__("api_reference", e)

    async def api_run(self, report: str, portfolio: str,
                      year_from: int, year_to: int = None):
        """
        Streams the MCP run as Server-Sent Events. The connection stays open
        until the server sends its 'completed' event, which is exactly the
        contract ReportMCPClient enforces.
        """
        generator = self.logic.stream_report(report, portfolio, year_from, year_to)
        return StreamingResponse(generator, media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })

    async def api_calendar(self, year_from: int, year_to: int = None,
                           symbol: str = None):
        try:
            rows = self.__calendar_mgr__().get_calendar_rows(
                from_year=int(year_from),
                to_year=int(year_to or year_from),
                symbol=symbol or None)
            return JSONResponse({"ok": True, "count": len(rows), "items": rows})
        except Exception as e:
            return self.__fail__("api_calendar", e)
