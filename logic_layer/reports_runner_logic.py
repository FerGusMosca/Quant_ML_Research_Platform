# FILE: logic_layer/reports_runner_logic.py
# Orchestration for the "Reports Runner" screen.
#
# This replaces the PowerShell script that opened a websocket by hand against
# run_report_mcp_server. All the protocol knowledge already lives in
# ReportMCPClient, so this layer only decides which report to call, with which
# arguments, and turns the stream into Server-Sent Events the browser can read.
#
# Nothing here talks to the websocket directly on purpose: if the MCP contract
# changes, it changes in the service layer and this file does not move.

import json
import traceback

from framework.common.logger.message_type import MessageType
from service_layer.client.mcp.mcp_report_client import ReportMCPClient


class ReportsRunnerLogic:

    # Reports the screen knows how to launch. Adding one is a line here plus a
    # card in the template — no new endpoint.
    REPORTS = {
        "download_k10": {
            "label": "Download 10-K",
            "description": "Downloads the annual filings for the portfolio and year range.",
        },
        "download_q10": {
            "label": "Download 10-Q",
            "description": "Downloads the quarterly filings for the portfolio and year range.",
        },
        "download_securities_reports_calendar": {
            "label": "Securities Calendar",
            "description": "Fills in the filing dates of every security, so a quarter "
                           "can be read as a calendar window instead of a fiscal one.",
        },
    }

    def __init__(self, config_settings: dict, logger):
        self.config = config_settings
        self.logger = logger
        self.mcp_uri = (config_settings.get("MCP_REPORTS_URI") or "").strip()

    # ── Reference data ────────────────────────────────────────────────────────

    def get_reports(self):
        return [{"report": key, **value} for key, value in self.REPORTS.items()]

    def is_configured(self) -> bool:
        return bool(self.mcp_uri)

    # ── Validation ────────────────────────────────────────────────────────────

    def build_arguments(self, report: str, portfolio: str, year_from, year_to) -> dict:
        if report not in self.REPORTS:
            raise Exception(f"Unknown report '{report}'")

        if not self.mcp_uri:
            raise Exception("MCP_REPORTS_URI is missing from configs/commands_mgr.ini "
                            "([SETTINGS] section). Without it there is no server to call.")

        portfolio = (portfolio or "").strip()
        if not portfolio:
            raise Exception("portfolio is required")

        try:
            year_from = int(year_from)
            year_to = int(year_to or year_from)
        except Exception:
            raise Exception("year_from and year_to must be years, e.g. 2026")

        if year_to < year_from:
            year_from, year_to = year_to, year_from

        # This is the exact shape the PowerShell script was sending
        return {"portfolio": portfolio, "year": f"{year_from}-{year_to}"}

    # ── Execution ─────────────────────────────────────────────────────────────

    async def stream_report(self, report: str, portfolio: str, year_from, year_to):
        """
        Async generator of Server-Sent Events. One event per websocket message,
        plus a terminal 'done' event carrying the outcome.
        """
        try:
            arguments = self.build_arguments(report, portfolio, year_from, year_to)
        except Exception as e:
            yield self.__sse__({"event": "error", "error": str(e)})
            yield self.__sse__({"event": "done", "ok": False, "error": str(e)})
            return

        self.logger.do_log(
            f"[REPORTS_RUNNER] START | report={report} | args={arguments} | uri={self.mcp_uri}",
            MessageType.INFO)

        yield self.__sse__({"event": "started", "report": report,
                            "arguments": arguments, "uri": self.mcp_uri})

        client = ReportMCPClient(uri=self.mcp_uri, report=report, arguments=arguments)

        try:
            async for raw in client.execute_and_stream():
                yield self.__sse__({"event": "message", "raw": self.__shorten__(raw)})
        except Exception as e:
            print(traceback.format_exc())
            self.logger.do_log(f"[REPORTS_RUNNER] ❌ {report}: {str(e)}", MessageType.ERROR)
            yield self.__sse__({"event": "error", "error": str(e)})
            yield self.__sse__({"event": "done", "ok": False, "error": str(e)})
            return

        if client.success:
            self.logger.do_log(f"[REPORTS_RUNNER] COMPLETED | report={report}", MessageType.INFO)
        else:
            self.logger.do_log(f"[REPORTS_RUNNER] FAILED | report={report} | "
                               f"{client.last_error}", MessageType.ERROR)

        yield self.__sse__({"event": "done",
                            "ok": bool(client.success),
                            "error": client.last_error,
                            "summary": client.summary,
                            "report": client.completed_report or report})

    # ── SSE plumbing ──────────────────────────────────────────────────────────

    @staticmethod
    def __sse__(payload: dict) -> str:
        return f"data: {json.dumps(payload, default=str)}\n\n"

    @staticmethod
    def __shorten__(raw, limit: int = 4000) -> str:
        text = raw if isinstance(raw, str) else str(raw)
        return text if len(text) <= limit else text[:limit] + " …[truncated]"
