import os
import traceback
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from controllers.base_controller import BaseController
from data_access_layer.sec_securities_metadata_manager import SECSecuritiesMetadataManager
from framework.common.logger.message_type import MessageType
from logic_layer.sec_metadata_orchestation_logic import (
    SECMetadataOrchestationLogic,
    get_run_state,
    request_cancel,
    run_all_in_background,
)


class SECSecuritiesController(BaseController):
    """
    "SEC Securities" screen: fills in SEC_Securities metadata from the SEC,
    shows coverage by sector, and tags universes from a CSV.

    Routes (under the /sec_securities prefix):
        GET    /                          page
        GET    /status                    coverage + run progress
        POST   /run                       full sweep (in the background)
        POST   /run_single                a single security
        POST   /cancel                    stops the run
        POST   /reset_errors              puts failed rows back in the queue
        GET    /securities                filtered search
        GET    /tags                      tag list
        POST   /tags                      creates or updates a tag
        POST   /tags/apply_csv            uploads a CSV and applies the tag
        POST   /tags/apply_symbols        applies a tag to a list of symbols
        POST   /tags/apply_sector         tags a whole sector
        POST   /tags/remove               removes a tag from a security
    """

    def __init__(self, config_settings: dict, logger):
        super().__init__()
        self.config = config_settings
        self.logger = logger
        self.ml_reports_conn_str = config_settings["ml_reports_conn_str"]

        # Identifying User-Agent: the SEC blocks requests without one.
        self.user_agent = (config_settings.get("SEC_USER_AGENT")
                           or os.getenv("SEC_USER_AGENT")
                           or "")

        self.metadata_mgr = SECSecuritiesMetadataManager(self.ml_reports_conn_str, logger)

        self.router = APIRouter()
        self.templates = Jinja2Templates(
            directory=str(Path(__file__).parent.parent / "templates")
        )

        # ── Page ──────────────────────────────────────────────────────────────
        self.router.get("/", response_class=HTMLResponse)(self.display_page)

        # ── Metadata ──────────────────────────────────────────────────────────
        self.router.get("/status",       response_class=JSONResponse)(self.api_status)
        self.router.post("/run",         response_class=JSONResponse)(self.api_run)
        self.router.post("/run_single",  response_class=JSONResponse)(self.api_run_single)
        self.router.post("/cancel",      response_class=JSONResponse)(self.api_cancel)
        self.router.post("/reset_errors", response_class=JSONResponse)(self.api_reset_errors)
        self.router.get("/securities",   response_class=JSONResponse)(self.api_securities)

        # ── Tags ──────────────────────────────────────────────────────────────
        self.router.get("/tags",               response_class=JSONResponse)(self.api_get_tags)
        self.router.post("/tags",              response_class=JSONResponse)(self.api_persist_tag)
        self.router.post("/tags/apply_csv",    response_class=JSONResponse)(self.api_apply_tag_csv)
        self.router.post("/tags/apply_symbols", response_class=JSONResponse)(self.api_apply_tag_symbols)
        self.router.post("/tags/apply_sector", response_class=JSONResponse)(self.api_apply_tag_sector)
        self.router.post("/tags/remove",       response_class=JSONResponse)(self.api_remove_tag)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def __build_orchestation__(self):
        return SECMetadataOrchestationLogic(self.ml_reports_conn_str,
                                            self.logger, self.user_agent)

    @staticmethod
    def __serialize__(rows):
        for row in rows:
            for key, value in list(row.items()):
                if hasattr(value, "isoformat"):
                    row[key] = value.isoformat()
        return rows

    # ── Page ──────────────────────────────────────────────────────────────────

    async def display_page(self, request: Request):
        return self.templates.TemplateResponse("sec_securities.html", {
            "request": request,
            "user_agent_configured": bool(self.user_agent),
        })

    # ── Metadata API ──────────────────────────────────────────────────────────

    async def api_status(self):
        try:
            return JSONResponse({"ok": True,
                                 "run": get_run_state(),
                                 "summary": self.metadata_mgr.get_summary()})
        except Exception as e:
            print(traceback.format_exc())
            self.logger.do_log(f"api_status: ❌ {str(e)}", MessageType.ERROR)
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_run(self, request: Request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        if not self.user_agent:
            return JSONResponse({"ok": False,
                "error": "SEC_USER_AGENT is missing from configs/commands_mgr.ini. "
                         "The SEC rejects requests without an identifying User-Agent."},
                status_code=422)

        started = run_all_in_background(
            self.ml_reports_conn_str, self.logger, self.user_agent,
            top=payload.get("top"),
            include_errors=bool(payload.get("include_errors", False)))

        if not started:
            return JSONResponse({"ok": False, "error": "A run is already in progress"},
                                status_code=409)

        return JSONResponse({"ok": True, "run": get_run_state()})

    async def api_run_single(self, request: Request):
        try:
            payload = await request.json()
            symbol = (payload.get("symbol") or "").strip() or None
            cik = payload.get("cik")

            if not symbol and cik is None:
                return JSONResponse({"ok": False, "error": "Send either symbol or cik"},
                                    status_code=400)

            result = self.__build_orchestation__().process_download_single_metadata(
                symbol=symbol, cik=cik)

            if not result.get("ok"):
                return JSONResponse({"ok": False,
                                     "error": f"Could not update {symbol or cik}",
                                     "detail": result}, status_code=422)

            return JSONResponse({"ok": True, "result": result})

        except Exception as e:
            print(traceback.format_exc())
            self.logger.do_log(f"api_run_single: ❌ {str(e)}", MessageType.ERROR)
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_cancel(self):
        request_cancel()
        return JSONResponse({"ok": True})

    async def api_reset_errors(self):
        try:
            return JSONResponse({"ok": True, "reset": self.metadata_mgr.reset_errors()})
        except Exception as e:
            print(traceback.format_exc())
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_securities(self, sector_code: str = None, industry_code: str = None,
                             tag_code: str = None, text: str = None, top: int = 500):
        try:
            rows = self.metadata_mgr.search(sector_code=sector_code or None,
                                            industry_code=industry_code or None,
                                            tag_code=tag_code or None,
                                            text=text or None,
                                            top=top)
            return JSONResponse({"ok": True, "count": len(rows),
                                 "items": self.__serialize__(rows)})
        except Exception as e:
            print(traceback.format_exc())
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    # ── Tags API ──────────────────────────────────────────────────────────────

    async def api_get_tags(self):
        try:
            return JSONResponse({"ok": True,
                                 "items": self.__serialize__(self.metadata_mgr.get_tags())})
        except Exception as e:
            print(traceback.format_exc())
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_persist_tag(self, request: Request):
        try:
            payload = await request.json()
            tag_code = (payload.get("tag_code") or "").strip()
            if not tag_code:
                return JSONResponse({"ok": False, "error": "tag_code is empty"},
                                    status_code=400)

            tag_id = self.metadata_mgr.persist_tag(tag_code,
                                                   payload.get("tag_name"),
                                                   payload.get("tag_group") or "CUSTOM",
                                                   payload.get("color"))
            return JSONResponse({"ok": True, "id": tag_id})
        except Exception as e:
            print(traceback.format_exc())
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_apply_tag_csv(self,
                                tag_code: str = Form(...),
                                tag_name: str = Form(None),
                                tag_group: str = Form("CUSTOM"),
                                file: UploadFile = File(...)):
        try:
            raw = await file.read()
            result = self.__build_orchestation__().process_tag_securities_from_csv(
                tag_code, raw, tag_name, tag_group)
            return JSONResponse({"ok": True, "result": result})
        except Exception as e:
            print(traceback.format_exc())
            self.logger.do_log(f"api_apply_tag_csv: ❌ {str(e)}", MessageType.ERROR)
            return JSONResponse({"ok": False, "error": str(e)}, status_code=422)

    async def api_apply_tag_symbols(self, request: Request):
        try:
            payload = await request.json()
            tag_code = (payload.get("tag_code") or "").strip()
            symbols = payload.get("symbols") or []

            if not tag_code or not symbols:
                return JSONResponse({"ok": False, "error": "Missing tag_code or symbols"},
                                    status_code=400)

            result = self.metadata_mgr.apply_tag_to_symbols(tag_code, symbols)
            return JSONResponse({"ok": True, "result": result})
        except Exception as e:
            print(traceback.format_exc())
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_apply_tag_sector(self, request: Request):
        try:
            payload = await request.json()
            tagged = self.metadata_mgr.apply_tag_by_sector(payload.get("tag_code"),
                                                           payload.get("sector_code"))
            return JSONResponse({"ok": True, "tagged": tagged})
        except Exception as e:
            print(traceback.format_exc())
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_remove_tag(self, request: Request):
        try:
            payload = await request.json()
            removed = self.metadata_mgr.remove_tag_from_security(
                payload.get("tag_code"), int(payload.get("security_id")))
            return JSONResponse({"ok": True, "removed": removed})
        except Exception as e:
            print(traceback.format_exc())
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
