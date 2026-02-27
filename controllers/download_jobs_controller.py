import io
import traceback
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from common.enums.information_vendors import InformationVendors
from controllers.base_controller import BaseController
from data_access_layer.download_jobs_manager import DownloadJobsManager
from logic_layer.algos_orchestation_logic import AlgosOrchestationLogic


class DataDownloaderController(BaseController):
    """
    Tab 1 — Execute downloads / spreads grouped by category.
    Tab 2 — Status: last run per job, re-run button.
    """

    def __init__(self, config_settings, logger):
        super().__init__()
        self.config   = config_settings
        self.logger   = logger
        self.jobs_mgr = DownloadJobsManager(config_settings["ml_reports_conn_str"], logger)

        self.router = APIRouter()
        self.templates = Jinja2Templates(
            directory=str(Path(__file__).parent.parent / "templates")
        )

        self.router.get("/",             response_class=HTMLResponse)(self.display_page)
        self.router.get("/groups",       response_class=JSONResponse)(self.api_get_groups)
        self.router.get("/jobs",         response_class=JSONResponse)(self.api_get_all_jobs)
        self.router.get("/jobs_by_group",response_class=JSONResponse)(self.api_get_jobs_by_group)
        self.router.post("/run_job",     response_class=JSONResponse)(self.api_run_job)
        self.router.post("/run_group",   response_class=JSONResponse)(self.api_run_group)
        self.router.post("/reset_job",   response_class=JSONResponse)(self.api_reset_job)

    # ── Pages ─────────────────────────────────────────────────────────────────

    async def display_page(self, request: Request):
        return self.templates.TemplateResponse("data_downloader.html", {"request": request})

    # ── API ───────────────────────────────────────────────────────────────────

    async def api_get_groups(self, request: Request):
        groups = self.jobs_mgr.get_download_job_groups()
        return JSONResponse([
            {
                "group_id":      g.group_id,
                "group_name":    g.group_name,
                "job_type":      g.job_type,
                "display_order": g.display_order,
                "job_count":     g.job_count,
            }
            for g in groups
        ])

    async def api_get_jobs_by_group(self, request: Request, group_id: int):
        # Need job_type from the group
        groups   = self.jobs_mgr.get_download_job_groups()
        group    = next((g for g in groups if g.group_id == group_id), None)
        job_type = group.job_type if group else "DOWNLOAD"
        jobs     = self.jobs_mgr.get_download_jobs(group_id, job_type)
        return JSONResponse([self._job_to_dict(j) for j in jobs])

    async def api_get_all_jobs(self, request: Request):
        return JSONResponse(self.jobs_mgr.get_all_download_jobs())

    async def api_run_job(self, request: Request):
        body     = await request.json()
        job_id   = int(body["job_id"])
        group_id = int(body["group_id"])

        groups   = self.jobs_mgr.get_download_job_groups()
        group    = next((g for g in groups if g.group_id == group_id), None)
        job_type = group.job_type if group else "DOWNLOAD"
        jobs     = self.jobs_mgr.get_download_jobs(group_id, job_type)
        job      = next((j for j in jobs if j.job_id == job_id), None)

        if not job:
            return JSONResponse({"ok": False, "error": f"job_id {job_id} not found"}, status_code=404)

        log_id, log_text, error_msg, ok = self._run_job(job)
        return JSONResponse({"ok": ok, "log": log_text, "error": error_msg})

    async def api_run_group(self, request: Request):
        body     = await request.json()
        group_id = int(body["group_id"])

        groups   = self.jobs_mgr.get_download_job_groups()
        group    = next((g for g in groups if g.group_id == group_id), None)
        job_type = group.job_type if group else "DOWNLOAD"
        jobs     = self.jobs_mgr.get_download_jobs(group_id, job_type)

        if not jobs:
            return JSONResponse({"ok": False, "error": "No jobs found for group"}, status_code=404)

        results, all_ok = [], True
        for job in jobs:
            log_id, log_text, error_msg, ok = self._run_job(job)
            if not ok:
                all_ok = False
            results.append({"job_id": job.job_id, "symbol": job.symbol,
                             "ok": ok, "log": log_text, "error": error_msg})

        return JSONResponse({"ok": all_ok, "results": results})

    async def api_reset_job(self, request: Request):
        """Reset a stuck RUNNING job (or all stuck jobs if job_id omitted)."""
        body   = await request.json()
        job_id = body.get("job_id")   # optional — None = reset all
        rows   = self.jobs_mgr.reset_stuck_jobs(job_id)
        return JSONResponse({"ok": True, "rows_reset": rows})

    # ── Execution ─────────────────────────────────────────────────────────────

    def _run_job(self, job) -> tuple:
        """Execute one job. Returns (log_id, stdout_text, error_msg, ok)."""
        log_id = self.jobs_mgr.start_download_job_log(job.job_id, job.group_id)
        buf    = io.StringIO()
        ok     = True
        error_msg = None

        try:
            with redirect_stdout(buf):
                self._execute_job(job)
        except Exception as e:
            ok        = False
            error_msg = traceback.format_exc()
            buf.write(f"\n❌ {e}\n{error_msg}")

        log_text = buf.getvalue()
        self.jobs_mgr.finish_download_job_log(log_id, "OK" if ok else "ERROR", log_text, error_msg)
        return log_id, log_text, error_msg, ok

    # ── Symbol parsing ────────────────────────────────────────────────────────
    # No longer needed — symbol and exchange are stored separately in the DB.
    # Kept as a static helper in case it's useful for one-off migrations.

    @staticmethod
    def _parse_symbol(raw_symbol: str, vendor: str):
        """Legacy helper. DB now stores symbol and exchange as separate columns."""
        vendor = vendor.upper()
        for suffix in (".TRADINGVIEW", ".FRED"):
            if raw_symbol.upper().endswith(suffix):
                raw_symbol = raw_symbol[: -len(suffix)]
                break
        parts    = raw_symbol.split(".")
        ticker   = parts[0]
        exchange = parts[1] if len(parts) > 1 else None
        return ticker, exchange

    def _execute_job(self, job):
        d_from = str(job.d_from)
        d_to   = str(job.d_to) if job.d_to else datetime.today().strftime("%Y-%m-%d")

        aol = AlgosOrchestationLogic(
            self.config["hist_data_conn_str"],
            self.config["ml_reports_conn_str"],
            None,
            self.logger
        )

        if job.job_type == "SPREAD":
            print(f"[SPREAD] {job.symbol} → {job.output_symbol}  from={d_from}")
            aol.process_create_spread_varaible(
                diff_indicators=job.symbol,
                d_from=datetime.strptime(d_from, "%Y-%m-%d").date(),
                d_to=datetime.strptime(d_to,   "%Y-%m-%d").date(),
                output_symbol=job.output_symbol,
            )
            print(f"[SPREAD-DONE] {job.output_symbol}")

        else:
            vendor        = job.vendor.upper()
            ticker        = job.symbol      # already clean in DB
            exchange      = job.exchange    # already split in DB
            vendor_params = {}

            if vendor == InformationVendors.FRED.value:
                vendor_params["api_key"] = self.config.get("FRED_API_KEY", "")

            elif vendor == InformationVendors.TRADINGVIEW.value:
                vendor_params["tradingview_user"] = self.config.get("TRADING_VIEW_USER", "")
                vendor_params["tradingview_pwd"]  = self.config.get("TRADING_VIEW_PWD",  "")
                if exchange:
                    vendor_params["exchange"] = exchange

            else:
                raise Exception(f"Unsupported vendor: {vendor}")

            print(f"[DOWNLOAD] symbol={ticker}  exchange={exchange}  vendor={vendor}  from={d_from}  to={d_to}")
            aol.process_download_financial_data_bulk(
                symbol=ticker,
                d_from=d_from,
                d_to=d_to,
                algo_params={
                    "vendor":        vendor,
                    "vendor_params": vendor_params,
                    "interval":      job.interval_code,
                },
            )
            print(f"[DOWNLOAD-DONE] {ticker}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _job_to_dict(self, job) -> dict:
        return {
            "job_id":           job.job_id,
            "group_id":         job.group_id,
            "job_type":         job.job_type,
            "symbol":           job.symbol,
            "exchange":         job.exchange,
            "output_symbol":    job.output_symbol,
            "vendor":           job.vendor,
            "d_from":           job.d_from,
            "d_to":             job.d_to,
            "interval_code":    job.interval_code,
            "last_status":      job.last_status,
            "last_run_at":      job.last_run_at,
            "last_finished_at": job.last_finished_at,
            "last_error":       job.last_error,
        }