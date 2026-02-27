import io
import traceback
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from common.enums.information_vendors import InformationVendors
from controllers.base_controller import BaseController
from data_access_layer.download_jobs_manager import DownloadJobsManager, CandleManager
from logic_layer.algos_orchestation_logic import AlgosOrchestationLogic

MANUAL_VENDOR = "MANUAL_VARIABLE"
SPREAD_VENDOR = "SPREAD"


class DataDownloaderController(BaseController):

    def __init__(self, config_settings, logger):
        super().__init__()
        self.config      = config_settings
        self.logger      = logger
        self.jobs_mgr    = DownloadJobsManager(config_settings["ml_reports_conn_str"], logger)
        self.candle_mgr  = CandleManager(config_settings["hist_data_conn_str"], logger)

        self.router    = APIRouter()
        self.templates = Jinja2Templates(
            directory=str(Path(__file__).parent.parent / "templates")
        )

        # Pages
        self.router.get("/", response_class=HTMLResponse)(self.display_page)

        # Groups & jobs read
        self.router.get("/groups",        response_class=JSONResponse)(self.api_get_groups)
        self.router.get("/jobs",          response_class=JSONResponse)(self.api_get_all_jobs)
        self.router.get("/jobs_by_group", response_class=JSONResponse)(self.api_get_jobs_by_group)

        # CRUD jobs  (#1)
        self.router.post("/add_job",    response_class=JSONResponse)(self.api_add_job)
        self.router.post("/edit_job",   response_class=JSONResponse)(self.api_edit_job)
        self.router.post("/delete_job", response_class=JSONResponse)(self.api_delete_job)

        # Execute
        self.router.post("/run_job",   response_class=JSONResponse)(self.api_run_job)
        self.router.post("/run_group", response_class=JSONResponse)(self.api_run_group)
        self.router.post("/reset_job", response_class=JSONResponse)(self.api_reset_job)

        # Data health  (#3 + #4)
        self.router.get("/last_values",          response_class=JSONResponse)(self.api_last_values)
        self.router.get("/manual_candles",        response_class=JSONResponse)(self.api_get_manual_candles)
        self.router.post("/save_manual_candle",   response_class=JSONResponse)(self.api_save_manual_candle)

    # ── Pages ─────────────────────────────────────────────────────────────────

    async def display_page(self, request: Request):
        return self.templates.TemplateResponse("data_downloader.html", {"request": request})

    # ── Groups / Jobs read ────────────────────────────────────────────────────

    async def api_get_groups(self, request: Request):
        groups = self.jobs_mgr.get_download_job_groups()
        return JSONResponse([{
            "group_id":      g.group_id,
            "group_name":    g.group_name,
            "job_type":      g.job_type,
            "display_order": g.display_order,
            "job_count":     g.job_count,
        } for g in groups])

    async def api_get_jobs_by_group(self, request: Request, group_id: int):
        groups   = self.jobs_mgr.get_download_job_groups()
        group    = next((g for g in groups if g.group_id == group_id), None)
        job_type = group.job_type if group else "DOWNLOAD"
        jobs     = self.jobs_mgr.get_download_jobs(group_id, job_type)
        return JSONResponse([self._job_to_dict(j) for j in jobs])

    async def api_get_all_jobs(self, request: Request):
        return JSONResponse(self.jobs_mgr.get_all_download_jobs())

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _s(val, fallback=None) -> Optional[str]:
        """Safely clean a JSON string field.
        Handles None/null from JSON, empty strings, and whitespace.
        Falls back to `fallback` when val is None (used in edit to keep existing value)."""
        v = val if val is not None else fallback
        if v is None:
            return None
        cleaned = str(v).strip().upper()
        return cleaned if cleaned else None

    # ── CRUD jobs  (#1) ───────────────────────────────────────────────────────

    async def api_add_job(self, request: Request):
        """Add a new job to a group."""
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Bad JSON body: {e}"}, status_code=400)
        try:
            symbol = self._s(body.get("symbol"))
            vendor = self._s(body.get("vendor"))
            if not symbol:
                return JSONResponse({"ok": False, "error": "symbol is required"}, status_code=400)
            if not vendor:
                return JSONResponse({"ok": False, "error": "vendor is required"}, status_code=400)

            job_id = self.jobs_mgr.persist_download_job(
                job_id        = None,
                group_id      = int(body["group_id"]),
                symbol        = symbol,
                exchange      = self._s(body.get("exchange")),
                output_symbol = self._s(body.get("output_symbol")),
                vendor        = vendor,
                d_from        = body["d_from"],
                d_to          = body.get("d_to") or None,
                interval_code = body.get("interval_code") or "1d",
            )
            return JSONResponse({"ok": True, "job_id": job_id})
        except KeyError as e:
            return JSONResponse({"ok": False, "error": f"Missing field: {e}"}, status_code=400)
        except Exception as e:
            self.logger.do_log(f"api_add_job: {traceback.format_exc()}", MessageType.ERROR)
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_edit_job(self, request: Request):
        """Edit symbol, exchange, output_symbol, d_from, d_to. Vendor is immutable."""
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Bad JSON body: {e}"}, status_code=400)
        try:
            job_id = int(body["job_id"])
        except (KeyError, TypeError, ValueError) as e:
            return JSONResponse({"ok": False, "error": f"Invalid job_id: {e}"}, status_code=400)

        # Load current job to preserve immutable fields
        groups   = self.jobs_mgr.get_download_job_groups()
        all_jobs = []
        for g in groups:
            all_jobs.extend(self.jobs_mgr.get_download_jobs(g.group_id, g.job_type))
        job = next((j for j in all_jobs if j.job_id == job_id), None)
        if not job:
            return JSONResponse({"ok": False, "error": f"Job {job_id} not found"}, status_code=404)

        try:
            self.jobs_mgr.persist_download_job(
                job_id        = job_id,
                group_id      = job.group_id,
                symbol        = self._s(body.get("symbol"),        job.symbol),
                exchange      = self._s(body.get("exchange"),      job.exchange),
                output_symbol = self._s(body.get("output_symbol"), job.output_symbol),
                vendor        = job.vendor,   # IMMUTABLE — never overwrite
                d_from        = body.get("d_from") or job.d_from,
                d_to          = body.get("d_to") or None,
                interval_code = job.interval_code,
            )
            return JSONResponse({"ok": True})
        except Exception as e:
            self.logger.do_log(f"api_edit_job: {traceback.format_exc()}", MessageType.ERROR)
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_delete_job(self, request: Request):
        body   = await request.json()
        job_id = int(body["job_id"])
        try:
            self.jobs_mgr.delete_download_job(job_id)
            return JSONResponse({"ok": True})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    # ── Execute ───────────────────────────────────────────────────────────────

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
        body   = await request.json()
        job_id = body.get("job_id")
        rows   = self.jobs_mgr.reset_stuck_jobs(job_id)
        return JSONResponse({"ok": True, "rows_reset": rows})

    # ── Data Health  (#3) ─────────────────────────────────────────────────────

    async def api_last_values(self, request: Request):
        """
        Returns last candle per symbol, joined with job metadata.
        Jobs that have no candle entry get last_date=None, days_ago=None.
        """
        # All active jobs
        groups   = self.jobs_mgr.get_download_job_groups()
        all_jobs = []
        for g in groups:
            for j in self.jobs_mgr.get_download_jobs(g.group_id, g.job_type):
                all_jobs.append((g, j))

        # Last candle index keyed by symbol
        candles  = self.candle_mgr.get_last_candle_per_symbol()
        candle_map = {c.symbol: c for c in candles}

        result = []
        for g, j in all_jobs:
            # Determine which symbol to look up in candles
            lookup = j.output_symbol if j.output_symbol else j.symbol
            c = candle_map.get(lookup)
            result.append({
                "job_id":       j.job_id,
                "group_name":   g.group_name,
                "symbol":       j.symbol,
                "exchange":     j.exchange,
                "output_symbol":j.output_symbol,
                "vendor":       j.vendor,
                "last_date":    c.last_date  if c else None,
                "last_close":   c.last_close if c else None,
                "days_ago":     c.days_ago   if c else None,
            })

        return JSONResponse(result)

    # ── Manual variables  (#4) ────────────────────────────────────────────────

    async def api_get_manual_candles(self, request: Request, symbol: str):
        candles = self.candle_mgr.get_recent_candles(symbol, top=5)
        return JSONResponse([{"symbol": c.symbol, "date": c.date, "value": c.value} for c in candles])

    async def api_save_manual_candle(self, request: Request):
        body  = await request.json()
        symbol = body["symbol"].strip().upper()
        d      = body["date"]
        value  = float(body["value"])
        try:
            self.candle_mgr.persist_manual_candle(symbol, d, value)
            return JSONResponse({"ok": True})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    # ── Execution engine ──────────────────────────────────────────────────────

    def _run_job(self, job) -> tuple:
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

    def _execute_job(self, job):
        d_from = str(job.d_from)
        d_to   = str(job.d_to) if job.d_to else datetime.today().strftime("%Y-%m-%d")

        aol = AlgosOrchestationLogic(
            self.config["hist_data_conn_str"],
            self.config["ml_reports_conn_str"],
            None,
            self.logger
        )

        vendor = job.vendor.upper()

        # ── SPREAD  (#2) ──────────────────────────────────────────────────────
        if vendor == SPREAD_VENDOR or job.job_type == "SPREAD":
            print(f"[SPREAD] {job.symbol} → {job.output_symbol}  from={d_from}")
            aol.process_create_spread_varaible(
                diff_indicators=job.symbol,
                d_from=datetime.strptime(d_from, "%Y-%m-%d").date(),
                d_to=datetime.strptime(d_to,   "%Y-%m-%d").date(),
                output_symbol=job.output_symbol,
            )
            print(f"[SPREAD-DONE] {job.output_symbol}")

        # ── MANUAL_VARIABLE — skip execution (user manages manually) ──────────
        elif vendor == MANUAL_VENDOR:
            print(f"[MANUAL] {job.symbol} — manual variable, skipping automated download")

        # ── FRED ──────────────────────────────────────────────────────────────
        elif vendor == InformationVendors.FRED.value:
            vendor_params = {"api_key": self.config.get("FRED_API_KEY", "")}
            print(f"[DOWNLOAD] symbol={job.symbol}  vendor=FRED  from={d_from}  to={d_to}")
            aol.process_download_financial_data_bulk(
                symbol=job.symbol,
                d_from=d_from, d_to=d_to,
                algo_params={"vendor": vendor, "vendor_params": vendor_params,
                             "interval": job.interval_code},
            )
            print(f"[DOWNLOAD-DONE] {job.symbol}")

        # ── TRADINGVIEW ───────────────────────────────────────────────────────
        elif vendor == InformationVendors.TRADINGVIEW.value:
            vendor_params = {
                "tradingview_user": self.config.get("TRADING_VIEW_USER", ""),
                "tradingview_pwd":  self.config.get("TRADING_VIEW_PWD",  ""),
            }
            if job.exchange:
                vendor_params["exchange"] = job.exchange
            print(f"[DOWNLOAD] symbol={job.symbol}  exchange={job.exchange}  vendor=TV  from={d_from}  to={d_to}")
            aol.process_download_financial_data_bulk(
                symbol=job.symbol,
                d_from=d_from, d_to=d_to,
                algo_params={"vendor": vendor, "vendor_params": vendor_params,
                             "interval": job.interval_code},
            )
            print(f"[DOWNLOAD-DONE] {job.symbol}")

        else:
            raise Exception(f"Unsupported vendor: {vendor}")

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