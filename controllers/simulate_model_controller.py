"""
controllers/model_runner_controller.py
Controller for the Model Runner page — manages running_model_configs,
executes XGBoost backtests, and returns prices + results for the frontend.
"""
from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from common.enums.intervals import Intervals
from controllers.base_controller import BaseController
from data_access_layer.economic_series_manager import EconomicSeriesManager
from data_access_layer.running_models_manager import RunningModelsManager
from logic_layer.algos_orchestation_logic import AlgosOrchestationLogic

_DAY_INTERVAL = Intervals.DAY.value


class SimulateModelController(BaseController):
    """
    Routes:
        GET  /                          → HTML page
        GET  /models                    → list all model configs
        GET  /model/{model_id}          → single model config
        POST /add_model                 → create model config + series
        POST /edit_model                → update model config + series
        POST /delete_model              → delete model config
        POST /run_model                 → execute backtest, return results
        GET  /prices                    → OHLC daily prices for chart
    """

    def __init__(self, config_settings, logger):
        super().__init__()
        self.config  = config_settings
        self.logger  = logger

        self.models_mgr = RunningModelsManager(
            config_settings["ml_reports_conn_str"], logger
        )
        self.econ_mgr = EconomicSeriesManager(
            config_settings["hist_data_conn_str"]
        )

        self.router    = APIRouter()
        self.templates = Jinja2Templates(
            directory=str(Path(__file__).parent.parent / "templates")
        )

        # Page
        self.router.get("/", response_class=HTMLResponse)(self.display_page)

        # Model CRUD
        self.router.get("/models",          response_class=JSONResponse)(self.api_get_models)
        self.router.get("/model/{model_id}", response_class=JSONResponse)(self.api_get_model)
        self.router.post("/add_model",      response_class=JSONResponse)(self.api_add_model)
        self.router.post("/edit_model",     response_class=JSONResponse)(self.api_edit_model)
        self.router.post("/delete_model",   response_class=JSONResponse)(self.api_delete_model)

        # Execution
        self.router.post("/run_model",      response_class=JSONResponse)(self.api_run_model)

        # Prices for chart
        self.router.get("/prices",          response_class=JSONResponse)(self.api_get_prices)

    # ── Page ──────────────────────────────────────────────────────────────────

    async def display_page(self, request: Request):
        return self.templates.TemplateResponse(
            "simulate_model.html", {"request": request}
        )

    # ── Model CRUD ────────────────────────────────────────────────────────────

    async def api_get_models(self, request: Request):
        models = self.models_mgr.get_running_model_configs(is_active=True)
        return JSONResponse([self._model_to_dict(m) for m in models])

    async def api_get_model(self, request: Request, model_id: int):
        m = self.models_mgr.get_running_model_by_id(model_id)
        if not m:
            return JSONResponse({"ok": False, "error": f"Model {model_id} not found"}, status_code=404)
        return JSONResponse(self._model_to_dict(m))

    async def api_add_model(self, request: Request):
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Bad JSON: {e}"}, status_code=400)

        try:
            model_id = self.models_mgr.persist_running_model_config(
                model_name             = body["model_name"].strip(),
                algo_type              = body.get("algo_type", "XGBOOST").strip().upper(),
                model_path             = body["model_path"].strip(),
                symbol                 = body["symbol"].strip().upper(),
                bias                   = body.get("bias", "LONG").strip().upper(),
                d_from                 = body["d_from"],
                d_to                   = body["d_to"],
                lower_percentile_limit = float(body.get("lower_percentile_limit", 0.3)),
                n_flip                 = int(body.get("n_flip", 3)),
                make_stationary        = bool(body.get("make_stationary", True)),
                draw_predictions       = False,   # always False — frontend handles chart
                init_portf_size        = float(body.get("init_portf_size", 100000.0)),
                trade_comm             = float(body.get("trade_comm", 0.0)),
                series_csv             = body.get("series_csv", ""),
                display_order          = int(body.get("display_order", 0)),
                is_active              = True,
            )
            return JSONResponse({"ok": True, "model_id": model_id})
        except KeyError as e:
            return JSONResponse({"ok": False, "error": f"Missing field: {e}"}, status_code=400)
        except Exception as e:
            self._log_error("api_add_model", e)
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_edit_model(self, request: Request):
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Bad JSON: {e}"}, status_code=400)

        model_id = body.get("model_id")
        if not model_id:
            return JSONResponse({"ok": False, "error": "model_id required"}, status_code=400)

        existing = self.models_mgr.get_running_model_by_id(int(model_id))
        if not existing:
            return JSONResponse({"ok": False, "error": f"Model {model_id} not found"}, status_code=404)

        try:
            self.models_mgr.persist_running_model_config(
                model_name             = body.get("model_name", existing.model_name).strip(),
                algo_type              = body.get("algo_type", existing.algo_type).strip().upper(),
                model_path             = body.get("model_path", existing.model_path).strip(),
                symbol                 = body.get("symbol", existing.symbol).strip().upper(),
                bias                   = body.get("bias", existing.bias).strip().upper(),
                d_from                 = body.get("d_from", existing.d_from),
                d_to                   = body.get("d_to", existing.d_to),
                lower_percentile_limit = float(body.get("lower_percentile_limit", existing.lower_percentile_limit)),
                n_flip                 = int(body.get("n_flip", existing.n_flip)),
                make_stationary        = bool(body.get("make_stationary", existing.make_stationary)),
                draw_predictions       = False,
                init_portf_size        = float(body.get("init_portf_size", existing.init_portf_size)),
                trade_comm             = float(body.get("trade_comm", existing.trade_comm)),
                series_csv             = body.get("series_csv", existing.series_csv),
                display_order          = int(body.get("display_order", existing.display_order)),
                is_active              = bool(body.get("is_active", existing.is_active)),
            )
            return JSONResponse({"ok": True})
        except Exception as e:
            self._log_error("api_edit_model", e)
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_delete_model(self, request: Request):
        try:
            body = await request.json()
            model_id = int(body["model_id"])
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        try:
            self.models_mgr.delete_running_model_config(model_id)
            return JSONResponse({"ok": True})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    # ── Execution ─────────────────────────────────────────────────────────────
    @staticmethod
    def _safe_json_response(data, status_code: int = 200):
        """JSONResponse que convierte NaN/Inf/float-subclasses a null."""
        import math
        import json

        class _SafeEncoder(json.JSONEncoder):
            def iterencode(self, o, _one_shot=False):
                # patch floats on the fly
                return super().iterencode(self._sanitize(o), _one_shot)

            def _sanitize(self, obj):
                if isinstance(obj, float):
                    if math.isnan(obj) or math.isinf(obj):
                        return None
                    return obj
                if isinstance(obj, dict):
                    return {k: self._sanitize(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [self._sanitize(v) for v in obj]
                return obj

        from starlette.responses import Response
        body = json.dumps(data, cls=_SafeEncoder)
        return Response(content=body, status_code=status_code,
                        media_type="application/json")

    async def api_run_model(self, request: Request):
        """
        Runs process_test_scalping_XGBoost and returns a serialised result.
        Body fields (all optional — override model defaults):
            model_id, d_from, d_to, bias, lower_percentile_limit, n_flip,
            make_stationary, init_portf_size, trade_comm
        """
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Bad JSON: {e}"}, status_code=400)

        model_id = body.get("model_id")
        if not model_id:
            return JSONResponse({"ok": False, "error": "model_id required"}, status_code=400)

        m = self.models_mgr.get_running_model_by_id(int(model_id))
        if not m:
            return JSONResponse({"ok": False, "error": f"Model {model_id} not found"}, status_code=404)

        # Merge model defaults with any request overrides
        d_from_str = body.get("d_from", m.d_from)
        d_to_str   = body.get("d_to",   m.d_to)

        n_algo_param_dict = {
            "bias":                   body.get("bias",                   m.bias),
            "lower_percentile_limit": float(body.get("lower_percentile_limit", m.lower_percentile_limit)),
            "n_flip":                 int(body.get("n_flip",             m.n_flip)),
            "make_stationary":        bool(body.get("make_stationary",   m.make_stationary)),
            "draw_predictions":       False,   # always off — frontend draws the chart
            "init_portf_size":        float(body.get("init_portf_size",  m.init_portf_size)),
            "trade_comm":             float(body.get("trade_comm",       m.trade_comm)),
            "classif_key":            body.get("classif_key", "signal"),
        }

        try:
            d_from = datetime.strptime(d_from_str, "%Y-%m-%d")
            d_to   = datetime.strptime(d_to_str,   "%Y-%m-%d")
        except ValueError as e:
            return JSONResponse({"ok": False, "error": f"Invalid date format: {e}"}, status_code=400)

        try:
            aol = AlgosOrchestationLogic(
                self.config["hist_data_conn_str"],
                self.config["ml_reports_conn_str"],
                None,
                self.logger,
            )

            results: dict = aol.process_test_scalping_XGBoost(
                symbol       = m.symbol,
                series_csv   = m.series_csv,
                model_to_use = m.model_path,
                d_from       = d_from,
                d_to         = d_to,
                n_algo_param_dict = n_algo_param_dict,
            )

            # results is {"DAILY_XGB": PortfSummary}
            summary = results.get("DAILY_XGB")
            if not summary:
                return JSONResponse({"ok": False, "error": "No DAILY_XGB result returned"}, status_code=500)

            return self._safe_json_response({
                "ok": True,
                "symbol": m.symbol,
                "d_from": d_from_str,
                "d_to": d_to_str,
                "summary": self._summary_to_dict(summary),
            })

        except Exception as e:
            self._log_error("api_run_model", e)
            return JSONResponse({"ok": False, "error": str(e), "trace": traceback.format_exc()}, status_code=500)

    # ── Prices ────────────────────────────────────────────────────────────────

    async def api_get_prices(
            self,
            request: Request,
            symbol: str,
            d_from: str,
            d_to: str,
    ):
        import math
        import json

        def _safe(v):
            if v is None:
                return None
            try:
                f = float(v)
                return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
            except Exception:
                return None

        try:
            from datetime import datetime as _dt
            dfrom = _dt.strptime(d_from, "%Y-%m-%d")
            dto = _dt.strptime(d_to, "%Y-%m-%d")
        except ValueError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

        try:
            candles = self.econ_mgr.get_economic_values(symbol, _DAY_INTERVAL, dfrom, dto)
            result = []
            for c in candles:
                date_val = c.date
                if hasattr(date_val, "strftime"):
                    date_str = date_val.strftime("%Y-%m-%d")
                else:
                    date_str = str(date_val)[:10]

                result.append({
                    "date": date_str,
                    "open": _safe(c.open),
                    "high": _safe(c.high),
                    "low": _safe(c.low),
                    "close": _safe(c.close),
                })

            # Serializar manualmente para evitar NaN/Inf residuales
            safe_payload = json.dumps({"ok": True, "prices": result})
            return self._safe_json_response({"ok": True, "prices": result})

        except Exception as e:
            self._log_error("api_get_prices", e)
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    # ── Serialisation helpers ─────────────────────────────────────────────────

    @staticmethod
    def _model_to_dict(m) -> dict:
        return {
            "model_id":               m.model_id,
            "model_name":             m.model_name,
            "algo_type":              m.algo_type,
            "model_path":             m.model_path,
            "symbol":                 m.symbol,
            "bias":                   m.bias,
            "d_from":                 m.d_from,
            "d_to":                   m.d_to,
            "lower_percentile_limit": m.lower_percentile_limit,
            "n_flip":                 m.n_flip,
            "make_stationary":        m.make_stationary,
            "draw_predictions":       m.draw_predictions,
            "init_portf_size":        m.init_portf_size,
            "trade_comm":             m.trade_comm,
            "series_csv":             m.series_csv,
            "display_order":          m.display_order,
            "is_active":              m.is_active,
        }

    @staticmethod
    def _summary_to_dict(s) -> dict:
        """
        Serialise a PortfSummary object into a JSON-safe dict.
        Handles Timestamp / date fields safely.
        """
        def _fmt(v):
            if v is None:
                return None
            if hasattr(v, "strftime"):
                return v.strftime("%Y-%m-%d")
            return str(v)[:10]

        def _float(v, default=0.0):
            try:
                return round(float(v), 4)
            except Exception:
                return default

        positions = []
        for p in (s.portf_pos_summary or []):
            positions.append({
                "symbol":      getattr(p, "symbol",      None),
                "side":        getattr(p, "side",         None),
                "price_open":  _float(getattr(p, "price_open",  None)),
                "price_close": _float(getattr(p, "price_close", None)),
                "date_open":   _fmt(getattr(p,  "date_open",    None)),
                "date_close":  _fmt(getattr(p,  "date_close",   None)),
                "units":       _float(getattr(p, "units",       None)),
                "nom_profit":  _float(getattr(p, "total_net_profit", None)
                                      if hasattr(p, "total_net_profit")
                                      else getattr(p, "calculate_th_nom_profit", lambda: None)()),
                "pct_profit":  _float(getattr(p, "calculate_pct_profit", lambda: None)()),
                "max_drawdown": _float(getattr(p, "calculate_max_drawdown", lambda: None)()
                                        or getattr(p, "max_drawdown", None)
                                        or 0.0
                                    ),
                # MTM series for chart — list of floats
                "daily_mtms":  [_float(x) for x in (getattr(p, "daily_MTMs", None) or [])],
            })

        # Last prediction signal — from n_algo_params if present
        last_signal_raw = None
        last_signal_date = None
        try:
            sigs = getattr(s, "last_signal_signals", None)
            if sigs:
                last_signal_raw = " → ".join(str(x) for x in sigs)
                last_signal_date = getattr(s, "last_signal_date", None)
        except Exception:
            pass

        return {
            "symbol":          s.symbol,
            "algo":            getattr(s, "trading_algo", "DAILY_XGB"),
            "period":          getattr(s, "period", None),
            "portf_init":      _float(s.portf_init_MTM),
            "portf_final":     _float(s.portf_final_MTM),
            "total_profit":    _float(s.total_net_profit),
            "profit_pct":      _float((s.portf_final_MTM - s.portf_init_MTM) / s.portf_init_MTM * 100)
                               if s.portf_init_MTM else 0.0,
            "max_drawdown":    _float(getattr(s, "drawdown_pct", None)
                          or getattr(s, "max_drawdown", None) or 0),
            "cagr":            _float(getattr(s, "cagr_pct", 0)),
            "positions":       positions,
            "last_signal":      last_signal_raw,   # "LONG → LONG → LONG"
            "last_signal_date": last_signal_date,  # "2026-02-23"
            "daily_profits":   [_float(x) for x in (s.daily_profits or [])],
        }

    def _log_error(self, where: str, exc: Exception):
        msg = f"{where}: {traceback.format_exc()}"
        if self.logger:
            try:
                from framework.common.logger.message_type import MessageType
                self.logger.do_log(msg, MessageType.ERROR)
            except Exception:
                print(msg)