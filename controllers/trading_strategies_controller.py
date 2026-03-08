import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from business_entities.trading_strategy_entities import TradingStrategy, StrategyDatabase, SymbolConfig
from controllers.base_controller import BaseController
from data_access_layer.day_turtle_manager import DailyTurtleManager


class TradingStrategiesController(BaseController):

    def __init__(self, config_settings: dict, logger):
        super().__init__()
        self.config  = config_settings
        self.logger  = logger
        self.mgr     = DailyTurtleManager(config_settings["fund_mgmt_dashboard_cs"])

        self.router    = APIRouter()
        self.templates = Jinja2Templates(
            directory=str(Path(__file__).parent.parent / "templates")
        )

        self.router.get("/",  response_class=HTMLResponse)(self.display_page)

        # Strategy CRUD
        self.router.get("/strategies",                      response_class=JSONResponse)(self.api_get_strategies)
        self.router.post("/strategies",                     response_class=JSONResponse)(self.api_persist_strategy)
        self.router.delete("/strategies/{strategy_id}",     response_class=JSONResponse)(self.api_delete_strategy)

        # StrategyDatabase CRUD
        self.router.get("/strategy_databases",              response_class=JSONResponse)(self.api_get_strategy_databases)
        self.router.post("/strategy_databases",             response_class=JSONResponse)(self.api_persist_strategy_database)
        self.router.delete("/strategy_databases/{db_id}",   response_class=JSONResponse)(self.api_delete_strategy_database)

        # Symbol config
        self.router.get("/symbol_configs",                  response_class=JSONResponse)(self.api_get_symbol_configs)
        self.router.post("/symbol_configs",                 response_class=JSONResponse)(self.api_persist_symbol_config)

        # Securities & trades
        self.router.get("/securities",                      response_class=JSONResponse)(self.api_get_securities)
        self.router.get("/trades",                          response_class=JSONResponse)(self.api_get_trades)
        self.router.get("/today_signals",                   response_class=JSONResponse)(self.api_today_signals)
        self.router.get("/monthly_performance",             response_class=JSONResponse)(self.api_monthly_performance)

        # OHLCV for chart
        self.router.get("/ohlcv",                           response_class=JSONResponse)(self.api_get_ohlcv)
        self.router.get("/ohlcv_debug",                     response_class=JSONResponse)(self.api_ohlcv_debug)

    # ── Page ──────────────────────────────────────────────────────────────────

    async def display_page(self, request: Request):
        return self.templates.TemplateResponse(
            "trading_strategies.html", {"request": request}
        )

    # ── Strategy CRUD ─────────────────────────────────────────────────────────

    async def api_get_strategies(self, request: Request):
        try:
            return JSONResponse([self._strategy_to_dict(s) for s in self.mgr.get_all_strategies()])
        except Exception as e:
            self.logger.do_log(f"api_get_strategies: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_persist_strategy(self, request: Request):
        try:
            body   = await request.json()
            entity = TradingStrategy(
                strategy_id   = body.get("strategy_id"),
                strategy_name = body["strategy_name"].strip(),
                exchange      = body.get("exchange", "NYSE").strip() or "NYSE",
                description   = body.get("description", "").strip() or None,
                is_active     = bool(body.get("is_active", True)),
            )
            if not entity.strategy_name:
                return JSONResponse({"ok": False, "error": "strategy_name is required"}, status_code=400)
            new_id = self.mgr.persist_strategy(entity)
            return JSONResponse({"ok": True, "strategy_id": new_id})
        except Exception as e:
            self.logger.do_log(f"api_persist_strategy: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_delete_strategy(self, request: Request, strategy_id: int):
        try:
            self.mgr.delete_strategy(strategy_id)
            return JSONResponse({"ok": True})
        except Exception as e:
            self.logger.do_log(f"api_delete_strategy: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    # ── StrategyDatabase CRUD ─────────────────────────────────────────────────

    async def api_get_strategy_databases(self, request: Request, strategy_id: int):
        try:
            dbs = self.mgr.get_strategy_databases(strategy_id)
            return JSONResponse([self._sdb_to_dict(d) for d in dbs])
        except Exception as e:
            self.logger.do_log(f"api_get_strategy_databases: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_persist_strategy_database(self, request: Request):
        try:
            body   = await request.json()
            entity = StrategyDatabase(
                db_id         = body.get("db_id"),
                strategy_id   = int(body["strategy_id"]),
                database_name = body["database_name"].strip(),
                label         = body.get("label", "").strip() or None,
                is_default    = bool(body.get("is_default", False)),
            )
            if not entity.database_name:
                return JSONResponse({"ok": False, "error": "database_name is required"}, status_code=400)
            new_id = self.mgr.persist_strategy_database(entity)
            return JSONResponse({"ok": True, "db_id": new_id})
        except Exception as e:
            self.logger.do_log(f"api_persist_strategy_database: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_delete_strategy_database(self, request: Request, db_id: int):
        try:
            self.mgr.delete_strategy_database(db_id)
            return JSONResponse({"ok": True})
        except Exception as e:
            self.logger.do_log(f"api_delete_strategy_database: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    # ── Symbol config ─────────────────────────────────────────────────────────

    async def api_get_symbol_configs(self, request: Request):
        try:
            return JSONResponse([self._sym_to_dict(s) for s in self.mgr.get_all_symbol_configs()])
        except Exception as e:
            self.logger.do_log(f"api_get_symbol_configs: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_persist_symbol_config(self, request: Request):
        try:
            body   = await request.json()
            entity = SymbolConfig(
                symbol   = body["symbol"].strip().upper(),
                exchange = body["exchange"].strip().upper(),
                notes    = body.get("notes", "").strip() or None,
            )
            self.mgr.persist_symbol_config(entity)
            return JSONResponse({"ok": True})
        except Exception as e:
            self.logger.do_log(f"api_persist_symbol_config: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    # ── Securities ────────────────────────────────────────────────────────────

    async def api_get_securities(self, request: Request, strategy_id: int, database_name: str):
        try:
            strategy = self.mgr.get_strategy(strategy_id)
            if not strategy:
                return JSONResponse({"ok": False, "error": "Strategy not found"}, status_code=404)
            securities = self.mgr.get_securities_for_strategy(strategy.strategy_name, database_name)
            return JSONResponse({
                "strategy":   self._strategy_to_dict(strategy),
                "database":   database_name,
                "securities": [self._security_to_dict(s) for s in securities],
            })
        except Exception as e:
            self.logger.do_log(f"api_get_securities: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    # ── Trades ────────────────────────────────────────────────────────────────

    async def api_get_trades(self, request: Request, strategy_id: int, database_name: str,
                              symbol: str, date_from: str = None, date_to: str = None):
        try:
            strategy = self.mgr.get_strategy(strategy_id)
            if not strategy:
                return JSONResponse({"ok": False, "error": "Strategy not found"}, status_code=404)
            df = datetime.strptime(date_from, "%Y-%m-%d") if date_from else None
            dt = datetime.strptime(date_to,   "%Y-%m-%d") if date_to   else None
            trades = self.mgr.get_trades(strategy.strategy_name, database_name, symbol, df, dt)

            # Resolve exchange: symbol config overrides strategy default
            sym_cfg  = self.mgr.get_symbol_config(symbol.upper())
            exchange = sym_cfg.exchange if sym_cfg else strategy.exchange

            return JSONResponse({
                "strategy": self._strategy_to_dict(strategy),
                "database": database_name,
                "symbol":   symbol,
                "exchange": exchange,
                "trades":   [self._trade_to_dict(t) for t in trades],
            })
        except Exception as e:
            self.logger.do_log(f"api_get_trades: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_today_signals(self, request: Request, strategy_id: int, database_name: str):
        try:
            strategy = self.mgr.get_strategy(strategy_id)
            if not strategy:
                return JSONResponse({"ok": False, "error": "Strategy not found"}, status_code=404)
            trades = self.mgr.get_today_signals(strategy.strategy_name, database_name)
            return JSONResponse({
                "strategy": self._strategy_to_dict(strategy),
                "date":     datetime.today().strftime("%Y-%m-%d"),
                "signals":  [self._trade_to_dict(t) for t in trades],
            })
        except Exception as e:
            self.logger.do_log(f"api_today_signals: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_monthly_performance(self, request: Request, strategy_id: int,
                                       database_name: str, symbol: str = None):
        try:
            strategy = self.mgr.get_strategy(strategy_id)
            if not strategy:
                return JSONResponse({"ok": False, "error": "Strategy not found"}, status_code=404)
            rows = self.mgr.get_monthly_performance(strategy.strategy_name, database_name, symbol or None)
            return JSONResponse([self._monthly_to_dict(r) for r in rows])
        except Exception as e:
            self.logger.do_log(f"api_monthly_performance: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    # ── OHLCV endpoint ────────────────────────────────────────────────────────
    # Downloads 1-minute bars for a symbol+date using TradingViewDownloader.
    # Returns list of {time, open, high, low, close, volume}

    async def api_get_ohlcv(self, request: Request, symbol: str, exchange: str, date: str):
        """
        GET /ohlcv?symbol=GGAL&exchange=NASDAQ&date=2026-03-05
        Downloads 1-min bars from TradingView for the given day.
        tvDatafeed returns timestamps in UTC — we convert to UTC-3 (Argentina)
        so the chart axis matches the trade times stored in the DB.
        """
        try:
            from common.util.downloaders.tradingview_downloader import TradingViewDownloader

            params = {
                "tradingview_user": self.config["TRADING_VIEW_USER"],
                "tradingview_pwd":  self.config["TRADING_VIEW_PWD"],
                "interval":         "1m",
                "exchange":         exchange.upper(),
            }
            downloader = TradingViewDownloader(params)

            target_date = datetime.strptime(date, "%Y-%m-%d").date()
            df = downloader.download(symbol=symbol.upper())

            if df is None or df.empty:
                return JSONResponse({"ok": False, "error": f"No data returned for {symbol}"}, status_code=404)

            # tvDatafeed returns timestamps already in GMT-3. No conversion needed.
            # Filter by date as-is.
            df = df[df.index.date == target_date]

            if df.empty:
                return JSONResponse({"ok": False, "error": f"No data for {symbol} on {date}"}, status_code=404)

            bars = []
            for ts, row in df.iterrows():
                bars.append({
                    "time":   int(ts.timestamp()),
                    "open":   round(float(row["open"]),  4),
                    "high":   round(float(row["high"]),  4),
                    "low":    round(float(row["low"]),   4),
                    "close":  round(float(row["close"]), 4),
                    "volume": int(row.get("volume", 0)),
                })

            return JSONResponse({"ok": True, "symbol": symbol, "exchange": exchange,
                                  "date": date, "bars": bars})

        except Exception as e:
            self.logger.do_log(f"api_get_ohlcv: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    # ── Timezone debug endpoint ───────────────────────────────────────────────
    # GET /ohlcv_debug?symbol=GGAL&exchange=NASDAQ&date=2026-03-05
    # Returns first 5 bars with both utc_unix and local_unix for easy comparison

    async def api_ohlcv_debug(self, request: Request, symbol: str, exchange: str, date: str):
        try:
            import pytz
            from common.util.downloaders.tradingview_downloader import TradingViewDownloader

            params = {
                "tradingview_user": self.config["TRADING_VIEW_USER"],
                "tradingview_pwd":  self.config["TRADING_VIEW_PWD"],
                "interval":         "1m",
                "exchange":         exchange.upper(),
            }
            downloader  = TradingViewDownloader(params)
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
            df          = downloader.download(symbol=symbol.upper())

            tz_utc = pytz.utc
            tz_arg = pytz.timezone("America/Argentina/Buenos_Aires")
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize(tz_utc)
            df.index = df.index.tz_convert(tz_arg)
            df = df[df.index.date == target_date].head(5)

            rows = []
            for ts, row in df.iterrows():
                utc_unix   = int(ts.timestamp())
                local_unix = utc_unix - 3 * 3600
                rows.append({
                    "ts_raw":         str(ts),
                    "ts_utc_str":     ts.astimezone(tz_utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "ts_art_str":     ts.strftime("%Y-%m-%d %H:%M:%S ART"),
                    "utc_unix":       utc_unix,
                    "local_unix":     local_unix,
                    "bar_time_sent":  local_unix,
                    "chart_will_show": datetime.utcfromtimestamp(local_unix).strftime("%H:%M"),
                })
            return JSONResponse({"ok": True, "rows": rows,
                                  "note": "chart_will_show = what LightweightCharts displays on axis"})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e), "trace": traceback.format_exc()}, status_code=500)

    # ── Serializers ───────────────────────────────────────────────────────────

    @staticmethod
    def _strategy_to_dict(s: TradingStrategy) -> dict:
        return {
            "strategy_id":   s.strategy_id,
            "strategy_name": s.strategy_name,
            "exchange":      s.exchange,
            "description":   s.description,
            "is_active":     s.is_active,
            "created_at":    s.created_at.isoformat() if s.created_at else None,
            "updated_at":    s.updated_at.isoformat() if s.updated_at else None,
        }

    @staticmethod
    def _sdb_to_dict(d: StrategyDatabase) -> dict:
        return {
            "db_id":         d.db_id,
            "strategy_id":   d.strategy_id,
            "database_name": d.database_name,
            "label":         d.label,
            "is_default":    d.is_default,
            "created_at":    d.created_at.isoformat() if d.created_at else None,
        }

    @staticmethod
    def _sym_to_dict(s: SymbolConfig) -> dict:
        return {
            "symbol":     s.symbol,
            "exchange":   s.exchange,
            "notes":      s.notes,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
        }

    @staticmethod
    def _security_to_dict(s) -> dict:
        return {
            "symbol":        s.symbol,
            "trade_count":   s.trade_count,
            "first_trade":   s.first_trade.isoformat() if s.first_trade else None,
            "last_trade":    s.last_trade.isoformat()  if s.last_trade  else None,
            "closed_trades": s.closed_trades,
            "open_trades":   s.open_trades,
            "total_profit":  round(s.total_profit, 4),
            "avg_profit":    round(s.avg_profit,   4),
        }

    @staticmethod
    def _trade_to_dict(t) -> dict:
        def _fmt(dt): return dt.isoformat() if dt else None
        return {
            "id":                   t.id,
            "strategy_name":        t.strategy_name,
            "opening_date":         _fmt(t.opening_date),
            "closing_date":         _fmt(t.closing_date),
            "symbol":               t.symbol,
            "qty":                  t.qty,
            "trade_direction":      t.trade_direction,
            "opening_price":        t.opening_price,
            "closing_price":        t.closing_price,
            "last_price":           t.last_price,
            "total_fee":            t.total_fee,
            "initial_cap":          t.initial_cap,
            "final_cap":            t.final_cap,
            "profit":               t.profit,
            "nominal_profit":       t.nominal_profit,
            "fee_type":             t.fee_type,
            "fee_value":            t.fee_value,
            "trendline_start_date": _fmt(t.trendline_start_date),
            "trendline_end_date":   _fmt(t.trendline_end_date),
            "is_closed":            t.is_closed,
            "duration_minutes":     t.duration_minutes,
        }

    @staticmethod
    def _monthly_to_dict(m) -> dict:
        return {
            "symbol":               m.symbol,
            "yr":                   m.yr,
            "mo":                   m.mo,
            "month_label":          m.month_label,
            "trade_count":          m.trade_count,
            "total_profit":         round(m.total_profit,         4),
            "total_nominal_profit": round(m.total_nominal_profit, 4),
            "avg_profit":           round(m.avg_profit,           4),
            "winning_trades":       m.winning_trades,
            "losing_trades":        m.losing_trades,
            "win_rate":             m.win_rate,
        }