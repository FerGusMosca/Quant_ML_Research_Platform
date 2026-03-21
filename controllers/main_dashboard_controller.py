from fastapi import FastAPI, Request
from fastapi.params import Form
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from itsdangerous import TimestampSigner
from starlette.responses import RedirectResponse
from controllers.account_controller import AccountController
from controllers.auth_middleware import AuthMiddleware
from controllers.chunk_management_controller import ChunkManagementController
from controllers.display_custom_etf_controller import DisplayCustomETFController
from controllers.display_series_controller import DisplaySeriesController
from controllers.download_jobs_controller import DataDownloaderController
from controllers.global_m2_indicator_controller import GlobalM2IndicatorController
from controllers.load_series_controller import LoadSeriesController
from controllers.portfolio_views_controller import PortfolioViewController
from controllers.routing_dashboard_controller import RoutingDashboardController
from controllers.simulate_indicator_strategy_controller import SimulateIndicatorStrategy
from POC.stripe_ACH_POC_controller import StripeAchDemoController
from POC.stripe_USDC_POC_controller import StripeUSDCDemoController
from controllers.simulate_model_controller import SimulateModelController
from controllers.trading_strategies_controller import TradingStrategiesController
from data_access_layer.account_data_manager import AccountDataManager
from data_access_layer.account_manager import AccountManager
from data_access_layer.ib_portfolio_manager import IBPortfolioManager
from data_access_layer.instruction_manager import InstructionManager
from data_access_layer.user_manager import UserManager
from framework.common.logger.message_type import MessageType
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


class MainDashboardController:
    def __init__(self, logger, config_settings):
        self.logger = logger
        self.fred_api_key = config_settings.get("FRED_API_KEY", "")

        self.valid_users = {"admin": "Test123123"}

        self.signer = TimestampSigner("my_super_secret")

        # ✅ Create the main FastAPI instance
        self.app = FastAPI()

        ib_prod_ws = config_settings["IB_PROD_WS"]
        primary_prod_ws = config_settings["PRIMARY_PROD_WS"]
        ib_dev_ws = config_settings["IB_DEV_WS"]
        fund_mgmt_dashboard_cs = config_settings["fund_mgmt_dashboard_cs"]
        secret_key = "my_super_secret"

        #managers
        account_mgr  = AccountManager(fund_mgmt_dashboard_cs)
        account_data_mgr =AccountDataManager(fund_mgmt_dashboard_cs)
        ib_portfolio_manager = IBPortfolioManager(fund_mgmt_dashboard_cs)
        instr_mgr = InstructionManager(fund_mgmt_dashboard_cs)

        self.user_manager = UserManager(fund_mgmt_dashboard_cs, secret_key)

        self.routing_dashboard = RoutingDashboardController(logger, ib_prod_ws, primary_prod_ws, ib_dev_ws, fund_mgmt_dashboard_cs)
        self.app.include_router(self.routing_dashboard.router, prefix="/routing_dashboard")

        self.custom_etf_controller = DisplayCustomETFController(config_settings, logger)
        self.app.include_router(self.custom_etf_controller.router, prefix="/display_custom_etf")

        self.load_series_controller = LoadSeriesController(config_settings, logger)
        self.app.include_router(self.load_series_controller.router, prefix="/load_series")

        self.simulate_indicator_strategy = SimulateIndicatorStrategy(config_settings, logger)
        self.app.include_router(self.simulate_indicator_strategy.router, prefix="/simulate_indicator_strategy")

        self.display_series_controller = DisplaySeriesController(config_settings, logger)
        self.app.include_router(self.display_series_controller.router, prefix="/display_series")

        self.account_controller = AccountController(account_mgr,account_data_mgr)
        self.app.include_router(self.account_controller.router)

        self.stripe_ACH_POC_controller = StripeAchDemoController(config_settings, logger)
        self.app.include_router(self.stripe_ACH_POC_controller.router, prefix="/stripe_ACH_POC")

        self.stripe_SUDC_POC_controller = StripeUSDCDemoController(config_settings, logger)
        self.app.include_router(self.stripe_SUDC_POC_controller.router, prefix="/stripe_USDC_POC")

        global_m2_controller = GlobalM2IndicatorController(config_settings, logger)
        self.app.include_router(global_m2_controller.router, prefix="/global_m2_indicator")

        self.data_downloader_controller = DataDownloaderController(config_settings, logger)
        self.app.include_router(self.data_downloader_controller.router, prefix="/data_downloader")

        self.model_runner_controller = SimulateModelController(config_settings, logger)
        self.app.include_router(self.model_runner_controller.router, prefix="/simulate_model")

        self.chunk_mgmt_ctrl = ChunkManagementController(config_settings, logger)
        self.app.include_router(self.chunk_mgmt_ctrl.router,prefix="/chunk_management")

        portfolio_view_controller = PortfolioViewController(account_manager=account_mgr,  account_data_manager=account_data_mgr,
                                                            ib_portfolio_manager=ib_portfolio_manager,instruction_manager=instr_mgr)
        self.app.include_router(portfolio_view_controller.router)

        self.trading_strategies_ctrl = TradingStrategiesController(config_settings, logger)
        self.app.include_router(self.trading_strategies_ctrl.router, prefix="/trading_strategies")


        # ── FRED proxy router ──
        self.app.include_router(self._build_fred_router(), prefix="/api")

        # ✅ Templates & static
        templates_path = Path(__file__).parent.parent / "templates"
        self.templates = Jinja2Templates(directory=templates_path)

        self.app.get("/", response_class=HTMLResponse)(self.main_dashboard)
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        self.app.post("/login")(self.login)
        self.app.get("/login", response_class=HTMLResponse)(self.login_form)
        self.app.get("/logout")(self.logout)
        self.app.add_event_handler("startup", self.startup_event)
        self.app.add_middleware(
            AuthMiddleware,
            secret_key="my_super_secret",
            exempt_paths=["/login", "/login/", "/static", "/static/", "/favicon.ico"]
        )

    def _build_fred_router(self):
        """
        Proxy server-side para FRED API.
        El browser llama a /api/fred/{series_id} → FastAPI llama a FRED → devuelve JSON.
        Evita CORS completamente ya que el request sale del servidor, no del browser.
        """
        router = APIRouter()
        fred_api_key = self.fred_api_key

        @router.get("/fred/{series_id}")
        async def fred_proxy(series_id: str, start: str = "2010-01-01"):
            fred_url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id":         series_id,
                "api_key":           fred_api_key,
                "file_type":         "json",
                "sort_order":        "asc",
                "observation_start": start,
            }
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(fred_url, params=params)
                    resp.raise_for_status()
                    return JSONResponse(content=resp.json())
            except httpx.HTTPStatusError as e:
                return JSONResponse(
                    status_code=502,
                    content={"error": f"FRED error {e.response.status_code}", "observations": []}
                )
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e), "observations": []}
                )

        return router

    async def startup_event(self):
        await self.routing_dashboard.initialize()
        self.app.include_router(self.routing_dashboard.router)

    async def login_form(self, request: Request):
        return self.templates.TemplateResponse("login.html", {"request": request})

    async def main_dashboard(self, request: Request):
        return self.templates.TemplateResponse("main_dashboard.html", {"request": request})

    def display(self, port=8000):
        """Starts the main dashboard server with all integrated dashboards."""
        def run():
            self.logger.do_log(f"Starting Main Dashboard on port {port}...", MessageType.INFO)
            import uvicorn
            uvicorn.run(self.app, host="0.0.0.0", port=port)

        import threading
        threading.Thread(target=run, daemon=True).start()

    from fastapi import Request

    async def login(self, request: Request, username: str = Form(...), password: str = Form(...)):
        if self.user_manager.authenticate_user(username, password):
            token = self.signer.sign("session").decode()
            response = RedirectResponse(url="/", status_code=302)
            response.set_cookie(key="session", value=token, httponly=True, max_age=3600)
            return response
        else:
            error_message = "Invalid credentials: The username or password you entered is incorrect. Please check your input and try again."
            return self.templates.TemplateResponse("login.html", {"request": request, "error_message": error_message})

    async def logout(self, request: Request):
        response = RedirectResponse(url="/login", status_code=302)
        response.delete_cookie("session")
        return response