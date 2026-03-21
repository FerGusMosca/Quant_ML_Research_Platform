# portfolio_view_controller.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from data_access_layer.account_manager import AccountManager
from data_access_layer.account_data_manager import AccountDataManager
from data_access_layer.ib_portfolio_manager import IBPortfolioManager
from data_access_layer.instruction_manager import InstructionManager, SyncBroker


class PortfolioViewController:
    def __init__(
        self,
        account_manager:      AccountManager,
        account_data_manager: AccountDataManager,
        ib_portfolio_manager: IBPortfolioManager,
        instruction_manager:  InstructionManager,
    ):
        self.router               = APIRouter()
        self.templates            = Jinja2Templates(directory="templates")
        self.account_manager      = account_manager
        self.account_data_manager = account_data_manager
        self.ib_portfolio_manager = ib_portfolio_manager
        self.instruction_manager  = instruction_manager

        self.router.get("/portfolio_views",                           response_class=HTMLResponse)(self.portfolio_view)
        self.router.get("/portfolio_views/{account_id}/holdings",     response_class=JSONResponse)(self.get_holdings)
        self.router.post("/portfolio_views/{account_id}/sync",        response_class=JSONResponse)(self.sync_portfolio)

    # ── Views ──────────────────────────────────────────────────────────────────

    async def portfolio_view(self, request: Request):
        accounts = self.account_manager.get_all_accounts()
        return self.templates.TemplateResponse("portfolio_views.html", {
            "request":  request,
            "accounts": accounts,
        })

    # ── API ────────────────────────────────────────────────────────────────────

    async def get_holdings(self, account_id: int):
        accounts = self.account_manager.get_all_accounts()
        account  = next((a for a in accounts if a.id == account_id), None)

        if not account:
            return JSONResponse({"ok": False, "error": "Account not found."}, status_code=404)

        broker = account.broker

        if broker == "IB_PROD":
            entries   = self.account_data_manager.get_by_account_id(account_id)
            id_entry  = next((e for e in entries if e.data_key == "account_id"), None)

            if not id_entry or not id_entry.data_value:
                return JSONResponse({"ok": False,
                    "error": "No account_id key found. Please add it in Manage Accounts."},
                    status_code=422)

            try:
                ib_account_id = int(id_entry.data_value)
            except ValueError:
                return JSONResponse({"ok": False,
                    "error": f"account_id value '{id_entry.data_value}' is not a valid integer."},
                    status_code=422)

            try:
                holdings = self.ib_portfolio_manager.fetch_ib_account_holdings(ib_account_id)
                return JSONResponse({"ok": True, "broker": broker,
                                     "holdings": [h.to_dict() for h in holdings]})
            except ValueError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
            except RuntimeError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        return JSONResponse({"ok": False, "broker": broker,
                             "error": f"Broker '{broker}' is not implemented yet.",
                             "not_implemented": True})

    async def sync_portfolio(self, account_id: int):
        """
        Inserts a SYNC_POS instruction for the account.
        Only available for IB_PROD and IB_DEV.
        """
        accounts = self.account_manager.get_all_accounts()
        account  = next((a for a in accounts if a.id == account_id), None)

        if not account:
            return JSONResponse({"ok": False, "error": "Account not found."}, status_code=404)

        if not SyncBroker.is_supported(account.broker):
            return JSONResponse({"ok": False,
                "error": f"Sync not supported for broker '{account.broker}'."}, status_code=422)

        try:
            entries = self.account_data_manager.get_by_account_id(account_id)
            id_entry = next((e for e in entries if e.data_key == "account_id"), None)

            if not id_entry or not id_entry.data_value:
                return JSONResponse({"ok": False,
                                     "error": "No account_id key found."}, status_code=422)

            ib_account_id = int(id_entry.data_value)
            instruction_id = self.instruction_manager.create_sync_instruction(ib_account_id)
            return JSONResponse({"ok": True, "instruction_id": instruction_id,
                                 "message": f"Sync instruction created (id={instruction_id})"})
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)