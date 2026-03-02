from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from business_entities.account import Account
from business_entities.account_data import AccountData
from data_access_layer.account_data_manager import AccountDataManager
from data_access_layer.account_manager import AccountManager



class AccountController:
    def __init__(self, account_manager: AccountManager, account_data_manager: AccountDataManager):
        self.router = APIRouter()
        self.templates = Jinja2Templates(directory="templates")
        self.account_manager = account_manager
        self.account_data_manager = account_data_manager

        # Account CRUD
        self.router.get("/accounts", response_class=HTMLResponse)(self.account_list_view)
        self.router.post("/accounts/save")(self.save_account)
        self.router.post("/accounts/delete")(self.delete_account)

        # Account data (EAV) — called inline from the accounts page
        self.router.get("/accounts/{account_id}/data", response_class=HTMLResponse)(self.get_account_data)
        self.router.post("/accounts/data/save")(self.save_account_data)
        self.router.post("/accounts/data/delete")(self.delete_account_data)

    # ── Account views ─────────────────────────────────────────────────────────

    async def account_list_view(self, request: Request):
        """Displays all accounts and a form to insert/update."""
        accounts = self.account_manager.get_all_accounts()
        return self.templates.TemplateResponse("manage_accounts.html", {
            "request":  request,
            "accounts": accounts,
        })

    async def save_account(self,
                           account_number: str = Form(...),
                           account_name:   str = Form(...),
                           broker:         str = Form(...)):
        account = Account(account_number=account_number,
                          account_name=account_name,
                          broker=broker)
        self.account_manager.persist_account(account)
        return JSONResponse({"ok": True})

    async def delete_account(self, account_number: str = Form(...)):
        self.account_manager.delete_account(account_number)
        return JSONResponse({"ok": True})

    # ── Account data (EAV) ────────────────────────────────────────────────────

    async def get_account_data(self, account_id: int):
        """
        Returns the key-value entries for one account as a JSON list.
        Called by the accordion JS when a row is expanded for the first time.
        """
        entries = self.account_data_manager.get_by_account_id(account_id)
        return JSONResponse([
            {"data_id": e.data_id, "data_key": e.data_key, "data_value": e.data_value}
            for e in entries
        ])

    async def save_account_data(self,
                                account_id: int = Form(...),
                                data_key:   str = Form(...),
                                data_value: str = Form(...)):
        if not data_key.strip():
            return JSONResponse({"ok": False, "error": "data_key cannot be empty"}, status_code=400)

        entity = AccountData(
            account_id = account_id,
            data_key   = data_key.strip(),
            data_value = data_value.strip(),
        )
        self.account_data_manager.persist(entity)

        # Fetch the saved row back so we can return its data_id
        entries = self.account_data_manager.get_by_account_id(account_id)
        saved   = next((e for e in entries if e.data_key == entity.data_key), None)
        return JSONResponse({
            "ok":        True,
            "data_id":   saved.data_id if saved else None,
            "data_key":  entity.data_key,
            "data_value": entity.data_value,
        })

    async def delete_account_data(self, data_id: int = Form(...)):
        self.account_data_manager.delete(data_id)
        return JSONResponse({"ok": True, "data_id": data_id})