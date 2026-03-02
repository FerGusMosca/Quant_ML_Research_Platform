import pyodbc
from business_entities.account_data import AccountData


class AccountDataManager:
    """
    Data Access Layer for dbo.account_data (EAV store).

    Instantiate with the connection string from config_settings:
        mgr = AccountDataManager(config_settings["fund_mgmt_dashboard_cs"])
    """

    def __init__(self, connection_string: str):
        self._cs = connection_string

    # ── helpers ──────────────────────────────────────────────────────────────

    def _connect(self) -> pyodbc.Connection:
        return pyodbc.connect(self._cs)

    @staticmethod
    def _row_to_entity(row) -> AccountData:
        return AccountData(

            data_id        = row.data_id,
            account_id     = row.account_id,
            data_key       = row.data_key,
            data_value     = row.data_value or "",
            account_number = row.account_number,
            account_name   = row.account_name,
            broker         = row.broker,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def get_all(self) -> list[AccountData]:
        """
        Returns every key/value entry across all accounts.
        Calls: dbo.get_all_account_data
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("EXEC dbo.get_all_account_data")
            return [self._row_to_entity(r) for r in cursor.fetchall()]

    def get_by_account_id(self, account_id: int) -> list[AccountData]:
        """
        Returns all key/value entries for a single account.
        Calls: dbo.get_account_data
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("EXEC dbo.get_account_data @account_id = ?", account_id)
            return [self._row_to_entity(r) for r in cursor.fetchall()]

    def persist(self, entity: AccountData) -> None:
        """
        Upserts a single key/value entry (insert or update by account_id + data_key).
        Calls: dbo.persist_account_data
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "EXEC dbo.persist_account_data "
                "@account_id = ?, @data_key = ?, @data_value = ?",
                entity.account_id,
                entity.data_key,
                entity.data_value,
            )
            conn.commit()

    def delete(self, data_id: int) -> None:
        """
        Deletes a single key/value entry by its PK.
        Calls: dbo.delete_account_data
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("EXEC dbo.delete_account_data @data_id = ?", data_id)
            conn.commit()

    def delete_all_for_account(self, account_id: int) -> None:
        """
        Deletes ALL entries for an account.
        Calls: dbo.delete_all_account_data
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "EXEC dbo.delete_all_account_data @account_id = ?", account_id
            )
            conn.commit()