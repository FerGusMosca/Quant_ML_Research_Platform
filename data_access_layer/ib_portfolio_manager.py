import pyodbc

from business_entities.portfolio_holding import PortfolioHolding


class IBPortfolioManager:
    """
    Data-access / logic layer for Interactive Brokers portfolio data.

    Connects to the AutPortfolio database via autportfolio_cs and retrieves
    holdings for a given IB account id.

    Instantiate with the connection string from config_settings:
        mgr = IBPortfolioManager(config_settings["autportfolio_cs"])
    """

    def __init__(self, connection_string: str):
        self._cs = connection_string

    # ── helpers ───────────────────────────────────────────────────────────────

    def _connect(self) -> pyodbc.Connection:
        return pyodbc.connect(self._cs)

    @staticmethod
    def _row_to_entity(row) -> PortfolioHolding:
        return PortfolioHolding(
            symbol          = row.symbol,
            name            = row.name,
            qty             = float(row.Qty),
            purchase_price  = float(row.Purchase_Price)  if row.Purchase_Price  is not None else None,
            purchase_amount = float(row.Purchase_Amount) if row.Purchase_Amount is not None else None,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def fetch_ib_account_holdings(self, ib_account_id: int) -> list[PortfolioHolding]:
        """
        Returns the portfolio holdings for the given IB account numeric id.

        Calls: dbo.get_portfolio_holdings @account_id = ?

        Raises:
            ValueError   — if no active holdings are found for the given account_id
            RuntimeError — on any database error
        """
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "EXEC dbo.get_portfolio_holdings @account_id = ?",
                    ib_account_id,
                )
                rows = cursor.fetchall()
        except pyodbc.Error as exc:
            raise RuntimeError(
                f"Database error while fetching IB holdings: {exc}"
            ) from exc

        if not rows:
            raise ValueError(
                f"No active holdings found for account_id {ib_account_id}. "
                "Verify the account_id key is correct and that active positions "
                "exist in AutPortfolio.dbo.account_positions."
            )

        return [self._row_to_entity(r) for r in rows]