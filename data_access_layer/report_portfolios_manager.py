import pyodbc

from framework.common.logger.message_type import MessageType


class ReportPortfoliosManager:
    """
    Data Access Layer for dbo.report_portfolios (machine_learning_research).

    This is the portfolio catalogue: the same list the Document Tagger offers
    and the same code that travels in the "portfolio" argument of the MCP
    report commands.

    Requires db/report_portfolios/01_report_portfolios_sps.sql to be applied.
    """

    def __init__(self, connection_string: str, logger):
        self.connection_string = connection_string
        self.logger = logger
        self._connection = None

    # ── Connection ────────────────────────────────────────────────────────────

    @property
    def connection(self):
        """Reconnects on demand: the screen sits idle and pyodbc drops the socket."""
        if self._connection is None:
            self._connection = pyodbc.connect(self.connection_string)
            return self._connection

        try:
            with self._connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchall()
        except Exception:
            self._connection = pyodbc.connect(self.connection_string)

        return self._connection

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def __rows_to_dicts__(cursor):
        columns = [c[0] for c in cursor.description]
        rows = []
        for row in cursor.fetchall():
            item = dict(zip(columns, row))
            for key, value in list(item.items()):
                if isinstance(value, str):
                    item[key] = value.strip()
                elif hasattr(value, "isoformat"):
                    item[key] = value.isoformat()
            rows.append(item)
        return rows

    # ── Reads ─────────────────────────────────────────────────────────────────

    def get_portfolios(self):
        """Every portfolio in the catalogue, with its code, name and description."""
        with self.connection.cursor() as cursor:
            cursor.execute("EXEC dbo.Get_ReportPortfolios")
            rows = self.__rows_to_dicts__(cursor)

        self.logger.do_log(f"[PORTFOLIOS] Retrieved {len(rows)} portfolios",
                           MessageType.INFO)
        return rows

    def get_portfolio_codes(self):
        """Just the codes, which is what the combos and the MCP arguments use."""
        return [row["portfolio_code"] for row in self.get_portfolios()
                if row.get("portfolio_code")]
