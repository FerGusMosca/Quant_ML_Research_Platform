import pyodbc

from framework.common.logger.message_type import MessageType


class StockMonitorPortfolioManager:
    """
    Lee los activos de un portfolio del monitor, con los mismos procedimientos
    que usa la pantalla: primero busca el portfolio por nombre y despues trae
    sus activos.
    """

    def __init__(self, connection_string, logger=None):
        self.connection_string = connection_string
        self.logger = logger

    def _connect(self):
        return pyodbc.connect(self.connection_string)

    def _log(self, msg, msg_type=MessageType.INFO):
        if self.logger is not None:
            self.logger.do_log(msg, msg_type)
        else:
            print(msg)

    def get_portfolio_id(self, portfolio_name):
        conn = self._connect()
        cursor = conn.cursor()

        try:
            cursor.execute("EXEC dbo.sm_get_portfolios")

            wanted = str(portfolio_name).strip().upper()

            for row in cursor.fetchall():
                if str(row.name).strip().upper() == wanted:
                    return int(row.id)

            raise Exception(f"Portfolio '{portfolio_name}' not found")

        finally:
            cursor.close()
            conn.close()

    def get_symbols(self, portfolio_name):
        """
        Devuelve la lista de tickers del portfolio, en el orden en que los
        devuelve la base.
        """
        portfolio_id = self.get_portfolio_id(portfolio_name)

        conn = self._connect()
        cursor = conn.cursor()

        try:
            cursor.execute("EXEC dbo.sm_get_assets @portfolio_id=?", portfolio_id)

            symbols = [str(row.symbol).strip().upper() for row in cursor.fetchall()]

            self._log(f"[MONITOR] {len(symbols)} assets read from portfolio '{portfolio_name}'")

            return symbols

        finally:
            cursor.close()
            conn.close()
