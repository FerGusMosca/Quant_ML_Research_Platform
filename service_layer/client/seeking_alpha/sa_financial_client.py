import requests
from framework.common.logger.message_type import MessageType


class SAFinancialsClient:
    """
    Client responsible for fetching structured financial fundamentals
    from Seeking Alpha (SA).

    This class is intentionally isolated so the vendor can be swapped
    without touching report orchestration or ratio logic.
    """

    @staticmethod
    def fetch_fundamentals(symbol, logger, job_id=None):
        """
        Fetch core financial data needed to compute basic ratios.
        Currently designed for single-symbol exploratory usage.

        Returns a dict with raw numeric values or None if unavailable.
        """

        logger.do_log(
            f"[SA] Fetching fundamentals | symbol={symbol}",
            MessageType.INFO,
            job_id
        )

        try:
            # NOTE:
            # Endpoints are placeholders and must be aligned
            # with the actual SA API contract / subscription level.
            # Keep calls separated and explicit.

            income_stmt = SAFinancialsClient._fetch_income_statement(symbol, logger, job_id)
            balance_sheet = SAFinancialsClient._fetch_balance_sheet(symbol, logger, job_id)
            valuation = SAFinancialsClient._fetch_valuation(symbol, logger, job_id)

            data = {
                "gross_profit": income_stmt.get("gross_profit"),
                "revenue": income_stmt.get("revenue"),
                "net_income": income_stmt.get("net_income"),
                "total_assets": balance_sheet.get("total_assets"),
                "total_debt": balance_sheet.get("total_debt"),
                "equity": balance_sheet.get("equity"),
                "pe": valuation.get("pe"),
            }

            logger.do_log(
                f"[SA] Fundamentals fetched successfully | symbol={symbol}",
                MessageType.INFO,
                job_id
            )

            return data

        except Exception as e:
            logger.do_log(
                f"[SA] Failed fetching fundamentals | symbol={symbol} | error={e}",
                MessageType.ERROR,
                job_id
            )
            return {}

    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_income_statement(symbol, logger, job_id):
        """
        Fetch last available income statement (TTM or FY).
        """
        logger.do_log(
            f"[SA] Fetching income statement | symbol={symbol}",
            MessageType.INFO,
            job_id
        )

        try:
            # Placeholder
            return {
                "gross_profit": None,
                "revenue": None,
                "net_income": None,
            }

        except Exception as e:
            logger.do_log(
                f"[SA] Income statement fetch failed | symbol={symbol} | error={e}",
                MessageType.WARNING,
                job_id
            )
            return {}

    @staticmethod
    def _fetch_balance_sheet(symbol, logger, job_id):
        """
        Fetch last available balance sheet snapshot.
        """
        logger.do_log(
            f"[SA] Fetching balance sheet | symbol={symbol}",
            MessageType.INFO,
            job_id
        )

        try:
            # Placeholder
            return {
                "total_assets": None,
                "total_debt": None,
                "equity": None,
            }

        except Exception as e:
            logger.do_log(
                f"[SA] Balance sheet fetch failed | symbol={symbol} | error={e}",
                MessageType.WARNING,
                job_id
            )
            return {}

    @staticmethod
    def _fetch_valuation(symbol, logger, job_id):
        """
        Fetch valuation metrics (P/E, etc.).
        """
        logger.do_log(
            f"[SA] Fetching valuation metrics | symbol={symbol}",
            MessageType.INFO,
            job_id
        )

        try:
            # Placeholder
            return {
                "pe": None,
            }

        except Exception as e:
            logger.do_log(
                f"[SA] Valuation fetch failed | symbol={symbol} | error={e}",
                MessageType.WARNING,
                job_id
            )
            return {}
