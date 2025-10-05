import pyodbc

from common.dto.sec_security_dto import SecSecurityDTO
from framework.common.logger.message_type import MessageType


class PortfolioSecuritiesManager:
    """
    Data Access Layer for fetching securities linked to a specific portfolio
    """
    def __init__(self, connection_string, logger):
        self.connection = pyodbc.connect(connection_string)
        self.logger = logger

    def get_portfolio_securities(self, portfolio_code: str):
        """
        Retrieve securities associated with a given portfolio code,
        enriched with SEC_Securities attributes.
        """
        securities = []
        with self.connection.cursor() as cursor:
            try:
                cursor.execute("EXEC GetPortfolioSecurities ?", (portfolio_code,))
                for row in cursor.fetchall():
                    dto = SecSecurityDTO(
                        cik=row[0],
                        ticker=row[1],
                        name=row[2],
                        exchange=row[3],
                        category=row[4],
                        sic=row[5],
                        entityType=row[6]
                    )
                    securities.append(dto)

                self.logger.do_log(
                    f"get_portfolio_securities: Retrieved {len(securities)} securities for portfolio {portfolio_code}",
                    MessageType.INFO
                )

            except Exception as e:
                self.logger.do_log(
                    f"get_portfolio_securities: ❌ Failed for portfolio {portfolio_code} - {str(e)}",
                    MessageType.ERROR
                )
        return securities
