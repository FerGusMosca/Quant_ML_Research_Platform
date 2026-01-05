import pyodbc

from common.dto.sec_security_dto import SecSecurityDTO
from framework.common.logger.message_type import MessageType


class ReportSecuritiesManager:
    """
    Data Access Layer for fetching securities linked to a specific report
    """
    def __init__(self, connection_string, logger):
        self.connection = pyodbc.connect(connection_string)
        self.logger = logger

    def get_report_securities(self, report_key: str):
        """
        Retrieve securities associated with a given report key, enriched with SEC_Securities attributes
        """
        securities = []
        with self.connection.cursor() as cursor:
            try:
                cursor.execute("EXEC GetReportSecurities ?", (report_key,))
                for row in cursor.fetchall():
                    dto = SecSecurityDTO(
                        cik=row[0],
                        ticker=row[1],
                        symbol=row[2],
                        name=row[3],
                        exchange=row[4],
                        category=row[5],
                        sic=row[6],
                        entityType=row[7]
                    )
                    securities.append(dto)

                self.logger.do_log(
                    f"get_report_securities: Retrieved {len(securities)} securities for report {report_key}",
                    MessageType.INFO
                )

            except Exception as e:
                self.logger.do_log(
                    f"get_report_securities: ❌ Failed for {report_key} - {str(e)}",
                    MessageType.ERROR
                )
        return securities

