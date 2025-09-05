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
                        name=row[2],
                        exchange=row[3],
                        category=row[4],
                        sic=row[5],
                        entityType=row[6]
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

