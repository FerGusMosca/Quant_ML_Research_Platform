import pyodbc

from framework.common.logger.message_type import MessageType


class SECSecuritiesManager:
    """
    Data Access Layer for SEC Securities
    """
    def __init__(self, connection_string, logger):
        self.connection = pyodbc.connect(connection_string)
        self.logger = logger

    def persist_bulk(self, sec_dtos: list):
        """
        Persist a list of SEC Securities using stored procedure Persist_SECSecurity
        """
        if not sec_dtos:
            self.logger.do_log("persist_sec_securities: empty DTO list, nothing to persist.", MessageType.WARNING)
            return

        with self.connection.cursor() as cursor:
            for sec_dto in sec_dtos:
                try:
                    cursor.execute("""
                        EXEC Persist_SECSecurity ?, ?, ?, ?, ?, ?, ?
                    """, (
                        sec_dto.cik,
                        sec_dto.ticker,
                        sec_dto.name,
                        sec_dto.exchange,
                        sec_dto.category,
                        sec_dto.sic,
                        sec_dto.entityType
                    ))
                    self.logger.do_log(f"Successfully logged sec {sec_dto.ticker} (Name ={sec_dto.name} Exch={sec_dto.exchange} category={sec_dto.category} )....",MessageType.INFO)
                except Exception as e:
                    self.logger.do_log(f"persist_sec_securities: failed to persist {sec_dto.ticker} ({sec_dto.cik}) - {str(e)}",
                                       MessageType.ERROR)
        self.connection.commit()
        self.logger.do_log(f"persist_sec_securities: successfully persisted {len(sec_dtos)} securities.",
                           MessageType.INFO)
