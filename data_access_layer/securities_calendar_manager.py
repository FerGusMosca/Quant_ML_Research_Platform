import pyodbc

from common.dto.security_report_calendar import SecurityReportCalendar


class SecuritiesCalendarManager:
    """
    Handles persistence of filing calendar data into SQL Server using
    the stored procedure upsert_securities_reports_calendar.
    """

    def __init__(self, connection_string):
        self.connection = pyodbc.connect(connection_string)

    def upsert_calendar_entry(self, entry: SecurityReportCalendar):
        """
        Insert or update a single security report calendar record.
        """
        with self.connection.cursor() as cursor:
            params = (
                entry.cik,
                entry.symbol,
                entry.fiscal_year,
                entry.q1_filing_date,
                entry.q2_filing_date,
                entry.q3_filing_date,
                entry.k10_filing_date
            )
            cursor.execute("{CALL upsert_securities_reports_calendar (?,?,?,?,?,?,?)}", params)
            self.connection.commit()
