import pyodbc

from common.dto.security_report_calendar import SecurityReportCalendar


class SecuritiesCalendarManager:
    """
    Handles persistence of filing calendar data into SQL Server using
    the stored procedure upsert_securities_reports_calendar.
    """

    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = pyodbc.connect(connection_string)

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

    def __reconnect_if_needed__(self):
        """The screen sits idle for long stretches and pyodbc drops the socket."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchall()
        except Exception:
            self.connection = pyodbc.connect(self.connection_string)

    # ── Reads ─────────────────────────────────────────────────────────────────

    def get_calendars_by_range(self, from_year: int, to_year: int):
        """
        Retrieve all calendar entries between two years (inclusive).
        """
        entries = {}
        with self.connection.cursor() as cursor:
            params = (from_year, to_year)
            cursor.execute("{CALL get_securities_reports_calendar_by_year (?,?)}", params)
            for row in cursor.fetchall():
                symbol = row.symbol.strip()
                entries[(symbol, row.fiscal_year)] = True
        return entries

    def get_calendar_rows(self, from_year: int, to_year: int, symbol: str = None):
        """
        Same stored procedure, full rows instead of a presence map. This is what
        the Reports Runner screen shows: which filing date each security actually
        has per fiscal year, which is the only way to translate a calendar
        quarter into the reports that landed inside it.
        """
        self.__reconnect_if_needed__()

        with self.connection.cursor() as cursor:
            cursor.execute("{CALL get_securities_reports_calendar_by_year (?,?)}",
                           (int(from_year), int(to_year)))
            rows = self.__rows_to_dicts__(cursor)

        if symbol:
            wanted = symbol.strip().upper()
            rows = [row for row in rows
                    if str(row.get("symbol", "")).upper() == wanted]

        rows.sort(key=lambda item: (str(item.get("symbol", "")),
                                    item.get("fiscal_year") or 0))
        return rows

    # ── Writes ────────────────────────────────────────────────────────────────

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
