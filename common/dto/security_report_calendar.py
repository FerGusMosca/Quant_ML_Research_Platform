# DTO for securities filing calendar entries

class SecurityReportCalendar:
    """
    Represents SEC filing dates for a given symbol and fiscal year.
    Used to persist data into securities_reports_calendar table.
    """

    def __init__(self, cik, symbol, fiscal_year,
                 q1=None, q2=None, q3=None, k10=None):
        self.cik = cik
        self.symbol = symbol
        self.fiscal_year = fiscal_year
        self.q1_filing_date = q1
        self.q2_filing_date = q2
        self.q3_filing_date = q3
        self.k10_filing_date = k10
