import os
import json
import time
import random
import requests
from datetime import datetime

from common.dto.security_report_calendar import SecurityReportCalendar
from common.enums.folders import Folders

class SecuritiesCalendarDownloader:
    """
    Downloads filing calendars from the SEC for each symbol.
    It retrieves all 10-Q and 10-K filings and maps them to Q1, Q2, Q3, and K10.
    """

    BASE_URL = "https://data.sec.gov/submissions/CIK{}.json"
    HEADERS = {
        "User-Agent": "SeekingBiasResearchBot/1.0 contact@seekingbias.com",
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov"
    }

    @staticmethod
    def download(symbol, cik, year, pause=1.0):
        """
        Download SEC filing dates for a single company by CIK.
        """
        url = SecuritiesCalendarDownloader.BASE_URL.format(str(cik).zfill(10))
        response = requests.get(url, headers=SecuritiesCalendarDownloader.HEADERS, timeout=20)
        if response.status_code != 200:
            raise RuntimeError(f"[SecuritiesCalendarDownloader] Failed for {symbol} ({cik}), HTTP {response.status_code}")

        data = response.json()
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        filing_dates = filings.get("filingDate", [])

        q1, q2, q3, k10 = None, None, None, None

        for form, fdate in zip(forms, filing_dates):
            if not fdate or not form:
                continue
            if "10-Q" in form and not q1:
                q1 = fdate
            elif "10-Q" in form and q1 and not q2:
                q2 = fdate
            elif "10-Q" in form and q2 and not q3:
                q3 = fdate
            elif "10-K" in form and not k10:
                k10 = fdate

        # Save JSON snapshot for auditing
        output_dir = os.path.join(Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value, "SecuritiesCalendar", str(year))
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{symbol}_{year}_calendar.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"symbol": symbol, "cik": cik, "year": year,
                       "Q1": q1, "Q2": q2, "Q3": q3, "K10": k10}, f, indent=2)

        time.sleep(pause + random.random())
        return SecurityReportCalendar(cik, symbol, year, q1, q2, q3, k10)
