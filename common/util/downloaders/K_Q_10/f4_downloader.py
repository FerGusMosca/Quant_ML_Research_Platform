from datetime import datetime, timedelta
import os
import random
import re
import time
from typing import Optional, List

import requests

from framework.common.logger.message_type import MessageType


class F4Downloader:
    """
    Utility class to download Form 4 filings from SEC EDGAR.
    """
    BASE_SEARCH_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

    def __init__(self, logger=None):
        self.logger = logger
        self.headers = {"User-Agent": "F4Downloader/1.0 (fer.mosca@example.com)"}

    def _resolve_f4_start_date(self, base_path: str, year: int) -> str:
        DATE_RE = re.compile(r".*_(\d{4}-\d{2}-\d{2})_4\.html")
        if not os.path.exists(base_path): return f"{year}-01-01"

        dates = []
        for fname in os.listdir(base_path):
            m = DATE_RE.match(fname)
            if m: dates.append(datetime.fromisoformat(m.group(1)))

        if not dates: return f"{year}-01-01"
        return (max(dates) + timedelta(days=1)).strftime("%Y-%m-%d")

    def download_f4_range(self, symbol: str, cik: str, year: int, output_dir: str, job_id: Optional[int] = None) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        downloaded = []

        start_date = self._resolve_f4_start_date(output_dir, year)
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(f"{year}-12-31")

        cik_padded = f"{int(cik):010d}"
        url = self.BASE_SEARCH_URL.format(cik=cik_padded)

        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            self._log(f"[F4] ❌ failed fetch {symbol}", e, job_id)
            return downloaded

        filings = data.get("filings", {}).get("recent", {})
        for form, fdate, acc, doc in zip(filings.get("form", []), filings.get("filingDate", []),
                                         filings.get("accessionNumber", []), filings.get("primaryDocument", [])):

            # Filtro estricto para Form 4
            if form != "4":
                continue

            filed_dt = datetime.fromisoformat(fdate)
            if not (start_dt <= filed_dt <= end_dt):
                continue

            acc_nodash = acc.replace("-", "")
            file_name = f"{symbol}_{fdate}_4.html"
            file_path = os.path.join(output_dir, file_name)

            if os.path.exists(file_path):
                continue

            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"

            try:
                time.sleep(0.2 + random.random() * 0.3)  # SEC rate limits are strict
                r = requests.get(filing_url, headers=self.headers, timeout=20)
                r.raise_for_status()

                with open(file_path, "wb") as f:
                    f.write(r.content)

                downloaded.append(file_path)
                self._log(f"[F4] DOWNLOADED | {symbol} | {fdate}", None, job_id)

            except Exception as e:
                self._log(f"[F4] ❌ download failed {fdate}", e, job_id)

        return downloaded

    def _log(self, msg: str, exc: Exception | None, job_id: Optional[int]):
        if self.logger:
            level = MessageType.ERROR if exc else MessageType.INFO
            self.logger.do_log(f"{msg} | {exc}" if exc else msg, level, job_id)