import os
import re
import time
import random
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from framework.common.logger.message_type import MessageType


class K8Downloader:
    """
    Utility class to download Form 8-K filings from SEC EDGAR by date range.
    Designed for market-moving event ingestion (M&A, guidance, layoffs, financing).
    """

    BASE_SEARCH_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

    def __init__(self, logger=None):
        self.logger = logger
        self.headers = {
            "User-Agent": "K8Downloader/1.0 (fer.mosca@example.com)",
            "Accept-Encoding": "gzip, deflate",
        }

    def _resolve_k8_start_date(self, base_path: str, year: int) -> str:

        DATE_RE = re.compile(r".*_(\d{4}-\d{2}-\d{2})_8-K\.html")

        if not os.path.exists(base_path):
            return f"{year}-01-01"

        dates = []
        for fname in os.listdir(base_path):
            m = DATE_RE.match(fname)
            if m:
                try:
                    dates.append(datetime.fromisoformat(m.group(1)))
                except ValueError:
                    pass

        if not dates:
            return f"{year}-01-01"

        last_date = max(dates) + timedelta(days=1)
        return last_date.strftime("%Y-%m-%d")

    def download_k8_range(
        self,
        symbol: str,
        cik: str,
        year: int,
        output_dir: str,
        job_id: Optional[int] = None,
    ) -> List[str]:
        """
        Download all 8-K filings for a company between start_date and end_date (YYYY-MM-DD).
        Returns list of downloaded file paths.
        """

        os.makedirs(output_dir, exist_ok=True)
        downloaded = []

        start_date=self._resolve_k8_start_date(output_dir,year)
        start_dt = datetime.fromisoformat(start_date)
        end_date = f"{year}-12-31"
        end_dt = datetime.fromisoformat(end_date)

        cik_padded = f"{int(cik):010d}"
        url = self.BASE_SEARCH_URL.format(cik=cik_padded)

        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            self._log(f"[K8] ❌ failed to fetch submissions for {symbol}", e, job_id)
            return downloaded

        time.sleep(0.5 + random.random())

        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accession_numbers = filings.get("accessionNumber", [])
        primary_docs = filings.get("primaryDocument", [])

        for form, fdate, acc, doc in zip(forms, dates, accession_numbers, primary_docs):
            if form != "8-K":
                continue

            filed_dt = datetime.fromisoformat(fdate)
            if not (start_dt <= filed_dt <= end_dt):
                continue

            acc_nodash = acc.replace("-", "")
            file_name = f"{symbol}_{fdate}_8-K.html"
            file_path = os.path.join(output_dir, file_name)

            # ⚠️ Skip if already downloaded
            if os.path.exists(file_path):
                self._log(f"[K8] SKIP exists | {file_name}", None, job_id)
                continue

            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{acc_nodash}/{doc}"
            )

            try:
                r = requests.get(filing_url, headers=self.headers, timeout=20)
                if r.status_code == 404:
                    self._log(f"[K8] ⚠️ 404 | {filing_url}", None, job_id)
                    continue

                r.raise_for_status()

                with open(file_path, "wb") as f:
                    f.write(r.content)

                downloaded.append(file_path)

                self._log(
                    f"[K8] DOWNLOADED | {symbol} | {fdate} | {doc}",
                    None,
                    job_id,
                )

                time.sleep(0.5 + random.random())

            except Exception as e:
                self._log(f"[K8] ❌ download failed | {symbol} | {fdate}", e, job_id)

        return downloaded

    def _log(self, msg: str, exc: Exception | None, job_id: Optional[int]):
        if self.logger:
            if exc:
                self.logger.do_log(msg + f" | {exc}", MessageType.ERROR, job_id)
            else:
                self.logger.do_log(msg, MessageType.INFO, job_id)
