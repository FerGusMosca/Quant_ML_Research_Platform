import os
import time
import requests

from common.enums.folders import Folders
from common.util.std_in_out.root_locator import RootLocator
from framework.common.logger.message_type import MessageType


class ThirteenFGraphDownloader:
    """
    Discover and download ALL 13F XML filings for a given quarter.
    """

    SEC_ARCHIVES = "https://www.sec.gov/Archives"
    SEC_INDEX = "https://www.sec.gov/Archives/edgar/full-index"

    def __init__(self, logger, out_folder, job_id):
        self.logger = logger
        self.job_id = job_id
        self.out_folder = out_folder
        self.headers = {
            "User-Agent": "zzLotteryTicket research contact@yourmail.com"
        }

    # ------------------------------------------------------------------
    # STEP 1 — Load ALL 13F filings from form.idx (no CIK guessing)
    # ------------------------------------------------------------------
    def _load_all_13f_filings(self, year, quarter):
        idx_url = f"{self.SEC_INDEX}/{year}/QTR{quarter}/form.idx"

        self.logger.do_log(
            f"[13F] ▶ Downloading index {idx_url}",
            MessageType.INFO,
            self.job_id
        )

        resp = requests.get(idx_url, headers=self.headers, timeout=30)
        resp.raise_for_status()

        filings = []
        for line in resp.text.splitlines():
            if not line.startswith("13F-HR"):
                continue

            form = line[0:12].strip()
            company = line[12:62].strip()
            cik = line[62:74].strip()
            date = line[74:86].strip()
            path = line[86:].strip()

            filings.append({
                "cik": cik,
                "company": company,
                "form": form,
                "date": date,
                "path": path
            })

        self.logger.do_log(
            f"[13F] ✔ Found {len(filings)} filings",
            MessageType.INFO,
            self.job_id
        )

        return filings

    # ------------------------------------------------------------------
    # STEP 2 — Download primary XML (information table)
    # ------------------------------------------------------------------
    def _download_filing(self, filing, out_dir):
        filing_base = filing["path"].replace(".txt", "")
        filing_dir_url = f"{self.SEC_ARCHIVES}/{filing_base}/"

        index_url = filing_dir_url + "index.json"
        r = requests.get(index_url, headers=self.headers, timeout=30)
        r.raise_for_status()

        files = r.json()["directory"]["item"]

        xml_files = [
            f["name"] for f in files
            if f["name"].endswith(".xml")
            and ("information" in f["name"].lower() or "infotable" in f["name"].lower())
        ]

        if not xml_files:
            return

        xml_name = xml_files[0]
        xml_url = filing_dir_url + xml_name

        out_file = os.path.join(
            out_dir,
            f"{filing['cik']}_{os.path.basename(filing_base)}.xml"
        )

        if os.path.exists(out_file):
            return

        xml_resp = requests.get(xml_url, headers=self.headers, timeout=30)
        xml_resp.raise_for_status()

        with open(out_file, "wb") as f:
            f.write(xml_resp.content)

        time.sleep(0.12)  # SEC rate limit

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def download(self, year, quarter):
        out_dir = os.path.join(
            RootLocator.get_root(),
            Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
            self.out_folder,
            "13f",
            f"{year}_Q{quarter}"
        )
        os.makedirs(out_dir, exist_ok=True)

        filings = self._load_all_13f_filings(year, quarter)

        for filing in filings:
            try:
                self._download_filing(filing, out_dir)
            except Exception as e:
                self.logger.do_log(
                    f"[13F] ⚠️ Failed {filing.get('cik')} | {str(e)}",
                    MessageType.WARNING,
                    self.job_id
                )

        self.logger.do_log(
            f"[13F] 🏁 Finished download | dir={out_dir}",
            MessageType.INFO,
            self.job_id
        )

        return out_dir
