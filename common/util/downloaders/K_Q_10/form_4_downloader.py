import json
import os
import time
import requests
from common.enums.folders import Folders
from common.util.std_in_out.root_locator import RootLocator
from framework.common.logger.message_type import MessageType


class Form4Downloader:
    """
    Identifies and downloads structured XML Form 4 (Insider Trading) filings
    for a specific ticker from the SEC EDGAR system.
    """

    SEC_ARCHIVES = "https://www.sec.gov/Archives"
    SEC_SUBMISSIONS = "https://data.sec.gov/submissions"

    def __init__(self, logger, out_folder, job_id):
        self.logger = logger
        self.job_id = job_id
        self.out_folder = out_folder
        # SEC requires a descriptive User-Agent
        self.headers = {
            "User-Agent": "zzLotteryTicket research contact@yourmail.com"
        }

    def _get_cik_from_symbol(self, symbol):
        """
        Maps a stock ticker symbol to its unique 10-digit SEC Central Index Key (CIK).
        """
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for k, v in data.items():
            if v['ticker'].upper() == symbol.upper():
                return str(v['cik_str']).zfill(10)
        return None

    def download_insider_trades(self, symbol, limit=10):
        """
        Fetches the most recent N Form 4 filings for the given security.
        Downloads the XML version for direct data ingestion.
        """
        cik = self._get_cik_from_symbol(symbol)
        if not cik:
            self.logger.do_log(f"CIK not found for {symbol}", MessageType.ERROR, self.job_id)
            return None, []

        # Construct output directory using project standard locator
        out_dir = os.path.join(
            RootLocator.get_root(),
            Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
            self.out_folder,
            "form4",
            symbol.upper()
        )
        os.makedirs(out_dir, exist_ok=True)

        # Retrieve recent filing history metadata
        url = f"{self.SEC_SUBMISSIONS}/CIK{cik}.json"
        resp = requests.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        downloaded_paths = []

        for i, form in enumerate(recent.get("form", [])):
            if form == "4":
                acc = recent["accessionNumber"][i].replace("-", "")
                # Convert the primary HTML document name to the .xml data file name
                doc_name = recent["primaryDocument"][i]
                xml_name = doc_name.replace(".html", ".xml")

                # SEC Archive Path: Archives/edgar/data/{CIK_INT}/{ACC_STIPPED}/{FILE}
                xml_url = f"{self.SEC_ARCHIVES}/edgar/data/{int(cik)}/{acc}/{xml_name}"
                out_file = os.path.join(out_dir, f"{recent['filingDate'][i]}_{acc}.xml")

                if os.path.exists(out_file):
                    downloaded_paths.append(out_file)
                    continue

                self.logger.do_log(f"[Form 4] ⬇️ Downloading | {symbol} | {acc}", MessageType.INFO, self.job_id)

                try:
                    r = requests.get(xml_url, headers=self.headers, timeout=30)
                    if r.status_code == 200:
                        with open(out_file, "wb") as f:
                            f.write(r.content)
                        downloaded_paths.append(out_file)

                    # Store transaction metadata (who/what/when)
                    meta = {
                        "ticker": symbol,
                        "date": recent['filingDate'][i],
                        "accession": acc,
                        "url": xml_url
                    }
                    with open(out_file + ".meta.json", "w") as m:
                        json.dump(meta, m)

                except Exception as e:
                    self.logger.do_log(f"Failed to download {acc}: {e}", MessageType.WARNING, self.job_id)

                if len(downloaded_paths) >= limit:
                    break

                # Respect SEC rate limits (10 requests per second)
                time.sleep(0.1)

        return out_dir, downloaded_paths