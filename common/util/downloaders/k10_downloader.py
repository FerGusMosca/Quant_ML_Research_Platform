import os
import random
import time
import requests
from bs4 import BeautifulSoup


class K10Downloader:
    """
    Utility class to handle 10-K downloads from SEC EDGAR
    """

    @staticmethod
    def download_k10(symbol, cik, year, output_dir):
        """
        Download the latest 10-K filing (HTML + XBRL) for a given company (CIK) and year
        from SEC EDGAR and save it into output_dir.
        """
        if not cik:
            raise ValueError(f"[K10Downloader] Missing CIK for {symbol}")

        os.makedirs(output_dir, exist_ok=True)

        # --- Submissions JSON (always data.sec.gov) ---
        cik_padded = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        headers = {
            "User-Agent": "K10Downloader/1.0 (fer.mosca@example.com)",
            "Accept-Encoding": "gzip, deflate",
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        time.sleep(0.5 + random.random())

        # --- Find correct 10-K for given year ---
        filings = data.get("filings", {}).get("recent", {})
        accession_numbers = filings.get("accessionNumber", [])
        filing_dates = filings.get("filingDate", [])
        forms = filings.get("form", [])
        primary_docs = filings.get("primaryDocument", [])

        target_url, acc_nodash = None, None
        for acc, fdate, form, doc in zip(accession_numbers, filing_dates, forms, primary_docs):
            if form == "10-K" and fdate.startswith(str(year)):
                acc_nodash = acc.replace("-", "")
                target_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"
                break

        if not target_url:
            raise FileNotFoundError(f"[K10Downloader] No 10-K found for {symbol} ({cik}) in {year}")

        # --- Download HTML filing ---
        filing_resp = requests.get(target_url, headers=headers)
        filing_resp.raise_for_status()
        time.sleep(0.5 + random.random())
        file_path_html = os.path.join(output_dir, f"{symbol}_{year}_10-K.html")
        with open(file_path_html, "wb") as f:
            f.write(filing_resp.content)

        # --- Download XBRL files ---
        xbrl_files = []
        index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/"
        index_resp = requests.get(index_url, headers=headers)
        index_resp.raise_for_status()
        time.sleep(0.5 + random.random())

        soup = BeautifulSoup(index_resp.text, "html.parser")
        for link in soup.find_all("a"):
            href = link.get("href", "")
            if any(href.lower().endswith(ext) for ext in [".xml", ".xsd"]):
                file_url = f"https://www.sec.gov{href}" if href.startswith("/") else f"{index_url}{href}"
                fname = os.path.join(output_dir, os.path.basename(href))
                r = requests.get(file_url, headers=headers)
                if r.status_code == 200:
                    with open(fname, "wb") as f:
                        f.write(r.content)
                    xbrl_files.append(fname)
                time.sleep(0.2 + random.random())

        return {"html": file_path_html, "xbrl": xbrl_files}
