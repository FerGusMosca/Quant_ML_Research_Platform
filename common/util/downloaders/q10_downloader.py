import os
import random
import time
import requests

class Q10Downloader:
    """
    Utility class to handle 10-Q downloads from SEC EDGAR
    """

    @staticmethod
    def download_q10s(symbol, cik, year, output_dir):
        headers = {
            "User-Agent": "Q10Downloader/1.0 (fer.mosca@example.com)",
            "Accept-Encoding": "gzip, deflate",
        }

        if not cik:
            raise ValueError(f"[Q10Downloader] Missing CIK for {symbol}")

        os.makedirs(output_dir, exist_ok=True)
        downloaded_files = []
        existing_all = True
        q_counter = 1

        url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 404:
                return "NOT_FOUND"
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.RequestException:
            return "NOT_FOUND"

        filings = data.get("filings", {}).get("recent", {})
        for acc, fdate, form, doc in zip(
                filings.get("accessionNumber", []),
                filings.get("filingDate", []),
                filings.get("form", []),
                filings.get("primaryDocument", []),
        ):
            if form == "10-Q" and fdate.startswith(str(year)):
                acc_nodash = acc.replace("-", "")
                target_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"
                file_name = f"{symbol}_{year}_Q{q_counter}_10-Q.html"
                file_path = os.path.join(output_dir, file_name)

                # ⚠️ Skip existing files
                if os.path.exists(file_path):
                    q_counter += 1
                    continue

                resp = requests.get(target_url, headers=headers, timeout=15)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                time.sleep(0.5 + random.random())

                with open(file_path, "wb") as f:
                    f.write(resp.content)

                downloaded_files.append(file_path)
                existing_all = False
                q_counter += 1

        if not downloaded_files and existing_all:
            return "EXISTS"
        if not downloaded_files and not existing_all:
            return "NOT_FOUND"

        return downloaded_files

