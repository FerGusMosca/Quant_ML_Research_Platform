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
        """
        Download all 10-Q filings for a given company (CIK) and year
        from SEC EDGAR and save them into output_dir.
        """
        if not cik:
            raise ValueError(f"[Q10Downloader] Missing CIK for {symbol}")

        cik_padded = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        headers = {
            "User-Agent": "Q10Downloader/1.0 (fer.mosca@example.com)",
            "Accept-Encoding": "gzip, deflate",
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        time.sleep(0.5 + random.random())

        filings = data.get("filings", {}).get("recent", {})
        accession_numbers = filings.get("accessionNumber", [])
        filing_dates = filings.get("filingDate", [])
        forms = filings.get("form", [])
        primary_docs = filings.get("primaryDocument", [])

        os.makedirs(output_dir, exist_ok=True)

        downloaded_files = []
        q_counter = 1

        for acc, fdate, form, doc in zip(accession_numbers, filing_dates, forms, primary_docs):
            if form == "10-Q" and fdate.startswith(str(year)):
                acc_nodash = acc.replace("-", "")
                target_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"

                file_name = f"{symbol}_{year}_Q{q_counter}_10-Q.html"
                file_path = os.path.join(output_dir, file_name)

                filing_resp = requests.get(target_url, headers=headers)
                filing_resp.raise_for_status()
                time.sleep(0.5 + random.random())

                with open(file_path, "wb") as f:
                    f.write(filing_resp.content)

                print(f"✅ Downloaded {file_name}")
                downloaded_files.append(file_path)
                q_counter += 1

        if not downloaded_files:
            raise FileNotFoundError(f"[Q10Downloader] No 10-Qs found for {symbol} ({cik}) in {year}")

        return downloaded_files
