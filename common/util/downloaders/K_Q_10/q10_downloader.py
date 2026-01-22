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

        # Determine which quarters are missing locally
        missing_qs = []
        for q in range(1, 5):
            file_path = os.path.join(output_dir, f"{symbol}_{year}_Q{q}_10-Q.html")
            if not os.path.exists(file_path):
                missing_qs.append(q)

        if not missing_qs:
            return "EXISTS"

        # Fetch SEC submissions
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

        # Map real filings by quarter
        filings_by_q = {}
        for acc, fdate, form, doc in zip(
                filings.get("accessionNumber", []),
                filings.get("filingDate", []),
                filings.get("form", []),
                filings.get("primaryDocument", []),
        ):
            if form != "10-Q" or not fdate.startswith(str(year)):
                continue

            month = int(fdate.split("-")[1])
            quarter = (month - 1) // 3 + 1
            filings_by_q[quarter] = (acc, doc)

        downloaded = []
        found_any = False

        for q in missing_qs:
            filing = filings_by_q.get(q)
            if not filing:
                continue

            found_any = True
            acc, doc = filing
            acc_nodash = acc.replace("-", "")
            target_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"

            file_path = os.path.join(output_dir, f"{symbol}_{year}_Q{q}_10-Q.html")

            resp = requests.get(target_url, headers=headers, timeout=15)
            if resp.status_code == 404:
                continue

            resp.raise_for_status()
            time.sleep(0.5 + random.random())

            with open(file_path, "wb") as f:
                f.write(resp.content)

            downloaded.append(file_path)

        if downloaded:
            return "FOUND"

        if found_any:
            return "EXISTS"

        return "NOT_FOUND"


