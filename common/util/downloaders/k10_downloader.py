import os
import requests

class K10Downloader:
    """
    Utility class to handle K-10 downloads from SEC EDGAR
    """

    @staticmethod
    def download_k10(symbol, cik, year, output_dir):
        """
        Download the latest 10-K filing for a given company (CIK) and year
        from SEC EDGAR and save it into output_dir.
        """
        if not cik:
            raise ValueError(f"[K10Downloader] Missing CIK for {symbol}")

        # Build SEC submissions endpoint (JSON of all filings for this CIK)
        cik_padded = str(cik).zfill(10)  # SEC requires 10-digit CIK with leading zeros
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        headers = {"User-Agent": "YourAppName/1.0 (contact@example.com)"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()

        # Find the correct 10-K for the given year
        filings = data.get("filings", {}).get("recent", {})
        accession_numbers = filings.get("accessionNumber", [])
        filing_dates = filings.get("filingDate", [])
        forms = filings.get("form", [])
        primary_docs = filings.get("primaryDocument", [])

        target_url = None
        for acc, fdate, form, doc in zip(accession_numbers, filing_dates, forms, primary_docs):
            if form == "10-K" and fdate.startswith(str(year)):
                acc_nodash = acc.replace("-", "")
                target_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"
                break

        if not target_url:
            raise FileNotFoundError(f"[K10Downloader] No 10-K found for {symbol} ({cik}) in {year}")

        # Download the filing
        filing_resp = requests.get(target_url, headers=headers)
        filing_resp.raise_for_status()

        file_path = os.path.join(output_dir, f"{symbol}_{year}_10-K.html")
        with open(file_path, "wb") as f:
            f.write(filing_resp.content)

        return file_path
