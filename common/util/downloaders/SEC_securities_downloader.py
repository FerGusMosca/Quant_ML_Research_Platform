import requests


class SECSecuritiesDownloader:
    """
    Utility class for downloading the list of SEC securities from EDGAR.

    REPLACES the original file. What changed: the old method read
    company_tickers.json, which ONLY carries cik_str, ticker and title. That is
    why process_download_sec_securities called item.get("exchange"),
    item.get("sic") and item.get("entityType") and always got None -> those four
    columns stayed NULL in SEC_Securities.

    The initial load now uses company_tickers_exchange.json, which also carries
    exchange. sic, sicDescription and entityType are not in any bulk file: those
    are filled in by SECSecuritiesMetadataDownloader against the submissions
    endpoint.

    download_security_list_from_edgar() keeps the same signature so nothing that
    already calls it breaks.
    """

    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"

    @staticmethod
    def __headers__(user_agent=None):
        return {"User-Agent": user_agent or "Seeking Bias Research alien.zimzum@gmail.com"}

    @staticmethod
    def download_security_list_from_edgar(user_agent=None):
        """
        Full security list, exchange included.

        company_tickers_exchange.json comes in columnar format:
            {"fields": ["cik","name","ticker","exchange"], "data": [[...], [...]]}

        Returns a list of dicts with the same keys the old code used
        (cik_str, ticker, title) plus exchange, so process_download_sec_securities
        needs no change.
        """
        response = requests.get(
            SECSecuritiesDownloader.TICKERS_EXCHANGE_URL,
            headers=SECSecuritiesDownloader.__headers__(user_agent),
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        fields = payload.get("fields") or []
        rows = payload.get("data") or []

        if not fields or not rows:
            # Fall back to the legacy file if the SEC changes the format
            return SECSecuritiesDownloader.download_security_list_legacy(user_agent)

        index = {name: pos for pos, name in enumerate(fields)}

        securities = []
        for row in rows:
            def value(key):
                pos = index.get(key)
                return row[pos] if pos is not None and pos < len(row) else None

            securities.append({
                "cik_str": value("cik"),
                "ticker": value("ticker"),
                "title": value("name"),
                "exchange": value("exchange"),
                "category": None,
                "sic": None,
                "entityType": None,
            })

        return securities

    @staticmethod
    def download_security_list_legacy(user_agent=None):
        """Legacy behaviour: cik_str, ticker and title only."""
        response = requests.get(
            SECSecuritiesDownloader.TICKERS_URL,
            headers=SECSecuritiesDownloader.__headers__(user_agent),
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        return [value for key, value in data.items()]
