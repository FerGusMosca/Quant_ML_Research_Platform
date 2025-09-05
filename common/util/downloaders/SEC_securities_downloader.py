import requests

class SECSecuritiesDownloader:
    """
    Utility class for downloading the list of SEC securities from EDGAR or alternative sources
    """

    @staticmethod
    def download_security_list_from_edgar():
        """
        Download the JSON file containing the full list of securities from SEC EDGAR.
        Returns a list of dictionaries with the securities information.
        """
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {"User-Agent": "YourAppName/1.0 (contact@example.com)"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()

        # SEC delivers a dict with numeric keys, so convert it to a list
        return [value for key, value in data.items()]
