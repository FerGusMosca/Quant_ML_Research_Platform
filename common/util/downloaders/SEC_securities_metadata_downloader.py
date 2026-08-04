import time

import requests

from common.dto.sec_security_metadata_dto import SecSecurityMetadataDTO
from common.util.classifiers.sic_sector_classifier import SICSectorClassifier


class SECSecuritiesMetadataDownloader:
    """
    Downloads per-security metadata from the SEC submissions endpoint:

        https://data.sec.gov/submissions/CIK{cik:010d}.json

    It returns sic, sicDescription, exchanges[], entityType, fiscalYearEnd and
    stateOfIncorporation. The original downloader (SECSecuritiesDownloader) reads
    company_tickers.json, which ONLY carries cik_str/ticker/title: that is why
    exchange, category, sic and entity_type ended up NULL in SEC_Securities.

    The SEC allows 10 requests per second and requires an identifying User-Agent.
    Default is 7 rps to keep some headroom.
    """

    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

    def __init__(self, user_agent, requests_per_second=7.0, max_retries=4, timeout=20):
        if not user_agent or "example.com" in user_agent:
            raise Exception("SECSecuritiesMetadataDownloader: a real identifying "
                            "User-Agent is required (name + email).")

        self.headers = {
            "User-Agent": user_agent,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }
        self.min_interval = 1.0 / max(float(requests_per_second), 0.5)
        self.max_retries = max_retries
        self.timeout = timeout
        self.last_call = 0.0
        self.session = requests.Session()

    # ── HTTP ──────────────────────────────────────────────────────────────────

    def __throttle__(self):
        wait = self.min_interval - (time.monotonic() - self.last_call)
        if wait > 0:
            time.sleep(wait)
        self.last_call = time.monotonic()

    def download_submissions(self, cik):
        """
        Raw submissions JSON for a given CIK.
        Retries with growing backoff on 429/5xx/timeouts.
        Raises FileNotFoundError when the CIK does not exist (404).
        """
        url = self.SUBMISSIONS_URL.format(cik=int(cik))
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            self.__throttle__()
            try:
                response = self.session.get(url, headers=self.headers, timeout=self.timeout)

                if response.status_code == 404:
                    raise FileNotFoundError(f"CIK {cik} has no submissions (404)")

                if response.status_code in (403, 429, 500, 502, 503, 504):
                    last_error = f"HTTP {response.status_code}"
                    time.sleep(min(2 ** attempt, 30))
                    continue

                response.raise_for_status()
                return response.json()

            except FileNotFoundError:
                raise
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                time.sleep(min(2 ** attempt, 30))

        raise Exception(f"CIK {cik}: retries exhausted ({last_error})")

    # ── Parsing ───────────────────────────────────────────────────────────────

    @staticmethod
    def build_metadata_dto(cik, json_data, symbol=None):
        """Turns the submissions JSON into a classified SecSecurityMetadataDTO."""
        sic = (json_data.get("sic") or "").strip() or None
        sic_description = (json_data.get("sicDescription") or "").strip() or None

        exchanges = json_data.get("exchanges") or []
        if isinstance(exchanges, str):
            exchanges = [exchanges]
        exchange = ",".join([str(x).strip() for x in exchanges if str(x).strip()]) or None

        entity_type = (json_data.get("entityType") or "").strip() or None
        fiscal_year_end = (json_data.get("fiscalYearEnd") or "").strip() or None
        state_of_inc = (json_data.get("stateOfIncorporation") or "").strip() or None

        sector_code, sector_name, industry_code, industry_name = SICSectorClassifier.classify(sic)

        return SecSecurityMetadataDTO(
            cik=cik,
            symbol=symbol,
            sic=sic,
            sic_description=sic_description,
            exchange=exchange,
            entity_type=entity_type,
            fiscal_year_end=fiscal_year_end,
            state_of_incorporation=state_of_inc,
            sector_code=sector_code,
            sector_name=sector_name,
            industry_code=industry_code,
            industry_name=industry_name,
        )

    def download_metadata(self, cik, symbol=None):
        """Downloads and parses in a single step."""
        json_data = self.download_submissions(cik)
        return self.build_metadata_dto(cik, json_data, symbol)
