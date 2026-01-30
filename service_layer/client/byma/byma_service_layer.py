import requests
import os
import json
import time
from datetime import datetime
from common.dto.byma_rate_dto import BYMARateDTO


class BYMAServiceLayer:
    """
    Robust client for BYMA Market Data API.
    Uses local cache, retry logic and date filters.
    """

    BASE_URL = "https://api.byma.com.ar"
    CACHE_DIR = "./cache/byma"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def _cache_path(self, name: str):
        safe_name = name.replace("/", "_").replace(" ", "_").lower()
        return os.path.join(self.CACHE_DIR, f"{safe_name}.json")

    def _get_with_cache(self, endpoint: str, name: str, retries=3, backoff=2):
        """
        Fetches endpoint with local caching, retry, timeout and logging.
        """
        cache_file = self._cache_path(endpoint)
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        url = f"{self.BASE_URL}{endpoint}"
        print(f"🔍 Fetching BYMA endpoint: {url}")

        for attempt in range(1, retries + 1):
            try:
                resp = self.session.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    return data
                else:
                    print(f"⚠️ Attempt {attempt}: {resp.status_code} {resp.text[:200]}")
            except requests.exceptions.Timeout:
                print(f"⏱️ Timeout (attempt {attempt}) for {url}")
            except requests.exceptions.RequestException as e:
                print(f"❌ Network error (attempt {attempt}): {str(e)}")

            sleep_time = backoff ** attempt
            print(f"   Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)

        print(f"❌ Failed to fetch data from {url} after {retries} attempts.")
        return []

    def get_interest_rates(self, d_from=None, d_to=None):
        """
        Retrieves short-term interest rates (cauciones) from BYMA.
        Only fetches the last available term (1-day term).
        """
        endpoint = "/market-data/v1/rates/cauciones?term=1"
        name = "Tasa Cauciones BYMA (1 día)"
        raw_data = self._get_with_cache(endpoint, name)

        if not raw_data:
            print(f"⚠️ No data received for {name}.")
            return []

        results = []
        for row in raw_data:
            try:
                date_str = row.get("date") or row.get("fecha")
                value = row.get("rate") or row.get("valor") or row.get("tasa")

                if not date_str or value is None:
                    continue

                date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
                if d_from and date < datetime.strptime(d_from, "%Y-%m-%d").date():
                    continue
                if d_to and date > datetime.strptime(d_to, "%Y-%m-%d").date():
                    continue

                results.append(BYMARateDTO(name=name, date=date, value=value))
            except Exception as e:
                print(f"⚠️ Skipping invalid row: {e}")

        print(f"✅ Retrieved {len(results)} BYMA rates (filtered by date).")
        return results
