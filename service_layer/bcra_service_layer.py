import requests
import os
import json
import time
from datetime import datetime
from common.dto.bcra_rate_dto import BCRARateDTO


class BCRAServiceLayer:
    """
    Stable client for BCRA data using Argentina's Open Data API (https://datos.gob.ar).
    This API is maintained and updated with 2025 data.
    """

    BASE_URL = "https://api.datos.gob.ar/series/api/series"
    CACHE_DIR = "./cache/bcra"

    SERIES = {
        "Tasa Política Monetaria (LELIQ)": "101.1_TISLELIQ_POR",
        "Tasa BADLAR Promedio Bancos Privados": "101.1_TIBADLAR_POR",
        "Tasa TM20": "101.1_TITM20_POR",
        "Tasa Pases Pasivos 1 día": "101.1_TIPPASIVA_POR",
        "Tasa Pases Activos 1 día": "101.1_TIPACTIVA_POR",
        "Tasa de Depósitos a 30 días": "101.1_TIPRESTOS30_POR"
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def _cache_path(self, name: str):
        safe_name = name.replace(" ", "_").replace("/", "_").lower()
        return os.path.join(self.CACHE_DIR, f"{safe_name}.json")

    def _get_with_cache(self, series_id: str, name: str):
        cache_file = self._cache_path(series_id)
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        url = f"{self.BASE_URL}/?ids={series_id}"
        time.sleep(1)
        resp = self.session.get(url)

        if resp.status_code != 200:
            print(f"⚠️ Failed to fetch {name}: {resp.status_code} {resp.text}")
            return {}

        data = resp.json()
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return data

    def get_interest_rates(self, d_from=None, d_to=None):
        results = []
        for name, series_id in self.SERIES.items():
            data = self._get_with_cache(series_id, name)
            series_data = data.get("data", [])

            for row in series_data:
                date_str, value = row[0], row[1]
                date = datetime.strptime(date_str, "%Y-%m-%d").date()
                if d_from and date < datetime.strptime(d_from, "%Y-%m-%d").date():
                    continue
                if d_to and date > datetime.strptime(d_to, "%Y-%m-%d").date():
                    continue

                results.append(BCRARateDTO(name=name, date=date, value=value))

        return results
