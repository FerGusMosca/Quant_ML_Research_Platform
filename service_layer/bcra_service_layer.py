import requests
from datetime import datetime
from common.dto.bcra_rate_dto import BCRARateDTO


class BCRAServiceLayer:
    """
    Handles all interactions with the unofficial but stable BCRA API (https://api.estadisticasbcra.com).
    Requires an API token via Authorization: BEARER {TOKEN}.
    """

    BASE_URL = "https://api.estadisticasbcra.com"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Missing BCRA API token.")
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Authorization": f"BEARER {api_key}"
        })

    def get_interest_rates(self, d_from=None, d_to=None):
        """
        Fetches key BCRA interest rate series.
        Returns a list of BCRARateDTO.
        """
        # Stable endpoints for each rate
        endpoints = {
            "Tasa Política Monetaria (LELIQ)": "/tasa_leliq",
            "Tasa BADLAR Promedio Bancos Privados": "/tasa_badlar",
            "Tasa BAIBAR": "/tasa_baibar",
            "Tasa TM20": "/tasa_tm20",
            "Tasa Pases Activos 1 día": "/tasa_pase_activas_1_dia",
            "Tasa Pases Pasivos 1 día": "/tasa_pase_pasivas_1_dia",
            "Tasa de Depósitos a 30 días": "/tasa_depositos_30_dias",
            "Tasa Préstamos Personales": "/tasa_prestamos_personales",
        }

        results = []

        for name, path in endpoints.items():
            url = f"{self.BASE_URL}{path}"
            resp = self.session.get(url)

            if resp.status_code != 200:
                print(f"⚠️ Failed to fetch {name}: {resp.status_code} {resp.text}")
                continue

            data = resp.json()

            for row in data:
                date_str = row.get("d")
                value = row.get("v")

                if not date_str or value is None:
                    continue

                date = datetime.strptime(date_str, "%Y-%m-%d").date()
                if d_from and date < datetime.strptime(d_from, "%Y-%m-%d").date():
                    continue
                if d_to and date > datetime.strptime(d_to, "%Y-%m-%d").date():
                    continue

                results.append(BCRARateDTO(name=name, date=date, value=value))

        return results
