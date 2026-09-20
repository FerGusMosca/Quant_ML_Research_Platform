import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class BymaCaucionDownloader:
    """
    Tasas de caucion por plazo, tomadas del panel publico de BYMA.

    Es el unico lugar donde aparece la tasa abierta por vencimiento: el dato
    del banco central es una sola tasa de un dia, y las otras fuentes no traen
    cauciones.

    No pide clave ni usuario. El certificado del sitio no valida contra el
    paquete estandar de Python, asi que se pide sin validar la cadena: la
    conexion sigue siendo cifrada.
    """

    BASE_URL = "https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free"

    PANEL_PATH = "cauciones"

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # Se pide todo de una: el panel entero son poco mas de cien renglones.
    PAGE_SIZE = 5000

    TIMEOUT_SECONDS = 30

    VERIFY_SSL = False

    # Como se llama cada moneda del lado de BYMA.
    CCY_PESOS = "ARS"
    CCY_DOLAR = "USD"

    # De donde se lee la tasa, en orden: primero lo operado, despues el cierre.
    RATE_FIELDS = ["trade", "closingPrice", "previousClosingPrice", "settlementPrice"]

    # De donde se lee el volumen, en orden.
    VOLUME_FIELDS = ["volumeAmount", "tradeVolume", "volume"]

    def __init__(self, timeout_seconds=None):
        self.timeout = timeout_seconds if timeout_seconds is not None else self.TIMEOUT_SECONDS

    # ==================================================================

    def get_panel(self):
        """
        Devuelve el panel crudo de cauciones, tal como lo publica BYMA.
        """
        response = requests.post(
            f"{self.BASE_URL}/{self.PANEL_PATH}",
            headers=self.HEADERS,
            json={"page_size": self.PAGE_SIZE},
            timeout=self.timeout,
            verify=self.VERIFY_SSL,
        )

        if response.status_code != 200:
            raise Exception(f"BYMA returned status {response.status_code}")

        data = response.json()

        # Segun el panel, a veces viene la lista sola y a veces adentro de una
        # caja con otros datos.
        if isinstance(data, dict):
            return data.get("data") or data.get("content") or []

        return data or []

    def get_rates_by_currency(self, currency):
        """
        Devuelve la lista de plazos de una moneda, cada uno con su fecha de
        vencimiento, sus dias, su tasa y su volumen.
        """
        rows = []

        for item in self.get_panel():

            if str(item.get("denominationCcy") or "").strip().upper() != str(currency).strip().upper():
                continue

            rate = self._first_value(item, self.RATE_FIELDS)

            rows.append({
                "symbol": str(item.get("symbol") or ""),
                "maturity_date": str(item.get("maturityDate") or ""),
                "days": int(item.get("daysToMaturity") or 0),
                "rate": rate,
                "volume": self._first_value(item, self.VOLUME_FIELDS),
            })

        return rows

    # ==================================================================

    @staticmethod
    def _first_value(item, fields):
        """
        Se queda con el primer campo que traiga algo distinto de cero.
        """
        for field in fields:

            value = item.get(field)

            if value is None:
                continue

            try:
                number = float(value)
            except Exception:
                continue

            if number != 0:
                return number

        return None
