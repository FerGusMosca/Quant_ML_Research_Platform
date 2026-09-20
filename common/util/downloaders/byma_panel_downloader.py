import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class BymaPanelDownloader:
    """
    Trae los paneles de renta fija de la pantalla publica de BYMA.

    Sirve para los papeles que no aparecen en las otras fuentes, que son
    justamente los del segmento del exterior: bonos de otros paises y letras
    que cotizan aca pero no figuran en los paneles de soberanas argentinas.

    No pide clave ni usuario. El certificado del sitio no valida contra el
    paquete estandar de Python, asi que se pide sin validar la cadena: la
    conexion sigue siendo cifrada.
    """

    BASE_URL = "https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free"

    # Paneles de pantalla que se piden, en este orden. El primero que trae el
    # papel gana.
    PANEL_PATHS = [
        "public-bonds",                 # soberanas y provinciales
        "corporate-bonds",              # obligaciones negociables
        "short-term-government-bonds",  # letras
    ]

    # Paneles del segmento bilateral, donde suele aparecer lo que no figura en
    # pantalla. Se piden siempre, despues de los de arriba.
    SENEBI_PATHS = [
        "senebi-public-bonds",
        "senebi-corporate-bonds",
        "senebi-short-term-government-bonds",
        "senebi-bonds",
    ]

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    PAYLOAD = {"excludeZeroPxAndQty": False, "T1": True, "T0": False, "Content-Type": "application/json"}

    PAGE_SIZE = 5000

    TIMEOUT_SECONDS = 30

    VERIFY_SSL = False

    # De donde se lee el precio, en orden: primero lo operado del dia, despues
    # el cierre anterior.
    PRICE_FIELDS = ["trade", "closingPrice", "settlementPrice", "previousClosingPrice"]

    # De donde se lee el volumen, en orden.
    VOLUME_FIELDS = ["volume", "volumeAmount", "tradeVolume", "quantityOperated"]

    # Como viene el nombre del papel en cada panel.
    SYMBOL_FIELDS = ["symbol", "denominationCcy", "securityId", "shortTicker"]

    def __init__(self, timeout_seconds=None):
        self.timeout = timeout_seconds if timeout_seconds is not None else self.TIMEOUT_SECONDS

        # Se baja una sola vez por corrida y queda indexado por papel.
        self._price_map = None

    # ==================================================================

    def get_price_map(self):
        """
        Devuelve un diccionario papel -> precio y volumen, juntando todos los
        paneles de renta fija.
        """
        if self._price_map is not None:
            return self._price_map

        result = {}

        for path in list(self.PANEL_PATHS) + list(self.SENEBI_PATHS):

            try:
                items = self._get_panel(path)
            except Exception:
                continue

            for item in items:

                if not isinstance(item, dict):
                    continue

                symbol = self._read_symbol(item)

                if symbol == "" or symbol in result:
                    continue

                price = self._first_value(item, self.PRICE_FIELDS)

                if price is None:
                    continue

                result[symbol] = {
                    "symbol": symbol,
                    "price": price,
                    "volume": self._first_value(item, self.VOLUME_FIELDS),
                    "panel": path,
                }

        self._price_map = result

        return result

    def get_quote(self, symbol):
        """
        Busca un papel puntual. Devuelve None si no esta en ningun panel.
        """
        clean = str(symbol).strip().upper()

        return self.get_price_map().get(clean)

    # ==================================================================

    def _get_panel(self, path):

        response = requests.post(
            f"{self.BASE_URL}/{path}",
            headers=self.HEADERS,
            json=dict(self.PAYLOAD, page_size=self.PAGE_SIZE),
            timeout=self.timeout,
            verify=self.VERIFY_SSL,
        )

        if response.status_code != 200:
            raise Exception(f"BYMA returned status {response.status_code} for {path}")

        data = response.json()

        # Segun el panel, a veces viene la lista sola y a veces adentro de una
        # caja con otros datos.
        if isinstance(data, dict):
            return data.get("data") or data.get("content") or []

        return data or []

    @classmethod
    def _read_symbol(cls, item):

        for field in cls.SYMBOL_FIELDS:

            value = item.get(field)

            if value is None:
                continue

            text = str(value).strip().upper()

            if text != "":
                return text

        return ""

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
