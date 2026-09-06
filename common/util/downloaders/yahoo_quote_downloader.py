import requests


class YahooQuoteDownloader:
    """
    Cotizacion puntual desde Yahoo Finance, con el mismo endpoint que ya usa la
    pantalla del monitor.

    La ventaja sobre TradingView es que alcanza con el ticker: no hace falta
    saber en que mercado cotiza cada instrumento, que es justo el dato que no
    tenemos para las solapas del maestro.
    """

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"

    HEADERS = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    def __init__(self, timeout=8.0):
        self.timeout = timeout

    def get_quote(self, symbol):
        """
        Devuelve precio, cierre anterior, las dos variaciones, el nombre y el
        volumen. Si algo falta viene en None: el que llama decide que hacer.
        """
        if symbol is None or str(symbol).strip() == "":
            raise Exception("Empty symbol")

        clean = str(symbol).strip().upper()

        url = f"{self.BASE_URL}/{clean}?interval=1d&range=1d"

        response = requests.get(url, headers=self.HEADERS, timeout=self.timeout)

        if response.status_code != 200:
            raise Exception(f"Yahoo returned status {response.status_code}")

        data = response.json()

        results = data.get("chart", {}).get("result") or []

        if len(results) == 0:
            raise Exception("No data returned")

        meta = results[0].get("meta", {})

        price = meta.get("regularMarketPrice")
        prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
        name = meta.get("longName") or meta.get("shortName") or clean
        volume = meta.get("regularMarketVolume")

        if price is None:
            raise Exception("No price returned")

        change = None
        change_pct = None

        if prev_close:
            change = round(float(price) - float(prev_close), 4)
            change_pct = round((change / float(prev_close)) * 100, 2)

        return {
            "symbol": clean,
            "name": name,
            "price": float(price),
            "prev_close": float(prev_close) if prev_close else None,
            "change": change,
            "change_pct": change_pct,
            "volume": volume,
        }
