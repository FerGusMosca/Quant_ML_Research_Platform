import asyncio
import time

from common.util.downloaders.byma_caucion_downloader import BymaCaucionDownloader
from common.util.downloaders.byma_panel_downloader import BymaPanelDownloader
from common.util.downloaders.data912_downloader import Data912Downloader
from framework.common.logger.message_type import MessageType


class MTMPriceSourceRouter:
    """
    Decide de donde sale el precio de cada instrumento segun la solapa en la
    que esta cargado.

    Hasta ahora todo salia de Yahoo (con un intento previo contra BYMA). Esta
    clase reemplaza esa logica por un mapa solapa -> lista de fuentes, que se
    prueban en orden hasta que alguna responde.

    Todo lo que se puede configurar vive aca adentro a proposito: es un
    desarrollo experimental, y recien cuando sirva se lleva a los archivos de
    configuracion.
    """

    # ==================================================================
    # Fuentes disponibles
    # ==================================================================

    SRC_ARG_BONDS = "arg_bonds"      # soberanas argentinas y CER
    SRC_ARG_NOTES = "arg_notes"      # letras / lecaps / boncaps
    SRC_ARG_CORP = "arg_corp"        # obligaciones negociables
    SRC_TV = "tradingview"           # precios de pantalla
    SRC_YAHOO = "yahoo"              # todo lo de afuera, alcanza el ticker
    SRC_BYMA_CAUCION = "byma_caucion"  # tasas de caucion abiertas por plazo
    SRC_BYMA_RF = "byma_renta_fija"    # paneles de renta fija de la pantalla de BYMA

    # ==================================================================
    # Mapa solapa -> fuentes, en orden de prueba
    # ==================================================================

    # La clave se compara sin mayusculas ni espacios.
    TAB_SOURCES = {
        "TICKER~1.CSV": [SRC_ARG_BONDS, SRC_ARG_NOTES, SRC_ARG_CORP, SRC_TV, SRC_YAHOO],
        "CABLERF": [SRC_BYMA_RF, SRC_TV, SRC_ARG_CORP, SRC_ARG_BONDS, SRC_YAHOO],
        "CABLERV": [SRC_YAHOO, SRC_TV],
        "OFFICIAL PORTFOLIO": [SRC_YAHOO, SRC_TV],
    }

    # Una solapa es de caucion si su nombre arranca con esta palabra. La
    # moneda sale del tag que viene en el mismo nombre (Caucion ARS, Caucion
    # USD); si no trae tag, se asume pesos.
    CAUCION_TAB_PREFIX = "CAUCION"

    CAUCION_CURRENCY_TAGS = {
        "USD": BymaCaucionDownloader.CCY_DOLAR,
        "DOLAR": BymaCaucionDownloader.CCY_DOLAR,
        "U$S": BymaCaucionDownloader.CCY_DOLAR,
        "ARS": BymaCaucionDownloader.CCY_PESOS,
        "PESOS": BymaCaucionDownloader.CCY_PESOS,
        "$": BymaCaucionDownloader.CCY_PESOS,
    }

    CAUCION_DEFAULT_CURRENCY = BymaCaucionDownloader.CCY_PESOS

    # Si la tasa viene en tanto por uno y la planilla la quiere en porcentaje,
    # se cambia aca. En 1 se escribe tal cual la publica BYMA.
    CAUCION_RATE_MULTIPLIER = 1

    # Solapas que se informan como "sin fuente automatica" en vez de como error.
    MANUAL_TABS = ["RESUMEN_VIGENCIA", "VENCIDOS"]

    # Si aparece una solapa que no esta en el mapa, se usa esto.
    DEFAULT_SOURCES = [SRC_YAHOO, SRC_TV]

    # ==================================================================
    # Mercados de pantalla por solapa
    # ==================================================================

    # Se prueban en orden. BYMA cubre lo local y tambien los papeles del
    # exterior que cotizan aca; los otros dos cubren lo que solo esta afuera.
    TV_EXCHANGES_BY_TAB = {
        "TICKER~1.CSV": ["BYMA"],
        "CABLERF": ["BYMA"],
        "CABLERV": ["NASDAQ", "NYSE", "BYMA"],
        "OFFICIAL PORTFOLIO": ["NASDAQ", "NYSE", "AMEX"],
    }

    TV_DEFAULT_EXCHANGES = ["BYMA"]

    TV_INTERVAL = "1d"

    # Pausa entre pedidos de pantalla y reintentos. Sin esto el feed corta la
    # conexion cuando se le piden muchos seguidos.
    TV_PAUSE_SECONDS = 0.7
    TV_RETRIES = 1
    TV_RETRY_PAUSE_SECONDS = 3

    # ==================================================================
    # Marcado de instrumentos que parecen estar en la solapa equivocada
    # ==================================================================

    MARK_MISPLACED = True

    # Donde termina resolviendose cada tipo de instrumento, para poder avisar
    # a donde habria que moverlo.
    HOME_TAB_BY_SOURCE = {
        SRC_ARG_BONDS: "TICKER~1.CSV",
        SRC_ARG_NOTES: "TICKER~1.CSV",
        SRC_ARG_CORP: "CableRF",
        SRC_YAHOO: "CableRV",
    }

    MISPLACED_TEXT = "REVISAR: responde como {tipo}, iria en {solapa}"
    NO_SOURCE_TEXT = "SIN PRECIO: no responde ninguna fuente de esta solapa"
    NO_MAPPING_TEXT = "SIN FUENTE: la solapa no tiene fuente asignada"

    # ==================================================================

    def __init__(self, yahoo_downloader, tv_params=None, logger=None, job_id=None):

        self.yahoo = yahoo_downloader
        self.tv_params = tv_params or {}
        self.logger = logger
        self.job_id = job_id

        # Se baja una sola vez por corrida y se reusa para todas las solapas.
        self._snapshots = {}

        # Un bajador de pantalla por mercado.
        self._tv_downloaders = {}

        # Paneles de renta fija, una sola bajada por corrida.
        self._byma_panels = BymaPanelDownloader()

        # Panel de cauciones, una sola bajada por corrida y por moneda.
        self._caucion_downloader = BymaCaucionDownloader()
        self._caucion_rows = {}

    # ==================================================================
    # Entrada principal
    # ==================================================================

    def get_price_and_volume(self, symbol, tab_name):
        """
        Devuelve (precio, volumen, fuente, observacion).

        La observacion viene vacia cuando el instrumento contesto por la
        fuente que le corresponde a su solapa.
        """

        clean = str(symbol).strip().upper()

        sources = self._sources_for_tab(tab_name)

        if len(sources) == 0:
            return None, None, None, self.NO_MAPPING_TEXT

        for source in sources:

            try:
                price, volume = self._ask_source(source, clean, tab_name)
            except Exception as e:
                self._log(f"[MTM][{tab_name}] {clean}: {source} no contesto ({str(e)})",
                          MessageType.WARNING)
                continue

            if price is None:
                continue

            return price, volume, source, self._review_note(source, tab_name)

        # No contesto ninguna de las suyas: se prueban las demas solo para
        # poder avisar a donde iria.
        return None, None, None, self._look_elsewhere(clean, tab_name)

    # ==================================================================
    # Fuentes
    # ==================================================================

    def _ask_source(self, source, symbol, tab_name):

        if source == self.SRC_YAHOO:
            quote = self.yahoo.get_quote(symbol)
            return quote["price"], quote["volume"]

        if source == self.SRC_TV:
            return self._ask_tradingview(symbol, tab_name)

        if source == self.SRC_BYMA_RF:
            return self._ask_byma_panels(symbol)

        if source == self.SRC_BYMA_CAUCION:
            # Las cauciones no se buscan por ticker: las resuelve la solapa.
            return None, None

        return self._ask_data912(source, symbol)

    def _ask_byma_panels(self, symbol):

        quote = self._byma_panels.get_quote(symbol)

        if quote is None:
            return None, None

        return float(quote["price"]), quote["volume"]

    def _ask_data912(self, source, symbol):

        snapshot = self._get_snapshot(source)

        item = snapshot.get(symbol)

        if item is None:
            return None, None

        price = item.get("price")

        if price is None or float(price) == 0:
            return None, None

        return float(price), item.get("volume")

    def _get_snapshot(self, source):
        """
        Baja la lista completa de la fuente una sola vez y la deja indexada por
        ticker. Es un pedido por corrida en vez de uno por instrumento.
        """
        if source in self._snapshots:
            return self._snapshots[source]

        endpoint_by_source = {
            self.SRC_ARG_BONDS: Data912Downloader.ENDPOINT_ARG_BONDS,
            self.SRC_ARG_NOTES: Data912Downloader.ENDPOINT_ARG_NOTES,
            self.SRC_ARG_CORP: Data912Downloader.ENDPOINT_ARG_CORP,
        }

        endpoint = endpoint_by_source.get(source)

        if endpoint is None:
            self._snapshots[source] = {}
            return self._snapshots[source]

        downloader = Data912Downloader()

        items = asyncio.run(downloader.fetch(endpoint))

        snapshot = Data912Downloader.build_price_map(items)

        self._log(f"[MTM] {source}: {len(snapshot)} instrumentos disponibles")

        self._snapshots[source] = snapshot

        return snapshot

    def _ask_tradingview(self, symbol, tab_name):

        exchanges = self._exchanges_for_tab(tab_name)

        for exchange in exchanges:

            downloader = self._get_tv_downloader(exchange)

            df = None

            for attempt in range(self.TV_RETRIES + 1):

                if attempt > 0:
                    time.sleep(self.TV_RETRY_PAUSE_SECONDS)

                try:
                    df = downloader.download(symbol)
                except Exception:
                    df = None

                if df is not None and len(df) > 0:
                    break

            time.sleep(self.TV_PAUSE_SECONDS)

            if df is None or len(df) == 0:
                continue

            last = df.iloc[-1]

            price = float(last["close"])
            volume = float(last["volume"]) if "volume" in df.columns else None

            return price, volume

        return None, None

    def _get_tv_downloader(self, exchange):
        """
        El import va adentro del metodo a proposito: la libreria de pantalla no
        esta instalada en todos los entornos y si se importa arriba del archivo
        se cae todo el proceso de reportes apenas arranca.
        """
        if exchange not in self._tv_downloaders:

            from common.util.downloaders.tradingview_downloader import TradingViewDownloader

            params = dict(self.tv_params)
            params["exchange"] = exchange
            params.setdefault("interval", self.TV_INTERVAL)

            self._tv_downloaders[exchange] = TradingViewDownloader(params)

        return self._tv_downloaders[exchange]

    # ==================================================================
    # Marcado
    # ==================================================================

    def _review_note(self, source, tab_name):
        """
        Si contesto una fuente que no es la de la solapa, se deja el aviso para
        que muevan el instrumento a donde corresponde.
        """
        if not self.MARK_MISPLACED:
            return ""

        home = self.HOME_TAB_BY_SOURCE.get(source)

        if home is None:
            return ""

        if self._same_tab(home, tab_name):
            return ""

        # Lo que contesta por pantalla o por Yahoo desde una solapa de afuera
        # no dice nada raro: recien se marca cuando la solapa tiene otro tipo.
        if source == self.SRC_YAHOO and self._sources_for_tab(tab_name)[0] == self.SRC_YAHOO:
            return ""

        return self.MISPLACED_TEXT.format(tipo=source, solapa=home)

    def _look_elsewhere(self, symbol, tab_name):

        if not self.MARK_MISPLACED:
            return self.NO_SOURCE_TEXT

        own = self._sources_for_tab(tab_name)

        for source in [self.SRC_ARG_BONDS, self.SRC_ARG_NOTES, self.SRC_ARG_CORP]:

            if source in own:
                continue

            try:
                price, volume = self._ask_data912(source, symbol)
            except Exception:
                continue

            if price is not None:
                home = self.HOME_TAB_BY_SOURCE.get(source, "")
                return self.MISPLACED_TEXT.format(tipo=source, solapa=home)

        return self.NO_SOURCE_TEXT

    # ==================================================================
    # Helpers
    # ==================================================================

    def is_caucion_tab(self, tab_name):

        return self._normalize(tab_name).startswith(self.CAUCION_TAB_PREFIX)

    def get_caucion_currency(self, tab_name):
        """
        Lee el tag de moneda que viene en el nombre de la solapa.
        """
        key = self._normalize(tab_name)

        for tag, currency in self.CAUCION_CURRENCY_TAGS.items():
            if tag in key:
                return currency

        return self.CAUCION_DEFAULT_CURRENCY

    def get_caucion_rows(self, tab_name):
        """
        Devuelve los plazos de caucion de esa solapa, ya con la tasa lista para
        escribir.
        """
        key = self._normalize(tab_name)

        if key in self._caucion_rows:
            return self._caucion_rows[key]

        if not self.is_caucion_tab(tab_name):
            return []

        currency = self.get_caucion_currency(tab_name)

        rows = self._caucion_downloader.get_rates_by_currency(currency)

        for row in rows:
            if row["rate"] is not None:
                row["rate"] = row["rate"] * self.CAUCION_RATE_MULTIPLIER

        self._log(f"[MTM][{tab_name}] {len(rows)} plazos de caucion en BYMA")

        self._caucion_rows[key] = rows

        return rows

    def _sources_for_tab(self, tab_name):

        key = self._normalize(tab_name)

        for tab, sources in self.TAB_SOURCES.items():
            if self._normalize(tab) == key:
                return sources

        if self.is_caucion_tab(tab_name):
            return [self.SRC_BYMA_CAUCION]

        return self.DEFAULT_SOURCES

    def _exchanges_for_tab(self, tab_name):

        key = self._normalize(tab_name)

        for tab, exchanges in self.TV_EXCHANGES_BY_TAB.items():
            if self._normalize(tab) == key:
                return exchanges

        return self.TV_DEFAULT_EXCHANGES

    def _same_tab(self, one, other):
        return self._normalize(one) == self._normalize(other)

    @staticmethod
    def _normalize(tab_name):
        return str(tab_name).strip().upper()

    def _log(self, msg, msg_type=MessageType.INFO):
        if self.logger is not None:
            self.logger.do_log(msg, msg_type, self.job_id)
        else:
            print(msg)
