import os
import time
import traceback
from datetime import datetime

import pandas as pd

from common.util.downloaders.yahoo_quote_downloader import YahooQuoteDownloader
from common.util.std_in_out.google_drive_handler import GoogleDriveHandler
from common.util.std_in_out.root_locator import RootLocator
from data_access_layer.stock_monitor_portfolio_manager import StockMonitorPortfolioManager
from framework.common.logger.message_type import MessageType


class MTMPricesReport:
    """
    Actualiza la planilla de activos compartida en Drive.

    Son dos cosas distintas dentro del mismo archivo:

    - Solapas por segmento de mercado: la lista de instrumentos la pone el
      cliente y nosotros solo completamos el precio de cierre y el volumen.
    - Solapa del portfolio: no tiene lista previa, se arma entera con los
      activos del portfolio del monitor y se pisa completa todos los dias.

    Los precios salen de Yahoo con el mismo mecanismo que ya usa la pantalla
    del monitor, que alcanza con el ticker y no pide saber el mercado.
    """

    # Si no viene la ruta por parametro, se usa la clave que vive adentro del
    # proyecto. El parametro siempre gana, asi que en otra maquina o en un
    # container se puede apuntar a otro lado sin tocar el codigo.
    DEFAULT_CREDENTIALS_PATH = os.path.join("static", "config", "update-ml-prices-cmd-ca0736c87831.json")

    CONTROL_TAB = "CONTROL"
    PORTFOLIO_TAB = "Official Portfolio"

    # Mercado donde figuran los papeles argentinos en TradingView.
    LOCAL_EXCHANGE = "BYMA"

    # Pausa entre pedidos a TradingView y reintentos. Sin esto el feed corta la
    # conexion cuando se le piden muchos simbolos seguidos.
    LOCAL_PAUSE_SECONDS = 0.7
    LOCAL_RETRIES = 2
    LOCAL_RETRY_PAUSE_SECONDS = 3

    # Cada cuantas filas se vuelca lo que se lleva bajado. De a lotes y no fila
    # por fila, porque cada escritura es un pedido a Drive.
    WRITE_BATCH_ROWS = 10

    # La solapa de control tiene el encabezado en la fila 1 y las descripciones
    # de cada columna en la fila 2, asi que el estado se escribe en la fila 3.
    CONTROL_STATUS_CELL = "A3"

    STATUS_WRITING = "ESCRIBIENDO"
    STATUS_DONE = "LISTO"
    STATUS_ERROR = "ERROR"

    # Columna llave tal como figura hoy en el archivo. Se dejan las dos formas
    # porque la especificacion la nombra de una manera y el archivo de otra.
    SYMBOL_HEADERS = ["TICKER_BYMA", "Ticker_ID"]

    # Columnas donde van los datos que completamos nosotros. Cada solapa las
    # nombra a su manera, asi que se busca por cualquiera de estas formas y se
    # escribe SIEMPRE sobre la que ya existe: nunca se crea una columna nueva.
    PRICE_HEADERS = ["PRECIO", "Ultimo precio", "Ultimo Precio", "Precio_Cierre"]
    VOLUME_HEADERS = ["VOLUMEN", "Volumen", "Volumen operado", "Volumen_Operado"]

    PORTFOLIO_SCHEMA = [
        "Ticker_ID",
        "Descripcion",
        "Precio_Cierre",
        "Variacion_Nominal",
        "Variacion_Porcentual",
        "Volumen_Operado",
    ]


    def __init__(self, gdrive_url, input_file, output_file, credentials_file=None, portfolio=None,
                 monitor_conn_str=None, tv_params=None, work_folder=None, logger=None, job_id=None):

        self.gdrive_url = gdrive_url
        self.input_file = input_file
        self.output_file = output_file if output_file is not None else input_file
        self.credentials_file = credentials_file if credentials_file is not None else self._default_credentials_path()
        # Siempre es el portfolio publicado del monitor, asi que no hace falta
        # mandarlo en cada llamada.
        self.portfolio = portfolio if portfolio is not None else self.PORTFOLIO_TAB
        self.monitor_conn_str = monitor_conn_str
        self.work_folder = work_folder or os.path.join(".", "output", "mtm")
        self.logger = logger
        self.job_id = job_id

        self.tv_params = tv_params or {}

        self.drive = GoogleDriveHandler(self.credentials_file, logger)
        self.quotes = YahooQuoteDownloader()

        self._local_downloader = None

    # ==================================================================
    # Logging
    # ==================================================================

    def _log(self, msg, msg_type=MessageType.INFO):
        if self.logger is not None:
            self.logger.do_log(msg, msg_type, self.job_id)
        else:
            print(msg)

    # ==================================================================
    # Entrada principal
    # ==================================================================

    def run(self):

        folder_id = GoogleDriveHandler.extract_folder_id(self.gdrive_url)

        input_id = self.drive.get_file_id(folder_id, self.input_file)
        output_id = self.drive.get_file_id(folder_id, self.output_file)

        summary = {
            "input_file": self.input_file,
            "output_file": self.output_file,
            "tabs": {},
            "total_symbols": 0,
            "priced": 0,
            "errors": 0,
        }

        try:
            self._write_control(output_id, self.STATUS_WRITING, 0, "Descarga de precios en curso")

            # Copia de respaldo de lo que se leyo, antes de tocar nada. Queda en
            # disco local: la planilla de Drive nunca cambia de formato.
            backup_path = os.path.join(
                self.work_folder,
                f"{self.input_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            )
            self.drive.download_file(input_id, backup_path)

            tabs = self.drive.read_all_tabs(input_id, skip_tabs=[self.CONTROL_TAB])

            for tab_name, df in tabs.items():

                if self._is_portfolio_tab(tab_name):
                    continue

                tab_summary = self._process_market_tab(output_id, tab_name, df)

                if tab_summary is None:
                    continue

                summary["tabs"][tab_name] = tab_summary
                summary["total_symbols"] += tab_summary["symbols"]
                summary["priced"] += tab_summary["priced"]
                summary["errors"] += tab_summary["errors"]

            portfolio_summary = self._process_portfolio_tab(output_id)

            if portfolio_summary is not None:
                summary["tabs"][self.PORTFOLIO_TAB] = portfolio_summary
                summary["total_symbols"] += portfolio_summary["symbols"]
                summary["priced"] += portfolio_summary["priced"]
                summary["errors"] += portfolio_summary["errors"]

            message = (
                f"{summary['priced']} precios actualizados sobre "
                f"{summary['total_symbols']} instrumentos. Errores: {summary['errors']}"
            )

            self._write_control(output_id, self.STATUS_DONE, summary["priced"], message)

            self._log(f"[MTM] {message}")

            return summary

        except Exception as e:
            detail = f"{str(e)}"

            self._log(f"[MTM] {detail}", MessageType.ERROR)
            self._log(traceback.format_exc(), MessageType.ERROR)

            try:
                self._write_control(output_id, self.STATUS_ERROR, summary["priced"], detail)
            except Exception:
                pass

            raise

    # ==================================================================
    # Solapas por segmento de mercado
    # ==================================================================

    def _process_market_tab(self, output_id, tab_name, df):
        """
        Completa precio y volumen sobre la lista de instrumentos que ya esta
        cargada. Si la solapa esta vacia se deja solo el encabezado.
        """
        if df.empty:
            self._log(f"[MTM][{tab_name}] solapa vacia. Salteada.", MessageType.WARNING)
            return None

        symbol_col = self._find_column(df, self.SYMBOL_HEADERS)

        if symbol_col is None:
            # Las solapas de cauciones no tienen ticker: no hay nada que buscar
            # por simbolo y se dejan como estan.
            self._log(f"[MTM][{tab_name}] no tiene columna de ticker. Salteada.", MessageType.WARNING)
            return None

        result = df.copy()

        price_col = self._find_column(result, self.PRICE_HEADERS)
        volume_col = self._find_column(result, self.VOLUME_HEADERS)

        if price_col is None or volume_col is None:
            faltan = []
            if price_col is None:
                faltan.append("precio")
            if volume_col is None:
                faltan.append("volumen")

            self._log(
                f"[MTM][{tab_name}] no tiene columna de {' ni de '.join(faltan)}. "
                f"Columnas encontradas: {list(result.columns)}. Solapa salteada.",
                MessageType.WARNING,
            )
            return None

        self._log(
            f"[MTM][{tab_name}] ticker en '{symbol_col}', "
            f"precio en '{price_col}', volumen en '{volume_col}'"
        )

        tab_summary = {"symbols": 0, "priced": 0, "errors": 0}

        pending = 0

        for index, row in result.iterrows():

            symbol = str(row[symbol_col]).strip()

            if symbol == "" or symbol.lower() == "nan":
                continue

            tab_summary["symbols"] += 1

            try:
                price, volume, fuente = self._get_price_and_volume(symbol, tab_name)

                result.at[index, price_col] = price
                result.at[index, volume_col] = volume if volume is not None else ""

                tab_summary["priced"] += 1

                self._log(f"[MTM][{tab_name}] {symbol}: {price} (fuente {fuente})")

            except Exception as e:
                tab_summary["errors"] += 1
                self._log(f"[MTM][{tab_name}] {symbol}: {str(e)}", MessageType.WARNING)

            pending += 1

            if pending >= self.WRITE_BATCH_ROWS:
                self._log(
                    f"[MTM][{tab_name}] >>> ESCRIBIENDO EN DRIVE: {pending} filas nuevas, "
                    f"{tab_summary['symbols']} procesadas hasta {symbol}"
                )
                self.drive.create_tab_if_missing(output_id, tab_name)
                self.drive.write_tab(output_id, tab_name, result)
                self._log(f"[MTM][{tab_name}] <<< ESCRITO OK. Ya podes mirar la planilla.")
                pending = 0

        self._log(f"[MTM][{tab_name}] >>> ESCRITURA FINAL: {pending} filas pendientes")
        self.drive.create_tab_if_missing(output_id, tab_name)
        self.drive.write_tab(output_id, tab_name, result)
        self._log(f"[MTM][{tab_name}] <<< ESCRITURA FINAL OK")

        self._log(
            f"[MTM][{tab_name}] {tab_summary['priced']} con precio, "
            f"{tab_summary['errors']} con error, sobre {tab_summary['symbols']} instrumentos"
        )

        return tab_summary

    # ==================================================================
    # Solapa del portfolio
    # ==================================================================

    def _process_portfolio_tab(self, output_id):
        """
        Arma la foto del cierre del portfolio y pisa la solapa entera. La lista
        de activos sale del monitor, no de la planilla.
        """
        if self.portfolio is None or self.monitor_conn_str is None:
            self._log(
                "[MTM] Portfolio tab skipped: portfolio or connection string not provided.",
                MessageType.WARNING,
            )
            return None

        manager = StockMonitorPortfolioManager(self.monitor_conn_str, self.logger)

        symbols = manager.get_symbols(self.portfolio)

        rows = []
        tab_summary = {"symbols": 0, "priced": 0, "errors": 0}

        for symbol in symbols:

            if symbol == "":
                continue

            tab_summary["symbols"] += 1

            try:
                quote = self.quotes.get_quote(symbol)

                rows.append([
                    quote["symbol"],
                    quote["name"],
                    quote["price"],
                    quote["change"] if quote["change"] is not None else "",
                    quote["change_pct"] if quote["change_pct"] is not None else "",
                    quote["volume"] if quote["volume"] is not None else "",
                ])

                tab_summary["priced"] += 1

            except Exception as e:
                tab_summary["errors"] += 1
                self._log(f"[MTM][{self.PORTFOLIO_TAB}] {symbol}: {str(e)}", MessageType.WARNING)

                rows.append([symbol, "", "", "", "", ""])

        result = pd.DataFrame(rows, columns=self.PORTFOLIO_SCHEMA)

        self.drive.create_tab_if_missing(output_id, self.PORTFOLIO_TAB)
        self.drive.write_tab(output_id, self.PORTFOLIO_TAB, result)

        self._log(
            f"[MTM][{self.PORTFOLIO_TAB}] {tab_summary['priced']} con precio, "
            f"{tab_summary['errors']} con error, sobre {tab_summary['symbols']} instrumentos"
        )

        return tab_summary

    # ==================================================================
    # Solapa de control
    # ==================================================================

    def _write_control(self, spreadsheet_id, status, price_rows, message):

        now = datetime.now()

        row = [
            status,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M"),
            price_rows,
            0,
            message,
        ]

        self.drive.create_tab_if_missing(spreadsheet_id, self.CONTROL_TAB)
        self.drive.write_rows(spreadsheet_id, self.CONTROL_TAB, [row], self.CONTROL_STATUS_CELL)

    # ==================================================================
    # Helpers
    # ==================================================================

    def _is_portfolio_tab(self, tab_name):
        return tab_name.strip().upper() == self.PORTFOLIO_TAB.upper()

    def _get_price_and_volume(self, symbol, tab_name=""):
        """
        Primero se pregunta a TradingView contra BYMA, que es donde estan los
        papeles locales y sus tres cotizaciones. Si ahi no aparece, se cae a
        Yahoo, que resuelve con el ticker solo y cubre lo extranjero.
        """
        self._log(f"[MTM][{tab_name}] {symbol}: buscando en TradingView ({self.LOCAL_EXCHANGE})")

        try:
            price, volume = self._get_local_price_and_volume(symbol)
            return price, volume, self.LOCAL_EXCHANGE
        except Exception:
            pass

        self._log(f"[MTM][{tab_name}] {symbol}: no esta en {self.LOCAL_EXCHANGE}, buscando en Yahoo")

        quote = self.quotes.get_quote(symbol)

        return quote["price"], quote["volume"], "Yahoo"

    def _get_local_price_and_volume(self, symbol):

        downloader = self._get_local_downloader()

        df = None
        last_error = None

        for attempt in range(self.LOCAL_RETRIES + 1):

            if attempt > 0:
                time.sleep(self.LOCAL_RETRY_PAUSE_SECONDS)

            try:
                df = downloader.download(symbol)
            except Exception as e:
                last_error = e
                df = None

            if df is not None and len(df) > 0:
                break

        time.sleep(self.LOCAL_PAUSE_SECONDS)

        if df is None or len(df) == 0:
            raise Exception(str(last_error) if last_error is not None else "No rows returned")

        last = df.iloc[-1]

        price = float(last["close"])
        volume = float(last["volume"]) if "volume" in df.columns else None

        return price, volume

    def _get_local_downloader(self):
        """
        El import va adentro del metodo a proposito: la libreria de TradingView
        no esta instalada en todos los entornos y si se importa arriba del
        archivo se cae todo el proceso de reportes apenas arranca.
        """
        if self._local_downloader is None:
            from common.util.downloaders.tradingview_downloader import TradingViewDownloader

            params = dict(self.tv_params)
            params["exchange"] = self.LOCAL_EXCHANGE
            params.setdefault("interval", "1d")

            self._local_downloader = TradingViewDownloader(params)

        return self._local_downloader

    @staticmethod
    def _default_credentials_path():
        """
        Arma la ruta de la clave que esta guardada adentro del proyecto,
        arrancando desde la raiz, para que no dependa de desde donde se corra.
        """
        root = RootLocator.get_root(markers=["bias_mgmt_console.py", "README.md"])

        return os.path.join(root, MTMPricesReport.DEFAULT_CREDENTIALS_PATH)

    @staticmethod
    def _find_column(df, candidates):
        """
        Busca la columna por nombre de encabezado, sin importar mayusculas ni
        espacios. Si no encuentra ninguna devuelve None.
        """
        normalized = {str(c).strip().upper(): c for c in df.columns}

        for candidate in candidates:
            if candidate.upper() in normalized:
                return normalized[candidate.upper()]

        return None
