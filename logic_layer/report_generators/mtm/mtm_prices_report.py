import os
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

    # La solapa de control tiene el encabezado en la fila 1 y las descripciones
    # de cada columna en la fila 2, asi que el estado se escribe en la fila 3.
    CONTROL_STATUS_CELL = "A3"

    STATUS_WRITING = "ESCRIBIENDO"
    STATUS_DONE = "LISTO"
    STATUS_ERROR = "ERROR"

    # Columna llave, la misma en todas las solapas.
    SYMBOL_COL = "Ticker_ID"

    ASSET_SCHEMA = [
        "Ticker_ID",
        "ISIN",
        "Descripcion_Activo",
        "Vencimiento_Final",
        "Precio_Cierre",
        "Paridad",
        "Volumen_Operado",
        "Tiene_Calendario_Cargado",
    ]

    PORTFOLIO_SCHEMA = [
        "Ticker_ID",
        "Descripcion",
        "Precio_Cierre",
        "Variacion_Nominal",
        "Variacion_Porcentual",
        "Volumen_Operado",
    ]

    PRICE_COL = "Precio_Cierre"
    VOLUME_COL = "Volumen_Operado"

    def __init__(self, gdrive_url, input_file, output_file, credentials_file=None, portfolio=None,
                 monitor_conn_str=None, work_folder=None, logger=None, job_id=None):

        self.gdrive_url = gdrive_url
        self.input_file = input_file
        self.output_file = output_file if output_file is not None else input_file
        self.credentials_file = credentials_file if credentials_file is not None else self._default_credentials_path()
        self.portfolio = portfolio
        self.monitor_conn_str = monitor_conn_str
        self.work_folder = work_folder or os.path.join(".", "output", "mtm")
        self.logger = logger
        self.job_id = job_id

        self.drive = GoogleDriveHandler(self.credentials_file, logger)
        self.quotes = YahooQuoteDownloader()

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
            self._log(
                f"[MTM] Tab '{tab_name}' is empty. Header written, nothing to price.",
                MessageType.WARNING,
            )
            self.drive.create_tab_if_missing(output_id, tab_name)
            self.drive.write_tab(output_id, tab_name, pd.DataFrame(columns=self.ASSET_SCHEMA))
            return None

        symbol_col = self._find_column(df, [self.SYMBOL_COL])

        if symbol_col is None:
            self._log(
                f"[MTM] Tab '{tab_name}' has no {self.SYMBOL_COL} column. Skipped.",
                MessageType.WARNING,
            )
            return None

        result = self._apply_schema(df, self.ASSET_SCHEMA)

        tab_summary = {"symbols": 0, "priced": 0, "errors": 0}

        for index, row in result.iterrows():

            symbol = str(row[symbol_col]).strip()

            if symbol == "" or symbol.lower() == "nan":
                continue

            tab_summary["symbols"] += 1

            try:
                quote = self.quotes.get_quote(symbol)

                result.at[index, self.PRICE_COL] = quote["price"]
                result.at[index, self.VOLUME_COL] = quote["volume"] if quote["volume"] is not None else ""

                tab_summary["priced"] += 1

            except Exception as e:
                tab_summary["errors"] += 1
                self._log(f"[MTM] {tab_name} / {symbol}: {str(e)}", MessageType.WARNING)

        self.drive.create_tab_if_missing(output_id, tab_name)
        self.drive.write_tab(output_id, tab_name, result)

        self._log(
            f"[MTM] Tab '{tab_name}': {tab_summary['priced']} priced, "
            f"{tab_summary['errors']} errors over {tab_summary['symbols']} instruments"
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
                self._log(f"[MTM] {self.PORTFOLIO_TAB} / {symbol}: {str(e)}", MessageType.WARNING)

                rows.append([symbol, "", "", "", "", ""])

        result = pd.DataFrame(rows, columns=self.PORTFOLIO_SCHEMA)

        self.drive.create_tab_if_missing(output_id, self.PORTFOLIO_TAB)
        self.drive.write_tab(output_id, self.PORTFOLIO_TAB, result)

        self._log(
            f"[MTM] Tab '{self.PORTFOLIO_TAB}': {tab_summary['priced']} priced, "
            f"{tab_summary['errors']} errors over {tab_summary['symbols']} instruments"
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

    @staticmethod
    def _apply_schema(df, schema):
        """
        Agrega las columnas del esquema que falten, sin tocar el orden ni el
        nombre de las que ya estan y sin borrar las que el cliente sumo por su
        cuenta a la derecha.
        """
        result = df.copy()

        existing = {str(c).strip().upper() for c in result.columns}

        for col in schema:
            if col.upper() not in existing:
                result[col] = ""

        return result

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
