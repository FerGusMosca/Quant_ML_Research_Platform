import os
import time
import traceback
from datetime import datetime

import pandas as pd

from common.util.downloaders.yahoo_quote_downloader import YahooQuoteDownloader
from logic_layer.report_generators.mtm.mtm_price_source_router import MTMPriceSourceRouter
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

    De donde sale el precio de cada solapa ya no se decide aca: eso lo resuelve
    el ruteador de fuentes, que tiene el mapa solapa por solapa.
    """

    # Si no viene la ruta por parametro, se usa la clave que vive adentro del
    # proyecto. El parametro siempre gana, asi que en otra maquina o en un
    # container se puede apuntar a otro lado sin tocar el codigo.
    DEFAULT_CREDENTIALS_PATH = os.path.join("static", "config", "update-ml-prices-cmd-ca0736c87831.json")

    CONTROL_TAB = "[CONTROL]"
    PORTFOLIO_TAB = "Official Portfolio"

    # Marcas que el cliente pone en el nombre de la solapa:
    #   <Nombre>  -> la lista de activos sale del portfolio con ese nombre
    #   [Nombre]  -> la solapa se ignora entera
    PORTFOLIO_TAB_OPEN = "<"
    PORTFOLIO_TAB_CLOSE = ">"
    IGNORED_TAB_OPEN = "["
    IGNORED_TAB_CLOSE = "]"

    # Encabezados de las solapas de caucion, que no van por ticker sino por
    # plazo contra el panel de BYMA.
    CAUCION_TERM_HEADERS = ["Plazo", "PLAZO"]
    CAUCION_MATURITY_HEADERS = ["Vencimiento", "VENCIMIENTO"]
    CAUCION_RATE_HEADERS = ["Ultima tasa", "Ultima Tasa", "TASA", "Tasa"]
    CAUCION_VOLUME_HEADERS = ["Volumen operado", "Volumen Operado", "VOLUMEN"]

    # Lo que queda escrito en la planilla cuando ese plazo no tuvo rueda.
    CAUCION_NO_TRADE_TEXT = "SIN OPERAR: ese plazo no opero hoy"

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

    # Columna propia que se agrega a la derecha para dejar marcado lo que no
    # responde o lo que parece estar en la solapa equivocada.
    OBSERVATION_HEADER = "Observacion"
    WRITE_OBSERVATIONS = True

    PORTFOLIO_SCHEMA = [
        "Ticker_ID",
        "Descripcion",
        "Precio_Cierre",
        "Variacion_Nominal",
        "Variacion_Porcentual",
        "Volumen_Operado",
    ]


    def __init__(self, gdrive_url, input_file, output_file, credentials_file=None, portfolio=None,
                 monitor_conn_str=None, tv_params=None, tab=None, work_folder=None, logger=None, job_id=None):

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

        # Si viene un nombre de solapa, se procesa solo esa y se dejan las
        # demas como estan. Vacio significa todas.
        self.tab = str(tab).strip() if tab is not None and str(tab).strip() != "" else None

        self.drive = GoogleDriveHandler(self.credentials_file, logger)
        self.quotes = YahooQuoteDownloader()

        # Quien decide de donde sale el precio de cada solapa.
        self.router = MTMPriceSourceRouter(
            yahoo_downloader=self.quotes,
            tv_params=self.tv_params,
            logger=self.logger,
            job_id=self.job_id,
        )

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
            "failed_tabs": [],
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

                if self._is_ignored_tab(tab_name):
                    self._log(f"[MTM][{tab_name}] marcada para ignorar. Salteada.")
                    continue

                if not self._is_wanted_tab(tab_name):
                    self._log(f"[MTM][{tab_name}] no es la solapa pedida. Salteada.")
                    continue

                # Lo que falle en una solapa no frena a las demas: se anota el
                # motivo y se sigue con la siguiente.
                try:

                    if self._is_portfolio_tab(tab_name):
                        tab_summary = self._process_portfolio_tab(
                            output_id, tab_name, self._portfolio_name_of(tab_name)
                        )

                    elif self.router.is_caucion_tab(tab_name):
                        tab_summary = self._process_caucion_tab(output_id, tab_name, df)

                    else:
                        tab_summary = self._process_market_tab(output_id, tab_name, df)

                except Exception as e:
                    summary["failed_tabs"].append(tab_name)

                    self._log(f"[MTM][{tab_name}] la solapa fallo entera: {str(e)}",
                              MessageType.ERROR)
                    self._log(traceback.format_exc(), MessageType.ERROR)

                    continue

                if tab_summary is None:
                    continue

                summary["tabs"][tab_name] = tab_summary
                summary["total_symbols"] += tab_summary["symbols"]
                summary["priced"] += tab_summary["priced"]
                summary["errors"] += tab_summary["errors"]

            message = (
                f"{summary['priced']} precios actualizados sobre "
                f"{summary['total_symbols']} instrumentos. Errores: {summary['errors']}"
            )

            if len(summary["failed_tabs"]) > 0:
                message += f". Solapas caidas: {', '.join(summary['failed_tabs'])}"

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

        observation_col = None

        if self.WRITE_OBSERVATIONS:
            observation_col = self._find_column(result, [self.OBSERVATION_HEADER])

            if observation_col is None:
                observation_col = self.OBSERVATION_HEADER
                result[observation_col] = ""

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
                price, volume, fuente, observacion = self.router.get_price_and_volume(symbol, tab_name)

                if observation_col is not None:
                    result.at[index, observation_col] = observacion

                if price is None:
                    tab_summary["errors"] += 1
                    self._log(f"[MTM][{tab_name}] {symbol}: {observacion}", MessageType.WARNING)
                else:
                    result.at[index, price_col] = price
                    result.at[index, volume_col] = volume if volume is not None else ""

                    tab_summary["priced"] += 1

                    self._log(f"[MTM][{tab_name}] {symbol}: {price} (fuente {fuente}) {observacion}")

            except Exception as e:
                tab_summary["errors"] += 1

                if observation_col is not None:
                    result.at[index, observation_col] = str(e)

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

    def _process_portfolio_tab(self, output_id, tab_name=None, portfolio_name=None):
        """
        Arma la foto del cierre de un portfolio y pisa la solapa entera. La
        lista de activos sale de la app, no de la planilla: el nombre del
        portfolio es el que viene entre los signos de mayor y menor en el
        nombre de la solapa.
        """
        tab_name = tab_name if tab_name is not None else self.PORTFOLIO_TAB

        portfolio_name = portfolio_name if portfolio_name is not None else self.portfolio

        if portfolio_name is None or self.monitor_conn_str is None:
            self._log(
                f"[MTM][{tab_name}] sin portfolio o sin conexion a la app. Salteada.",
                MessageType.WARNING,
            )
            return None

        manager = StockMonitorPortfolioManager(self.monitor_conn_str, self.logger)

        symbols = manager.get_symbols(portfolio_name)

        self._log(f"[MTM][{tab_name}] {len(symbols)} activos del portfolio '{portfolio_name}'")

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

                continue

            except Exception as e:
                primer_error = str(e)

            # Si no contesto Yahoo, se prueba con el resto de las fuentes.
            try:
                price, volume, fuente, observacion = self.router.get_price_and_volume(
                    symbol, self.PORTFOLIO_TAB
                )
            except Exception:
                price, volume = None, None

            if price is not None:
                rows.append([symbol, "", price, "", "", volume if volume is not None else ""])
                tab_summary["priced"] += 1
            else:
                tab_summary["errors"] += 1
                self._log(f"[MTM][{tab_name}] {symbol}: {primer_error}", MessageType.WARNING)
                rows.append([symbol, "", "", "", "", ""])

        result = pd.DataFrame(rows, columns=self.PORTFOLIO_SCHEMA)

        self.drive.create_tab_if_missing(output_id, tab_name)
        self.drive.write_tab(output_id, tab_name, result)

        self._log(
            f"[MTM][{tab_name}] {tab_summary['priced']} con precio, "
            f"{tab_summary['errors']} con error, sobre {tab_summary['symbols']} instrumentos"
        )

        return tab_summary

    # ==================================================================
    # Solapas de caucion
    # ==================================================================

    def _process_caucion_tab(self, output_id, tab_name, df):
        """
        Completa tasa y volumen de cada plazo contra el panel de BYMA. La
        moneda sale del tag que trae el nombre de la solapa.

        No hay ticker: cada fila se engancha por fecha de vencimiento y, si esa
        columna no esta, por cantidad de dias.
        """
        if df.empty:
            self._log(f"[MTM][{tab_name}] solapa vacia. Salteada.", MessageType.WARNING)
            return None

        rate_col = self._find_column(df, self.CAUCION_RATE_HEADERS)
        volume_col = self._find_column(df, self.CAUCION_VOLUME_HEADERS)

        if rate_col is None:
            self._log(
                f"[MTM][{tab_name}] no tiene columna de tasa. "
                f"Columnas encontradas: {list(df.columns)}. Solapa salteada.",
                MessageType.WARNING,
            )
            return None

        maturity_col = self._find_column(df, self.CAUCION_MATURITY_HEADERS)
        term_col = self._find_column(df, self.CAUCION_TERM_HEADERS)

        rows = self.router.get_caucion_rows(tab_name)

        by_maturity = {}
        by_days = {}

        for row in rows:

            if row["rate"] is None:
                continue

            fecha = self._to_date_text(row["maturity_date"])

            if fecha != "":
                by_maturity[fecha] = row

            by_days[int(row["days"])] = row

        result = df.copy()

        observation_col = None

        if self.WRITE_OBSERVATIONS:
            observation_col = self._find_column(result, [self.OBSERVATION_HEADER])

            if observation_col is None:
                observation_col = self.OBSERVATION_HEADER
                result[observation_col] = ""

        tab_summary = {"symbols": 0, "priced": 0, "errors": 0}

        for index, sheet_row in result.iterrows():

            fecha = self._to_date_text(sheet_row[maturity_col]) if maturity_col is not None else ""

            dias = self._days_from_text(sheet_row[term_col]) if term_col is not None else None

            if fecha == "" and dias is None:
                continue

            tab_summary["symbols"] += 1

            found = by_maturity.get(fecha)

            if found is None and dias is not None:
                found = by_days.get(dias)

            if found is None:
                tab_summary["errors"] += 1

                if observation_col is not None:
                    result.at[index, observation_col] = self.CAUCION_NO_TRADE_TEXT

                self._log(f"[MTM][{tab_name}] plazo {dias} ({fecha}): no opero hoy",
                          MessageType.WARNING)
                continue

            result.at[index, rate_col] = found["rate"]

            if volume_col is not None:
                result.at[index, volume_col] = found["volume"] if found["volume"] is not None else ""

            if observation_col is not None:
                result.at[index, observation_col] = ""

            tab_summary["priced"] += 1

        self.drive.create_tab_if_missing(output_id, tab_name)
        self.drive.write_tab(output_id, tab_name, result)

        self._log(
            f"[MTM][{tab_name}] {tab_summary['priced']} con tasa, "
            f"{tab_summary['errors']} sin operar, sobre {tab_summary['symbols']} plazos"
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
        """
        Es solapa de portfolio cuando el nombre viene entre los signos de mayor
        y menor, por ejemplo <Official Portfolio>.
        """
        clean = str(tab_name).strip()

        return clean.startswith(self.PORTFOLIO_TAB_OPEN) and clean.endswith(self.PORTFOLIO_TAB_CLOSE)

    def _portfolio_name_of(self, tab_name):
        """
        Saca las marcas y deja el nombre del portfolio tal como figura en la app.
        """
        clean = str(tab_name).strip()

        return clean[len(self.PORTFOLIO_TAB_OPEN):-len(self.PORTFOLIO_TAB_CLOSE)].strip()

    def _is_wanted_tab(self, tab_name):
        """
        Cuando llega el nombre de una solapa por parametro, se procesa solo esa.
        Se compara con y sin las marcas, asi vale tanto Caucion ARS como
        <Official Portfolio>.
        """
        if self.tab is None:
            return True

        wanted = self.tab.strip().upper().strip("<>[]")

        current = str(tab_name).strip().upper().strip("<>[]")

        return wanted == current

    def _is_ignored_tab(self, tab_name):
        """
        Es solapa a ignorar cuando el nombre viene entre corchetes, por ejemplo
        [Vencidos].
        """
        clean = str(tab_name).strip()

        return clean.startswith(self.IGNORED_TAB_OPEN) and clean.endswith(self.IGNORED_TAB_CLOSE)

    @staticmethod
    def _to_date_text(value):
        """
        Deja cualquier fecha en el mismo formato de texto, para poder comparar
        la planilla contra el panel de BYMA.
        """
        if value is None:
            return ""

        text = str(value).strip()

        if text == "" or text.lower() == "nan":
            return ""

        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%m/%d/%Y"]:
            try:
                return datetime.strptime(text[:len(datetime.now().strftime(fmt))], fmt).strftime("%Y-%m-%d")
            except Exception:
                continue

        return text

    @staticmethod
    def _days_from_text(value):
        """
        Lee la cantidad de dias de textos del estilo "13 DIAS".
        """
        if value is None:
            return None

        digits = ""

        for character in str(value):
            if character.isdigit():
                digits += character
            elif digits != "":
                break

        if digits == "":
            return None

        return int(digits)

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
