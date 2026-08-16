# FILE: logic_layer/vectorization_history_logic.py
# Orchestration for the "Vectorizations" screen.
#
# Two sources are combined here and nowhere else:
#   - Postgres (bias_research): what was actually vectorized and what it weighs.
#   - SQL Server (machine_learning_research): the sector and tag catalogue, so
#     the sector field of a manual run is picked from the same list the
#     SEC Securities screen shows instead of being typed free hand.
#
# The SQL Server side is optional on purpose: if that database is down the
# screen still answers with everything Postgres knows.

import traceback

from data_access_layer.report_portfolios_manager import ReportPortfoliosManager
from data_access_layer.sec_securities_metadata_manager import SECSecuritiesMetadataManager
from data_access_layer.vectors.vectorization_history_manager import VectorizationHistoryManager
from framework.common.logger.message_type import MessageType


class VectorizationHistoryLogic:

    # Fallback catalogue for the model combo. These are the same names the
    # tagger already knows in POOLING_BY_MODEL, copied here instead of imported
    # because that module pulls torch and transformers on import and this is a
    # screen. EMBEDDING_MODELS in the ini overrides the list when it is set.
    DEFAULT_MODEL_OPTIONS = [
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",
        "distilbert-base-uncased",
    ]

    # Hard ceiling for a listing. The screen used to cut at its own limit and
    # say nothing, which is how "the file count looks wrong" starts.
    MAX_TOP = 5000

    def __init__(self, config_settings: dict, logger):
        self.config = config_settings
        self.logger = logger
        self.history_mgr = VectorizationHistoryManager(config_settings, logger)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def __build_metadata_mgr__(self):
        """Built per call: pyodbc connections do not survive an idle screen."""
        return SECSecuritiesMetadataManager(self.config["ml_reports_conn_str"], self.logger)

    def __sql_server_sectors__(self):
        try:
            summary = self.__build_metadata_mgr__().get_summary()
            return summary.get("sectors") or []
        except Exception as e:
            print(traceback.format_exc())
            self.logger.do_log(f"[VECTORIZE][HISTORY] SEC sector catalogue unavailable: {e}",
                               MessageType.WARNING)
            return []

    def __clamp_top__(self, top):
        try:
            top = int(top)
        except Exception:
            top = 500
        return max(1, min(top, self.MAX_TOP))

    @staticmethod
    def __clean_year__(value):
        if value is None or str(value).strip() == "":
            return None
        return int(value)

    @staticmethod
    def __clean_quarter__(value):
        """
        Vacio = sin filtro. NONE = los archivos anuales, que en la base tienen
        el quarter en blanco. Sin esto no habria forma de pedir solo los K10.
        """
        value = (value or "").strip().upper()
        if not value:
            return None
        return "" if value == "NONE" else value

    # ── Screen bootstrap ──────────────────────────────────────────────────────

    def get_reference_data(self) -> dict:
        """Everything the combos need, in one round trip."""
        vectorized_sectors = self.history_mgr.get_sectors()
        vectorized_codes = {row["sector_code"] for row in vectorized_sectors}

        catalogue = []
        for row in self.__sql_server_sectors__():
            code = (row.get("sector_code") or row.get("sectorCode") or "").strip()
            if not code:
                continue
            catalogue.append({
                "sector_code": code,
                "securities": row.get("securities") or row.get("total") or 0,
                "vectorized": code in vectorized_codes,
            })

        # Sectors present in Postgres but absent from the catalogue still belong
        # in the list, otherwise a run could never be filtered back.
        known = {row["sector_code"] for row in catalogue}
        for row in vectorized_sectors:
            if row["sector_code"] not in known:
                catalogue.append({"sector_code": row["sector_code"],
                                  "securities": row["securities"],
                                  "vectorized": True})

        catalogue.sort(key=lambda item: item["sector_code"])

        return {
            "totals": self.history_mgr.get_totals(),
            "sectors": catalogue,
            "portfolios": self.get_portfolio_options(),
            "embedding_models": self.history_mgr.get_embedding_models(),
            "model_options": self.get_model_options(),
            "report_types": self.history_mgr.get_report_types(),
            "years": self.history_mgr.get_years(),
            "quarters": self.history_mgr.get_quarters(),
        }

    def get_portfolio_options(self):
        """
        The portfolio catalogue: dbo.report_portfolios in machine_learning_research.
        That table is what the Document Tagger reads and what the MCP commands
        expect in their "portfolio" argument.

        Whatever already appears in the vector store is added after it, so a run
        recorded against a portfolio later removed from the catalogue never
        loses its own name.
        """
        options = []

        try:
            options = ReportPortfoliosManager(self.config["ml_reports_conn_str"],
                                              self.logger).get_portfolio_codes()
        except Exception as e:
            print(traceback.format_exc())
            self.logger.do_log(f"[VECTORIZE][HISTORY] Portfolio catalogue unavailable: {e}",
                               MessageType.WARNING)

        for row in self.history_mgr.get_known_portfolios():
            if row["portfolio"] not in options:
                options.append(row["portfolio"])

        return options

    def get_model_options(self):
        """
        The list the model combo offers. Two sources, in this order:
          - EMBEDDING_MODELS from the [EMBEDDINGS] section of the ini, so the
            catalogue is edited in one place instead of inside a screen. When
            that key is absent, DEFAULT_MODEL_OPTIONS takes its place, so the
            combo is never empty on a fresh install;
          - whatever already has chunks in the vector store, so a model that
            ran before anyone updated the ini is never missing from the list.
        """
        options = [name.strip() for name in
                   (self.config.get("EMBEDDING_MODELS") or "").split(",")
                   if name.strip()]

        if not options:
            options = list(self.DEFAULT_MODEL_OPTIONS)

        for row in self.history_mgr.get_embedding_models():
            if row["embedding_model"] not in options:
                options.append(row["embedding_model"])

        return sorted(options)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_overview(self, embedding_model=None, sector_code=None, symbol=None,
                     report_type=None, fiscal_year=None, quarter=None) -> dict:
        """
        The header of the screen. Totals honour the same filters as the listing,
        so the number on top and the rows below always tell the same story.
        """
        return {
            "totals": self.history_mgr.get_totals(
                embedding_model=(embedding_model or None),
                sector_code=(sector_code or None),
                symbol=(symbol or None),
                report_type=(report_type or None),
                fiscal_year=self.__clean_year__(fiscal_year),
                quarter=self.__clean_quarter__(quarter)),
            "by_sector": self.history_mgr.get_sector_summary(
                embedding_model=(embedding_model or None),
                report_type=(report_type or None),
                fiscal_year=self.__clean_year__(fiscal_year),
                quarter=self.__clean_quarter__(quarter)),
            "coverage": self.history_mgr.get_coverage(
                sector_code=(sector_code or None),
                report_type=(report_type or None),
                fiscal_year=self.__clean_year__(fiscal_year),
                quarter=self.__clean_quarter__(quarter)),
        }

    def search_symbols(self, text=None, top: int = 500):
        return self.history_mgr.get_symbols(text=text, top=self.__clamp_top__(top))

    def get_symbol_detail(self, symbol: str, embedding_model=None, report_type=None,
                          fiscal_year=None, quarter=None, include_pending=False,
                          top: int = 1000) -> dict:
        symbol = (symbol or "").strip().upper()
        if not symbol:
            raise Exception("symbol is empty")

        return {
            "symbol": symbol,
            "summary": self.history_mgr.get_symbol_summary(symbol, embedding_model or None),
            "documents": self.get_storage(symbol=symbol,
                                          embedding_model=embedding_model,
                                          report_type=report_type,
                                          fiscal_year=fiscal_year,
                                          quarter=quarter,
                                          include_pending=include_pending,
                                          top=top),
            "total": self.count_storage(symbol=symbol,
                                        embedding_model=embedding_model,
                                        report_type=report_type,
                                        fiscal_year=fiscal_year,
                                        quarter=quarter,
                                        include_pending=include_pending),
            "runs": self.history_mgr.get_runs(symbol=symbol),
        }

    def get_sector_detail(self, sector_code: str, embedding_model=None, report_type=None,
                          fiscal_year=None, quarter=None, include_pending=False,
                          top: int = 1000) -> dict:
        sector_code = (sector_code or "").strip().upper()
        if not sector_code:
            raise Exception("sector_code is empty")

        return {
            "sector_code": sector_code,
            "documents": self.get_storage(sector_code=sector_code,
                                          embedding_model=embedding_model,
                                          report_type=report_type,
                                          fiscal_year=fiscal_year,
                                          quarter=quarter,
                                          include_pending=include_pending,
                                          top=top),
            "total": self.count_storage(sector_code=sector_code,
                                        embedding_model=embedding_model,
                                        report_type=report_type,
                                        fiscal_year=fiscal_year,
                                        quarter=quarter,
                                        include_pending=include_pending),
            "coverage": self.history_mgr.get_coverage(
                sector_code=sector_code,
                report_type=(report_type or None),
                fiscal_year=self.__clean_year__(fiscal_year),
                quarter=self.__clean_quarter__(quarter)),
            "runs": self.history_mgr.get_runs(sector_code=sector_code),
        }

    def get_storage(self, symbol=None, sector_code=None, embedding_model=None,
                    report_type=None, fiscal_year=None, quarter=None,
                    include_pending=False, top: int = 500):
        return self.history_mgr.get_storage(
            symbol=(symbol or None),
            sector_code=(sector_code or None),
            embedding_model=(embedding_model or None),
            report_type=(report_type or None),
            fiscal_year=self.__clean_year__(fiscal_year),
            quarter=self.__clean_quarter__(quarter),
            include_pending=bool(include_pending),
            top=self.__clamp_top__(top))

    def count_storage(self, symbol=None, sector_code=None, embedding_model=None,
                      report_type=None, fiscal_year=None, quarter=None,
                      include_pending=False) -> int:
        return self.history_mgr.count_storage(
            symbol=(symbol or None),
            sector_code=(sector_code or None),
            embedding_model=(embedding_model or None),
            report_type=(report_type or None),
            fiscal_year=self.__clean_year__(fiscal_year),
            quarter=self.__clean_quarter__(quarter),
            include_pending=bool(include_pending))

    def get_runs(self, symbol=None, sector_code=None, portfolio=None,
                 run_source=None, top: int = 300):
        """
        Run history. The progress columns need the events view; when that script
        was not applied yet the join fails, and the history is still worth
        showing without it.
        """
        try:
            return self.history_mgr.get_runs(symbol=(symbol or None),
                                             sector_code=(sector_code or None),
                                             portfolio=(portfolio or None),
                                             run_source=(run_source or None),
                                             top=self.__clamp_top__(top))
        except Exception as e:
            self.logger.do_log(f"[VECTORIZE] historial sin progreso en vivo: {e}",
                               MessageType.WARNING)
            self.history_mgr.close()
            return self.history_mgr.get_runs_basic(symbol=(symbol or None),
                                                   sector_code=(sector_code or None),
                                                   portfolio=(portfolio or None),
                                                   run_source=(run_source or None),
                                                   top=self.__clamp_top__(top))

    # ── Round robin log (#II.1) ───────────────────────────────────────────────

    def get_run_events(self, run_id=None, sector_code=None, symbol=None,
                       event_type=None, top: int = 200) -> dict:
        """
        What the vectorization is doing right now. Returns available=False when
        the events table was never created, so the screen can say "corré el
        script 04" instead of showing an empty panel that looks like a bug.
        """
        try:
            if not self.history_mgr.events_table_exists():
                return {"available": False, "items": []}

            return {"available": True,
                    "items": self.history_mgr.get_run_events(
                        run_id=(run_id or None),
                        sector_code=(sector_code or None),
                        symbol=(symbol or None),
                        event_type=(event_type or None),
                        top=self.__clamp_top__(top))}
        except Exception as e:
            print(traceback.format_exc())
            self.logger.do_log(f"[VECTORIZE][EVENTS] no se pudo leer el log: {e}",
                               MessageType.WARNING)
            return {"available": False, "items": []}

    # ── Manual register ───────────────────────────────────────────────────────

    VALID_STATUS = ("STARTED", "FINISHED", "ERROR")
    VALID_REPORT_TYPES = ("K10", "Q10")

    def persist_manual_run(self, payload: dict) -> int:
        report_type = (payload.get("report_type") or "").strip().upper()
        if report_type not in self.VALID_REPORT_TYPES:
            raise Exception(f"report_type must be one of {self.VALID_REPORT_TYPES}")

        status = (payload.get("status") or "FINISHED").strip().upper()
        if status not in self.VALID_STATUS:
            raise Exception(f"status must be one of {self.VALID_STATUS}")

        fiscal_year = payload.get("fiscal_year")
        if not fiscal_year:
            raise Exception("fiscal_year is required")

        embedding_model = (payload.get("embedding_model") or "").strip()
        if not embedding_model:
            raise Exception("embedding_model is required")

        symbols_csv = (payload.get("symbols_csv") or "").strip() or None
        if symbols_csv:
            symbols_csv = ",".join(s.strip().upper()
                                   for s in symbols_csv.replace(";", ",").split(",")
                                   if s.strip())

        return self.history_mgr.persist_manual_run(
            portfolio=(payload.get("portfolio") or "").strip() or None,
            sector_code=(payload.get("sector_code") or "").strip().upper() or None,
            report_type=report_type,
            fiscal_year=int(fiscal_year),
            quarter=(payload.get("quarter") or "").strip().upper(),
            embedding_model=embedding_model,
            status=status,
            files_found=payload.get("files_found") or 0,
            files_processed=payload.get("files_processed") or 0,
            started_at=(payload.get("started_at") or None),
            finished_at=(payload.get("finished_at") or None),
            symbols_csv=symbols_csv,
            notes=(payload.get("notes") or "").strip() or None,
            run_id=payload.get("run_id") or None)

    def delete_runs(self, run_ids) -> int:
        """
        Point #1.a: any run can be removed, manual or written by the job. Test
        runs and half finished executions are noise on the screen and there is
        no reason to keep them.
        """
        if run_ids is None:
            raise Exception("run_ids is required")

        if not isinstance(run_ids, (list, tuple, set)):
            run_ids = [run_ids]

        cleaned = []
        for run_id in run_ids:
            try:
                cleaned.append(int(run_id))
            except Exception:
                raise Exception(f"'{run_id}' is not a valid run_id")

        if not cleaned:
            raise Exception("No run_id was received")

        return self.history_mgr.delete_runs(cleaned)

    def delete_manual_run(self, run_id: int) -> int:
        """Kept so nothing that already called it breaks."""
        return self.delete_runs([run_id])
