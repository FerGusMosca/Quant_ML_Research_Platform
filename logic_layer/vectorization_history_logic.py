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

    def get_overview(self, embedding_model=None) -> dict:
        return {
            "totals": self.history_mgr.get_totals(),
            "by_sector": self.history_mgr.get_sector_summary(embedding_model),
        }

    def search_symbols(self, text=None, top: int = 500):
        return self.history_mgr.get_symbols(text=text, top=top)

    def get_symbol_detail(self, symbol: str, embedding_model=None) -> dict:
        symbol = (symbol or "").strip().upper()
        if not symbol:
            raise Exception("symbol is empty")

        return {
            "symbol": symbol,
            "summary": self.history_mgr.get_symbol_summary(symbol, embedding_model),
            "documents": self.history_mgr.get_storage(symbol=symbol,
                                                      embedding_model=embedding_model),
            "runs": self.history_mgr.get_runs(symbol=symbol),
        }

    def get_sector_detail(self, sector_code: str, embedding_model=None) -> dict:
        sector_code = (sector_code or "").strip().upper()
        if not sector_code:
            raise Exception("sector_code is empty")

        return {
            "sector_code": sector_code,
            "documents": self.history_mgr.get_storage(sector_code=sector_code,
                                                      embedding_model=embedding_model,
                                                      top=1000),
            "runs": self.history_mgr.get_runs(sector_code=sector_code),
        }

    def get_storage(self, symbol=None, sector_code=None, embedding_model=None,
                    report_type=None, fiscal_year=None, top: int = 500):
        return self.history_mgr.get_storage(
            symbol=(symbol or None),
            sector_code=(sector_code or None),
            embedding_model=(embedding_model or None),
            report_type=(report_type or None),
            fiscal_year=(int(fiscal_year) if fiscal_year else None),
            top=top)

    def get_runs(self, symbol=None, sector_code=None, portfolio=None,
                 run_source=None, top: int = 300):
        return self.history_mgr.get_runs(symbol=(symbol or None),
                                         sector_code=(sector_code or None),
                                         portfolio=(portfolio or None),
                                         run_source=(run_source or None),
                                         top=top)

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

    def delete_manual_run(self, run_id: int) -> int:
        return self.history_mgr.delete_manual_run(run_id)
