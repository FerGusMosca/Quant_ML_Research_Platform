# FILE: data_access_layer/vectors/vectorization_history_manager.py
# Read side of the pgvector store for the "Vectorizations" screen, plus the
# manual run register.
#
# Kept apart from FilingVectorsManager on purpose: that one is the write path
# used by the vectorization job, this one is the screen. Same database, same
# [VECTOR_DB] settings, opposite direction.
#
# Requires db/vectors/02_vectorization_history.sql to be applied.

import psycopg2

from framework.common.logger.message_type import MessageType


class VectorizationHistoryManager:

    REQUIRED_KEYS = ["VECTORS_PG_HOST", "VECTORS_PG_PORT", "VECTORS_PG_DB",
                     "VECTORS_PG_USER", "VECTORS_PG_SCHEMA"]

    def __init__(self, vectors_db_config, logger=None):
        self.logger = logger
        self.config = vectors_db_config or {}

        missing = [k for k in self.REQUIRED_KEYS if not self.config.get(k)]
        if missing:
            raise Exception(
                f"Missing Postgres settings {missing}. Check the [VECTOR_DB] section "
                f"in configs/commands_mgr.ini"
            )

        self.schema = self.config["VECTORS_PG_SCHEMA"]
        self._connection = None

    # ── Connection ────────────────────────────────────────────────────────────

    @property
    def connection(self):
        """Reconnects on demand: the screen is long lived, the socket is not."""
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(
                host=self.config["VECTORS_PG_HOST"],
                port=int(self.config["VECTORS_PG_PORT"]),
                dbname=self.config["VECTORS_PG_DB"],
                user=self.config["VECTORS_PG_USER"],
                password=self.config.get("VECTORS_PG_PWD"),
            )
            self._connection.autocommit = False
            with self._connection.cursor() as cursor:
                cursor.execute(f"SET search_path TO {self.schema}, public")
            self._connection.commit()
        return self._connection

    def close(self):
        try:
            if self._connection and not self._connection.closed:
                self._connection.close()
        except Exception:
            pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def __rows_to_dicts__(cursor):
        columns = [c[0] for c in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _log(self, message, level=None):
        if self.logger:
            self.logger.do_log(message, level or MessageType.INFO)

    def _query(self, sql, params=None):
        with self.connection.cursor() as cursor:
            cursor.execute(sql, params or ())
            return self.__rows_to_dicts__(cursor)

    def ping(self) -> str:
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT version()")
            return cursor.fetchone()[0].split(",")[0]

    # ── Reference data for the combos ─────────────────────────────────────────

    def get_symbols(self, text=None, top: int = 500):
        return self._query("""
            SELECT symbol,
                   COALESCE(sector_code, 'UNCLASSIFIED') AS sector_code,
                   COUNT(*) AS documents
              FROM filing_documents
             WHERE (%s IS NULL OR symbol ILIKE '%%' || %s || '%%')
             GROUP BY symbol, COALESCE(sector_code, 'UNCLASSIFIED')
             ORDER BY symbol
             LIMIT %s
        """, (text, text, top))

    def get_sectors(self):
        """Sectors that already have something vectorized."""
        return self._query("""
            SELECT COALESCE(sector_code, 'UNCLASSIFIED') AS sector_code,
                   COUNT(DISTINCT symbol)      AS securities,
                   COUNT(DISTINCT document_id) AS documents
              FROM filing_documents
             GROUP BY COALESCE(sector_code, 'UNCLASSIFIED')
             ORDER BY 1
        """)

    def get_report_types(self):
        """
        Report types actually present. Feeds the Type filter of #1.d instead of
        a hardcoded K10 / Q10, so a new family shows up on its own.
        """
        return self._query("""
            SELECT report_type, COUNT(*) AS documents
              FROM filing_documents
             WHERE report_type IS NOT NULL AND report_type <> ''
             GROUP BY report_type
             ORDER BY 1
        """)

    def get_years(self):
        """Fiscal years present. Feeds the Year filter of #1.d."""
        return self._query("""
            SELECT fiscal_year, COUNT(*) AS documents
              FROM filing_documents
             WHERE fiscal_year IS NOT NULL
             GROUP BY fiscal_year
             ORDER BY 1 DESC
        """)

    def get_quarters(self):
        """
        Quarters presentes. El valor vacio es el de los K10, que son anuales
        y no tienen quarter: viaja como NONE para que el combo pueda pedirlo
        sin confundirse con "todos".
        """
        return self._query("""
            SELECT CASE WHEN COALESCE(quarter, '') = '' THEN 'NONE' ELSE quarter END AS quarter,
                   COUNT(*) AS documents
              FROM filing_documents
             GROUP BY 1
             ORDER BY 1
        """)

    def get_known_portfolios(self):
        """
        Portfolio codes already seen in this database. Used to feed the combo
        together with the SEC tags, so the value is never typed twice.
        """
        return self._query("""
            SELECT portfolio, COUNT(*) AS uses
              FROM (
                    SELECT portfolio FROM filing_documents  WHERE portfolio IS NOT NULL AND portfolio <> ''
                    UNION ALL
                    SELECT portfolio FROM vectorization_runs WHERE portfolio IS NOT NULL AND portfolio <> ''
                   ) t
             GROUP BY portfolio
             ORDER BY 2 DESC, 1
        """)

    def get_embedding_models(self):
        return self._query("""
            SELECT embedding_model, COUNT(*) AS chunks
              FROM filing_chunks
             GROUP BY embedding_model
             ORDER BY 2 DESC
        """)

    # ── Global picture ────────────────────────────────────────────────────────

    def get_totals(self, embedding_model=None, sector_code=None, symbol=None,
                   report_type=None, fiscal_year=None, quarter=None):
        """
        The headline numbers, and the answer to #1.b.

        Three counts, on purpose, because they are not the same thing and the
        old screen showed only one of them:
          - documents_registered: every file the store knows about;
          - documents:            the files that actually have vectors;
          - documents_pending:    registered minus vectorized.

        Nothing here comes from the counters the runs write. The runs count
        what one execution did; this counts what is really stored.
        """
        rows = self._query("""
            SELECT COUNT(*)                                  AS documents_registered,
                   COUNT(*) FILTER (WHERE chunks > 0)        AS documents,
                   COUNT(*) FILTER (WHERE chunks = 0)        AS documents_pending,
                   COUNT(DISTINCT symbol)                    AS securities_registered,
                   COUNT(DISTINCT symbol) FILTER (WHERE chunks > 0) AS securities,
                   COALESCE(SUM(chunks), 0)::bigint          AS chunks,
                   COALESCE(SUM(bytes), 0)::bigint           AS bytes,
                   pg_size_pretty(COALESCE(SUM(bytes), 0)::bigint) AS pretty_size
              FROM (
                    SELECT d.document_id,
                           d.symbol,
                           COUNT(c.chunk_id) AS chunks,
                           COALESCE(SUM(pg_column_size(c.embedding)), 0)::bigint AS bytes
                      FROM filing_documents d
                      LEFT JOIN filing_chunks c
                             ON c.document_id = d.document_id
                            AND (%s IS NULL OR c.embedding_model = %s)
                     WHERE (%s IS NULL OR COALESCE(d.sector_code, 'UNCLASSIFIED') = %s)
                       AND (%s IS NULL OR d.symbol = %s)
                       AND (%s IS NULL OR d.report_type = %s)
                       AND (%s IS NULL OR d.fiscal_year = %s)
                       AND (%s IS NULL OR d.quarter = %s)
                     GROUP BY d.document_id, d.symbol
                   ) t
        """, (embedding_model, embedding_model,
              sector_code, sector_code,
              symbol, symbol,
              report_type, report_type,
              fiscal_year, fiscal_year,
              quarter, quarter))
        return rows[0] if rows else {}

    def get_coverage(self, sector_code=None, report_type=None, fiscal_year=None,
                     quarter=None):
        """
        Registered vs vectorized vs pending. Abierto por tipo de archivo, anio
        y quarter, que es lo que la solapa By Sector muestra en sus columnas.
        """
        return self._query("""
            SELECT * FROM v_vectorization_coverage_detail
             WHERE (%s IS NULL OR sector_code = %s)
               AND (%s IS NULL OR report_type = %s)
               AND (%s IS NULL OR fiscal_year = %s)
               AND (%s IS NULL OR quarter = %s)
             ORDER BY documents_registered DESC
        """, (sector_code, sector_code,
              report_type, report_type,
              fiscal_year, fiscal_year,
              quarter, quarter))

    def get_sector_summary(self, embedding_model=None, report_type=None,
                           fiscal_year=None, quarter=None):
        """Una fila por sector, tipo de archivo, anio, quarter y modelo."""
        return self._query("""
            SELECT * FROM v_vectorization_by_sector_detail
             WHERE (%s IS NULL OR embedding_model = %s)
               AND (%s IS NULL OR report_type = %s)
               AND (%s IS NULL OR fiscal_year = %s)
               AND (%s IS NULL OR quarter = %s)
             ORDER BY sector_code, fiscal_year DESC, report_type, quarter
        """, (embedding_model, embedding_model,
              report_type, report_type,
              fiscal_year, fiscal_year,
              quarter, quarter))

    # ── Per security / per sector detail ──────────────────────────────────────

    def get_symbol_summary(self, symbol, embedding_model=None):
        return self._query("""
            SELECT * FROM v_vectorization_by_symbol
             WHERE symbol = %s
               AND (%s IS NULL OR embedding_model = %s)
             ORDER BY embedding_model
        """, (symbol, embedding_model, embedding_model))

    # ── Filings listing ───────────────────────────────────────────────────────
    #
    # Two sources, one signature:
    #   include_pending = False -> v_vectorization_storage,   one row per
    #                              document AND model, only what has vectors.
    #   include_pending = True  -> v_vectorization_documents, one row per
    #                              document, vectorized or not. This is what
    #                              makes the missing files visible (#1.b).

    __STORAGE_FILTER__ = """
             WHERE (%s IS NULL OR symbol = %s)
               AND (%s IS NULL OR COALESCE(sector_code, 'UNCLASSIFIED') = %s)
               AND (%s IS NULL OR embedding_model = %s)
               AND (%s IS NULL OR report_type = %s)
               AND (%s IS NULL OR fiscal_year = %s)
               AND (%s IS NULL OR quarter = %s)
    """

    __DOCUMENTS_FILTER__ = """
             WHERE (%s IS NULL OR symbol = %s)
               AND (%s IS NULL OR sector_code = %s)
               AND (%s IS NULL OR report_type = %s)
               AND (%s IS NULL OR fiscal_year = %s)
               AND (%s IS NULL OR quarter = %s)
    """

    @staticmethod
    def __storage_params__(symbol, sector_code, embedding_model, report_type,
                           fiscal_year, quarter):
        return (symbol, symbol,
                sector_code, sector_code,
                embedding_model, embedding_model,
                report_type, report_type,
                fiscal_year, fiscal_year,
                quarter, quarter)

    @staticmethod
    def __documents_params__(symbol, sector_code, report_type, fiscal_year, quarter):
        return (symbol, symbol,
                sector_code, sector_code,
                report_type, report_type,
                fiscal_year, fiscal_year,
                quarter, quarter)

    def get_storage(self, symbol=None, sector_code=None, embedding_model=None,
                    report_type=None, fiscal_year=None, quarter=None,
                    include_pending=False, top: int = 500):
        """
        The #1.b query, filterable by symbol, sector, model, type and year.
        Ordered by weight, heaviest first, which is the order the original
        query used.
        """
        if include_pending:
            return self._query(f"""
                SELECT document_id, symbol, file_name, report_type, fiscal_year,
                       quarter, sector_code, portfolio,
                       NULL::text AS embedding_model,
                       models, chunks, bytes, pretty_size,
                       first_chunk_at, last_chunk_at, vector_status
                  FROM v_vectorization_documents
                  {self.__DOCUMENTS_FILTER__}
                 ORDER BY bytes DESC, symbol, fiscal_year DESC
                 LIMIT %s
            """, self.__documents_params__(symbol, sector_code, report_type,
                                          fiscal_year, quarter) + (top,))

        return self._query(f"""
            SELECT *, 1 AS models, 'VECTORIZED' AS vector_status
              FROM v_vectorization_storage
              {self.__STORAGE_FILTER__}
             ORDER BY bytes DESC
             LIMIT %s
        """, self.__storage_params__(symbol, sector_code, embedding_model,
                                     report_type, fiscal_year, quarter) + (top,))

    def count_storage(self, symbol=None, sector_code=None, embedding_model=None,
                      report_type=None, fiscal_year=None, quarter=None,
                      include_pending=False) -> int:
        """
        How many rows the same filters really match, ignoring the LIMIT. The
        screen needs it to say "showing 1000 of 3.641" instead of quietly
        cutting the list, which is half of what #1.b is about.
        """
        if include_pending:
            rows = self._query(f"""
                SELECT COUNT(*) AS total
                  FROM v_vectorization_documents
                  {self.__DOCUMENTS_FILTER__}
            """, self.__documents_params__(symbol, sector_code, report_type,
                                          fiscal_year, quarter))
        else:
            rows = self._query(f"""
                SELECT COUNT(*) AS total
                  FROM v_vectorization_storage
                  {self.__STORAGE_FILTER__}
            """, self.__storage_params__(symbol, sector_code, embedding_model,
                                         report_type, fiscal_year, quarter))

        return int(rows[0]["total"]) if rows else 0

    # ── Runs ──────────────────────────────────────────────────────────────────

    def get_runs(self, symbol=None, sector_code=None, portfolio=None,
                 run_source=None, top: int = 300):
        """
        Run history. When a symbol is given, a run counts as related to it if
        the run declares it in symbols_csv or if it shares the sector.
        """
        return self._query("""
            SELECT r.*
              FROM vectorization_runs r
             WHERE (%s IS NULL OR COALESCE(r.sector_code, 'UNCLASSIFIED') = %s)
               AND (%s IS NULL OR r.portfolio = %s)
               AND (%s IS NULL OR r.run_source = %s)
               AND (%s IS NULL
                    OR r.symbols_csv ILIKE '%%' || %s || '%%'
                    OR r.sector_code IN (SELECT DISTINCT sector_code
                                           FROM filing_documents
                                          WHERE symbol = %s))
             ORDER BY r.started_at DESC
             LIMIT %s
        """, (sector_code, sector_code,
              portfolio, portfolio,
              run_source, run_source,
              symbol, symbol, symbol,
              top))

    def persist_manual_run(self, portfolio, sector_code, report_type, fiscal_year,
                           quarter, embedding_model, status, files_found,
                           files_processed, started_at, finished_at,
                           symbols_csv, notes, run_id=None) -> int:
        """
        Creates or updates a manually registered run. This is what lets the old
        corridas — the ones nobody logged — end up in the same history.
        """
        if run_id:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE vectorization_runs
                       SET portfolio       = %s,
                           sector_code     = %s,
                           report_type     = %s,
                           fiscal_year     = %s,
                           quarter         = %s,
                           embedding_model = %s,
                           status          = %s,
                           files_found     = %s,
                           files_processed = %s,
                           started_at      = COALESCE(%s, started_at),
                           finished_at     = %s,
                           symbols_csv     = %s,
                           notes           = %s
                     WHERE run_id = %s AND run_source = 'MANUAL'
                 RETURNING run_id
                """, (portfolio, sector_code, report_type, int(fiscal_year),
                      quarter or "", embedding_model, status,
                      int(files_found or 0), int(files_processed or 0),
                      started_at, finished_at, symbols_csv, notes, int(run_id)))
                row = cursor.fetchone()
            self.connection.commit()
            if not row:
                raise Exception(f"Run {run_id} does not exist or is not MANUAL")
            self._log(f"[VECTORIZE][HISTORY] Manual run UPDATED | run_id={run_id}")
            return row[0]

        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO vectorization_runs
                    (job_id, portfolio, sector_code, report_type, fiscal_year,
                     quarter, embedding_model, status, files_found, files_processed,
                     started_at, finished_at, run_source, symbols_csv, notes)
                VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        COALESCE(%s, now()), %s, 'MANUAL', %s, %s)
             RETURNING run_id
            """, (portfolio, sector_code, report_type, int(fiscal_year),
                  quarter or "", embedding_model, status,
                  int(files_found or 0), int(files_processed or 0),
                  started_at, finished_at, symbols_csv, notes))
            new_id = cursor.fetchone()[0]
        self.connection.commit()
        self._log(f"[VECTORIZE][HISTORY] Manual run CREATED | run_id={new_id} | "
                  f"sector={sector_code} | year={fiscal_year}")
        return new_id

    def delete_runs(self, run_ids) -> int:
        """
        Removes runs, whatever their source (#1.a). The old rule of only
        deleting MANUAL rows left every test run of the job stuck on the
        screen forever, which is exactly what has to go.

        Deleting a run never touches filing_documents or filing_chunks: the
        vectors stay where they are, only the log entry disappears.
        """
        ids = [int(run_id) for run_id in (run_ids or []) if str(run_id).strip()]
        if not ids:
            return 0

        with self.connection.cursor() as cursor:
            cursor.execute("""
                DELETE FROM vectorization_runs
                 WHERE run_id = ANY(%s)
            """, (ids,))
            deleted = cursor.rowcount
        self.connection.commit()
        self._log(f"[VECTORIZE][HISTORY] Runs DELETED | ids={ids} | deleted={deleted}")
        return deleted

    def delete_manual_run(self, run_id: int) -> int:
        """Kept so nothing that already called it breaks."""
        return self.delete_runs([run_id])
