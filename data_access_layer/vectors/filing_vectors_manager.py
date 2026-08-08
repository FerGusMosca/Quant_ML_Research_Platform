# FILE: data_access_layer/vectors/filing_vectors_manager.py
# Data Access Layer for the pgvector store (PostgreSQL 16, schema bias_research).
#
# Kept apart from the SQL Server managers on purpose: this one talks to Postgres
# through psycopg2, while everything else in the project uses pyodbc.
#
# Credentials never live in this file. The caller passes the [VECTOR_DB] values
# read from configs/commands_mgr.ini; assembling them into a psycopg2 DSN is this
# layer's job, because this is the only layer that knows the driver.

import json

import psycopg2
from psycopg2.extras import execute_values

from framework.common.logger.message_type import MessageType


class FilingVectorsManager:

    EMBEDDING_DIM = 768          # matches vector(768) in the schema
    INSERT_PAGE_SIZE = 200       # rows per round trip when bulk inserting chunks

    # The password is deliberately out of this list: a trust/peer authenticated
    # server takes an empty one, and a missing password should fail at connect
    # time with the server's own message, not here.
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

        self.connection = psycopg2.connect(
            host=self.config["VECTORS_PG_HOST"],
            port=int(self.config["VECTORS_PG_PORT"]),
            dbname=self.config["VECTORS_PG_DB"],
            user=self.config["VECTORS_PG_USER"],
            password=self.config.get("VECTORS_PG_PWD"),
        )
        self.connection.autocommit = False

        with self.connection.cursor() as cursor:
            # public stays on the path: the vector type and the <=> operator live there
            cursor.execute(f"SET search_path TO {self.schema}, public")
        self.connection.commit()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def to_vector_literal(values) -> str:
        """pgvector accepts the '[0.1,0.2,...]' text form, so no extra driver is needed."""
        return "[" + ",".join(f"{float(v):.7f}" for v in values) + "]"

    @staticmethod
    def __rows_to_dicts__(cursor):
        columns = [c[0] for c in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _log(self, message, job_id=None, level=None):
        if self.logger:
            self.logger.do_log(message, level or MessageType.INFO, job_id)

    def close(self):
        try:
            self.connection.close()
        except Exception:
            pass

    # ── Health check ──────────────────────────────────────────────────────────

    def ping(self) -> str:
        """Confirms the connection and that pgvector is actually enabled."""
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT version(), extversion FROM pg_extension WHERE extname = 'vector'")
            row = cursor.fetchone()
        return f"{row[0].split(',')[0]} | pgvector={row[1]}" if row else "vector extension NOT installed"

    # ── Documents ─────────────────────────────────────────────────────────────

    def upsert_document(self, symbol, cik, report_type, fiscal_year, quarter,
                        portfolio, sector_code, source_folder, file_name,
                        file_path, content_hash) -> int:
        """Creates or refreshes the document row and returns its id."""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO filing_documents
                    (symbol, cik, report_type, fiscal_year, quarter, portfolio,
                     sector_code, source_folder, file_name, file_path, content_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, report_type, fiscal_year, quarter, file_name)
                DO UPDATE SET portfolio     = EXCLUDED.portfolio,
                              sector_code   = EXCLUDED.sector_code,
                              source_folder = EXCLUDED.source_folder,
                              file_path     = EXCLUDED.file_path,
                              content_hash  = EXCLUDED.content_hash,
                              cik           = EXCLUDED.cik,
                              updated_at    = now()
                RETURNING document_id
            """, (symbol, cik, report_type, int(fiscal_year), quarter or "", portfolio,
                  sector_code, source_folder, file_name, file_path, content_hash))
            document_id = cursor.fetchone()[0]
        self.connection.commit()
        return document_id

    def set_section_count(self, document_id: int, section_count: int):
        with self.connection.cursor() as cursor:
            cursor.execute("""
                UPDATE filing_documents
                   SET section_count = %s, updated_at = now()
                 WHERE document_id = %s
            """, (section_count, document_id))
        self.connection.commit()

    def is_already_vectorized(self, symbol, report_type, fiscal_year, quarter,
                              file_name, embedding_model, content_hash=None) -> bool:
        """
        Skip rule for a resumable run: the file already has chunks for this model
        and, when a hash is supplied, the file has not changed since.
        """
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(c.chunk_id), MAX(d.content_hash)
                  FROM filing_documents d
                  JOIN filing_chunks c ON c.document_id = d.document_id
                                      AND c.embedding_model = %s
                 WHERE d.symbol = %s AND d.report_type = %s
                   AND d.fiscal_year = %s AND d.quarter = %s AND d.file_name = %s
            """, (embedding_model, symbol, report_type, int(fiscal_year),
                  quarter or "", file_name))
            count, stored_hash = cursor.fetchone()

        if not count:
            return False
        if content_hash and stored_hash and stored_hash != content_hash:
            return False
        return True

    # ── Chunks ────────────────────────────────────────────────────────────────

    def delete_chunks(self, document_id: int, embedding_model: str) -> int:
        """Clears a previous vectorization of the same document with the same model."""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                DELETE FROM filing_chunks
                 WHERE document_id = %s AND embedding_model = %s
            """, (document_id, embedding_model))
            deleted = cursor.rowcount
        self.connection.commit()
        return deleted

    def persist_chunks(self, document_id: int, embedding_model: str, chunks: list,
                       job_id=None) -> int:
        """
        chunks: list of dicts with section_label, chunk_index, chunk_text,
        word_count and embedding (an iterable of floats).
        """
        if not chunks:
            return 0

        rows = []
        for chunk in chunks:
            embedding = chunk["embedding"]
            if len(embedding) != self.EMBEDDING_DIM:
                raise Exception(
                    f"Embedding dimension mismatch: got {len(embedding)}, "
                    f"the schema stores vector({self.EMBEDDING_DIM}). "
                    f"Model '{embedding_model}' does not fit this table."
                )
            rows.append((
                document_id,
                embedding_model,
                chunk["section_label"][:60],
                int(chunk["chunk_index"]),
                chunk["chunk_text"],
                int(chunk.get("word_count", 0)),
                self.to_vector_literal(embedding),
            ))

        with self.connection.cursor() as cursor:
            execute_values(cursor, """
                INSERT INTO filing_chunks
                    (document_id, embedding_model, section_label, chunk_index,
                     chunk_text, word_count, embedding)
                VALUES %s
                ON CONFLICT (document_id, embedding_model, chunk_index)
                DO UPDATE SET chunk_text    = EXCLUDED.chunk_text,
                              section_label = EXCLUDED.section_label,
                              word_count    = EXCLUDED.word_count,
                              embedding     = EXCLUDED.embedding
            """, rows, page_size=self.INSERT_PAGE_SIZE)
        self.connection.commit()

        self._log(f"[VECTORIZE][DB] Persisted {len(rows)} chunks | document_id={document_id}", job_id)
        return len(rows)

    # ── Runs ──────────────────────────────────────────────────────────────────

    def start_run(self, job_id, portfolio, sector_code, report_type,
                  fiscal_year, quarter, embedding_model, files_found) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO vectorization_runs
                    (job_id, portfolio, sector_code, report_type, fiscal_year,
                     quarter, embedding_model, files_found, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'STARTED')
                RETURNING run_id
            """, (str(job_id) if job_id else None, portfolio, sector_code, report_type,
                  int(fiscal_year), quarter or "", embedding_model, files_found))
            run_id = cursor.fetchone()[0]
        self.connection.commit()
        return run_id

    def finish_run(self, run_id, files_processed, files_skipped, files_failed,
                   chunks_persisted, status="FINISHED", error_message=None):
        with self.connection.cursor() as cursor:
            cursor.execute("""
                UPDATE vectorization_runs
                   SET files_processed  = %s,
                       files_skipped    = %s,
                       files_failed     = %s,
                       chunks_persisted = %s,
                       status           = %s,
                       error_message    = %s,
                       finished_at      = now()
                 WHERE run_id = %s
            """, (files_processed, files_skipped, files_failed, chunks_persisted,
                  status, error_message, run_id))
        self.connection.commit()

    # ── Read side (semantic search) ───────────────────────────────────────────

    def search_similar(self, query_embedding, embedding_model, top_k=10,
                       section_label=None, sector_code=None, report_type=None,
                       fiscal_year=None, symbol=None):
        """
        Nearest chunks by cosine distance. Filters are optional and additive,
        which is what makes the sector segmentation useful at query time.
        """
        filters = ["embedding_model = %s"]
        params = [embedding_model]

        if section_label:
            filters.append("section_label = %s")
            params.append(section_label)
        if sector_code:
            filters.append("sector_code = %s")
            params.append(sector_code)
        if report_type:
            filters.append("report_type = %s")
            params.append(report_type)
        if fiscal_year:
            filters.append("fiscal_year = %s")
            params.append(int(fiscal_year))
        if symbol:
            filters.append("symbol = %s")
            params.append(symbol)

        where = " AND ".join(filters)
        vector_literal = self.to_vector_literal(query_embedding)

        with self.connection.cursor() as cursor:
            cursor.execute(f"""
                SELECT symbol, report_type, fiscal_year, quarter, section_label,
                       chunk_index, chunk_text,
                       1 - (embedding <=> %s::vector) AS similarity
                  FROM v_filing_chunks
                 WHERE {where}
                 ORDER BY embedding <=> %s::vector
                 LIMIT %s
            """, [vector_literal] + params + [vector_literal, top_k])
            return self.__rows_to_dicts__(cursor)

    def get_document_chunks(self, symbol, report_type, fiscal_year, quarter,
                            file_name, embedding_model):
        """
        Returns the stored chunks of one filing, in chunk order, each with its
        embedding already parsed into a list of floats.
        This is what lets tagging run without re-encoding anything.
        """
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT c.section_label, c.chunk_index, c.chunk_text,
                       c.word_count, c.embedding::text AS embedding
                  FROM filing_documents d
                  JOIN filing_chunks c ON c.document_id = d.document_id
                                      AND c.embedding_model = %s
                 WHERE d.symbol = %s AND d.report_type = %s
                   AND d.fiscal_year = %s AND d.quarter = %s AND d.file_name = %s
                 ORDER BY c.chunk_index
            """, (embedding_model, symbol, report_type, int(fiscal_year),
                  quarter or "", file_name))
            rows = self.__rows_to_dicts__(cursor)

        for row in rows:
            # pgvector hands the value back as '[0.1,0.2,...]', which is valid JSON
            row["embedding"] = json.loads(row["embedding"])

        return rows

    def get_coverage(self, embedding_model=None):
        """Row per model / report type / year with document and chunk counts."""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT c.embedding_model,
                       d.report_type,
                       d.fiscal_year,
                       d.quarter,
                       COUNT(DISTINCT d.document_id) AS documents,
                       COUNT(c.chunk_id)             AS chunks
                  FROM filing_documents d
                  JOIN filing_chunks c ON c.document_id = d.document_id
                 WHERE (%s IS NULL OR c.embedding_model = %s)
                 GROUP BY c.embedding_model, d.report_type, d.fiscal_year, d.quarter
                 ORDER BY d.fiscal_year DESC, d.report_type, d.quarter
            """, (embedding_model, embedding_model))
            return self.__rows_to_dicts__(cursor)
