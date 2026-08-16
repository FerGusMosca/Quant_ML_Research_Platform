# FILE: data_access_layer/vectors/vectorization_events_manager.py
# Write side of the round robin log of a vectorization run (#II.1).
#
# Kept apart from FilingVectorsManager because it has the opposite priority:
# that one must never lose a vector, this one must never slow the run down or
# break it. Everything here is best effort — if the log fails, the run keeps
# going and only the visibility is lost.
#
# Round robin, applied on every flush:
#   1) whatever is older than today is deleted;
#   2) only the newest MAX_RECORDS rows survive.
# The table is a window on what is happening now, not a history. The history
# of runs stays in vectorization_runs.
#
# Requires db/vectors/04_vectorization_events.sql to be applied.

import psycopg2
from psycopg2.extras import execute_values

from framework.common.logger.message_type import MessageType


class VectorizationEventsManager:

    # Not big on purpose: this is a window, not an archive. Overridable from
    # the [VECTOR_DB] section of configs/commands_mgr.ini with
    # VECTORS_EVENTS_MAX_RECORDS.
    DEFAULT_MAX_RECORDS = 500

    # Events are buffered and written in batches. One insert per file would add
    # a round trip to the VPS for every document, and a document takes seconds
    # to encode but the run walks thousands of them.
    FLUSH_EVERY = 10

    REQUIRED_KEYS = ["VECTORS_PG_HOST", "VECTORS_PG_PORT", "VECTORS_PG_DB",
                     "VECTORS_PG_USER", "VECTORS_PG_SCHEMA"]

    COLUMNS = ("run_id", "job_id", "event_type", "sector_code", "portfolio",
               "symbol", "file_name", "report_type", "fiscal_year", "quarter",
               "position", "total", "chunks", "elapsed_sec", "message")

    def __init__(self, vectors_db_config, logger=None):
        self.logger = logger
        self.config = vectors_db_config or {}
        self.enabled = all(self.config.get(k) for k in self.REQUIRED_KEYS)

        self.schema = self.config.get("VECTORS_PG_SCHEMA")
        self.max_records = self.__read_max_records__()

        self._connection = None
        self._buffer = []
        self._since_prune = 0

        # Context shared by every event of the run, so the caller does not have
        # to repeat it on each call.
        self._context = {}

    def __read_max_records__(self) -> int:
        try:
            value = int(self.config.get("VECTORS_EVENTS_MAX_RECORDS")
                        or self.DEFAULT_MAX_RECORDS)
            return max(50, value)
        except Exception:
            return self.DEFAULT_MAX_RECORDS

    # ── Connection ────────────────────────────────────────────────────────────

    @property
    def connection(self):
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
            self.flush()
        except Exception:
            pass
        try:
            if self._connection and not self._connection.closed:
                self._connection.close()
        except Exception:
            pass

    def _log(self, message, level=None):
        if self.logger:
            self.logger.do_log(message, level or MessageType.INFO)

    # ── Context ───────────────────────────────────────────────────────────────

    def set_context(self, run_id=None, job_id=None, sector_code=None, portfolio=None,
                    report_type=None, fiscal_year=None, quarter=None, total=None):
        """
        Fields every event of this run shares. Called once when the run starts;
        each event only carries what changes (the file, the position, the result).
        """
        self._context = {
            "run_id": run_id,
            "job_id": str(job_id) if job_id else None,
            "sector_code": sector_code,
            "portfolio": portfolio,
            "report_type": report_type,
            "fiscal_year": int(fiscal_year) if fiscal_year else None,
            "quarter": quarter or "",
            "total": total,
        }

    # ── Writing ───────────────────────────────────────────────────────────────

    def log_event(self, event_type, symbol=None, file_name=None, position=None,
                  chunks=None, elapsed_sec=None, message=None, total=None,
                  report_type=None, flush=False):
        """
        Queues one event. Never raises: a failure here must not take a run down
        that has been going for hours.
        """
        if not self.enabled:
            return

        try:
            ctx = self._context
            self._buffer.append((
                ctx.get("run_id"),
                ctx.get("job_id"),
                event_type,
                ctx.get("sector_code"),
                ctx.get("portfolio"),
                symbol,
                file_name,
                report_type or ctx.get("report_type"),
                ctx.get("fiscal_year"),
                ctx.get("quarter"),
                position,
                total if total is not None else ctx.get("total"),
                chunks,
                round(float(elapsed_sec), 2) if elapsed_sec is not None else None,
                (message or "")[:500] or None,
            ))

            if flush or len(self._buffer) >= self.FLUSH_EVERY:
                self.flush()
        except Exception as e:
            self._log(f"[VECTORIZE][EVENTS] ⚠ could not queue event: {e}",
                      MessageType.WARNING)

    def flush(self):
        """Writes the buffer and applies the round robin. Best effort."""
        if not self.enabled or not self._buffer:
            return

        rows, self._buffer = self._buffer, []

        try:
            with self.connection.cursor() as cursor:
                execute_values(cursor, f"""
                    INSERT INTO vectorization_run_events ({", ".join(self.COLUMNS)})
                    VALUES %s
                """, rows)
            self.connection.commit()

            self._since_prune += len(rows)
            if self._since_prune >= self.FLUSH_EVERY * 5:
                self.prune()
        except Exception as e:
            try:
                self.connection.rollback()
            except Exception:
                pass
            self._log(f"[VECTORIZE][EVENTS] ⚠ could not write {len(rows)} events: {e}",
                      MessageType.WARNING)

    def prune(self):
        """
        The round robin itself: yesterday's rows go, and only the newest
        max_records survive. Cheap enough to run every few flushes.
        """
        if not self.enabled:
            return

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM vectorization_run_events
                     WHERE log_date < CURRENT_DATE
                """)
                by_date = cursor.rowcount

                cursor.execute("""
                    DELETE FROM vectorization_run_events
                     WHERE event_id < (
                           SELECT MIN(event_id)
                             FROM (SELECT event_id
                                     FROM vectorization_run_events
                                    ORDER BY event_id DESC
                                    LIMIT %s) t)
                """, (self.max_records,))
                by_size = cursor.rowcount

            self.connection.commit()
            self._since_prune = 0

            if by_date or by_size:
                self._log(f"[VECTORIZE][EVENTS] round robin | borrados por fecha={by_date} | "
                          f"borrados por tamano={by_size} | maximo={self.max_records}")
        except Exception as e:
            try:
                self.connection.rollback()
            except Exception:
                pass
            self._log(f"[VECTORIZE][EVENTS] ⚠ round robin failed: {e}", MessageType.WARNING)
