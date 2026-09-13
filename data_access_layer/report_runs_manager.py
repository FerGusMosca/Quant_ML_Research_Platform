import pyodbc
from typing import Optional, List

from business_entities.report_run import ReportRun
from framework.common.logger.message_type import MessageType


class ReportRunsManager:
    """
    Data Access Layer for the report_runs table.

    Same shape as TagRunManager: all access through stored procedures, lazy
    connection, id=0 --> INSERT / id!=0 --> UPDATE.
    """

    def __init__(self, connection_string: str, logger):
        self.connection_string = connection_string
        self.logger = logger
        self._connection = None

    @property
    def connection(self):
        if self._connection is None or self._connection.closed:
            self._connection = pyodbc.connect(self.connection_string)
            self._connection.autocommit = False
        return self._connection

    # -- write ----------------------------------------------------------------

    def persist_report_run(self, run: ReportRun) -> int:
        cursor = None
        try:
            cursor = self.connection.cursor()

            cursor.execute(
                """
                EXEC persist_report_run ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                """,
                (
                    run.id or 0,        # @id
                    run.job_id,         # @job_id
                    run.report_key,     # @report_key
                    run.portfolio,      # @portfolio
                    run.symbol,         # @symbol
                    run.year,           # @year
                    run.quarter,        # @quarter
                    run.source,         # @source
                    run.params_json,    # @params_json
                    run.status,         # @status
                    run.last_error,     # @last_error
                )
            )

            new_id = cursor.fetchone()[0]
            self.connection.commit()

            action = "CREATED" if (run.id or 0) == 0 else "UPDATED"
            self.logger.do_log(
                f"[REPORT_RUN] {action} | id={new_id} | report={run.report_key} | "
                f"portfolio={run.portfolio} | status={run.status}",
                MessageType.INFO
            )

            run.id = new_id
            return new_id

        except Exception as e:
            try:
                if self._connection:
                    self._connection.rollback()
            except Exception:
                pass

            self.logger.do_log(
                f"[REPORT_RUN] persist failed | report={run.report_key} | error={e}",
                MessageType.ERROR
            )
            raise

        finally:
            if cursor:
                cursor.close()

    def touch_report_run(self, run_id: int, last_error: Optional[str] = None):
        """
        Refresca la hora de la corrida sin cambiar su estado (y de paso guarda el
        ultimo error, si vino uno). Sirve para saber si sigue viva.
        """
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "EXEC persist_report_run ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?",
                (run_id, None, None, None, None, None, None, None, None, None, last_error)
            )
            cursor.fetchone()
            self.connection.commit()
        except Exception as e:
            try:
                if self._connection:
                    self._connection.rollback()
            except Exception:
                pass
            self.logger.do_log(f"[REPORT_RUN] touch failed | id={run_id} | error={e}",
                               MessageType.WARNING)
        finally:
            if cursor:
                cursor.close()

    def close_report_run(self, run_id: int, status: str, last_error: Optional[str] = None):
        """Cierra la corrida: finished, error o aborted."""
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "EXEC persist_report_run ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?",
                (run_id, None, None, None, None, None, None, None, None, status, last_error)
            )
            cursor.fetchone()
            self.connection.commit()
            self.logger.do_log(f"[REPORT_RUN] {status.upper()} | id={run_id}", MessageType.INFO)
        except Exception as e:
            try:
                if self._connection:
                    self._connection.rollback()
            except Exception:
                pass
            self.logger.do_log(f"[REPORT_RUN] close failed | id={run_id} | error={e}",
                               MessageType.WARNING)
        finally:
            if cursor:
                cursor.close()

    # -- read -----------------------------------------------------------------

    def get_report_runs(self, top: int = 50, status: Optional[str] = None,
                        report_key: Optional[str] = None) -> List[dict]:
        result = []
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute("EXEC get_report_runs ?, ?, ?", (top, status, report_key))
            cols = [d[0] for d in cursor.description]
            for row in cursor.fetchall():
                result.append(dict(zip(cols, [str(v) if v is not None else None for v in row])))
        except Exception as e:
            self.logger.do_log(f"[REPORT_RUN] get_report_runs failed | error={e}", MessageType.ERROR)
        finally:
            if cursor:
                cursor.close()
        return result

    def get_last_report_run(self, report_key: Optional[str] = None) -> Optional[dict]:
        runs = self.get_report_runs(top=1, status=None, report_key=report_key)
        return runs[0] if runs else None

    # -- housekeeping ---------------------------------------------------------

    def reset_stuck_report_runs(self, run_id: Optional[int] = None,
                                older_than_hours: Optional[int] = None) -> int:
        """
        Turns runs left in 'started' into 'aborted'. Use it after a crash or a
        restart so a truncated run does not look like it is still alive.
        """
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute("EXEC reset_stuck_report_runs ?, ?", (run_id, older_than_hours))
            row = cursor.fetchone()
            self.connection.commit()
            rows_reset = int(row[0]) if row else 0
            self.logger.do_log(f"[REPORT_RUN] reset_stuck_report_runs: {rows_reset} rows reset",
                               MessageType.INFO)
            return rows_reset
        except Exception as e:
            self.logger.do_log(f"[REPORT_RUN] reset_stuck_report_runs failed | error={e}",
                               MessageType.ERROR)
            raise
        finally:
            if cursor:
                cursor.close()

    def delete_report_run(self, run_id: int):
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute("EXEC delete_report_run ?", (run_id,))
            self.connection.commit()
            self.logger.do_log(f"[REPORT_RUN] deleted | id={run_id}", MessageType.INFO)
        except Exception as e:
            self.logger.do_log(f"[REPORT_RUN] delete failed | id={run_id} | error={e}",
                               MessageType.ERROR)
            raise
        finally:
            if cursor:
                cursor.close()
