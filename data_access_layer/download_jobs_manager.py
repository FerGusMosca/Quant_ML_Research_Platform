import pyodbc
from dataclasses import dataclass
from typing import Optional, List

from framework.common.logger.message_type import MessageType


# ── DTOs ──────────────────────────────────────────────────────────────────────

@dataclass
class DownloadJobGroupDTO:
    group_id:      int
    group_name:    str
    job_type:      str       # 'DOWNLOAD' | 'SPREAD'
    display_order: int
    is_active:     bool
    job_count:     int = 0


@dataclass
class DownloadJobDTO:
    job_id:           int
    group_id:         int
    job_type:         str    # forwarded from group — injected by the manager
    symbol:           str
    exchange:         Optional[str]   # TV exchange (None for FRED)
    output_symbol:    Optional[str]
    vendor:           str
    d_from:           str
    d_to:             Optional[str]
    interval_code:    str
    is_active:        bool
    last_status:      Optional[str] = None
    last_run_at:      Optional[str] = None
    last_finished_at: Optional[str] = None
    last_error:       Optional[str] = None


# ── Manager ───────────────────────────────────────────────────────────────────

class DownloadJobsManager:
    """
    DAL for download_job / download_job_group / download_job_log.
    All access via stored procedures.
    """

    def __init__(self, connection_string: str, logger):
        self.connection = pyodbc.connect(connection_string)
        self.logger = logger

    # ── Groups ────────────────────────────────────────────────────────────────

    def get_download_job_groups(self) -> List[DownloadJobGroupDTO]:
        result = []
        with self.connection.cursor() as cursor:
            try:
                cursor.execute("EXEC get_download_job_groups")
                for row in cursor.fetchall():
                    result.append(DownloadJobGroupDTO(
                        group_id      = row[0],
                        group_name    = row[1],
                        job_type      = row[2],
                        display_order = row[3],
                        is_active     = bool(row[4]),
                        job_count     = row[5],
                    ))
                self.logger.do_log(f"get_download_job_groups: {len(result)} groups", MessageType.INFO)
            except Exception as e:
                self.logger.do_log(f"get_download_job_groups: ❌ {e}", MessageType.ERROR)
        return result

    def persist_download_job_group(self, group_id: Optional[int], group_name: str,
                                   job_type: str, display_order: int) -> int:
        """group_id=None → INSERT, else → UPDATE. Returns persisted group_id."""
        with self.connection.cursor() as cursor:
            try:
                cursor.execute("EXEC persist_download_job_group ?, ?, ?, ?",
                               (group_id, group_name, job_type, display_order))
                row = cursor.fetchone()
                self.connection.commit()
                return int(row[0])
            except Exception as e:
                self.logger.do_log(f"persist_download_job_group: ❌ {e}", MessageType.ERROR)
                raise

    # ── Jobs ──────────────────────────────────────────────────────────────────

    def get_download_jobs(self, group_id: int, job_type: str = 'DOWNLOAD') -> List[DownloadJobDTO]:
        """
        Returns jobs for one group.
        job_type is supplied by the caller (from the group) because the SP
        doesn't join to the group table — keeps the query lean.
        """
        result = []
        with self.connection.cursor() as cursor:
            try:
                cursor.execute("EXEC get_download_jobs ?", (group_id,))
                for row in cursor.fetchall():
                    result.append(DownloadJobDTO(
                        job_id           = row[0],
                        group_id         = row[1],
                        job_type         = job_type,
                        symbol           = row[2],
                        exchange         = row[3],
                        output_symbol    = row[4],
                        vendor           = row[5],
                        d_from           = str(row[6]),
                        d_to             = str(row[7]) if row[7] else None,
                        interval_code    = row[8],
                        is_active        = bool(row[9]),
                        last_status      = row[10],
                        last_run_at      = str(row[11]) if row[11] else None,
                        last_finished_at = str(row[12]) if row[12] else None,
                        last_error       = row[13],
                    ))
            except Exception as e:
                self.logger.do_log(f"get_download_jobs: ❌ {e}", MessageType.ERROR)
        return result

    def get_all_download_jobs(self) -> List[dict]:
        """Flat list for the status tab — returns plain dicts ready for JSON."""
        result = []
        with self.connection.cursor() as cursor:
            try:
                cursor.execute("EXEC get_all_download_jobs")
                cols = [d[0] for d in cursor.description]
                for row in cursor.fetchall():
                    result.append(dict(zip(cols, [str(v) if v is not None else None for v in row])))
            except Exception as e:
                self.logger.do_log(f"get_all_download_jobs: ❌ {e}", MessageType.ERROR)
        return result

    def persist_download_job(self, job_id: Optional[int], group_id: int,
                             symbol: str, exchange: Optional[str],
                             output_symbol: Optional[str],
                             vendor: str, d_from: str, d_to: Optional[str],
                             interval_code: str) -> int:
        """job_id=None → INSERT, else → UPDATE. Returns persisted job_id."""
        with self.connection.cursor() as cursor:
            try:
                cursor.execute("EXEC persist_download_job ?, ?, ?, ?, ?, ?, ?, ?, ?",
                               (job_id, group_id, symbol, exchange, output_symbol,
                                vendor, d_from, d_to, interval_code))
                row = cursor.fetchone()
                self.connection.commit()
                return int(row[0])
            except Exception as e:
                self.logger.do_log(f"persist_download_job: ❌ {e}", MessageType.ERROR)
                raise

    def delete_download_job(self, job_id: int):
        with self.connection.cursor() as cursor:
            try:
                cursor.execute("EXEC delete_download_job ?", (job_id,))
                self.connection.commit()
            except Exception as e:
                self.logger.do_log(f"delete_download_job: ❌ {e}", MessageType.ERROR)
                raise

    # ── Log ───────────────────────────────────────────────────────────────────

    def start_download_job_log(self, job_id: int, group_id: int) -> int:
        """Open a RUNNING log row. Returns log_id."""
        with self.connection.cursor() as cursor:
            try:
                cursor.execute("EXEC persist_download_job_log ?, ?, ?, ?, ?, ?",
                               (None, job_id, group_id, None, None, None))
                row = cursor.fetchone()
                self.connection.commit()
                return int(row[0])
            except Exception as e:
                self.logger.do_log(f"start_download_job_log: ❌ {e}", MessageType.ERROR)
                raise

    def finish_download_job_log(self, log_id: int, status: str,
                                stdout_log: str, error_msg: Optional[str]):
        """Close an existing log row with final status."""
        with self.connection.cursor() as cursor:
            try:
                cursor.execute("EXEC persist_download_job_log ?, ?, ?, ?, ?, ?",
                               (log_id, None, None, status, stdout_log, error_msg))
                self.connection.commit()
            except Exception as e:
                self.logger.do_log(f"finish_download_job_log: ❌ {e}", MessageType.ERROR)
                raise

    def reset_stuck_jobs(self, job_id: Optional[int] = None) -> int:
        """
        Marks RUNNING rows as ERROR.
        job_id=None resets ALL stuck jobs, job_id=N resets only that job.
        Returns number of rows affected.
        """
        with self.connection.cursor() as cursor:
            try:
                cursor.execute("EXEC reset_download_job_log ?", (job_id,))
                row = cursor.fetchone()
                self.connection.commit()
                rows_reset = int(row[0]) if row else 0
                self.logger.do_log(f"reset_stuck_jobs: {rows_reset} rows reset (job_id={job_id})", MessageType.INFO)
                return rows_reset
            except Exception as e:
                self.logger.do_log(f"reset_stuck_jobs: ❌ {e}", MessageType.ERROR)
                raise