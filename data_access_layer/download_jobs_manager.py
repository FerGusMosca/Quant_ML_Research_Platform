import pyodbc
from dataclasses import dataclass
from datetime import date, datetime
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
    job_type:         str
    symbol:           str
    exchange:         Optional[str]
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


@dataclass
class LastCandleDTO:
    symbol:     str
    last_date:  Optional[str]
    last_close: Optional[float]
    days_ago:   Optional[int]   # computed in Python


@dataclass
class ManualCandleDTO:
    symbol: str
    date:   str
    value:  float


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
        with self.connection.cursor() as cursor:
            try:
                cursor.execute("EXEC persist_download_job_log ?, ?, ?, ?, ?, ?",
                               (log_id, None, None, status, stdout_log, error_msg))
                self.connection.commit()
            except Exception as e:
                self.logger.do_log(f"finish_download_job_log: ❌ {e}", MessageType.ERROR)
                raise

    def reset_stuck_jobs(self, job_id: Optional[int] = None) -> int:
        with self.connection.cursor() as cursor:
            try:
                cursor.execute("EXEC reset_download_job_log ?", (job_id,))
                row = cursor.fetchone()
                self.connection.commit()
                rows_reset = int(row[0]) if row else 0
                self.logger.do_log(f"reset_stuck_jobs: {rows_reset} rows reset", MessageType.INFO)
                return rows_reset
            except Exception as e:
                self.logger.do_log(f"reset_stuck_jobs: ❌ {e}", MessageType.ERROR)
                raise


# ── CandleManager ─────────────────────────────────────────────────────────────
# Uses hist_data_conn_str → SecuritiesHistoricalData_Light

class CandleManager:
    """
    Reads/writes candles from SecuritiesHistoricalData_Light.
    Uses existing PersistCandle and GetCandles SPs plus new helpers.
    """

    def __init__(self, connection_string: str, logger):
        self.connection = pyodbc.connect(connection_string)
        self.logger = logger

    def get_last_candle_per_symbol(self) -> List[LastCandleDTO]:
        """
        Returns the most recent date+close for every symbol in the candles table.
        Computes days_ago in Python (today - last_date).
        """
        result = []
        today = date.today()
        with self.connection.cursor() as cursor:
            try:
                cursor.execute("EXEC GetLastCandlePerSymbol")
                for row in cursor.fetchall():
                    symbol    = str(row[0])
                    last_date = row[1]
                    last_close= float(row[2]) if row[2] is not None else None
                    if last_date:
                        d = last_date.date() if hasattr(last_date, 'date') else datetime.strptime(str(last_date)[:10], "%Y-%m-%d").date()
                        days_ago = (today - d).days
                    else:
                        days_ago = None
                    result.append(LastCandleDTO(
                        symbol     = symbol,
                        last_date  = str(last_date)[:10] if last_date else None,
                        last_close = last_close,
                        days_ago   = days_ago,
                    ))
            except Exception as e:
                self.logger.do_log(f"GetLastCandlePerSymbol: ❌ {e}", MessageType.ERROR)
        return result

    def get_recent_candles(self, symbol: str, top: int = 5) -> List[ManualCandleDTO]:
        """Returns last `top` candles for a symbol — used by MANUAL_VARIABLE popup."""
        result = []
        with self.connection.cursor() as cursor:
            try:
                cursor.execute("EXEC GetRecentCandles ?, ?", (symbol, top))
                for row in cursor.fetchall():
                    result.append(ManualCandleDTO(
                        symbol = str(row[0]),
                        date   = str(row[1])[:10],
                        value  = float(row[2]) if row[2] is not None else 0.0,
                    ))
            except Exception as e:
                self.logger.do_log(f"GetRecentCandles: ❌ {e}", MessageType.ERROR)
        return result

    def persist_manual_candle(self, symbol: str, candle_date: str, value: float):
        """
        Upserts a manual value calling PersistCandle directly.
        Stores the same value in open/high/low/close/trade. No volume.
        """
        with self.connection.cursor() as cursor:
            try:
                params = (symbol, candle_date, '1d', value, value, value, value, None, None, None)
                cursor.execute("{CALL PersistCandle (?,?,?,?,?,?,?,?,?,?)}", params)
                self.connection.commit()
                self.logger.do_log(f"persist_manual_candle: {symbol} @ {candle_date} = {value}", MessageType.INFO)
            except Exception as e:
                self.logger.do_log(f"persist_manual_candle: ❌ {e}", MessageType.ERROR)
                raise