import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any


@dataclass(frozen=False)
class ReportRun:
    """
    Entity/DTO representing a single report execution launched through the MCP
    server (or through the Reports Runner screen).

    It only tracks what matters to know, one day later, whether a run really
    finished: when it started, when it ended and how it ended.
    """

    _STARTED = "started"
    _FINISHED = "finished"
    _ERROR = "error"
    _ABORTED = "aborted"

    id: Optional[int] = None
    job_id: Optional[str] = None
    report_key: str = ""
    portfolio: Optional[str] = None
    symbol: Optional[str] = None
    year: Optional[str] = None
    quarter: Optional[str] = None
    source: Optional[str] = None
    params_json: Optional[str] = None
    status: str = "started"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    last_error: Optional[str] = None
    last_update_time: Optional[datetime] = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()
        if self.last_update_time is None:
            self.last_update_time = datetime.now()

    @staticmethod
    def initialize_report_run(report_key, job_id=None, portfolio=None, symbol=None,
                              year=None, quarter=None, source=None, extra_params=None):
        """
        Builds a brand new run in 'started' state (id=0 --> INSERT).
        `extra_params` is any dict with the remaining arguments of the call, so
        nothing gets lost even for reports with exotic parameters.
        """
        now = datetime.now()

        params_json = None
        if extra_params:
            try:
                params_json = json.dumps(extra_params, default=str)
            except Exception:
                params_json = str(extra_params)

        return ReportRun(
            id=0,
            job_id=str(job_id) if job_id is not None else None,
            report_key=str(report_key) if report_key is not None else "",
            portfolio=portfolio,
            symbol=symbol,
            year=str(year) if year is not None else None,
            quarter=str(quarter) if quarter is not None else None,
            source=source,
            params_json=params_json,
            status=ReportRun._STARTED,
            start_time=now,
            last_update_time=now,
        )

    def set_finished(self):
        now = datetime.now()
        self.end_time = now
        self.last_update_time = now
        self.status = ReportRun._FINISHED

    def set_error(self, error):
        now = datetime.now()
        self.end_time = now
        self.last_update_time = now
        self.last_error = str(error)[:3900] if error is not None else None
        self.status = ReportRun._ERROR

    def set_aborted(self, msg=None):
        now = datetime.now()
        self.end_time = now
        self.last_update_time = now
        self.last_error = str(msg)[:3900] if msg is not None else None
        self.status = ReportRun._ABORTED
