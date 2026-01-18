import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Any


@dataclass(frozen=False)
class TagRun:

    _STARTED="started"
    _FINISHED ="finished"
    _ERROR = "error"
    _SKIPPED = "skipped"

    """
    Entity/DTO representing a single tag processing run
    """
    id: Optional[int] = None                    # None when it's a new record
    portfolio: str = ""
    report:str=""
    source: str = ""
    rank_folder: Optional[str] = None
    year: str=""
    quarter: str = "",
    sec_processed:int=None,
    timestamp: Optional[datetime] = None
    tag_file: Optional[str] = None
    tag_json: Optional[dict | list | str | Any] = None  # flexible - can be dict, list, json string, etc.
    tag_model: str = ""
    doc_type: str = ""
    status: str = "started"
    last_error: Optional[str] = None
    last_update_time: Optional[datetime] = None

    def __post_init__(self):
        """Set default values for timestamp fields if not provided"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.last_update_time is None:
            self.last_update_time = datetime.now()


    @staticmethod
    def initialize_tag_run(portfolio,report_type,source,rank_folder,year,quarter,sec_processed,tag_cfg,tag_dict):
        now = datetime.now(timezone.utc)

        tag_json = json.dumps(tag_dict)

        return TagRun(id=0,portfolio=portfolio,report=report_type,source=source,
               rank_folder=rank_folder,year=year,quarter=quarter,sec_processed=sec_processed,
               timestamp=now,tag_json=tag_json,
               tag_model=tag_cfg.tag_model,doc_type=tag_cfg.doc_type,tag_file=tag_cfg.tag_file,
               status=TagRun._STARTED)


    def set_finished(self):
        now = datetime.now(timezone.utc)
        self.last_update_time=now
        self.status=TagRun._FINISHED


    def set_error(self,error):
        now = datetime.now(timezone.utc)
        self.last_update_time = now
        self.last_error=error
        self.status = TagRun._ERROR

    def set_skipped(self,msg):
        now = datetime.now(timezone.utc)
        self.last_update_time = now
        self.last_error=msg
        self.status = TagRun._SKIPPED



