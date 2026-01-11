from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any


@dataclass(frozen=False)
class TagRun:
    """
    Entity/DTO representing a single tag processing run
    """
    id: Optional[int] = None                    # None when it's a new record
    portfolio: str = ""
    source: str = ""
    rank_folder: Optional[str] = None
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