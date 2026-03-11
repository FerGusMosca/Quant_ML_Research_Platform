from dataclasses import dataclass
from typing import Optional


@dataclass
class MetadataPoint:
    """Represents a single point from the zh_metadata collection."""

    id: str
    filename: Optional[str] = None
    path: Optional[str] = None
    source: Optional[str] = None
    status: Optional[str] = None
    sha256_file: Optional[str] = None
    sha256_text: Optional[str] = None

    @classmethod
    def from_qdrant_point(cls, point: dict) -> "MetadataPoint":
        payload = point.get("payload", {})
        return cls(
            id=str(point.get("id", "")),
            filename=payload.get("filename"),
            path=payload.get("path"),
            source=payload.get("source"),
            status=payload.get("status"),
            sha256_file=payload.get("sha256_file"),
            sha256_text=payload.get("sha256_text"),
        )

    def to_dict(self) -> dict:
        # source_pdf and chunk_text are used by the frontend table/modal
        # as the common display fields across collections
        return {
            "id":               self.id,
            "source_pdf":       self.filename,   # mapped for the table's Source PDF column
            "chunk_text":       None,
            "chunk_index":      None,
            "chunk_id":         None,
            "pdf_path":         None,
            "source_path":      self.path,
            "text_len":         None,
            "ingest_timestamp": None,
            "ingest_ts_epoch":  None,
            "ingest_run_id":    None,
            "extra_payload": {
                "filename":    self.filename,
                "path":        self.path,
                "source":      self.source,
                "status":      self.status,
                "sha256_file": self.sha256_file,
                "sha256_text": self.sha256_text,
            },
        }