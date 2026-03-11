from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChunkPoint:
    """Represents a single point from the zh_chunks collection."""

    id: str
    source_pdf: Optional[str] = None
    chunk_id: Optional[int] = None
    chunk_index: Optional[int] = None
    chunk_text: Optional[str] = None
    pdf_path: Optional[str] = None
    source_path: Optional[str] = None
    text_len: Optional[int] = None
    ingest_timestamp: Optional[str] = None
    ingest_ts_epoch: Optional[int] = None
    ingest_run_id: Optional[str] = None

    @classmethod
    def from_qdrant_point(cls, point: dict) -> "ChunkPoint":
        payload = point.get("payload", {})
        return cls(
            id=str(point.get("id", "")),
            source_pdf=payload.get("source_pdf"),
            chunk_id=payload.get("chunk_id"),
            chunk_index=payload.get("chunk_index"),
            chunk_text=payload.get("chunk_text"),
            pdf_path=payload.get("pdf_path"),
            source_path=payload.get("source_path"),
            text_len=payload.get("text_len"),
            ingest_timestamp=payload.get("ingest_timestamp"),
            ingest_ts_epoch=payload.get("ingest_ts_epoch"),
            ingest_run_id=payload.get("ingest_run_id"),
        )

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "source_pdf":       self.source_pdf,
            "chunk_id":         self.chunk_id,
            "chunk_index":      self.chunk_index,
            "chunk_text":       self.chunk_text,
            "pdf_path":         self.pdf_path,
            "source_path":      self.source_path,
            "text_len":         self.text_len,
            "ingest_timestamp": self.ingest_timestamp,
            "ingest_ts_epoch":  self.ingest_ts_epoch,
            "ingest_run_id":    self.ingest_run_id,
            "extra_payload":    {},
        }