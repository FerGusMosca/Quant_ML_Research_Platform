# ingest_metadata_manager.py
# All comments MUST be in English.

import os
import json
from datetime import datetime


class IngestMetadataManager:
    """
    Tracks ingestion metadata to support multi-layer skip logic:
    1) Detect whether a PDF was already processed.
    2) Detect whether the underlying corpus metadata says the file changed.
    3) Allow re-run detection (no-changes scenario).
    """

    def __init__(self, ingest_metadata_path, logger):
        self.logger = logger
        self.ingest_metadata_path = ingest_metadata_path

        # Load existing metadata or start new structure
        self.data = self._load()

    # -------------------------------------------------------------
    def _load(self):
        """Load ingestion metadata from disk."""
        if not os.path.exists(self.ingest_metadata_path):
            self.logger.do_log(f"[META] No ingest metadata found. Starting fresh.", 1)
            return {}

        try:
            with open(self.ingest_metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.do_log(f"[META] Failed to load ingest metadata: {e}", 0)
            return {}

    # -------------------------------------------------------------
    def save(self):
        """Persist ingestion metadata to disk."""
        try:
            os.makedirs(os.path.dirname(self.ingest_metadata_path), exist_ok=True)
            with open(self.ingest_metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            self.logger.do_log(f"[META] Failed to save ingest metadata: {e}", 0)

    # -------------------------------------------------------------
    def was_already_processed(self, pdf_path: str) -> bool:
        """Return True if this PDF was processed before."""
        return pdf_path in self.data

    # -------------------------------------------------------------
    def mark_processed(self, pdf_path: str):
        """Record that a PDF was processed successfully."""
        self.data[pdf_path] = {
            "processed_at": datetime.utcnow().isoformat()
        }

    # -------------------------------------------------------------
    def should_skip(self, pdf_path: str, corpus_meta: dict) -> bool:
        """
        Combined decision logic:

        IF corpus metadata says 'unchanged'
            AND ingest metadata says it was already processed
        THEN: safe to skip
        """

        corpus_status = corpus_meta.get("status", None)

        if corpus_status == "unchanged" and self.was_already_processed(pdf_path):
            return True

        return False
