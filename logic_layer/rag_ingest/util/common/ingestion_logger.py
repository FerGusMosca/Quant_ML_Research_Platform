# logic_layer/rag_ingest/util/ingestion_logger.py
# Standalone class - ONLY handles final run log + SINGLE last_ingestion.json in the logs directory
# All comments in English

import os
import json
from datetime import datetime
from typing import Dict, Optional, Any


class RAGIngestionLogger:
    """
    Handles ONLY the final steps of ingestion logging:
    - Saves the per-run summary JSON (the usual ingest_*.json file)
    - Updates a SINGLE file: last_ingestion.json (with history and last successful folder)
    Nothing else. One file for state.
    """

    def __init__(self, logger, logs_dir: str, chunk_name: str):
        """
        :param logger: Existing logger with .do_log()
        :param logs_dir: Path to ingest_data_logs/
        :param chunk_name: For context in logs
        """
        self.logger = logger
        self.logs_dir = logs_dir
        self.chunk_name = chunk_name
        self.last_ingestion_path = None

    def finalize_run_and_update_state(
        self,
        log_root_path:str,
        current_run_log: str,
        start_ts: str,
        end_ts: str,
        pdf_list: list,
        source_path: str,
        summary: Dict[str, int],
        log_posfix: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Does exactly two things:
        1. Saves the normal per-run summary (ingest_*.json)
        2. Updates the SINGLE last_ingestion.json with this run info
        """

        self.last_ingestion_path=os.path.join(log_root_path,"last_ingestion.json")

        # 1. Save per-run summary (unchanged behavior)
        run_summary = {
            "start": start_ts,
            "end": end_ts,
            "total": len(pdf_list),
            "processed": summary["processed"],
            "skipped": summary["skipped"],
            "errors": summary["errors"],
            "pdf_source_path": source_path,
            "run_path": current_run_log,
            "chunk_name": self.chunk_name,
            "log_posfix": log_posfix,
        }

        try:
            os.makedirs(self.logs_dir, exist_ok=True)
            with open(current_run_log, "w", encoding="utf-8") as f:
                json.dump(run_summary, f, indent=2)
            self.logger.do_log(f"[RAG] Run summary saved → {current_run_log}", 1)
        except Exception as e:
            self.logger.do_log(f"[RAG] Failed to save run summary: {e}", 0)

        # 2. Update the ONLY state file: last_ingestion.json
        self._update_last_ingestion(
            source_path=source_path,
            end_ts=end_ts,
            summary=summary,
            error_message=error_message,
        )

    def _update_last_ingestion(
        self,
        source_path: str,
        end_ts: str,
        summary: Dict[str, int],
        error_message: Optional[str],
    ) -> None:
        """Updates last_ingestion.json directly. No tmp file, no bullshit."""
        try:

            # Load existing or start fresh
            if os.path.exists(self.last_ingestion_path):
                try:
                    with open(self.last_ingestion_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    history = data.get("processed_history", [])
                except json.JSONDecodeError:
                    data = {}
                    history = []
            else:
                data = {}
                history = []

            # Status
            if summary["errors"] == 0 and summary["processed"] > 0:
                status = "success"
            elif summary["errors"] > 0:
                status = "error"
            else:
                status = "no_content"

            # New entry
            entry = {
                "folder": source_path,
                "status": status,
                "timestamp": end_ts,
                "processed": summary["processed"],
                "skipped": summary["skipped"],
                "errors": summary["errors"],
            }
            if error_message:
                entry["error_message"] = error_message

            # Update last successful only if success
            if status == "success":
                data["last_successful_folder"] = source_path
                data["last_successful_timestamp"] = end_ts

            # Add to history
            history.insert(0, entry)
            data["processed_history"] = history[:50]


            with open(self.last_ingestion_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            self.logger.do_log(f"[RAG] last_ingestion.json updated → {status} | {os.path.basename(source_path)}", 1)

        except Exception as e:
            self.logger.do_log(f"[RAG] Failed to update last_ingestion.json: {e}", 0)