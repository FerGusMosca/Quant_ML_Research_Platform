# ================================================================
# rag_ingest_orchestration_logic.py
# Stable orchestration logic for the new RAG ingestion pipeline
# ================================================================

import os
import traceback
from framework.common.logger.message_type import MessageType
from logic_layer.rag_ingest.rag_pipeline import RAGPipeline


class RAGIngestOrchestrationLogic:
    """
    Main entry point for triggering RAG ingestion tasks.
    Handles:
      - Parameter validation
      - PDF discovery
      - Pipeline execution
    """

    def __init__(self, config_settings, logger):
        """
        :param config_settings: dict loaded from commands_mgr.ini
        :param logger: Logger instance from your framework
        """
        self.config = config_settings
        self.logger = logger


    # ============================================================
    # INTERNAL: Compute output path (same FIX used in RAGPipeline)
    # ============================================================
    def _compute_output_path(self, pdf_path: str, dest_root: str) -> str:
        """
        Returns the directory path from dest_root down to the folder
        containing the PDF — EXCLUDING the PDF filename.
        Example:
        C:/zerohedge_docs/Archives/2025/Nov/Nov 6/AAPL.pdf
        → Archives/2025/Nov/Nov 6
        """
        # Normalize for cross-platform consistency
        normalized = os.path.normpath(pdf_path)
        parts = normalized.split(os.sep)

        if dest_root not in parts:
            raise ValueError(
                f"[RAG] dest_root='{dest_root}' not found in path: {pdf_path}"
            )

        # Starting index of dest_root
        idx = parts.index(dest_root)

        # Remove the filename (last element)
        folder_parts = parts[idx:-1]

        # Rebuild clean folder path
        return os.path.join(*folder_parts)


    # ============================================================
    # MAIN DISPATCH METHOD
    # ============================================================
    def process_rag_ingest(self, ingest_type, source_path=None, dest_root=None):
        """
        :param ingest_type: "full" / "incremental"
        :param source_path: folder where PDFs exist
        :param dest_root:    top-level folder name (e.g. "Archives")
        """
        try:
            self.logger.do_log(
                f"[RAG-INGEST] 🚀 Trigger received: ingest_type={ingest_type}, "
                f"source={source_path}, dest_root={dest_root}",
                MessageType.INFO
            )

            # ---------------------------
            # Validate source_path
            # ---------------------------
            if not source_path:
                self.logger.do_log("[RAG-INGEST] ❌ Missing source_path.", MessageType.ERROR)
                return False

            if not os.path.exists(source_path):
                self.logger.do_log(
                    f"[RAG-INGEST] ❌ Path does not exist: {source_path}",
                    MessageType.ERROR
                )
                return False

            if not dest_root:
                self.logger.do_log("[RAG-INGEST] ❌ Missing dest_root parameter.", MessageType.ERROR)
                return False

            # ---------------------------
            # Discover all PDFs
            # ---------------------------
            pdfs = self._discover_pdfs(source_path)

            self.logger.do_log(
                f"[RAG-INGEST] 📄 Found {len(pdfs)} PDF(s) to process.",
                MessageType.INFO
            )

            for i, pdf_file in enumerate(pdfs):
                self.logger.do_log(
                    f"[RAG-INGEST] [{i+1}/{len(pdfs)}] {pdf_file}",
                    MessageType.INFO
                )

            if len(pdfs) == 0:
                self.logger.do_log(
                    "[RAG-INGEST] ❌ No PDFs found. Nothing to process.",
                    MessageType.ERROR
                )
                return False

            # ---------------------------
            # Initialize pipeline ONCE
            # ---------------------------
            self.logger.do_log("[RAG-INGEST] 🔧 Initializing RAG pipeline...", MessageType.INFO)

            pipeline = RAGPipeline(dest_root, self.config, self.logger)
            pipeline.run(pdfs)

            self.logger.do_log(
                "[RAG-INGEST] ✅ RAG ingestion completed successfully.",
                MessageType.INFO
            )

            return True

        except Exception as e:
            self.logger.do_log(
                f"[RAG-INGEST] ❌ Exception: {str(e)}\n{traceback.format_exc()}",
                MessageType.ERROR
            )
            return False


    # ============================================================
    # INTERNAL: PDF DISCOVERY
    # ============================================================
    def _discover_pdfs(self, root_folder):
        """
        Recursively discover all PDF files inside root_folder.
        """
        pdfs = []
        for root, dirs, files in os.walk(root_folder):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdfs.append(os.path.join(root, f))
        return pdfs
