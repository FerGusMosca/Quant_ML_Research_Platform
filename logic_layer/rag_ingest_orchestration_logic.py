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
    CURRENT STATE:
      - Validates parameters
      - Discovers PDF files
      - Executes ingestion pipeline once
    """

    def __init__(self, config_settings, logger):
        """
        :param config_settings: dict loaded from commands_mgr.ini
        :param logger: Logger instance from your framework
        """
        self.config = config_settings
        self.logger = logger

    # ------------------------------------------------------------------
    # MAIN DISPATCH METHOD
    # ------------------------------------------------------------------
    def process_rag_ingest(self, ingest_type, source_path=None):
        """
        :param ingest_type: "full" / "incremental"
        :param source_path: folder containing PDFs
        """
        try:
            self.logger.do_log(
                f"[RAG-INGEST] 🚀 Trigger received: ingest_type={ingest_type}, source={source_path}",
                MessageType.INFO
            )

            # ------------------------------------------------------
            # Validate input path
            # ------------------------------------------------------
            if not source_path:
                self.logger.do_log("[RAG-INGEST] ❌ No source_path provided.", MessageType.ERROR)
                return False

            if not os.path.exists(source_path):
                self.logger.do_log(
                    f"[RAG-INGEST] ❌ Path does not exist: {source_path}",
                    MessageType.ERROR
                )
                return False

            # ------------------------------------------------------
            # Discover PDFs (recursive)
            # ------------------------------------------------------
            pdfs = self._discover_pdfs(source_path)

            self.logger.do_log(
                f"[RAG-INGEST] 📄 Found {len(pdfs)} PDF(s) to process.",
                MessageType.INFO
            )

            # Only log them — do NOT run pipeline inside the loop
            for i, pdf_file in enumerate(pdfs):
                self.logger.do_log(
                    f"[RAG-INGEST] [{i+1}/{len(pdfs)}] Found PDF: {pdf_file}",
                    MessageType.INFO
                )

            # ------------------------------------------------------
            # Initialize + run the pipeline ONCE
            # ------------------------------------------------------
            if len(pdfs) == 0:
                self.logger.do_log("[RAG-INGEST] ❌ No PDFs found. Nothing to process.", MessageType.ERROR)
                return False


            self.logger.do_log(
                "[RAG-INGEST] 🔧 Initializing RAG pipeline...",
                MessageType.INFO
            )

            pipeline = RAGPipeline(self.config, self.logger)
            pipeline.run(pdfs)

            self.logger.do_log(
                "[RAG-INGEST] ✅ RAG pipeline finished successfully.",
                MessageType.INFO
            )

            return True

        except Exception as e:
            self.logger.do_log(
                f"[RAG-INGEST] ❌ Error: {str(e)}\n{traceback.format_exc()}",
                MessageType.ERROR
            )
            return False

    # ------------------------------------------------------------------
    # INTERNAL UTILITY — PDF DISCOVERY
    # ------------------------------------------------------------------
    def _discover_pdfs(self, root_folder):
        """
        Recursively discover all PDFs under the given folder.
        """
        pdfs = []
        for root, dirs, files in os.walk(root_folder):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdfs.append(os.path.join(root, f))
        return pdfs
