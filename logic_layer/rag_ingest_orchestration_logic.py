# ================================================================
# rag_ingest_orchestration_logic.py
# Stable orchestration logic for the new RAG ingestion pipeline
# ================================================================
import asyncio
import os
import traceback

from common.dto.mcp.bootstrap_registry import  build_mcp_registry_ingest
from common.dto.mcp.dispatcher import JsonRpcDispatcher
from common.dto.mcp.progress_bus import ProgressBus
from framework.common.logger.message_type import MessageType
from logic_layer.rag_ingest.util.multi_stage_rag.rag_pipeline import RAGPipeline
from service_layer.server.mcp_server import MCPServer


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

    def process_start_mcp(self,server,port):
        """
        Starts the MCP WebSocket server.
        Minimal, blocking startup.
        """

        # Prevent double start
        if getattr(self, "_mcp_started", False):
            self.logger.do_log(
                "[MCP] Server already running – skipping",
                MessageType.WARNING
            )
            return

        self._mcp_started = True

        self.progress_bus = ProgressBus()
        self.mcp_registry = build_mcp_registry_ingest(orchestrator=self)
        self.mcp_dispatcher = JsonRpcDispatcher(self.mcp_registry, self.progress_bus)
        mcp_server=server
        mcp_port = int(port)

        try:
            # Log startup
            self.logger.do_log(
                f"[MCP] Starting server on {mcp_server}:{mcp_port}",
                MessageType.INFO
            )

            # Create MCP server instance (already configured elsewhere)
            server = MCPServer(
                host=mcp_server,
                port=mcp_port,
                dispatcher=self.mcp_dispatcher,  # existing dispatcher,
                bus=self.progress_bus,
                logger=self.logger
            )

            # Run async MCP server (blocks current thread)
            asyncio.run(server.start())

        except Exception as e:
            # Fatal startup error: log and propagate
            self.logger.do_log(
                f"[MCP] ❌ Fatal error while starting server: {e}",
                MessageType.ERROR
            )
            raise

    def process_rag_ingest(self, ingest_type, source_path=None,chunk_name=None, dest_root=None,log_posfix=None,
                           embedding_model=None,clustering_model=None,job_id=None):
        """
        :param ingest_type: "full" / "incremental"
        :param source_path: folder where PDFs exist
        :param dest_root:    top-level folder name (e.g. "Archives")
        """
        try:
            self.logger.do_log(
                f"[RAG-INGEST] 🚀 Trigger received: ingest_type={ingest_type}, "
                f"source={source_path}, dest_root={dest_root}",
                MessageType.INFO,job_id
            )


            # ---------------------------
            # Initialize pipeline ONCE
            # ---------------------------
            self.logger.do_log("[RAG-INGEST] 🔧 Initializing RAG pipeline...", MessageType.INFO,job_id)

            pipeline = RAGPipeline(chunk_name,dest_root, self.config,embedding_model,clustering_model, self.logger)
            pipeline.run(source_path,log_posfix,ingest_type=ingest_type,job_id=job_id)

            self.logger.do_log(
                "[RAG-INGEST] ✅ RAG ingestion completed successfully.",
                MessageType.INFO,job_id
            )

            return True

        except Exception as e:
            self.logger.do_log(
                f"[RAG-INGEST] ❌ Exception: {str(e)}\n{traceback.format_exc()}",
                MessageType.ERROR,job_id
            )
            return False



