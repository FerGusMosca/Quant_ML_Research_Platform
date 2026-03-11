# corpus_metadata_orchestration_logic.py
import asyncio
import os
import traceback

from common.dto.mcp.bootstrap_registry import build_mcp_registry_corpus_metadata
from common.dto.mcp.dispatcher import JsonRpcDispatcher
from common.dto.mcp.progress_bus import ProgressBus
from framework.common.logger.message_type import MessageType
from logic_layer.rag_corpus_metadata.corpus_metadata_pipeline import CorpusMetadataPipeline
from service_layer.server.mcp_server import MCPServer


class CorpusMetadataOrchestrationLogic:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    # ── File discovery ────────────────────────────────────────────────────────




    # ── Pipeline run ──────────────────────────────────────────────────────────

    def run(self, source_path, dest_root, chunk_name, tag_cfg=None, job_id=None):
        self.logger.do_log(f"[CORPUS] 🚀 Starting metadata: {source_path}", MessageType.INFO, job_id)

        if not os.path.exists(source_path):
            raise Exception(f"Source path does not exist: {source_path}")



        pipeline = CorpusMetadataPipeline(self.config, self.logger, dest_root, chunk_name, tag_cfg=tag_cfg)
        files = pipeline.discover_files(source_path)
        self.logger.do_log(f"[CORPUS] Found {len(files)} PDFs/TXTs/HTMLs", MessageType.INFO, job_id)
        pipeline.run(files,job_id)

        self.logger.do_log("[CORPUS] ✅ Completed.", MessageType.INFO, job_id)