# corpus_metadata_orchestration_logic.py
import os
from framework.common.logger.message_type import MessageType
from logic_layer.rag_corpus_metadata.corpus_metadata_pipeline import CorpusMetadataPipeline


class CorpusMetadataOrchestrationLogic:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def _discover_pdfs(self, root_folder):
        pdfs = []
        for root, _, files in os.walk(root_folder):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdfs.append(os.path.join(root, f))
        return pdfs

    def run(self, source_path, dest_root):
        self.logger.do_log(f"[CORPUS] 🚀 Starting metadata: {source_path}",
                           MessageType.INFO)

        if not os.path.exists(source_path):
            raise Exception(f"Source path does not exist: {source_path}")

        pdfs = self._discover_pdfs(source_path)
        self.logger.do_log(f"[CORPUS] Found {len(pdfs)} PDFs", MessageType.INFO)

        pipeline = CorpusMetadataPipeline(self.config, self.logger, dest_root)
        pipeline.run(pdfs)

        self.logger.do_log("[CORPUS] ✅ Completed.", MessageType.INFO)
