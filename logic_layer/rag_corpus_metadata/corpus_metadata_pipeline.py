import os
import json
from tqdm import tqdm

from logic_layer.rag_corpus_metadata.drift_detector import DriftDetector
from logic_layer.rag_corpus_metadata.file_hashing import FileHashing
from logic_layer.rag_corpus_metadata.metadata_inventory_builder import MetadataInventoryBuilder
from logic_layer.rag_corpus_metadata.pdf_metadata_extractor import PDFMetadataExtractor
from logic_layer.rag_corpus_metadata.topic_tagger import TopicTagger
from logic_layer.rag_corpus_metadata.run_logger import RunLogger


class CorpusMetadataPipeline:

    def __init__(self, config, logger, dest_root):
        self.config = config
        self.logger = logger
        self.dest_root = dest_root

        folder = os.path.join(config["RAG_OUTPUT_FOLDER"], "corpus_metadata")
        os.makedirs(folder, exist_ok=True)
        self.output_folder = folder

        self.extractor = PDFMetadataExtractor(logger)
        self.hasher = FileHashing(logger)
        self.drift = DriftDetector(logger, folder)
        self.inventory = MetadataInventoryBuilder(folder, logger)

        # NEW
        self.tagger = TopicTagger()
        self.runlog = RunLogger(folder)

    # ============================================================
    # PUBLIC ENTRYPOINT
    # ============================================================
    def run(self, pdf_list):
        self.logger.do_log("[PIPE] ▶ Starting corpus metadata pipeline", 1)

        items = self._process_all_pdfs(pdf_list)
        items = self._apply_drift(items)
        self._apply_tagging(items)
        self._save_inventory(items)

        summary = self.runlog.write_summary(items)
        self.runlog.write_log(f"RUN COMPLETED → {summary}")

        self.logger.do_log(f"[PIPE] ✔ Completed metadata run | {summary}", 1)

    # ============================================================
    # INTERNAL STEPS
    # ============================================================
    def _process_all_pdfs(self, pdf_list):
        out = []
        for pdf in tqdm(pdf_list):
            out.append(self._process_single_pdf(pdf))

        self.runlog.write_log(f"Processed PDFs: {len(out)}")
        return out

    def _process_single_pdf(self, pdf):
        try:
            # --- HASHES ---
            try:
                text_hash, file_hash, skipped_hash = self.hasher.compute_hashes(pdf)
            except Exception as e:
                self.runlog.write_log(f"[ERROR] Hash fail {pdf}: {e}")
                text_hash, file_hash, skipped_hash = None, None, True

            # --- METADATA ---
            try:
                meta = self.extractor.extract(pdf)
            except Exception as e:
                self.runlog.write_log(f"[ERROR] Metadata fail {pdf}: {e}")
                meta = {"path": pdf, "skipped": True}

            meta["sha256_file"] = file_hash
            meta["sha256_text"] = text_hash

            if meta.get("skipped") or skipped_hash:
                meta["status"] = "skipped"
            else:
                meta["status"] = "unknown"

            return meta

        except Exception as e:
            self.runlog.write_log(f"[FATAL] Pipeline error for {pdf}: {e}")
            return {"path": pdf, "skipped": True, "status": "error"}

    def _apply_drift(self, items):
        try:
            out = self.drift.apply_status(items)
            self.runlog.write_log("Drift detection OK")
            return out
        except Exception as e:
            self.runlog.write_log(f"[ERROR] Drift detection failed: {e}")
            return items

    def _apply_tagging(self, items):
        for m in items:
            title = m.get("title_guess", "")
            m["tags"] = self.tagger.classify(title)

        self.runlog.write_log("Topic tagging applied")

    def _save_inventory(self, items):
        try:
            self.inventory.save(items)
            self.runlog.write_log("Inventory saved")
        except Exception as e:
            self.runlog.write_log(f"[ERROR] Inventory save failed: {e}")
