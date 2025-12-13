import os
from tqdm import tqdm

from logic_layer.rag_corpus_metadata.tagger.transformers_topic_tagger import TransformersTopicTagger
from logic_layer.rag_corpus_metadata.drift_detector import DriftDetector
from logic_layer.rag_corpus_metadata.file_hashing import FileHashing
from logic_layer.rag_corpus_metadata.metadata_inventory_builder import MetadataInventoryBuilder
from logic_layer.rag_corpus_metadata.pdf_metadata_extractor import PDFMetadataExtractor
from logic_layer.rag_corpus_metadata.run_logger import RunLogger


class CorpusMetadataPipeline:

    def __init__(self, config, logger, dest_root):
        self.config = config
        self.logger = logger
        self.dest_root = dest_root

        folder = os.path.join(config["RAG_OUTPUT_FOLDER"], "corpus_metadata")
        os.makedirs(folder, exist_ok=True)
        self.output_folder = folder

        # NEW
        #self.tagger = NaiveTopicTagger()
        self.tagger = TransformersTopicTagger(logger=logger)
        self.runlog = RunLogger(folder)

        self.extractor = PDFMetadataExtractor(logger)
        self.hasher = FileHashing(logger)
        self.drift = DriftDetector(logger, folder)
        self.inventory = MetadataInventoryBuilder(folder, logger)


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
        self.logger.do_log(f"[PIPE] ▶ Processing {len(pdf_list)} PDFs", 1)

        out = []
        for pdf in tqdm(pdf_list):
            self.logger.do_log(f"[PDF] ▶ Start: {pdf}", 2)
            out.append(self._process_single_pdf(pdf))
            self.logger.do_log(f"[PDF] ✔ Done: {pdf}", 2)

        self.runlog.write_log(f"Processed PDFs: {len(out)}")
        return out

    def _process_single_pdf(self, pdf):
        self.logger.do_log(f"[HASH] ▶ Hashing: {pdf}", 3)
        try:
            text_hash, file_hash, skipped_hash = self.hasher.compute_hashes(pdf)
        except Exception as e:
            self.logger.do_log(f"[HASH] ❌ Failed {pdf} → {e}", 3)
            text_hash, file_hash, skipped_hash = None, None, True

        self.logger.do_log(f"[META] ▶ Extracting metadata: {pdf}", 3)
        try:
            meta = self.extractor.extract(pdf)
        except Exception as e:
            self.logger.do_log(f"[META] ❌ Failed {pdf} → {e}", 3)
            meta = {"path": pdf, "skipped": True}

        meta["sha256_file"] = file_hash
        meta["sha256_text"] = text_hash
        meta["status"] = "skipped" if meta.get("skipped") or skipped_hash else "unknown"

        self.logger.do_log(f"[PDF] ✔ Completed metadata: {pdf}", 3)
        return meta

    def _apply_drift(self, items):
        self.logger.do_log("[DRIFT] ▶ Running drift detection", 1)
        try:
            out = self.drift.apply_status(items)
            self.logger.do_log("[DRIFT] ✔ OK", 1)
            return out
        except Exception as e:
            self.logger.do_log(f"[DRIFT] ❌ Failed → {e}", 1)
            return items

    def _apply_tagging(self, items):
        self.logger.do_log("[TAG] ▶ Applying topic tagging", 1)

        skipped = 0
        processed = 0

        for m in items:
            status = m.get("status")

            if status == "unchanged":
                skipped += 1
                self.logger.do_log(
                    f"[TAG] ⏭ Skipped tagging (unchanged): {m.get('path', 'unknown')}",
                    3
                )
                continue

            full_text = m.get("full_text", "")
            m["tags"] = self.tagger.classify(full_text)
            processed += 1

        self.logger.do_log(
            f"[TAG] ✔ Completed | processed={processed} | skipped={skipped}",
            1
        )

    def _save_inventory(self, items):
        self.logger.do_log("[SAVE] ▶ Saving metadata inventory", 1)
        try:
            self.inventory.save(items)
            self.logger.do_log("[SAVE] ✔ Inventory saved", 1)
        except Exception as e:
            self.logger.do_log(f"[SAVE] ❌ Failed → {e}", 1)

