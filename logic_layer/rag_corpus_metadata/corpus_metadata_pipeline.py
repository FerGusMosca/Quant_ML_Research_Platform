# corpus_metadata_pipeline.py
import os
from tqdm import tqdm

from logic_layer.rag_corpus_metadata.drift_detector import DriftDetector
from logic_layer.rag_corpus_metadata.file_hashing import FileHashing
from logic_layer.rag_corpus_metadata.metadata_inventory_builder import MetadataInventoryBuilder
from logic_layer.rag_corpus_metadata.pdf_metadata_extractor import PDFMetadataExtractor


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
        self.drift = DriftDetector(logger)
        self.inventory = MetadataInventoryBuilder(folder, logger)

    def run(self, pdf_list):
        metadata_items = []

        for pdf in tqdm(pdf_list):

            try:
                # -------- HASHING --------
                try:
                    text_hash, file_hash, skipped_hash = self.hasher.compute_hashes(pdf)
                except Exception as e:
                    self.logger.do_log(f"[PIPE] ❌ Hashing exploded for {pdf}: {e}", 1)
                    text_hash, file_hash, skipped_hash = None, None, True

                # -------- METADATA --------
                try:
                    meta = self.extractor.extract(pdf)
                except Exception as e:
                    self.logger.do_log(f"[PIPE] ❌ Metadata exploded for {pdf}: {e}", 1)
                    meta = {"path": pdf, "skipped": True}

                # -------- Attach Hashes --------
                meta["sha256_file"] = file_hash
                meta["sha256_text"] = text_hash

                # -------- Status placeholder --------
                meta["status"] = "skipped" if meta.get("skipped") or skipped_hash else "unknown"

                metadata_items.append(meta)

            except Exception as e:
                # This catch prevents the pipeline from dying FOREVER.
                self.logger.do_log(f"[PIPE] ❌ Fatal pipeline error for {pdf}: {e}", 1)
                continue

        # -------- Apply drift detection --------
        try:
            final_items = self.drift.apply_status(metadata_items)
        except Exception as e:
            self.logger.do_log(f"[PIPE] ❌ Drift detection failed: {e}", 1)
            final_items = metadata_items

        # -------- Save inventory --------
        try:
            self.inventory.save(final_items)
        except Exception as e:
            self.logger.do_log(f"[PIPE] ❌ Inventory save failed: {e}", 1)
