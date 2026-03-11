import os

#from docling.document_converter import DocumentConverter
from tqdm import tqdm

from data_access_layer.qdrant.qdrant_manager import QdrantManager
from framework.common.logger.message_type import MessageType
from logic_layer.rag_corpus_metadata.extractors.html_metadata_extractor import HTMLMetadataExtractor
from logic_layer.rag_corpus_metadata.tagger.transformers_topic_tagger import TransformersTopicTagger
from logic_layer.rag_corpus_metadata.drift_detector import DriftDetector
from logic_layer.rag_corpus_metadata.file_hashing import FileHashing
from logic_layer.rag_corpus_metadata.metadata_inventory_builder import MetadataInventoryBuilder
from logic_layer.rag_corpus_metadata.extractors.pdf_metadata_extractor import PDFMetadataExtractor
from logic_layer.rag_corpus_metadata.run_logger import RunLogger


class CorpusMetadataPipeline:

    def __init__(self, config, logger, dest_root,chunk_name,tag_cfg=None):
        self.config = config
        self.logger = logger
        self.dest_root = dest_root
        self.chunk_name=chunk_name

        self.qdrant = QdrantManager(host=self.config["QDRANT_SERVER"],
                                    port=int(self.config["QDRANT_PORT"]),
                                    collection="zh_metadata")

        folder = os.path.join(config["RAG_OUTPUT_FOLDER"],self.chunk_name, "corpus_metadata")
        os.makedirs(folder, exist_ok=True)
        self.output_folder = folder
        self.tag_cfg=tag_cfg

        self.do_tag =tag_cfg is not None

        # NEW
        #self.tagger = NaiveTopicTagger()


        if self.do_tag:
            self.tagger = TransformersTopicTagger(logger=logger,tag_cfg=tag_cfg)


        self.runlog = RunLogger(folder)

        self.simple_extractor = PDFMetadataExtractor(logger)
        self.html_extarctor=HTMLMetadataExtractor(logger)
        self.hasher = FileHashing(logger)
        self.drift = DriftDetector(logger, folder)
        self.inventory = MetadataInventoryBuilder(folder, logger)


    # ============================================================
    # PUBLIC ENTRYPOINT
    # ============================================================

    def discover_files(self, root_folder):
        pdfs = []
        for root, _, files in os.walk(root_folder):
            for f in files:
                if f.lower().endswith((".pdf", ".txt", ".html")):
                    pdfs.append(os.path.join(root, f))
        return pdfs

    def run(self, input_file_list,job_id=None):
        self.logger.do_log("[PIPE] ▶ Starting corpus metadata pipeline", MessageType.INFO,job_id)

        items = self._process_all_simple_files(input_file_list,job_id)
        items = self._apply_drift(items,job_id)
        if self.do_tag:
            self._apply_tagging(items,job_id)
        self._save_inventory(items,job_id)

        summary = self.runlog.write_summary(items)
        self.runlog.write_log(f"RUN COMPLETED → {summary}")

        self.logger.do_log(f"[PIPE] ✔ Completed metadata run | {summary}", MessageType.INFO,job_id)

    # ============================================================
    # INTERNAL STEPS
    # ============================================================
    def _process_all_simple_files(self, file_list,job_id=None):
        self.logger.do_log(f"[PIPE] ▶ Processing {len(file_list)} PDFs", MessageType.INFO,job_id)

        out = []
        for file in tqdm(file_list):
            self.logger.do_log(f"[FILE] ▶ Start: {file}", MessageType.INFO,job_id)

            if self.tag_cfg is not None and self.tag_cfg.is_K_Q_10_doc():
                out.append(self._process_single_html_file(file,job_id))
            else:
                out.append(self._process_single_simple_file(file,job_id))
            self.logger.do_log(f"[FILE] ✔ Done: {file}", MessageType.INFO,job_id)

        self.runlog.write_log(f"Processed FILEs: {len(out)}")
        return out

    def _prrocess_all_html_files(self, file_list,job_id=None):
        self.logger.do_log(f"[HTML] ▶ Processing {len(file_list)} HTMLs", MessageType.INFO,job_id)

        out = []
        for file in tqdm(file_list):
            self.logger.do_log(f"[FILE] ▶ Start: {file}", MessageType.INFO,job_id)
            out.append(self._process_single_simple_file(file,job_id))
            self.logger.do_log(f"[FILE] ✔ Done: {file}", MessageType.INFO,job_id)

        self.runlog.write_log(f"Processed FILEs: {len(out)}")
        return out

    def _process_single_html_file(self, file,job_id=None):
        self.logger.do_log(f"[HASH] ▶ Hashing HTML: {file}", MessageType.INFO,job_id)
        try:
            text_hash, file_hash, skipped_hash = self.hasher.compute_hashes(file)
        except Exception as e:
            self.logger.do_log(f"[HASH] ❌ Failed {file} → {e}", MessageType.INFO,job_id)
            text_hash, file_hash, skipped_hash = None, None, True

        self.logger.do_log(f"[META] ▶ Extracting HTML metadata: {file}", MessageType.INFO,job_id)
        try:
            meta = self.html_extarctor.extract(file)
        except Exception as e:
            self.logger.do_log(f"[META] ❌ Failed {file} → {e}", MessageType.INFO,job_id)
            meta = {"path": file, "skipped": True}

        meta["sha256_file"] = file_hash
        meta["sha256_text"] = text_hash
        meta["status"] = "skipped" if meta.get("skipped") or skipped_hash else "unknown"

        self.logger.do_log(f"[FILE] ✔ Completed HTML metadata: {file}", MessageType.INFO,job_id)
        return meta

    def _process_single_simple_file(self,file,job_id):
        self.logger.do_log(f"[HASH] ▶ Hashing: {file}", MessageType.INFO,job_id)
        try:
            text_hash, file_hash, skipped_hash = self.hasher.compute_hashes(file)

        except Exception as e:
            self.logger.do_log(f"[HASH] ❌ Failed {file} → {e}", MessageType.INFO,job_id)
            text_hash, file_hash, skipped_hash = None, None, True

        self.logger.do_log(f"[META] ▶ Extracting metadata: {file}", MessageType.INFO,job_id)
        try:
            meta = self.simple_extractor.extract(file)
        except Exception as e:
            self.logger.do_log(f"[META] ❌ Failed {file} → {e}", MessageType.INFO,job_id)
            meta = {"path": file, "skipped": True}

        meta["sha256_file"] = file_hash
        meta["sha256_text"] = text_hash
        meta["status"] = "skipped" if meta.get("skipped") or skipped_hash else "unknown"

        self.logger.do_log(f"[FILE] ✔ Completed metadata: {file}", MessageType.INFO,job_id)
        return meta

    def _apply_drift(self, items,job_id=None):
        self.logger.do_log("[DRIFT] ▶ Running drift detection", MessageType.INFO,job_id)
        try:
            out = self.drift.apply_status(items)
            self.logger.do_log("[DRIFT] ✔ OK", 1)
            return out
        except Exception as e:
            self.logger.do_log(f"[DRIFT] ❌ Failed → {e}", MessageType.INFO,job_id)
            return items

    def _apply_tagging(self, items,job_id=None):
        self.logger.do_log("[TAG] ▶ Applying topic tagging", MessageType.INFO,job_id)


        old = self.inventory.load_existing()

        for m in items:
            status = m.get("status")
            sha = m.get("sha256_text")

            if status == "unchanged":
                if sha in old and "tags" in old[sha]:
                    m["tags"] = old[sha]["tags"]
                    self.logger.do_log(f"[TAG] ⏭ Reused tags → {sha}", MessageType.INFO,job_id)
                else:
                    m["tags"] = []
                    self.logger.do_log(f"[TAG] ⚠ No previous tags → {sha}", MessageType.INFO,job_id)

            else:  # new / changed
                m["tags"] = self.tagger.classify(m.get("full_text", ""),m["filename"])
                self.logger.do_log(f"[TAG] 🏷 Recomputed tags → {sha}", MessageType.INFO,job_id)

            m.pop("full_text", None)

        self.logger.do_log("[TAG] ✔ Completed", 1,job_id)

    def _save_inventory(self, items,job_id=None):
        self.logger.do_log("[SAVE] ▶ Saving metadata inventory", MessageType.INFO,job_id)
        try:
            self.inventory.save(items)

            for m in items:
                sha256_text = m.get("sha256_text")

                # Skip si no hay sha256_text
                if not sha256_text:
                    self.logger.do_log(f"[SAVE] ⚠ Skipped (no sha256_text): {m.get('path')}", MessageType.WARNING,job_id)
                    continue


                chunk_id = int(sha256_text[:16], 16)

                payload = {
                    "source": "ZH",
                    "status": m.get("status"),
                    "path": m.get("path"),
                    "sha256_file": m.get("sha256_file"),
                    "sha256_text": sha256_text,
                    "filename": m.get("filename"),
                }
                self.qdrant.upsert_metadata(chunk_id, payload)

            self.logger.do_log("[SAVE] ✔ Inventory + Qdrant updated", MessageType.INFO,job_id)
        except Exception as e:
            self.logger.do_log(f"[SAVE] ❌ Failed → {e}", MessageType.ERROR,job_id)

