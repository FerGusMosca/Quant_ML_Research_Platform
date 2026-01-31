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
                                    collection="zh_chunks")

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
    def run(self, input_file_list):
        self.logger.do_log("[PIPE] ▶ Starting corpus metadata pipeline", MessageType.INFO)

        items = self._process_all_simple_files(input_file_list)
        items = self._apply_drift(items)
        if self.do_tag:
            self._apply_tagging(items)
        self._save_inventory(items)

        summary = self.runlog.write_summary(items)
        self.runlog.write_log(f"RUN COMPLETED → {summary}")

        self.logger.do_log(f"[PIPE] ✔ Completed metadata run | {summary}", MessageType.INFO)

    # ============================================================
    # INTERNAL STEPS
    # ============================================================
    def _process_all_simple_files(self, file_list):
        self.logger.do_log(f"[PIPE] ▶ Processing {len(file_list)} PDFs", MessageType.INFO)

        out = []
        for file in tqdm(file_list):
            self.logger.do_log(f"[FILE] ▶ Start: {file}", 2)

            if self.tag_cfg is not None and self.tag_cfg.is_K_Q_10_doc():
                out.append(self._process_single_html_file(file))
            else:
                out.append(self._process_single_simple_file(file))
            self.logger.do_log(f"[FILE] ✔ Done: {file}", MessageType.INFO)

        self.runlog.write_log(f"Processed FILEs: {len(out)}")
        return out

    def _prrocess_all_html_files(self, file_list):
        self.logger.do_log(f"[HTML] ▶ Processing {len(file_list)} HTMLs", MessageType.INFO)

        out = []
        for file in tqdm(file_list):
            self.logger.do_log(f"[FILE] ▶ Start: {file}", MessageType.INFO)
            out.append(self._process_single_simple_file(file))
            self.logger.do_log(f"[FILE] ✔ Done: {file}", MessageType.INFO)

        self.runlog.write_log(f"Processed FILEs: {len(out)}")
        return out

    def _process_single_html_file(self, file):
        self.logger.do_log(f"[HASH] ▶ Hashing HTML: {file}", MessageType.INFO)
        try:
            text_hash, file_hash, skipped_hash = self.hasher.compute_hashes(file)
        except Exception as e:
            self.logger.do_log(f"[HASH] ❌ Failed {file} → {e}", MessageType.INFO)
            text_hash, file_hash, skipped_hash = None, None, True

        self.logger.do_log(f"[META] ▶ Extracting HTML metadata: {file}", MessageType.INFO)
        try:
            meta = self.html_extarctor.extract(file)
        except Exception as e:
            self.logger.do_log(f"[META] ❌ Failed {file} → {e}", MessageType.INFO)
            meta = {"path": file, "skipped": True}

        meta["sha256_file"] = file_hash
        meta["sha256_text"] = text_hash
        meta["status"] = "skipped" if meta.get("skipped") or skipped_hash else "unknown"

        self.logger.do_log(f"[FILE] ✔ Completed HTML metadata: {file}", MessageType.INFO)
        return meta

    def _process_single_simple_file(self,file):
        self.logger.do_log(f"[HASH] ▶ Hashing: {file}", MessageType.INFO)
        try:
            text_hash, file_hash, skipped_hash = self.hasher.compute_hashes(file)

        except Exception as e:
            self.logger.do_log(f"[HASH] ❌ Failed {file} → {e}", MessageType.INFO)
            text_hash, file_hash, skipped_hash = None, None, True

        self.logger.do_log(f"[META] ▶ Extracting metadata: {file}", MessageType.INFO)
        try:
            meta = self.simple_extractor.extract(file)
        except Exception as e:
            self.logger.do_log(f"[META] ❌ Failed {file} → {e}", MessageType.INFO)
            meta = {"path": file, "skipped": True}

        meta["sha256_file"] = file_hash
        meta["sha256_text"] = text_hash
        meta["status"] = "skipped" if meta.get("skipped") or skipped_hash else "unknown"

        self.logger.do_log(f"[FILE] ✔ Completed metadata: {file}", MessageType.INFO)
        return meta

    def _apply_drift(self, items):
        self.logger.do_log("[DRIFT] ▶ Running drift detection", MessageType.INFO)
        try:
            out = self.drift.apply_status(items)
            self.logger.do_log("[DRIFT] ✔ OK", 1)
            return out
        except Exception as e:
            self.logger.do_log(f"[DRIFT] ❌ Failed → {e}", MessageType.INFO)
            return items

    def _apply_tagging(self, items):
        self.logger.do_log("[TAG] ▶ Applying topic tagging", MessageType.INFO)


        old = self.inventory.load_existing()

        for m in items:
            status = m.get("status")
            sha = m.get("sha256_text")

            if status == "unchanged":
                if sha in old and "tags" in old[sha]:
                    m["tags"] = old[sha]["tags"]
                    self.logger.do_log(f"[TAG] ⏭ Reused tags → {sha}", MessageType.INFO)
                else:
                    m["tags"] = []
                    self.logger.do_log(f"[TAG] ⚠ No previous tags → {sha}", MessageType.INFO)

            else:  # new / changed
                m["tags"] = self.tagger.classify(m.get("full_text", ""),m["filename"])
                self.logger.do_log(f"[TAG] 🏷 Recomputed tags → {sha}", MessageType.INFO)

            m.pop("full_text", None)

        self.logger.do_log("[TAG] ✔ Completed", 1)

    def _save_inventory(self, items):
        self.logger.do_log("[SAVE] ▶ Saving metadata inventory", MessageType.INFO)
        try:
            self.inventory.save(items)

            for m in items:
                chunk_id = m["sha256_text"]  # ID estable
                payload = {
                    "source": "ZH",
                    "status": m.get("status"),
                    "path": m.get("path"),
                    "sha256_file": m.get("sha256_file"),
                    "filename": m.get("filename"),
                }
                self.qdrant.upsert_metadata(chunk_id, payload)

            self.logger.do_log("[SAVE] ✔ Inventory + Qdrant updated", MessageType.INFO)
        except Exception as e:
            self.logger.do_log(f"[SAVE] ❌ Failed → {e}", MessageType.ERROR)

