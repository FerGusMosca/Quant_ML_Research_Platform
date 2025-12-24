# logic_layer/rag_ingest/util/multi_stage_rag/rag_pipeline.py
# All comments MUST be in English.

import os
import json
import re
from datetime import datetime

import numpy as np

from framework.common.logger.message_type import MessageType
from logic_layer.rag_ingest.util.common.ingestion_logger import RAGIngestionLogger
from logic_layer.rag_ingest.util.common.zh_next_folder_generator import ZHNextFolderGenerator
from logic_layer.rag_ingest.util.multi_stage_rag.chunk_generation.transformers.ktransformers_chunk_generator import \
    KTransformersChunkGenerator
from logic_layer.rag_ingest.util.multi_stage_rag.chunk_generation.transformers.transfomers_semantic_dedupers import \
    TranfomersSemanticChunkDeduper
from logic_layer.rag_ingest.util.multi_stage_rag.chunk_generation.vainilla.vainilla_chunk_generator import \
    VainillaChunkGenerator
from logic_layer.rag_ingest.util.multi_stage_rag.chunk_generation.vainilla.vanilla_deduper import VanillaChunkDeduper
from logic_layer.rag_ingest.util.multi_stage_rag.pdf_text_extractor import PDFTextExtractor
from logic_layer.rag_ingest.util.multi_stage_rag.pdf_cleaner import PDFCleaner
from logic_layer.rag_ingest.util.multi_stage_rag.metadata_builder import MetadataBuilder
from logic_layer.rag_ingest.util.multi_stage_rag.embeddings_generator import EmbeddingsGenerator
from logic_layer.rag_ingest.util.multi_stage_rag.ingest_metadata_manager import IngestMetadataManager


class RAGPipeline:

    def __init__(self,chunk_name, dest_root, config,embedding_model,clustering_model, logger):
        self.logger = logger
        self.chunk_name=chunk_name
        self.dest_root = dest_root
        self.current_run_log =None
        #self.chunk_generator = VainillaChunkGenerator(self.logger)
        self.chunk_generator = KTransformersChunkGenerator(model_name= clustering_model,logger=self.logger)

        #self.chunk_deduper=VanillaChunkDeduper(logger=self.logger)
        self.chunk_deduper=TranfomersSemanticChunkDeduper(model_name=clustering_model,logger=self.logger)

        # ------- Embedding model -------
        try:
            self.embedder = EmbeddingsGenerator(embedding_model,logger=self.logger)
        except Exception as e:
            self.logger.do_log(f"[RAG] ❌ Failed to load embedding model: {e}", 0)
            raise

        # ------- Output base folder -------
        #self.output_base = config["RAG_OUTPUT_FOLDER"]
        self.output_base =os.path.join(config["RAG_OUTPUT_FOLDER"], self.chunk_name)

        try:
            os.makedirs(self.output_base, exist_ok=True)
        except Exception as e:
            logger.do_log(f"[RAG] ❌ Could not create output folder: {e}", 0)
            raise

        # ------- Metadata path -------
        self.corpus_metadata_path = config.get(
            "CORPUS_METADATA_PATH",
            os.path.join(self.output_base, "corpus_metadata", "corpus_inventory.json")
        )

        # ------- Load metadata for skip logic -------
        self.global_metadata = self._load_corpus_inventory()

        # ------- Logging folder -------
        self.logs_dir = os.path.join(self.output_base, "ingest_data_logs")
        os.makedirs(self.logs_dir, exist_ok=True)

        os.makedirs(os.path.join(self.output_base, "corpus_metadata"), exist_ok=True)

        self.ingest_metadata_path = os.path.join(self.output_base, "corpus_metadata", "ingest_metadata.json")
        self.ingest_meta = IngestMetadataManager(self.ingest_metadata_path, logger)


    def _load_corpus_inventory(self):
        """Load global corpus metadata to support incremental ingest."""
        try:
            with open(self.corpus_metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {item["path"]: item for item in data}
        except Exception as e:
            self.logger.do_log(f"[RAG] ⚠️ Could not load corpus inventory: {e}", 0)
            return {}

    # ==========================================================
    # Compute output folder relative to dest_root
    # ==========================================================

    def _format_folder(self,folder):
        clean = re.sub(r'[^a-zA-Z0-9._-]', '_', folder)
        # Collapse multiple underscores and strip leading/trailing ones
        clean = re.sub(r'_+', '_', clean.strip('_'))
        return  clean

    def _compute_output_path(self, pdf_path: str) -> str:
        """
        Compute the relative output folder path starting from dest_root.
        All folder names are sanitized: spaces and any weird characters are replaced with '_'.
        Example: 'Nov 6' to 'Nov_6', 'My (final) Report!' to 'My_final_Report'
        """
        try:
            normalized = os.path.normpath(pdf_path)
            parts = normalized.split(os.sep)

            # Safety check: dest_root must be present in the path
            if self.dest_root not in parts:
                raise ValueError(
                    f"[RAG] ERROR: dest_root='{self.dest_root}' not found in path: {pdf_path}"
                )

            # Everything from dest_root (included) up to the file's folder (file
            idx = parts.index(self.dest_root)
            folder_parts = parts[idx:-1]

            # Sanitize every folder name
            clean_parts = []
            for part in folder_parts:
                # Replace anything that is not letter/number/dot/hyphen with _
                clean=self._format_folder(part)


                clean_parts.append(clean)

            return os.path.join(*clean_parts)

        except Exception as e:
            self.logger.do_log(f"[RAG] compute_output_path failed: {e}", 0)
            raise

    # ==========================================================
    # Safe filename sanitizer
    # ==========================================================
    def _sanitize_filename(self, name: str) -> str:
        try:
            original = name
            sanitized = re.sub(r'[<>:"/\\|?*]', '', name)
            sanitized = sanitized.replace("...", "")
            sanitized = re.sub(r'\.{2,}', '.', sanitized)
            sanitized = re.sub(r'\s+', '_', sanitized)
            sanitized = sanitized.rstrip('. ')
            if len(sanitized) == 0:
                sanitized = "doc_" + str(abs(hash(original)) % 100000)
            if len(sanitized) > 120:
                sanitized = sanitized[:110] + "_" + str(abs(hash(sanitized)) % 100000)
            return sanitized
        except Exception as e:
            self.logger.do_log(f"[RAG] ❌ sanitize_filename failed: {e}", 0)
            return "unnamed_document"

    # ==========================================================
    # PROCESS ONE PDF
    # ==========================================================
    def process_pdf(self, pdf_path: str):
        self.logger.do_log(f"[RAG] Extracting text: {pdf_path}", 1)

        # ----- Extract -----
        try:
            raw_text = PDFTextExtractor.extract_text(pdf_path, logger=self.logger)
        except Exception as e:
            self.logger.do_log(f"[RAG] ❌ PDF extraction failed: {e}", 0)
            return None

        # ----- Clean -----
        try:
            clean_text = PDFCleaner.clean(raw_text, logger=self.logger)
        except Exception as e:
            self.logger.do_log(f"[RAG] ❌ PDF cleaning failed: {e}", 0)
            return None

        # ----- Chunking -----
        try:
            chunks = self.chunk_generator.chunk(clean_text)

            # ===== DEDUP LAYER (safe, exact duplicates only) =====

            chunks = self.chunk_deduper.dedup_chunks(chunks)

        except Exception as e:
            self.logger.do_log(f"[RAG] ❌ Chunking failed: {e}", 0)
            return None



        if len(chunks) == 0:
            self.logger.do_log("[RAG] ❌ No chunks generated.", 0)
            return None

        # ----- Metadata -----
        try:
            metadata = [MetadataBuilder.build(pdf_path, idx) for idx in range(len(chunks))]
        except Exception as e:
            self.logger.do_log(f"[RAG] ❌ Metadata generation failed: {e}", 0)
            return None

        # ----- Embeddings -----
        try:
            embeddings = self.embedder.embed(chunks)
        except Exception as e:
            self.logger.do_log(f"[RAG] ❌ Embedding generation failed: {e}", 0)
            return None

        # ----- Output path -----
        try:
            rel_folder = os.path.normpath(self._compute_output_path(pdf_path))
        except Exception as e:
            self.logger.do_log(
                f"[RAG][PATH] Failed to compute output path | pdf_path={pdf_path} | error={repr(e)}",0
            )
            return None

        base_raw = os.path.basename(pdf_path).replace(".pdf", "")
        sanitized_name = self._sanitize_filename(base_raw)

        out_dir = os.path.normpath(
            os.path.join(self.output_base, rel_folder, sanitized_name)
        )
        os.makedirs(out_dir, exist_ok=True)

        # ----- Save artifacts -----
        try:
            with open(os.path.join(out_dir, "chunks.txt"), "w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(c + "\n\n")

            with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)

            self.logger.do_log(f"[RAG] ✅ Artifacts saved → {out_dir}", 1)

        except Exception as e:
            self.logger.do_log(f"[RAG] ❌ Saving artifacts failed: {e}", 0)
            return None

        return chunks, metadata, embeddings,out_dir

    def _make_run_log(self, source_path,log_posfix=None):

        if log_posfix is None:
            """Build early-run log filename using the ingest source."""

            clean = re.sub(r"[^A-Za-z0-9_\-]", "_", os.path.basename(source_path))
            ts = datetime.utcnow().isoformat().replace(":", "-")
            fn = f"ingest_{clean}_{ts}.json"
            path = os.path.join(self.logs_dir, fn)
            with open(path, "w", encoding="utf-8") as f: f.write("{}")
            return path
        else:
            ts = datetime.utcnow().isoformat().replace(":", "-")
            fn = f"ingest_{log_posfix}_{ts}.json"
            path = os.path.join(self.logs_dir, fn)
            with open(path, "w", encoding="utf-8") as f: f.write("{}")
            return path

    def _validate_source_paths(self,source_path,dest_root):
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

        return True


    # ============================================================
    # INTERNAL: PDF DISCOVERY
    # ============================================================
    def _discover_ext(self, root_folder,format=".pdf"):
        """
        Recursively discover all PDF files inside root_folder.
        """
        pdfs = []
        for root, dirs, files in os.walk(root_folder):
            for f in files:
                if f.lower().endswith(format):
                    pdfs.append(os.path.join(root, f))
        return pdfs

    def _discover_all_pdfs(self,source_path):
        # ---------------------------
        # Discover all PDFs/Oth
        # ---------------------------
        pdfs = self._discover_ext(source_path)
        txts = self._discover_ext(source_path, ".txt")

        pdfs.extend(txts)

        self.logger.do_log(
            f"[RAG-INGEST] 📄 Found {len(pdfs)} PDF/TXT(s) to process.",
            MessageType.INFO
        )

        for i, pdf_file in enumerate(pdfs):
            self.logger.do_log(
                f"[RAG-INGEST] [{i + 1}/{len(pdfs)}] {pdf_file}",
                MessageType.INFO
            )

        if len(pdfs) == 0:
            self.logger.do_log(
                "[RAG-INGEST] ❌ No PDF/txt(s) found. Nothing to process.",
                MessageType.ERROR
            )
            return []

        return pdfs

    def _get_dest_root_folder(self,source_path,dest_root,out_folder):
        # --------------------------------------------------
        # Extract everything from dest_root onward (inclusive)
        # Result: \Archives\2025\November\Nov 6
        # --------------------------------------------------
        idx = source_path.find(dest_root)
        dest_footer = source_path[idx:] if idx != -1 else ""


        #Sanitize dest_footer
        normalized = os.path.normpath(dest_footer)
        parts = normalized.split(os.sep)

        clean_parts=[]
        for part in parts:
            part = self._format_folder(part)
            clean_parts.append(part)

        dest_footer=os.path.join(*clean_parts) #We need to remove Nov 6 --> Nov_6

        # --------------------------------------------------
        # Remove dest_footer (and anything after) from out_folder
        # Result: output\rag\ZERO_HEDGE_CHUNKS\Archives\2025\November\Nov_6
        # --------------------------------------------------
        idx2 = out_folder.find(dest_footer)
        out_folder_clean = out_folder[:idx2] if idx2 != -1 else out_folder

        return  os.path.join(out_folder_clean,dest_footer)

    # ==========================================================
    # PROCESS MULTIPLE PDFs
    # ==========================================================
    def run(self,source_path,log_posfix=None,ingest_type=None):
        """Run ingestion with two-layer skip logic + per-file logging + final summary."""

        self.current_run_log = self._make_run_log(source_path,log_posfix)
        ingestion_logger = RAGIngestionLogger(self.logger,source_path,log_posfix)

        start_ts = self.logger.now_iso() if hasattr(self.logger, "now_iso") else \
            __import__("datetime").datetime.utcnow().isoformat()

        summary = {"processed": 0, "skipped": 0, "errors": 0}

        details_path =None
        if log_posfix is None:
            details_path = os.path.join(self.logs_dir, "ingest_details.log")
        else:
            ts = datetime.utcnow().isoformat().replace(":", "-")
            details_path = os.path.join(self.logs_dir, f"ingest_details_{log_posfix}_{ts}.log")


        if ingest_type=="recurrent" and source_path=="*":
            last_sucessful_folder= ingestion_logger.get_last_successful_folder(self.logs_dir)
            next_folder= ingestion_logger.get_next_folder(last_sucessful_folder,self.dest_root)

            if not next_folder:
                self.logger.do_log(f"[RAG] No folders with PDFs → next to {last_sucessful_folder} --> Nothing to process.", 1)
                return
            else:
                self.logger.do_log(f"[RAG] Next Folder Found: {next_folder}", 1)
                source_path=next_folder


        #Validate Source Path
        if not self._validate_source_paths(source_path,self.dest_root):
            return

        #Fetch all pdfs
        pdf_list= self._discover_all_pdfs(source_path)
        out_folder=None
        for pdf_path in pdf_list:

            # -------- Initial QUEUED log --------
            with open(details_path, "a", encoding="utf-8", buffering=1) as lf:
                lf.write(f"{start_ts} | queued | {pdf_path}\n")

            # -------- Combined skip logic --------
            corpus_meta = self.global_metadata.get(pdf_path, {})

            if self.ingest_meta.should_skip(pdf_path, corpus_meta):
                self.logger.do_log(f"[RAG] ⏩ FULL-SKIP: {pdf_path}", 1)
                summary["skipped"] += 1

                with open(details_path, "a", encoding="utf-8", buffering=1) as lf:
                    lf.write(f"{start_ts} | full-skip | {pdf_path}\n")

                continue

            # -------- Process PDF --------
            self.logger.do_log(f"[RAG] 🔥 Processing PDF: {pdf_path}", 1)
            res = self.process_pdf(pdf_path)

            if res:
                summary["processed"] += 1
                self.ingest_meta.mark_processed(pdf_path)

                with open(details_path, "a", encoding="utf-8", buffering=1) as lf:
                    lf.write(f"{start_ts} | processed | {pdf_path}\n")

                if out_folder is None:
                    out_folder=res[3]
                    out_folder=self._get_dest_root_folder(source_path,self.dest_root,out_folder)

            else:
                summary["errors"] += 1

                with open(details_path, "a", encoding="utf-8", buffering=1) as lf:
                    lf.write(f"{start_ts} | error | {pdf_path}\n")

        # -------- Save ingest metadata --------
        self.ingest_meta.save()

        # -------- Final summary --------
        end_ts = self.logger.now_iso() if hasattr(self.logger, "now_iso") else \
            __import__("datetime").datetime.utcnow().isoformat()

        ingestion_logger.finalize_run_and_update_state(log_root_path=self.logs_dir,
                                                       current_run_log=self.current_run_log,out_folder=out_folder,
                                                        start_ts=start_ts,
                                                       end_ts=end_ts, pdf_list=pdf_list, source_path=source_path,
                                                       summary=summary, log_posfix=log_posfix)



        self.logger.do_log("[RAG] ✅ Completed full batch ingestion.", 1)

