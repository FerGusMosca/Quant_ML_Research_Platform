# logic_layer/rag_ingest/util/multi_stage_rag/rag_pipeline.py
# All comments MUST be in English.

import os
import json
import re
import numpy as np

from logic_layer.rag_ingest.util.multi_stage_rag.pdf_text_extractor import PDFTextExtractor
from logic_layer.rag_ingest.util.multi_stage_rag.pdf_cleaner import PDFCleaner
from logic_layer.rag_ingest.util.multi_stage_rag.chunk_generator import ChunkGenerator
from logic_layer.rag_ingest.util.multi_stage_rag.metadata_builder import MetadataBuilder
from logic_layer.rag_ingest.util.multi_stage_rag.embeddings_generator import EmbeddingsGenerator


class RAGPipeline:

    def __init__(self, dest_root, config, logger):
        self.logger = logger
        self.dest_root = dest_root

        try:
            self.embedder = EmbeddingsGenerator(logger=self.logger)
        except Exception as e:
            self.logger.do_log(f"[RAG] ❌ Failed to load embedding model: {e}", 0)
            raise

        self.output_base = config["RAG_OUTPUT_FOLDER"]
        try:
            os.makedirs(self.output_base, exist_ok=True)
        except Exception as e:
            logger.do_log(f"[RAG] ❌ Could not create output folder: {e}", 0)
            raise

    # ==========================================================
    # Compute output folder relative to dest_root
    # ==========================================================
    def _compute_output_path(self, pdf_path: str) -> str:
        try:
            normalized = os.path.normpath(pdf_path)
            parts = normalized.split(os.sep)

            if self.dest_root not in parts:
                raise ValueError(
                    f"[RAG] ERROR: dest_root='{self.dest_root}' not found in path: {pdf_path}"
                )

            idx = parts.index(self.dest_root)
            folder_parts = parts[idx:-1]

            return os.path.join(*folder_parts)

        except Exception as e:
            self.logger.do_log(f"[RAG] ❌ compute_output_path failed: {e}", 0)
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

        # ----- Chunking (Multi-Stage) -----
        try:
            chunks = ChunkGenerator.chunk(clean_text, logger=self.logger)
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

        # ----- Output paths -----
        try:
            rel_folder = os.path.normpath(self._compute_output_path(pdf_path))
        except Exception:
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

        return chunks, metadata, embeddings

    # ==========================================================
    # PROCESS MULTIPLE PDFs
    # ==========================================================
    def run(self, pdf_list):
        for pdf_path in pdf_list:
            self.logger.do_log(f"[RAG] 🔥 Processing PDF: {pdf_path}", 1)
            self.process_pdf(pdf_path)

        self.logger.do_log("[RAG] ✅ Completed full batch ingestion.", 1)
