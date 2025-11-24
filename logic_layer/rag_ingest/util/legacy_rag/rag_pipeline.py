"""
RAGPipeline
PDF → text → clean → chunks → metadata → embeddings → save
"""

import os
import json
import re
import numpy as np

from logic_layer.rag_ingest.util.legacy_rag.pdf_text_extractor import PDFTextExtractor
from logic_layer.rag_ingest.util.legacy_rag.pdf_cleaner import PDFCleaner
from logic_layer.rag_ingest.util.legacy_rag.chunk_generator import ChunkGenerator
from logic_layer.rag_ingest.util.legacy_rag.metadata_builder import MetadataBuilder
from logic_layer.rag_ingest.util.legacy_rag.embeddings_generator import EmbeddingsGenerator


class RAGPipeline:

    def __init__(self, dest_root, config, logger):
        self.logger = logger
        self.dest_root = dest_root
        self.embedder = EmbeddingsGenerator()

        self.output_base = config["RAG_OUTPUT_FOLDER"]
        os.makedirs(self.output_base, exist_ok=True)

    # ==========================================================
    # Compute output folder relative to dest_root
    # ==========================================================
    def _compute_output_path(self, pdf_path: str) -> str:
        """
        Returns folder path starting at dest_root and EXCLUDING the PDF filename.
        Example:
        Archives/2025/Nov/Nov 6
        """
        normalized = os.path.normpath(pdf_path)
        parts = normalized.split(os.sep)

        if self.dest_root not in parts:
            raise ValueError(
                f"[RAG] ERROR: dest_root='{self.dest_root}' not found in path: {pdf_path}"
            )

        idx = parts.index(self.dest_root)

        # === CRITICAL FIX ===
        # Extract ONLY directories → remove the last element (PDF filename)
        folder_parts = parts[idx:-1]

        return os.path.join(*folder_parts)

    # ==========================================================
    # SAFE WINDOWS FOLDER NAME SANITIZER
    # ==========================================================
    def _sanitize_filename(self, name: str) -> str:
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

    # ==========================================================
    # PROCESS ONE PDF
    # ==========================================================
    def process_pdf(self, pdf_path: str):
        self.logger.do_log(f"[RAG] Extracting text: {pdf_path}", 1)

        raw_text = PDFTextExtractor.extract_text(pdf_path)
        clean_text = PDFCleaner.clean(raw_text)
        chunks = ChunkGenerator.chunk(clean_text)
        metadata = [MetadataBuilder.build(pdf_path, idx)
                    for idx in range(len(chunks))]
        embeddings = self.embedder.embed(chunks)

        # --------------------------------------------
        # Determine output folder structure
        # --------------------------------------------
        try:
            # rel_folder MUST NOT include the PDF filename
            rel_folder = os.path.normpath(self._compute_output_path(pdf_path))
        except Exception as e:
            self.logger.do_log(str(e), 0)
            raise

        # sanitize PDF filename
        base_raw = os.path.basename(pdf_path).replace(".pdf", "")
        sanitized_name = self._sanitize_filename(base_raw)

        out_dir = os.path.join(self.output_base, rel_folder, sanitized_name)
        out_dir = os.path.normpath(
            os.path.join(self.output_base, rel_folder, sanitized_name)
        )
        os.makedirs(out_dir, exist_ok=True)

        # --------------------------------------------
        # Save artifacts
        # --------------------------------------------
        with open(os.path.join(out_dir, "chunks.txt"), "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c + "\n\n")

        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)

        self.logger.do_log(f"[RAG] ✅ Artifacts saved → {out_dir}", 1)

        return chunks, metadata, embeddings

    # ==========================================================
    # PROCESS MULTIPLE PDFs
    # ==========================================================
    def run(self, pdf_list):
        for pdf_path in pdf_list:
            self.logger.do_log(f"[RAG] 🔥 Processing PDF: {pdf_path}", 1)
            self.process_pdf(pdf_path)

        self.logger.do_log("[RAG] ✅ Completed full batch ingestion.", 1)
