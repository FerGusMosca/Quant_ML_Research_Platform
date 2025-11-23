"""
RAGPipeline
-----------
PDF → text → clean → chunks → metadata → embeddings → save
"""

import os
import json
import re
import numpy as np

from logic_layer.rag_ingest.util.pdf_text_extractor import PDFTextExtractor
from logic_layer.rag_ingest.util.pdf_cleaner import PDFCleaner
from logic_layer.rag_ingest.util.chunk_generator import ChunkGenerator
from logic_layer.rag_ingest.util.metadata_builder import MetadataBuilder
from logic_layer.rag_ingest.util.embeddings_generator import EmbeddingsGenerator


class RAGPipeline:

    def __init__(self, config, logger):
        """
        :param config: loaded from commands_mgr.ini
        :param logger: shared framework logger
        """
        self.logger = logger
        self.embedder = EmbeddingsGenerator()

        self.output_folder = config["RAG_OUTPUT_FOLDER"]
        os.makedirs(self.output_folder, exist_ok=True)

    # -------------------------------------------------------------------
    # SAFE WINDOWS FOLDER NAME SANITIZER
    # -------------------------------------------------------------------
    def _sanitize_filename(self, name: str) -> str:
        """
        Produces a Windows-safe folder name:
            - removes illegal characters
            - removes repeated dots / triple dots
            - collapses whitespace to '_'
            - strips trailing dots/spaces (Windows forbidden)
            - enforces length limit
            - guarantees non-empty output
        """
        original = name

        # Remove forbidden chars
        sanitized = re.sub(r'[<>:"/\\|?*]', '', name)

        # Remove triple dots and repeated dots
        sanitized = sanitized.replace("...", "")
        sanitized = re.sub(r'\.{2,}', '.', sanitized)

        # Collapse whitespace into underscore
        sanitized = re.sub(r'\s+', '_', sanitized)

        # Strip trailing forbidden chars (🔥 Windows requirement)
        sanitized = sanitized.rstrip('. ')

        # Ensure not empty
        if len(sanitized) == 0:
            sanitized = "doc_" + str(abs(hash(original)) % 100000)

        # Length guard
        if len(sanitized) > 120:
            sanitized = sanitized[:110] + "_" + str(abs(hash(sanitized)) % 100000)

        return sanitized

    # -------------------------------------------------------------------
    # PROCESS ONE PDF
    # -------------------------------------------------------------------
    def process_pdf(self, pdf_path: str):
        self.logger.do_log(f"[RAG] Extracting text: {pdf_path}", 1)

        # Extract → clean → chunk
        raw_text = PDFTextExtractor.extract_text(pdf_path)
        clean_text = PDFCleaner.clean(raw_text)
        chunks = ChunkGenerator.chunk(clean_text)

        metadata = [MetadataBuilder.build(pdf_path, idx)
                    for idx in range(len(chunks))]

        embeddings = self.embedder.embed(chunks)

        # --------------------------------------------------------
        # SANITIZE OUTPUT FOLDER NAME
        # --------------------------------------------------------
        base_raw = os.path.basename(pdf_path).replace(".pdf", "")
        base = self._sanitize_filename(base_raw)

        self.logger.do_log(
            f"[RAG] Output folder name sanitized: "
            f"original='{base_raw}' → sanitized='{base}'",
            1
        )

        out_dir = os.path.join(self.output_folder, base)
        os.makedirs(out_dir, exist_ok=True)

        # Save chunks
        with open(os.path.join(out_dir, "chunks.txt"), "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c + "\n\n")

        # Save metadata
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Save embeddings
        np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)

        self.logger.do_log(f"[RAG] ✅ Artifacts saved → {out_dir}", 1)

        return chunks, metadata, embeddings

    # -------------------------------------------------------------------
    # PROCESS MULTIPLE PDFs
    # -------------------------------------------------------------------
    def run(self, pdf_list):
        for pdf_path in pdf_list:
            self.logger.do_log(f"[RAG] 🔥 Processing PDF: {pdf_path}", 1)
            self.process_pdf(pdf_path)

        self.logger.do_log("[RAG] ✅ Completed full batch ingestion.", 1)
