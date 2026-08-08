# FILE: logic_layer/rag_corpus_metadata/vectorization/document_vectorization_processor.py
# Owns the whole vectorization of a batch of documents: section extraction,
# chunking, encoding and persistence into Postgres/pgvector.
#
# Mirrors the role TransformersTopicTagger.rank() plays for tagging: the
# orchestration layer resolves securities and files, then hands the batch over
# here and gets a summary back.
#
# The document family is resolved through SectionExtractorRegistry, so adding
# transcripts or PDFs later never touches this class.

import hashlib
import os
from datetime import datetime

from common.util.extractors.section_extractors.section_extractor_registry import SectionExtractorRegistry
from common.util.std_in_out.raw_file_reader import RawFileReader
from data_access_layer.vectors.filing_vectors_manager import FilingVectorsManager
from framework.common.logger.message_type import MessageType
from logic_layer.rag_corpus_metadata.tagger.transformers_topic_base import TransformersTopicBase


class DocumentVectorizationProcessor(TransformersTopicBase):

    def __init__(self, logger, tag_cfg, vectors_db_config=None):
        super().__init__(logger, tag_cfg)

        self.section_extractor = SectionExtractorRegistry.get(tag_cfg.doc_type)
        self.vectors_mgr = FilingVectorsManager(vectors_db_config, logger)
        self.embedding_model = tag_cfg.tag_model

    # ------------------------------------------------------
    # Helpers
    # ------------------------------------------------------

    @staticmethod
    def file_hash(raw_text: str) -> str:
        """Lets a rerun skip files that have not changed since the last vectorization."""
        return hashlib.sha256(raw_text.encode("utf-8", errors="ignore")).hexdigest()

    def close(self):
        self.vectors_mgr.close()

    def ping(self) -> str:
        return self.vectors_mgr.ping()

    # ------------------------------------------------------
    # Single document
    # ------------------------------------------------------

    def read_document(self, file_path: str, file_name: str):
        """
        Cheap first pass: reads the file and derives its identity (sub type + hash).
        No model, no chunking. This is what the skip check runs against, so an
        already vectorized file costs a file read instead of a full encode.
        """
        raw_text = RawFileReader.get_raw_text(file_path)
        if not raw_text:
            return None, None, None

        return self.section_extractor.resolve_sub_type(file_name), self.file_hash(raw_text), raw_text

    def vectorize_file(self, file_path: str, file_name: str, job_id=None,
                       raw_text: str = None, sub_type: str = None, content_hash: str = None):
        """
        Returns (sub_type, content_hash, chunks). Each chunk carries the section it
        came from, its text and its embedding. Returns empty chunks when the file
        has nothing worth vectorizing.

        raw_text / sub_type / content_hash can be passed in when the caller already
        read the file through read_document(), to avoid reading it twice.
        """
        start_time = datetime.now()

        if raw_text is None:
            sub_type, content_hash, raw_text = self.read_document(file_path, file_name)
            if not raw_text:
                return None, None, []

        sections = self.section_extractor.extract_sections(raw_text, file_name)

        if self.logger:
            detail = " | ".join(f"{k}={len(v)}c" for k, v in sections.items()) or "none"
            self.logger.do_log(
                f"[VECTORIZE] 📑 {file_name} | type={sub_type} | sections={len(sections)} | {detail}",
                MessageType.INFO, job_id
            )

        if not sections:
            return sub_type, content_hash, []

        # 1) chunk every section on its own, so the label is never lost
        labelled_texts = []
        for label, section_text in sections.items():
            try:
                sub_chunks = self.chunk_generator.chunk(
                    section_text, tag_dedup=self.tag_cfg.tag_dedup, job_id=job_id
                )
            except Exception as e:
                if self.logger:
                    self.logger.do_log(
                        f"[VECTORIZE] ⚠ chunking failed | section={label} | {e}",
                        MessageType.INFO, job_id
                    )
                continue

            for text in sub_chunks:
                if text and text.strip():
                    labelled_texts.append((label, text.strip()))

        if not labelled_texts:
            return sub_type, content_hash, []

        # 2) one batched forward pass for the whole document
        embeddings = self._encode_batch([text for _, text in labelled_texts])

        chunks = []
        for index, ((label, text), embedding) in enumerate(zip(labelled_texts, embeddings)):
            chunks.append({
                "section_label": label,
                "chunk_index": index,
                "chunk_text": text,
                "word_count": len(text.split()),
                "embedding": embedding.tolist(),
            })

        if self.logger:
            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.do_log(
                f"[VECTORIZE] ✔ {file_name} | chunks={len(chunks)} | "
                f"dim={len(chunks[0]['embedding'])} | {elapsed:.1f}s "
                f"({len(chunks) / max(elapsed, 0.001):.1f} chunks/s)",
                MessageType.INFO, job_id
            )

        return sub_type, content_hash, chunks

    # ------------------------------------------------------
    # Batch — this is the entry point the orchestration layer calls
    # ------------------------------------------------------

    def vectorize(self, sec_w_files: list, portfolio, sector, source, fiscal_year,
                  quarter, overwrite=False, job_id=None) -> dict:
        """
        Vectorizes and persists every matched file. Resumable: a file already
        stored for this model with the same content hash is skipped.
        """
        stats = {"files_found": len(sec_w_files), "processed": 0, "skipped": 0,
                 "failed": 0, "chunks": 0}

        if not sec_w_files:
            return stats

        # report_type is only known per file, so the run row uses the first one
        run_sub_type = self.section_extractor.resolve_sub_type(
            os.path.basename(sec_w_files[0].file)
        )

        run_id = self.vectors_mgr.start_run(
            job_id=job_id, portfolio=portfolio, sector_code=sector,
            report_type=run_sub_type, fiscal_year=fiscal_year, quarter=quarter or "",
            embedding_model=self.embedding_model, files_found=len(sec_w_files),
        )

        for idx, sec_w_file in enumerate(sec_w_files, start=1):
            file_path = sec_w_file.file
            file_name = os.path.basename(file_path)
            symbol = sec_w_file.security.symbol

            if self.logger:
                self.logger.do_log(
                    f"[VECTORIZE] ▶ ({idx}/{len(sec_w_files)}) {symbol} | file={file_name}",
                    MessageType.INFO, job_id
                )

            try:
                # 1) cheap pass: identify the file without loading the model.
                #    A nightly rerun over an unchanged corpus never gets past here.
                sub_type, content_hash, raw_text = self.read_document(file_path, file_name)

                if not raw_text:
                    stats["failed"] += 1
                    if self.logger:
                        self.logger.do_log(
                            f"[VECTORIZE] ❌ Empty or unreadable file | file={file_name}",
                            MessageType.WARNING, job_id
                        )
                    continue

                if not overwrite and self.vectors_mgr.is_already_vectorized(
                        symbol, sub_type, fiscal_year, quarter or "", file_name,
                        self.embedding_model, content_hash):
                    stats["skipped"] += 1
                    if self.logger:
                        self.logger.do_log(
                            f"[VECTORIZE] ⏭ Already vectorized, skipping encode | "
                            f"{symbol} | file={file_name}",
                            MessageType.INFO, job_id
                        )
                    continue

                # 2) expensive pass: only files that actually need it get encoded
                sub_type, content_hash, chunks = self.vectorize_file(
                    file_path, file_name, job_id,
                    raw_text=raw_text, sub_type=sub_type, content_hash=content_hash
                )

                if not chunks:
                    stats["failed"] += 1
                    if self.logger:
                        self.logger.do_log(
                            f"[VECTORIZE] ❌ No chunks generated | file={file_name}",
                            MessageType.WARNING, job_id
                        )
                    continue

                document_id = self.vectors_mgr.upsert_document(
                    symbol=symbol,
                    cik=getattr(sec_w_file.security, "cik", None),
                    report_type=sub_type,
                    fiscal_year=fiscal_year,
                    quarter=quarter or "",
                    portfolio=portfolio,
                    sector_code=sector,
                    source_folder=source,
                    file_name=file_name,
                    file_path=file_path,
                    content_hash=content_hash,
                )

                # A rerun replaces the previous chunks of this model instead of
                # leaving orphans behind when the chunk count changes.
                self.vectors_mgr.delete_chunks(document_id, self.embedding_model)

                persisted = self.vectors_mgr.persist_chunks(
                    document_id, self.embedding_model, chunks, job_id
                )
                self.vectors_mgr.set_section_count(
                    document_id, len({c["section_label"] for c in chunks})
                )

                stats["processed"] += 1
                stats["chunks"] += persisted

            except Exception as e:
                stats["failed"] += 1
                if self.logger:
                    self.logger.do_log(
                        f"[VECTORIZE] ❌ Failed | {symbol} | file={file_name} | "
                        f"{e.__class__.__name__}: {e}",
                        MessageType.ERROR, job_id
                    )

        self.vectors_mgr.finish_run(
            run_id=run_id,
            files_processed=stats["processed"],
            files_skipped=stats["skipped"],
            files_failed=stats["failed"],
            chunks_persisted=stats["chunks"],
            status="FINISHED" if stats["failed"] == 0 else "FINISHED_WITH_ERRORS",
        )

        return stats

    def encode_query(self, query_text: str):
        """Embeds free text with the same model, for semantic search."""
        return self._encode_batch([query_text])[0].tolist()
