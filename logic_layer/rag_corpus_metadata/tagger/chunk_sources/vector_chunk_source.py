# FILE: logic_layer/rag_corpus_metadata/tagger/chunk_sources/vector_chunk_source.py
# Chunks and embeddings read from the pgvector store.
#
# Nothing is re-encoded here: the vectors were computed once by
# vectorize_documents and are reused as they are. That is the whole point -
# tagging a document a second time costs a database read instead of a full
# forward pass over the filing.
#
# The embedding model is part of the lookup, so tagging with mpnet only ever
# sees mpnet vectors even when the same filing was also vectorized with another
# model.

import torch

from common.util.extractors.K_Q_10.k_q_10_html_structured_block_extractor import (
    KQ10HtmlStructuredBlockExtractor,
)
from data_access_layer.vectors.filing_vectors_manager import FilingVectorsManager
from framework.common.logger.message_type import MessageType
from logic_layer.rag_corpus_metadata.tagger.chunk_sources.base_chunk_source import BaseChunkSource


class VectorChunkSource(BaseChunkSource):

    MODE = "VECTORS"
    DESCRIPTION = "chunks and embeddings read from pgvector, nothing re-encoded"

    def __init__(self, tagger, logger=None, vectors_db_config=None):
        super().__init__(tagger, logger)

        self.embedding_model = tagger.tag_cfg.tag_model
        self.vectors_mgr = FilingVectorsManager(vectors_db_config, logger)
        self.extractor = KQ10HtmlStructuredBlockExtractor()

        self._log(f"[RANK][VECTORS] 🔌 Connected | {self.vectors_mgr.ping()} | "
                  f"model={self.embedding_model}")

    def get_chunks(self, sec_w_file, file_name, fiscal_year, quarter, job_id=None):
        symbol = sec_w_file.security.symbol
        report_type = self.extractor.resolve_report_type(file_name)

        rows = self.vectors_mgr.get_document_chunks(
            symbol=symbol,
            report_type=report_type,
            fiscal_year=fiscal_year,
            quarter=quarter or "",
            file_name=file_name,
            embedding_model=self.embedding_model,
        )

        if not rows:
            # Not an error: this filing simply has not been vectorized yet with
            # this model. Saying so explicitly beats a silent zero-score row.
            self._log(
                f"[RANK][VECTORS] ⚠ No stored vectors | {symbol} | file={file_name} | "
                f"model={self.embedding_model} - run vectorize_documents for it first",
                job_id, MessageType.WARNING
            )
            return [], None

        texts = [row["chunk_text"] for row in rows]
        embeddings = torch.tensor([row["embedding"] for row in rows], dtype=torch.float32)

        sections = sorted({row["section_label"] for row in rows})
        self._log(
            f"[RANK][VECTORS] ✔ {symbol} | file={file_name} | chunks={len(texts)} | "
            f"dim={embeddings.shape[1]} | sections={', '.join(sections)}",
            job_id
        )

        return texts, embeddings

    def close(self):
        self.vectors_mgr.close()
