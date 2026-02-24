# FILE: semantic_chunk_deduper.py
# All comments MUST be in English.

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from logic_layer.rag_ingest.util.multi_stage_rag.perf_optimization.model_registry import ModelRegistry


class TranfomersSemanticChunkDeduper:
    """
    Removes semantically duplicated chunks using transformer embeddings + cosine similarity.
    This runs AFTER the vanilla hash-based deduper.
    """

    def __init__(
        self,
        logger,
        model_name=None,
        similarity_threshold=0.95
    ):
        self.logger = logger
        self.similarity_threshold = similarity_threshold
        self.model_name=model_name if model_name is not None else "sentence-transformers/all-MiniLM-L6-v2"
        self.model = ModelRegistry.get(self.model_name)

        if self.logger:
            self.logger.do_log(
                f"[SEM-DEDUP] 🧠 Semantic deduper initialized | model={model_name} | threshold={similarity_threshold}",
                1
            )

    def dedup_chunks(self, chunks):
        if not chunks:
            return []

        if self.logger:
            self.logger.do_log(
                f"[SEM-DEDUP] 🔍 Starting semantic deduplication for {len(chunks)} chunks",
                1
            )

        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        kept_chunks = []
        kept_embeddings = []

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            is_duplicate = False

            for kept_idx, kept_emb in enumerate(kept_embeddings):
                sim = float(np.dot(emb, kept_emb))  # cosine (normalized)

                if sim >= self.similarity_threshold:
                    is_duplicate = True
                    if self.logger:
                        self.logger.do_log(
                            f"[SEM-DEDUP] 🔁 Chunk #{idx} dropped (cosine={sim:.3f}) vs kept #{kept_idx}",
                            2
                        )
                    break

            if not is_duplicate:
                kept_chunks.append(chunk)
                kept_embeddings.append(emb)

        if self.logger:
            self.logger.do_log(
                f"[SEM-DEDUP] ✅ Semantic unique chunks kept: {len(kept_chunks)} / {len(chunks)}",
                1
            )

        return kept_chunks
