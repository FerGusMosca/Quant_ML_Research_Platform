# multi_stage_rag/transformers_chunk_overlapping.py
# All comments MUST be in English.

import numpy as np
from sentence_transformers import SentenceTransformer


class TransformersChunkOverlapping:
    """
    Decides whether two chunks should be merged based on semantic similarity.
    Uses a fast transformer (MiniLM) for cheap similarity decisions.
    """

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold=0.85,
        logger=None
    ):
        self.logger = logger
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer(model_name)

    def should_merge(self, chunk_a: str, chunk_b: str) -> bool:
        vecs = self.model.encode([chunk_a, chunk_b], normalize_embeddings=True)
        similarity = float(np.dot(vecs[0], vecs[1]))

        if self.logger:
            self.logger.do_log(
                f"[TCO] 🔗 Chunk similarity={similarity:.4f} (threshold={self.similarity_threshold})",
                3
            )

        return similarity >= self.similarity_threshold
