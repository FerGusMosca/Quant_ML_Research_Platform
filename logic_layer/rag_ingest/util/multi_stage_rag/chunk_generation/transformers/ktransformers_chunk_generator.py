# multi_stage_rag/ktransformers_chunk_generator.py
# All comments MUST be in English.

import nltk

from framework.common.logger.message_type import MessageType
from logic_layer.rag_ingest.util.multi_stage_rag.chunk_generation.transformers.kmeans_sentence_clustering import \
    KMeansSentenceClustering
from logic_layer.rag_ingest.util.multi_stage_rag.chunk_generation.transformers.transformers_chunk_overlapping import \
    TransformersChunkOverlapping


class KTransformersChunkGenerator:
    """
    High-level chunk generator.
    Orchestrates sentence clustering, token-budget chunking,
    and semantic-aware chunk merging.
    """

    def __init__(
        self,
        target_tokens=180,
        model_name=None,
        k=3,
        logger=None
    ):
        self.target_tokens = target_tokens
        self.k = k
        self.logger = logger
        self.model_name= model_name if model_name is not None else "BAAI/bge-small-en-v1.5"

        self.clusterer = KMeansSentenceClustering(model_name=self.model_name,logger=logger)
        self.overlapper = TransformersChunkOverlapping(model_name=self.model_name,logger=logger)

    def chunk(self, text: str,job_id:str=None):
        sentences = nltk.sent_tokenize(text)

        if self.logger:
            self.logger.do_log(f"[KTG] 📌 Total sentences: {len(sentences)}", 2)

        sentence_groups = self.clusterer.cluster(sentences, self.k,job_id)

        chunks = []

        for group_idx, group in enumerate(sentence_groups):
            if self.logger:
                self.logger.do_log(
                    f"[KTG] 📦 Processing sentence group {group_idx} ({len(group)} sentences)",
                    MessageType.INFO,job_id
                )

            current = []
            tok_count = 0

            for sentence in group:
                tokens = sentence.split()
                token_len = len(tokens)

                if tok_count + token_len > self.target_tokens:
                    new_chunk = " ".join(current)

                    if chunks and self.overlapper.should_merge(chunks[-1], new_chunk,job_id):
                        if self.logger:
                            self.logger.do_log("[KTG] 🔗 Merging chunks based on semantic similarity", MessageType.INFO,job_id)
                        chunks[-1] += " " + new_chunk
                    else:
                        chunks.append(new_chunk)

                    current = []
                    tok_count = 0

                current.append(sentence)
                tok_count += token_len

            if current:
                chunks.append(" ".join(current))

        if self.logger:
            self.logger.do_log(f"[KTG] ✅ Final chunks generated: {len(chunks)}", MessageType.INFO,job_id)

        return chunks
