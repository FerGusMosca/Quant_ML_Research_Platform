# FILE: transformers_single_security_topic_tagger.py

import os
import torch
from framework.common.logger.message_type import MessageType
from logic_layer.rag_corpus_metadata.tagger.transformers_topic_base import TransformersTopicBase


class TransformersSingleSecurityTopicTagger(TransformersTopicBase):

    def __init__(self, logger, tag_cfg=None):
        super().__init__(logger, tag_cfg)
        self.tag_dict=None

        self.logger.do_log("[BERT] Single Security Topic Tagger READY", MessageType.INFO)

    def analyze(
            self,
            security_symbol: str,
            file_path: str,
            job_id: int = None,
            top_k_chunks: int = 5,
    ):
        """
        Performs topic-based semantic ranking for a single security.

        - Sentences split with NLTK
        - Chunks generated via KMeans clustering + overlap
        - NO thresholds
        - Pure semantic ranking
        - Returns top-K most representative chunks per topic
        """

        file_name = os.path.basename(file_path)

        if self.logger:
            self.logger.do_log(
                f"[SINGLE][ANALYZE][START] security={security_symbol} | file={file_name}",
                MessageType.INFO,
                job_id
            )

        # --------------------------------------------------
        # Extract full text
        # --------------------------------------------------
        text = self._extract_text(file_path, job_id)
        if not text:
            if self.logger:
                self.logger.do_log(
                    f"[SINGLE][ANALYZE][EMPTY_TEXT] security={security_symbol}",
                    MessageType.WARNING,
                    job_id
                )
            return None

        if self.logger:
            self.logger.do_log(
                f"[SINGLE][ANALYZE][TEXT] chars={len(text)}",
                MessageType.INFO,
                job_id
            )

        # --------------------------------------------------
        # Generate semantically coherent chunks
        # --------------------------------------------------
        chunks = self.chunk_generator.chunk(text,self.tag_cfg.tag_dedup, job_id)
        if not chunks:
            if self.logger:
                self.logger.do_log(
                    f"[SINGLE][ANALYZE][NO_CHUNKS] security={security_symbol}",
                    MessageType.WARNING,
                    job_id
                )
            return None

        if self.logger:
            self.logger.do_log(
                f"[SINGLE][ANALYZE][CHUNKS] count={len(chunks)}",
                MessageType.INFO,
                job_id
            )

        # --------------------------------------------------
        # Encode chunks
        # --------------------------------------------------
        chunk_embeddings = []
        for idx, chunk in enumerate(chunks):
            emb = self._encode(chunk).squeeze(0)
            chunk_embeddings.append(emb)

        if self.logger:
            self.logger.do_log(
                f"[SINGLE][ANALYZE][CHUNK_EMBEDDINGS] count={len(chunk_embeddings)}",
                MessageType.INFO,
                job_id
            )

        report = {
            "security": security_symbol,
            "file": file_name,
            "topics": {}
        }

        # --------------------------------------------------
        # Encode topic phrases (semantic anchors)
        # --------------------------------------------------
        topic_embeddings = {}
        for topic, phrases in self.tag_dict.items():
            topic_embeddings[topic] = []
            for phrase in phrases:
                topic_embeddings[topic].append({
                    "phrase": phrase,
                    "embedding": self._encode(phrase).squeeze(0)
                })

            if self.logger:
                self.logger.do_log(
                    f"[SINGLE][TOPIC][PHRASES] topic={topic} | phrases={len(phrases)}",
                    MessageType.INFO,
                    job_id
                )

        # --------------------------------------------------
        # PURE semantic ranking per topic
        # --------------------------------------------------
        for topic, phrase_embs in topic_embeddings.items():
            matches = []

            for chunk_idx, chunk_emb in enumerate(chunk_embeddings):
                for pe in phrase_embs:
                    score = float(torch.dot(chunk_emb, pe["embedding"]))
                    matches.append({
                        "chunk_idx": chunk_idx,
                        "score": score,
                        "matched_phrase": pe["phrase"],
                        "chunk_text": chunks[chunk_idx],
                    })

            if not matches:
                if self.logger:
                    self.logger.do_log(
                        f"[SINGLE][TOPIC][NO_MATCHES] topic={topic}",
                        MessageType.WARNING,
                        job_id
                    )
                continue

            matches.sort(key=lambda x: x["score"], reverse=True)
            top_matches = matches[:top_k_chunks]

            report["topics"][topic] = {
                "top_score": top_matches[0]["score"],
                "matches": top_matches,
                "summary": (
                    f"Top {len(top_matches)} most semantically aligned text segments "
                    f"for topic '{topic}', ranked by embedding similarity."
                ),
            }

            if self.logger:
                self.logger.do_log(
                    f"[SINGLE][TOPIC][DONE] security={security_symbol} | topic={topic} | "
                    f"top_score={top_matches[0]['score']:.4f} | "
                    f"matches={len(top_matches)}",
                    MessageType.INFO,
                    job_id
                )

        if self.logger:
            self.logger.do_log(
                f"[SINGLE][ANALYZE][END] security={security_symbol} | topics={len(report['topics'])}",
                MessageType.INFO,
                job_id
            )

        return report


