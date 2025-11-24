# multi_stage_rag/chunk_generator.py
# All comments MUST be in English.

import nltk
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


class ChunkGenerator:

    @staticmethod
    def chunk(text: str, target_tokens=180, overlap_tokens=40, k=3, logger=None):
        try:
            if logger:
                logger.do_log("[MSC] 🔍 Starting Multi-Stage Chunking...", 1)

            # ===== Stage 1: Sentence segmentation =====
            try:
                sentences = nltk.sent_tokenize(text)
                if logger: logger.do_log(f"[MSC] 📌 Sentences extracted: {len(sentences)}", 2)
            except Exception as e:
                if logger: logger.do_log(f"[MSC] ❌ Sentence split failed: {e}", 0)
                return []

            if len(sentences) == 0:
                if logger: logger.do_log("[MSC] ❌ No sentences found.", 0)
                return []

            # ===== Stage 2: Sentence embeddings =====
            try:
                model = SentenceTransformer("BAAI/bge-small-en")
                embeddings = model.encode(sentences, normalize_embeddings=True)
                if logger: logger.do_log(f"[MSC] 📌 Embeddings computed: {embeddings.shape}", 2)
            except Exception as e:
                if logger: logger.do_log(f"[MSC] ❌ Embeddings failed: {e}", 0)
                return []

            # ===== Stage 3: K-means clustering (semantic grouping) =====
            try:
                real_k = min(k, len(embeddings))
                if logger: logger.do_log(f"[MSC] 🎯 Clustering k={real_k}", 2)

                kmeans = KMeans(n_clusters=real_k, n_init=5)
                labels = kmeans.fit_predict(embeddings)
            except Exception as e:
                if logger: logger.do_log(f"[MSC] ❌ K-means failed: {e}", 0)
                return []

            clusters = {}
            for s, lab in zip(sentences, labels):
                clusters.setdefault(lab, []).append(s)

            # ===== Stage 4: Dynamic chunking =====
            final_chunks = []

            for lab, sents in clusters.items():
                if logger:
                    logger.do_log(f"[MSC] 📦 Processing cluster {lab} ({len(sents)} sentences)", 2)

                current = []
                tok_count = 0

                for s in sents:
                    t = s.split()

                    if tok_count + len(t) > target_tokens:
                        final_chunks.append(" ".join(current))

                        # Overlap
                        overlap = current[-overlap_tokens:] if overlap_tokens < len(current) else current
                        current = overlap.copy()
                        tok_count = len(" ".join(current).split())

                    current.append(s)
                    tok_count += len(t)

                if current:
                    final_chunks.append(" ".join(current))

            if logger:
                logger.do_log(f"[MSC] ✅ Final chunks: {len(final_chunks)}", 1)

            return final_chunks

        except Exception as e:
            if logger:
                logger.do_log(f"[MSC] ❌ UNEXPECTED ERROR in chunk(): {e}", 0)
            return []
