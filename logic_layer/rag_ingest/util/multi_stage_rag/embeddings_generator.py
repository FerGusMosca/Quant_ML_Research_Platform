# multi_stage_rag/embeddings_generator.py

from sentence_transformers import SentenceTransformer


class EmbeddingsGenerator:

    def __init__(self, logger=None):
        self.logger = logger
        try:
            if self.logger: self.logger.do_log("[MSC] 🔧 Loading BGE-large-en-v1.5...", 1)
            self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        except Exception as e:
            if self.logger: self.logger.do_log(f"[MSC] ❌ Failed to load BGE-large: {e}", 0)
            raise

    def embed(self, texts):
        try:
            if self.logger: self.logger.do_log(f"[MSC] 🔍 Embedding {len(texts)} chunks...", 2)
            vecs = self.model.encode(texts, normalize_embeddings=True)
            if self.logger: self.logger.do_log("[MSC] ✅ Embeddings generated.", 1)
            return vecs
        except Exception as e:
            if self.logger: self.logger.do_log(f"[MSC] ❌ Embedding failed: {e}", 0)
            return []
