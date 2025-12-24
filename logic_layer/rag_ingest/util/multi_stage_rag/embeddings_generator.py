# multi_stage_rag/embeddings_generator.py

from sentence_transformers import SentenceTransformer


class EmbeddingsGenerator:

    def __init__(self,embedding_model="BAAI/bge-large-en-v1.5", logger=None):
        self.logger = logger
        try:
            if self.logger: self.logger.do_log(f"[MSC] 🔧 Loading {embedding_model}...", 1)
            self.model = SentenceTransformer(embedding_model)
        except Exception as e:
            if self.logger: self.logger.do_log(f"[MSC] ❌ Failed to load {embedding_model}: {e}", 0)
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
