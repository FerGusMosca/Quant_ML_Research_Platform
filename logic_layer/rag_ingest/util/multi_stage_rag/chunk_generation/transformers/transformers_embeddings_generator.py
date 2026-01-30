# multi_stage_rag/embeddings_generator.py

from sentence_transformers import SentenceTransformer


class TransformersEmbeddingsGenerator:

    def __init__(self,embedding_model=None, logger=None):
        self.logger = logger
        self.embedding_model = embedding_model if embedding_model is not None else "BAAI/bge-large-en-v1.5"
        try:
            if self.logger: self.logger.do_log(f"[MSC] 🔧 Loading {self.embedding_model}...", 1)
            self.model = SentenceTransformer(self.embedding_model)
        except Exception as e:
            if self.logger: self.logger.do_log(f"[MSC] ❌ Failed to load {self.embedding_model}: {e}", 0)
            raise

    def embed(self, texts):
        try:
            if self.logger: self.logger.do_log(f"[MSC] 🔍 Embedding {len(texts)} chunks with model {self.embedding_model}...", 2)
            vecs = self.model.encode(texts, normalize_embeddings=True)
            if self.logger: self.logger.do_log("[MSC] ✅ Embeddings generated.", 1)
            return vecs
        except Exception as e:
            if self.logger: self.logger.do_log(f"[MSC] ❌ Embedding failed: {e}", 0)
            return []
