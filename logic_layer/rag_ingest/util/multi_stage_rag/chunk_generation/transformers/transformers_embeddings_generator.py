# multi_stage_rag/embeddings_generator.py

from sentence_transformers import SentenceTransformer

from framework.common.logger.message_type import MessageType
from logic_layer.rag_ingest.util.multi_stage_rag.perf_optimization.model_registry import ModelRegistry


class TransformersEmbeddingsGenerator:

    def __init__(self,embedding_model=None, logger=None):
        self.logger = logger
        self.embedding_model = embedding_model if embedding_model is not None else "BAAI/bge-large-en-v1.5"
        try:
            if self.logger: self.logger.do_log(f"[MSC] 🔧 Loading {self.embedding_model}...", MessageType.INFO)
            self.model = ModelRegistry.get(self.embedding_model)
        except Exception as e:
            if self.logger: self.logger.do_log(f"[MSC] ❌ Failed to load {self.embedding_model}: {e}", MessageType.ERROR)
            raise

    def embed(self, texts):
        try:
            if self.logger: self.logger.do_log(f"[MSC] 🔍 Embedding {len(texts)} chunks with model {self.embedding_model}...", MessageType.INFO)
            vecs = self.model.encode(texts, normalize_embeddings=True)
            if self.logger: self.logger.do_log(f"[MSC] ✅ Embeddings generated. Shape: {vecs.shape}", MessageType.INFO)
            return vecs
        except Exception as e:
            if self.logger: self.logger.do_log(f"[MSC] ❌ Embedding failed: {e}", MessageType.ERROR)
            return []
