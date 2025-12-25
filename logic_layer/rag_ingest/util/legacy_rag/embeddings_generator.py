"""
TransformersEmbeddingsGenerator
-------------------
Computes embeddings using BGE-small-en.
"""

from sentence_transformers import SentenceTransformer

class EmbeddingsGenerator:

    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-small-en")

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)
