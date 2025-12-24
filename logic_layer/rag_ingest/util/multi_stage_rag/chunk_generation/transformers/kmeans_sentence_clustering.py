# multi_stage_rag/kmeans_sentence_clustering.py
# All comments MUST be in English.

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


class KMeansSentenceClustering:
    """
    Groups sentences by semantic similarity using sentence embeddings + K-Means.
    This class decides WHAT sentences belong together (semantic responsibility only).
    """

    def __init__(self, model_name=None,k_units=5, logger=None):
        self.logger = logger
        self.model_name=model_name if model_name is not None else "BAAI/bge-small-en"
        self.model = SentenceTransformer(self.model_name)
        self.k_units=k_units

    def cluster(self, sentences, k):
        if not sentences:
            return []

        if self.logger:
            self.logger.do_log(f"[KMSC] 🔍 Encoding {len(sentences)} sentences", 2)

        embeddings = self.model.encode(sentences, normalize_embeddings=True)

        real_k = min(k, len(sentences))
        if self.logger:
            self.logger.do_log(f"[KMSC] 🎯 Running K-Means with k={real_k}", 2)

        labels = KMeans(n_clusters=real_k, n_init=self.k_units).fit_predict(embeddings)

        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(sentences[idx])

        return list(clusters.values())
