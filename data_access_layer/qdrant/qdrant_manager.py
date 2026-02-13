from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct


class QdrantManager:
    def __init__(self, host="localhost", port=6333, collection="chunks"):
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection

        info = self.client.get_collection(collection)
        self.vector_size = info.config.params.vectors.size
        self.dummy_vector = [0.0] * self.vector_size

    def upsert_metadata(self, chunk_id, payload):
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=chunk_id,
                vector=self.dummy_vector,
                payload=payload
            )]
        )

    def upsert_vector(self, chunk_id, vector, payload):
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=chunk_id, vector=vector, payload=payload)]
        )

    def update_status(self, chunk_id, status):
        self.client.set_payload(
            collection_name=self.collection,
            payload={"status": status},
            points=[chunk_id],
        )