from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

class QdrantManager:
    DUMMY_VECTOR = [0.0] * 384
    def __init__(self, host="localhost", port=6333, collection="chunks"):
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection

    def upsert_metadata(self, chunk_id, payload):
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=chunk_id,
                vector=QdrantManager.DUMMY_VECTOR,
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
