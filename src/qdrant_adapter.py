
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, models
from qdrant_client.models import (
    PointStruct,
    FilterSelector,
    Filter,
    FieldCondition,
    MatchValue,
    Distance
)

from embed_anything import Adapter, EmbedData
from typing import List, Dict
import os
import uuid
import numpy as np

class QdrantAdapter(Adapter):

    def __init__(self):
        if os.getenv("QDRANT_HOST"):
            self.client = QdrantClient(
                host=os.getenv("QDRANT_HOST"), port=6333, grpc_port=6334
            )
        else:
            self.client = QdrantClient(host="localhost", port=6333, grpc_port=6334)

    def create_index(self, dimension, metric, index_name):
        self.client.create_collection(
            collection_name=index_name,
            sparse_vectors_config={"text-sparse": models.SparseVectorParams()},
            vectors_config={"text": VectorParams(size=dimension, distance=metric)},
        )

    def delete_index(self, index_name):
        self.client.delete_collection(collection_name=index_name)

    def convert(
        self, embeddings: List[EmbedData], sparse_embeddings: List[EmbedData]
    ) -> List[PointStruct]:

        points = []
        for i, embedding in enumerate(embeddings):
            payload = embedding.metadata
            payload["text"] = embedding.text
            sparse_embedding = get_sparse_embedding(sparse_embeddings[i].embedding)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "text": embedding.embedding,
                        "text-sparse": models.SparseVector(
                            indices=sparse_embedding["indices"],
                            values=sparse_embedding["values"],
                        )
                    },
                    payload=payload,
                )
            )

        return points

    def upsert(
        self, data: List[EmbedData], sparse_data: List[EmbedData], index_name: str
    ) -> None:
        data = self.convert(data, sparse_data)
        self.client.upsert(collection_name=index_name, points=data)

    def delete_points(self, file_path: str, collection_name: str):
        self.client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_name", match=MatchValue(value=file_path)
                        )
                    ]
                )
            ),
        )

    def search_hybrid(
        self,
        collection_name: str,
        query_vector: List[float],
        query_sparse_vector: models.SparseVector,
    ):
        return self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(query=query_sparse_vector, using="text-sparse", limit=20),
                models.Prefetch(query=query_vector, using="text", limit=20),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
        )

    def search(self, collection_name: str, query_vector: List[float]):
        return self.client.query_points(
            collection_name=collection_name, query=query_vector,
            using="text"
        )
    
def get_sparse_embedding(embedding):
    # Convert the embedding to a NumPy array
    embedding_array = np.array(embedding)

    # Get indices of non-zero elements
    non_zero_indices = np.nonzero(embedding_array)[0]

    # Get values of non-zero elements
    non_zero_values = embedding_array[non_zero_indices]

    # Create a dictionary with lists of indices and values
    non_zero_terms = {
        "indices": non_zero_indices.tolist(),
        "values": non_zero_values.tolist()
    }

    return non_zero_terms