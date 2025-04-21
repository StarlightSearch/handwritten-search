import lancedb
import os
from embed_anything import Adapter, EmbedData
from typing import List, Dict
import uuid
import pyarrow as pa

class LanceDBAdapter(Adapter):
    def __init__(self):
        self.db = lancedb.connect("data/lancedb")
        self._init_table()

    def _init_table(self):
        # Get vector dimension from the first embedding
        schema = pa.schema([
            pa.field("id", pa.string()),
            # Fixed-size list for vector embeddings
            pa.field("vector", pa.list_(pa.float32(), 768)),  # Jina embeddings are 768-dimensional
            pa.field("text", pa.string()),
            pa.field("file_path", pa.string())
        ])
        self.db.create_table("vectors", schema=schema, exist_ok=True)

    def delete_index(self, index_name):
        # We use this method to clear all data, ignoring the index_name parameter
        table = self.db["vectors"]
        table.delete("1=1")  # Delete all rows

    def create_index(self, dimension, metric, index_name):
        # Stub method required by Adapter interface
        pass

    def convert(self, embeddings: List[EmbedData], sparse_embeddings: List[EmbedData], index_name: str) -> List[Dict]:
        points = []
        for embedding in embeddings:
            points.append({
                "id": str(uuid.uuid4()),
                "vector": embedding.embedding,
                "text": embedding.text,
                "file_path": embedding.metadata.get("file_path", "")
            })
        return points

    def upsert(self, data: List[EmbedData], sparse_data: List[EmbedData], index_name: str) -> None:
        points = self.convert(data, sparse_data, index_name)
        table = self.db["vectors"]
        table.add(points)

    def search(self, collection_name: str, query_vector: List[float]):
        table = self.db["vectors"]
        # Convert numpy array to list if needed
        query = query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector
        
        results = (table.search(query, vector_column_name="vector")
                  .limit(20)
                  .to_list())
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=r["id"],
                    payload={"text": r["text"], "file_path": r["file_path"]},
                    score=r["_distance"]
                )
                for r in results
            ]
        )

    def search_hybrid(self, collection_name: str, query_vector: List[float], query_sparse_vector) -> List[Dict]:
        # Since we removed sparse vectors, just use regular vector search
        return self.search(collection_name, query_vector)

class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs) 