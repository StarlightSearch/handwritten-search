import lancedb
import os
from embed_anything import Adapter, EmbedData
from typing import List, Dict
import numpy as np
import uuid
import pyarrow as pa

class LanceDBAdapter(Adapter):
    def __init__(self):
        self.db = lancedb.connect("data/lancedb")
        self._init_table()

    def _init_table(self):
        schema = pa.schema([
            ("id", pa.string()),
            ("vector", pa.list_(pa.float32(), 1024)),  # Default dimension, will be updated on first create
            ("sparse_indices", pa.list_(pa.int64())),
            ("sparse_values", pa.list_(pa.float32())),
            ("text", pa.string()),
            ("file_path", pa.string())
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
        for i, embedding in enumerate(embeddings):
            sparse_embedding = get_sparse_embedding(sparse_embeddings[i].embedding)
            points.append({
                "id": str(uuid.uuid4()),
                "vector": embedding.embedding,
                "sparse_indices": sparse_embedding["indices"],
                "sparse_values": sparse_embedding["values"],
                "text": embedding.text,
                "file_path": embedding.metadata.get("file_path", "")
            })
        return points

    def upsert(self, data: List[EmbedData], sparse_data: List[EmbedData], index_name: str) -> None:
        points = self.convert(data, sparse_data, index_name)
        table = self.db["vectors"]
        table.add(points)

    def search_hybrid(self, collection_name: str, query_vector: List[float], query_sparse_vector) -> List[Dict]:
        table = self.db["vectors"]
        
        # Perform dense vector search
        dense_results = (table.search(query_vector)
                        .limit(20)
                        .to_list())
        
        # Perform sparse vector search using dot product of sparse vectors
        sparse_results = (table.search(
            query_vector=query_sparse_vector.values,
            vector_column_name="sparse_values",
            query_type="sparse"
        )
        .limit(20)
        .to_list())
        
        # Combine results using RRF (Reciprocal Rank Fusion)
        all_results = self._combine_results_rrf(dense_results, sparse_results)
        return SimpleNamespace(points=all_results)

    def search(self, collection_name: str, query_vector: List[float]):
        table = self.db["vectors"]
        results = (table.search(query_vector)
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

    def _combine_results_rrf(self, dense_results, sparse_results, k=60):
        # Create a dictionary to store combined scores
        combined_scores = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results):
            doc_id = result["id"]
            score = 1 / (k + rank + 1)
            combined_scores[doc_id] = {"score": score, "data": result}
        
        # Process sparse results
        for rank, result in enumerate(sparse_results):
            doc_id = result["id"]
            score = 1 / (k + rank + 1)
            if doc_id in combined_scores:
                combined_scores[doc_id]["score"] += score
            else:
                combined_scores[doc_id] = {"score": score, "data": result}
        
        # Sort by combined score
        sorted_results = sorted(
            [
                SimpleNamespace(
                    id=doc_id,
                    payload={"text": data["data"]["text"], "file_path": data["data"]["file_path"]},
                    score=data["score"]
                )
                for doc_id, data in combined_scores.items()
            ],
            key=lambda x: x.score,
            reverse=True
        )
        
        return sorted_results

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

class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs) 