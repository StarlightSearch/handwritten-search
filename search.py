from api.lancedb_adapter import SimpleNamespace
from embed_anything import embed_query
import numpy as np

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
        "values": non_zero_values.tolist(),
    }

    return non_zero_terms

def perform_search(query, embedding_model, sparse_model, adapter):
    dense_query_embedding = embed_query([query], embedding_model)
    sparse_query_embedding = embed_query([query], sparse_model)
    query_sparse_embeddings = get_sparse_embedding(
        sparse_query_embedding[0].embedding
    )

    query_sparse_embeddings = SimpleNamespace(
        indices=query_sparse_embeddings["indices"],
        values=query_sparse_embeddings["values"],
    )

    results = adapter.search_hybrid(
        collection_name="",  # Empty string as we don't use the collection_name parameter
        query_vector=dense_query_embedding[0].embedding,
        query_sparse_vector=query_sparse_embeddings,
    )

    return [
        {
            "text": result.payload.get("text"),
            "file_path": result.payload.get("file_path"),
            "score": result.score,
        }
        for result in results.points
    ] 