from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from embed_anything import embed_query, EmbeddingModel, WhichModel

from api.lancedb_adapter import LanceDBAdapter, SimpleNamespace
import numpy as np
import uvicorn

app = FastAPI()

# Initialize models
qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="float16", device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
)

embedding_model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Jina, "jinaai/jina-embeddings-v2-base-en"
)
sparse_model = EmbeddingModel.from_pretrained_hf(
    WhichModel.SparseBert, "prithivida/Splade_PP_en_v1"
)

adapter = LanceDBAdapter()

# Pydantic models for request validation
class SearchQuery(BaseModel):
    query: str

class CollectionDelete(BaseModel):
    collection_name: str

@app.post("/search")
async def search(request: SearchQuery):
    try:
        dense_query_embedding = embed_query([request.query], embedding_model)
        sparse_query_embedding = embed_query([request.query], sparse_model)
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

        return {
            "results": [
                {
                    "text": result.payload.get("text"),
                    "file_path": result.payload.get("file_path"),
                    "score": result.score,
                }
                for result in results.points
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collections/delete")
async def delete_collection(request: CollectionDelete):
    try:
        adapter.delete_index("")  # Empty string as we don't use the index_name parameter
        return {"message": "All data cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

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
