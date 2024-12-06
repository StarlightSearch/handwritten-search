from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from qdrant_client.models import Distance

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from embed_anything import embed_query, EmbeddingModel, ONNXModel, WhichModel

from api.qdrant_adapter import QdrantAdapter
from qdrant_client.models import SparseVector
import numpy as np
app = FastAPI()

# Initialize models
qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="float16", device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
)

embedding_model = EmbeddingModel.from_pretrained_onnx(
    WhichModel.Bert, ONNXModel.AllMiniLML12V2
)
sparse_model = EmbeddingModel.from_pretrained_hf(
    WhichModel.SparseBert, "prithivida/Splade_PP_en_v1"
)

adapter = QdrantAdapter()


# Pydantic models for request validation
class CollectionCreate(BaseModel):
    collection_name: str
    dimension: int = 384
    metric: str = "cosine"


class FileProcess(BaseModel):
    file_path: str
    collection_name: str


class SearchQuery(BaseModel):
    query: str
    collection_name: str


class CollectionDelete(BaseModel):
    collection_name: str


@app.post("/collections/create")
async def create_collection(request: CollectionCreate):
    try:
        if adapter.client.collection_exists(request.collection_name):
            raise HTTPException(status_code=400, detail="Collection already exists")

        metric = (
            Distance.COSINE if request.metric.lower() == "cosine" else Distance.EUCLID
        )
        adapter.create_index(
            dimension=request.dimension,
            metric=metric,
            index_name=request.collection_name,
        )
        return {"message": f"Collection {request.collection_name} created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process")
async def process_file(request: FileProcess):
    try:
        # Prepare messages for Qwen model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": request.file_path,
                    },
                    {"type": "text", "text": "Transcribe this image. Just give the transcription, no other information."},
                ],
            }
        ]

        # Process with Qwen
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate text
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Create embeddings
        dense_embedding = embed_query(output_text, embedding_model)
        sparse_embedding = embed_query(output_text, sparse_model)
        dense_embedding[0].metadata = {
            "text": output_text[0],
            "file_path": request.file_path,
        }

        # Upsert to database
        adapter.upsert(
            data=dense_embedding,
            sparse_data=sparse_embedding,
            index_name=request.collection_name,
        )

        return {
            "message": "File processed and stored successfully",
            "text": output_text[0],
            "file_path": request.file_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search(request: SearchQuery):
    try:
        dense_query_embedding = embed_query([request.query], embedding_model)
        sparse_query_embedding = embed_query([request.query], sparse_model)
        query_sparse_embeddings = get_sparse_embedding(sparse_query_embedding[0].embedding)

        query_sparse_embeddings = SparseVector(
        indices=query_sparse_embeddings["indices"],
        values=query_sparse_embeddings["values"],
    )


        results = adapter.search_hybrid(
            collection_name=request.collection_name,
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
        if not adapter.client.collection_exists(request.collection_name):
            raise HTTPException(status_code=404, detail="Collection does not exist")
        
        adapter.client.delete_collection(collection_name=request.collection_name)
        return {"message": f"Collection {request.collection_name} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/qdrant/collections")
async def list_collections():
    try:
        collections = adapter.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        return {"collections": collection_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

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
        "values": non_zero_values.tolist()
    }

    return non_zero_terms