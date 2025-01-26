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
class FileProcess(BaseModel):
    file_path: str

class SearchQuery(BaseModel):
    query: str

class CollectionDelete(BaseModel):
    collection_name: str

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
                    {
                        "type": "text",
                        "text": "Transcribe this image. Just give the transcription, no other information.",
                    },
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
            index_name=""  # Empty string as we don't use the index_name parameter
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
