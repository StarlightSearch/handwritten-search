import streamlit as st
import requests
from PIL import Image
import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from embed_anything import embed_query, EmbeddingModel, WhichModel
from api.lancedb_adapter import LanceDBAdapter, SimpleNamespace
import numpy as np

API_URL = "http://localhost:8000"

# Initialize embedding models
embedding_model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Jina, "jinaai/jina-embeddings-v2-base-en"
)
sparse_model = EmbeddingModel.from_pretrained_hf(
    WhichModel.SparseBert, "prithivida/Splade_PP_en_v1"
)
adapter = LanceDBAdapter()

st.title("Image Processing and Search System")

# Data Management
st.header("Data Management")
if st.button("Refresh Data", type="secondary"):
    response = requests.delete(
        f"{API_URL}/collections/delete",
        json={"collection_name": ""}  # Empty string as we don't use collection_name
    )
    if response.status_code == 200:
        st.success("All data cleared successfully")
    else:
        st.error(response.json()["detail"])

# Main content
tab1, tab2 = st.tabs(["Process Images", "Search"])

# Process Images Tab
with tab1:
    st.header("Process Images")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        # Save the uploaded file temporarily
        temp_path = f"temp/temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        if st.button("Process Image"):
            try:
                with st.spinner("Loading models and processing image..."):
                    # Initialize Qwen models
                    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="float16", device_map="auto"
                    )
                    processor = AutoProcessor.from_pretrained(
                        "Qwen/Qwen2-VL-2B-Instruct", min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
                    )

                    # Prepare messages for Qwen model
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": temp_path,
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
                    inputs = inputs.to(qwen_model.device)

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
                        "file_path": temp_path,
                    }

                    # Upsert to database
                    adapter.upsert(
                        data=dense_embedding,
                        sparse_data=sparse_embedding,
                        index_name=""  # Empty string as we don't use the index_name parameter
                    )

                st.success("Image processed successfully!")
                st.write("Extracted Text:", output_text[0])
            except Exception as e:
                st.error(str(e))
            
            # Clean up temporary file
            # os.remove(temp_path)

# Search Tab
with tab2:
    st.header("Search")
    search_query = st.text_input("Enter your search query")
    
    if st.button("Search"):
        try:
            with st.spinner("Searching..."):
                dense_query_embedding = embed_query([search_query], embedding_model)
                sparse_query_embedding = embed_query([search_query], sparse_model)
                
                # Get sparse values
                embedding_array = np.array(sparse_query_embedding[0].embedding)
                non_zero_values = embedding_array[np.nonzero(embedding_array)[0]]
                query_sparse_embeddings = SimpleNamespace(
                    values=non_zero_values.tolist()
                )

                results = adapter.search_hybrid(
                    collection_name="",
                    query_vector=dense_query_embedding[0].embedding,
                    query_sparse_vector=query_sparse_embeddings
                )

                for idx, result in enumerate(results.points, 1):
                    with st.expander(f"Result {idx} (Score: {result.score:.2f})"):
                        st.write("Text:", result.payload.get("text"))
                        st.write("File:", result.payload.get("file_path"))
                        try:
                            image = Image.open(result.payload.get("file_path"))
                            st.image(image, caption="Found Image", use_container_width=True)
                        except:
                            st.warning("Image file not found")
        except Exception as e:
            import traceback
            print(f"Error during search: {str(e)}")
            print(traceback.format_exc())  # Print full traceback
            st.error(str(e)) 