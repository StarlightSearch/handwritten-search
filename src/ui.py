import streamlit as st
import requests
import os
from PIL import Image
from streamlit_file_browser import st_file_browser

API_URL = "http://localhost:8000"

if 'current_collection' not in st.session_state:
    st.session_state.current_collection = None

st.title("Image Processing and Search System")

# Sidebar for collection management
with st.sidebar:
    st.header("Collection Management")
    
    # Get existing collections from Qdrant
    response = requests.get(f"{API_URL}/collections")
    collections = []
    if response.status_code == 200:
        collections = response.json()["collections"]
    
    # Collection Selector
    st.subheader("Select Collection")
    selected_collection = st.selectbox(
        "Choose a collection",
        options=collections,
        index=0 if st.session_state.current_collection is None 
        else collections.index(st.session_state.current_collection)
    )
    
    if selected_collection:
        st.session_state.current_collection = selected_collection
    
    # Create Collection
    st.subheader("Create Collection")
    collection_name = st.text_input("Collection Name")
    if st.button("Create Collection"):
        response = requests.post(
            f"{API_URL}/collections/create",
            json={"collection_name": collection_name}
        )
        if response.status_code == 200:
            st.success(response.json()["message"])
        else:
            st.error(response.json()["detail"])
    
    # Delete Collection
    st.subheader("Delete Collection")
    if st.button("Delete Collection"):
        response = requests.delete(
            f"{API_URL}/collections/delete",
            json={"collection_name": st.session_state.current_collection}
        )
        if response.status_code == 200:
            st.success(response.json()["message"])
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
        
        # Save the uploaded file temporarily
        temp_path = f"temp/temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        if st.button("Process Image"):
            response = requests.post(
                f"{API_URL}/process",
                json={
                    "file_path": temp_path,
                    "collection_name": st.session_state.current_collection
                }
            )
            
            if response.status_code == 200:
                st.success("Image processed successfully!")
                st.write("Extracted Text:", response.json()["text"])
            else:
                st.error(response.json()["detail"])
            
            # Clean up temporary file
            # os.remove(temp_path)

# Search Tab
with tab2:
    st.header("Search")
    search_query = st.text_input("Enter your search query")
    
    if st.button("Search"):
        response = requests.post(
            f"{API_URL}/search",
            json={
                "query": search_query,
                "collection_name": st.session_state.current_collection
            }
        )
        
        if response.status_code == 200:
            results = response.json()["results"]
            for idx, result in enumerate(results, 1):
                with st.expander(f"Result {idx} (Score: {result['score']:.2f})"):
                    st.write("Text:", result["text"])
                    st.write("File:", result["file_path"])
                    try:
                        image = Image.open(result["file_path"])
                        st.image(image, caption="Found Image", use_container_width=True)
                    except:
                        st.warning("Image file not found")
        else:
            st.error(response.json()["detail"]) 