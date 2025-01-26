import streamlit as st
import requests
from PIL import Image
import os

API_URL = "http://localhost:8000"

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
            response = requests.post(
                f"{API_URL}/process",
                json={"file_path": temp_path}
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
            json={"query": search_query}
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