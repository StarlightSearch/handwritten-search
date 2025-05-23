# Handwritten Search

> Click on the image below to view the demo.

[![Youtube Video](https://github.com/user-attachments/assets/1902e5f0-fc1e-458c-b248-3b0c2b3cf067)](https://www.youtube.com/watch?v=TDSDH8EQnUY)

A powerful application for searching through handwritten documents using advanced Vision Language Models (VLM) and vector search capabilities. This project combines state-of-the-art OCR using Qwen-VL with efficient vector search using LanceDB.

## Features

- OCR powered by Qwen2-VL-2B-Instruct model
- Vector embeddings generation using Jina Embeddings (jinaai/jina-embeddings-v2-base-en)
- Efficient vector search using LanceDB
- Streamlit-based user interface with integrated API functionality

## Prerequisites

- Python 3.11 or higher
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

## Installation

Install dependencies using Poetry:
```bash
poetry install
```

## Usage

Run the Streamlit application:

```bash
poetry run streamlit run search-handwritten/main.py
```

## Features

- **Process Images**: Upload and process handwritten documents
- **Search**: Search through processed documents using natural language queries
- **Data Management**: Clear and manage your document database

## Technologies Used

- **Qwen2-VL**: Vision Language Model for OCR
- **Jina Embeddings**: For generating text embeddings
- **LanceDB**: Vector database for efficient similarity search
- **Streamlit**: For building the user interface
- **Poetry**: Dependency management and packaging
