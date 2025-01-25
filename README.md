# Handwritten Search

A powerful application for searching through handwritten documents using advanced Vision Language Models (VLM) and vector search capabilities. This project combines state-of-the-art OCR using Qwen-VL with efficient vector search using Qdrant.

## Features

- OCR powered by Qwen2-VL-2B-Instruct model
- Vector embeddings generation using EmbedAnything
- Efficient vector search using Qdrant
- FastAPI backend for robust API endpoints
- Streamlit-based desktop application interface

## Prerequisites

- Python 3.11 or higher
- Poetry for dependency management

## Installation

Install dependencies using Poetry:
```bash
poetry install
```

## Usage

### Starting the API Server

```bash
poetry run uvicorn main:app --reload
```

### Running the Desktop Application

```bash
poetry run streamlit run desktop_app.py
```

## API Endpoints

- `POST /collections/create` - Create a new collection for storing document embeddings
- `POST /process` - Process and index a handwritten document
- `POST /search` - Search through indexed documents
- `DELETE /collections/delete` - Delete a collection
- `GET /qdrant/collections` - List all collections

## Technologies Used

- **Qwen2-VL**: Vision Language Model for OCR
- **EmbedAnything**: For generating embeddings
- **Qdrant**: Vector database for efficient similarity search
- **FastAPI**: Modern web framework for building APIs
- **Streamlit**: For building the desktop interface
- **Poetry**: Dependency management and packaging
