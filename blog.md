---
context: This is a blog post about the Handwritten Document Search App. Don't use references to the codebase while generating responses to any queries run on this document. Use simple language and avoid using multiple clauses in a single sentence.
---

# Handwritten Document Search App

In the world of digital information, we're still limited by our ability to search through our physical notes. Now, through the power of Vision Models and Embeddings, we can finally make this a reality. In this article, we're going to use the Qwen vision model via Huggingface Transformers to extract text from handwritten documents and then use the Embed-Anything pipeline to embed the text and store it in a vector database.

## System Workflow

The app runs on Streamlit for the UI. We first accept the user's handwritten images from their a file picker. Once uploaded, these documents are processed by a Vision Language Model (Qwen-VL) via Transformers that specializes in handwriting recognition. The extracted text then flows through the Embed-Anything pipeline, which transforms the raw text into vector embeddings while storing the text and filepath as metadata. 

These embeddings are stored in LanceDB, a vector database which runs locally, like SQLite. This enables fast retrieval and searching capabilities, avoiding network latency. When users want to search through their documents, they can input natural language queries - which undergo the same embedding process. LanceDB then performs nearest neighbor search operations on these vectors to find the most semantically similar documents.

## Core Technologies

### 1. Vision Processing with Huggingface Transformers

- Specialized VLM for handwritten text extraction (2B parameter model)
- Implements chat-style OCR prompts for accurate transcription:

### 2. Embedding Generation Pipeline

- **Embed-Anything** unified interface for:
  - Model management (Jina v2 base embeddings)
  - Batch processing
  - Metadata handling

### 3. Vector Search Infrastructure

- **LanceDB integration** through custom adapter pattern:
  - Columnar storage for fast retrieval
  - Hybrid search capabilities
  - Native Python SDK integration

## Key Design Decisions

1. **Streamlit Interface**

- Dual-purpose UI for processing and search
- Temporary file handling system for document batches

2. **Model Optimization**

- FP16 quantization for memory efficiency
- Device-aware loading (CPU/GPU)

3. **Data Pipeline**

- Text+image metadata preservation
- Automatic schema management

## Implementation Challenges

- Handwriting recognition consistency
- Embedding model memory footprint
- Cross-modal metadata alignment

The system demonstrates how modern AI components (VLMs, embedding APIs) can be combined with efficient data systems (LanceDB) to solve complex search problems.
