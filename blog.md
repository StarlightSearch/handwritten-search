# Handwritten Document Search App

In the world of digital information, we're still limited by our ability to search through our physical notes. Now, through the power of Vision Models and Embeddings, we can finally make this a reality. In this article, we're going to use the Qwen vision model via Huggingface Transformers to extract text from handwritten documents and then use the Embed-Anything pipeline to embed the text and store it in a vector database.

## System Workflow

1. **Image Upload** - Users submit handwritten documents through Streamlit interface  
2. **Qwen-VL Processing** - Vision Language Model extracts text with chat-style OCR prompts  
3. **Embed-Anything Pipeline** - Generates dense embeddings while preserving document metadata  
4. **LanceDB Storage** - Vector database stores embeddings with columnar optimization  
5. **Query Handling** - Natural language input converts to embedding via same pipeline  
6. **Semantic Search** - LanceDB performs nearest neighbor search on 768D vectors  
7. **Result Ranking** - Hybrid scoring returns most relevant documents with original text excerpts  

## Core Technologies

### 1. Vision Processing with Qwen-VL

- Specialized VLM for handwritten text extraction (2B parameter model)
- Implements chat-style OCR prompts for accurate transcription:

```54:59:search-handwritten/main.py
                    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="float16", device_map="auto"
                    )
                    processor = AutoProcessor.from_pretrained(
                        "Qwen/Qwen2-VL-2B-Instruct", min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
                    )
```

### 2. Embedding Generation Pipeline

- **Embed-Anything** unified interface for:
  - Model management (Jina v2 base embeddings)
  - Batch processing
  - Metadata handling

```13:15:search-handwritten/main.py
embedding_model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Jina, "jinaai/jina-embeddings-v2-base-en"
)
```

### 3. Vector Search Infrastructure

- **LanceDB integration** through custom adapter pattern:
  - Columnar storage for fast retrieval
  - Hybrid search capabilities
  - Native Python SDK integration

```44:47:search-handwritten/api/lancedb_adapter.py
    def upsert(self, data: List[EmbedData], sparse_data: List[EmbedData], index_name: str) -> None:
        points = self.convert(data, sparse_data, index_name)
        table = self.db["vectors"]
        table.add(points)
```

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

## Performance Characteristics

- 768-dimension embeddings (Jina v2 base)
- LanceDB's SIMD-optimized search
- Async-ready architecture

## Implementation Challenges

- Handwriting recognition consistency
- Embedding model memory footprint
- Cross-modal metadata alignment

The system demonstrates how modern AI components (VLMs, embedding APIs) can be combined with efficient data systems (LanceDB) to solve complex search problems.
