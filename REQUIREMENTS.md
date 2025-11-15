# System Requirements

This document outlines the system requirements for running the RAG system.

## Python Requirements

- **Python Version**: 3.8 or higher (tested with Python 3.10)
- **pip**: Latest version recommended

## Model Requirements

### SentenceTransformer Model: `stsb-roberta-large`

The `stsb-roberta-large` model is used for generating embeddings. Here are its requirements:

- **Model Size**: ~420 MB (downloads automatically on first use)
- **Memory (RAM)**: 
  - Minimum: 2 GB free RAM
  - Recommended: 4+ GB free RAM
  - The model loads into memory when initialized
- **Embedding Dimension**: 1024 dimensions per vector
- **Disk Space**: ~500 MB for model storage (in `~/.cache/torch/sentence_transformers/`)

**Note**: The model will be automatically downloaded from HuggingFace on first use. Ensure you have internet connection for the initial download.

### Alternative Models

If `stsb-roberta-large` is too resource-intensive, you can use smaller models:

- `all-MiniLM-L6-v2` (~80 MB, 384 dimensions) - Much faster, less memory
- `paraphrase-multilingual-MiniLM-L12-v2` (~420 MB, 384 dimensions) - Multilingual support
- `all-mpnet-base-v2` (~420 MB, 768 dimensions) - Good balance

To use an alternative model, change the `model_name` parameter:

```python
vector_store = VectorStore(model_name="all-MiniLM-L6-v2")
retriever = Retriever(model_name="all-MiniLM-L6-v2")
```

## Qdrant Requirements

### Docker Deployment (Recommended)

- **Docker**: Version 20.10 or higher
- **Memory**: 
  - Minimum: 512 MB for Qdrant container
  - Recommended: 1-2 GB for better performance
- **Disk Space**: 
  - Depends on your data size
  - Rough estimate: ~1-2 GB per 100,000 vectors (with 1024 dimensions)
  - For typical use: 2-5 GB should be sufficient

### Qdrant Memory Usage

- Each vector: ~4 KB (1024 dimensions × 4 bytes)
- Metadata overhead: ~1-2 KB per vector
- Total per vector: ~5-6 KB
- Example: 10,000 vectors ≈ 50-60 MB

## System Requirements Summary

### Minimum Requirements

- **CPU**: 2+ cores
- **RAM**: 4 GB total system RAM
  - 2 GB for Python process + model
  - 512 MB for Qdrant
  - 1.5 GB for OS and other processes
- **Disk Space**: 5 GB free space
  - 500 MB for model
  - 2-3 GB for Qdrant data
  - 1-2 GB for Python packages and data files
- **Network**: Internet connection for initial model download

### Recommended Requirements

- **CPU**: 4+ cores (for faster processing)
- **RAM**: 8 GB total system RAM
  - 4 GB for Python process + model
  - 1 GB for Qdrant
  - 3 GB for OS and other processes
- **Disk Space**: 10 GB free space
- **Network**: Stable internet connection

### For Production/Heavy Usage

- **CPU**: 8+ cores
- **RAM**: 16+ GB
- **Disk Space**: 50+ GB (SSD recommended)
- **GPU**: Optional but recommended for faster embedding generation
  - CUDA-compatible GPU with 4+ GB VRAM
  - Install `torch` with CUDA support

## Python Package Dependencies

All dependencies are listed in `requirements.txt`:

```
beautifulsoup4      # Web scraping (if needed)
requests            # HTTP requests
lxml                # XML/HTML parsing
langchain           # Text splitting and chain utilities
sentence-transformers  # Embedding models
qdrant-client       # Qdrant Python client
numpy               # Numerical operations
```

### Additional Dependencies (Installed Automatically)

- `torch` or `tensorflow` (for sentence-transformers)
- `transformers` (HuggingFace models)
- `scikit-learn` (for some utilities)
- `tqdm` (progress bars)

## Installation Size

Total disk space needed for Python packages: ~2-3 GB

## Performance Considerations

### Embedding Generation Speed

- **CPU**: ~10-50 sentences/second (depends on CPU and sentence length)
- **GPU**: ~100-500 sentences/second (with CUDA-enabled GPU)

### Query Speed

- **Qdrant KNN Search**: <10ms for top-k queries on typical datasets
- **Total Query Time**: ~50-200ms (including embedding generation)

## Network Requirements

- **Initial Setup**: 
  - Download model from HuggingFace (~420 MB)
  - Download Docker image for Qdrant (~200 MB)
- **Runtime**: 
  - No internet required after initial setup
  - Qdrant runs locally

## Operating System Support

- **Linux**: Fully supported
- **macOS**: Fully supported (tested on macOS)
- **Windows**: Supported (Docker Desktop required for Qdrant)

## Docker Requirements (for Qdrant)

If using Docker for Qdrant:

```bash
# Minimum Docker resources
docker run -p 6333:6333 \
  --memory="512m" \
  qdrant/qdrant

# Recommended Docker resources
docker run -p 6333:6333 \
  --memory="2g" \
  --cpus="2" \
  qdrant/qdrant
```

## Troubleshooting

### Out of Memory Errors

1. Use a smaller embedding model (e.g., `all-MiniLM-L6-v2`)
2. Reduce `chunk_size` in VectorStore
3. Process documents in batches
4. Increase system RAM or use swap space

### Slow Performance

1. Use GPU for embedding generation (if available)
2. Increase Qdrant memory allocation
3. Use smaller embedding dimensions
4. Optimize chunk size (larger chunks = fewer vectors = faster)

### Disk Space Issues

1. Clean up old Qdrant collections
2. Use model quantization (if supported)
3. Store models on external drive (symlink cache directory)

## Example Resource Usage

For a dataset with 1,000 documents (~100 KB each):

- **Documents**: ~100 MB
- **Chunks**: ~5,000 chunks (with 500 char chunks)
- **Vectors**: ~5,000 vectors × 1024 dims = ~20 MB
- **Qdrant Storage**: ~50-100 MB
- **Total**: ~200-300 MB

For 10,000 documents:

- **Total Storage**: ~2-3 GB

