# RAG System for Construction Materials

Simple RAG (Retrieval-Augmented Generation) system for searching through construction materials documents using semantic search.

## Features

- **Text Splitting**: Uses LangChain's `RecursiveCharacterTextSplitter` to split documents into chunks
- **Embeddings**: Uses SentenceTransformer model `stsb-roberta-large` for generating embeddings
- **Vector Store**: Uses Qdrant for storing and searching vectors
- **KNN Retrieval**: Finds top-k similar documents using cosine distance

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. **Choose one of the following options for Qdrant:**

### Option 1: In-Memory Mode (No Docker Required - Easiest for Testing)

No Docker or server setup needed! Perfect for testing and development:

```python
from src.rag.vectore_store import VectorStore

# Use in-memory mode (no Docker needed)
vector_store = VectorStore(use_in_memory=True)
vector_store.add_documents("data/raw/raw_materials.jsonl")
```

**Note**: Data is stored in memory and will be lost when the program ends.

### Option 2: Docker (Recommended for Production)

Start Qdrant server using Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

Then use the default settings (no `use_in_memory` parameter).

### Option 3: Native Installation

Install Qdrant natively (see [Qdrant documentation](https://qdrant.tech/documentation/guides/installation/)).

## System Requirements

**Minimum:**
- Python 3.8+
- 4 GB RAM
- 5 GB disk space
- Internet connection (for initial model download)

**Recommended:**
- Python 3.10+
- 8 GB RAM
- 10 GB disk space

**Model Requirements:**
- `stsb-roberta-large`: ~420 MB model size, ~2 GB RAM when loaded
- Embedding dimension: 1024

For detailed system requirements, see [REQUIREMENTS.md](REQUIREMENTS.md).

## Quick Start

### For LangChain Integration

If you're building a system with LangChain for user interaction, see:
```bash
python langchain_integration_example.py
```

This example shows:
- How to wrap the retriever for LangChain compatibility
- Integration with LangChain chains
- Building conversational systems

### 1. Build Vector Store

**Option A: In-Memory (No Docker)**
```python
from src.rag.vectore_store import VectorStore
from src.rag.retriver import Retriever

# Initialize vector store (no Docker needed)
vector_store = VectorStore(use_in_memory=True)

# Load and index documents
vector_store.add_documents("data/raw/raw_materials.jsonl")

# IMPORTANT: Share the same client for retriever in in-memory mode
retriever = Retriever(
    qdrant_client=vector_store.qdrant_client  # Share the client!
)
```

**Option B: With Docker/Server**
```python
from src.rag.vectore_store import VectorStore

# Initialize vector store
vector_store = VectorStore(
    collection_name="construction_materials",
    qdrant_host="localhost",
    qdrant_port=6333
)

# Load and index documents
vector_store.add_documents("data/raw/raw_materials.jsonl")
```

### 2. Query Documents

**Option A: In-Memory (No Docker)**
```python
from src.rag.retriver import Retriever

# Initialize retriever with shared client (from vector_store)
# IMPORTANT: In in-memory mode, you must share the same QdrantClient
retriever = Retriever(
    qdrant_client=vector_store.qdrant_client  # Share the client!
)

# Search for top 5 documents
results = retriever.retrieve("бетон М400 для фундамента", top_k=5)

for result in results:
    print(f"Score: {result['score']}")
    print(f"URL: {result['url']}")
    print(f"Text: {result['text'][:200]}...")
```

**Option B: With Docker/Server**
```python
from src.rag.retriver import Retriever

# Initialize retriever
retriever = Retriever(
    collection_name="construction_materials",
    qdrant_host="localhost",
    qdrant_port=6333
)

# Search for top 5 documents
results = retriever.retrieve("бетон М400 для фундамента", top_k=5)

for result in results:
    print(f"Score: {result['score']}")
    print(f"URL: {result['url']}")
    print(f"Text: {result['text'][:200]}...")
```

### 3. Use RAG Generator

```python
from src.rag.generator import RAGGenerator

generator = RAGGenerator(retriever)
result = generator.generate("бетон М300 характеристики", top_k=3)
print(result['response'])
```

## Running Examples

### Example Scripts

**Basic Example (in-memory, no Docker):**
```bash
python example_no_docker.py
```

**LangChain Integration Example:**
```bash
# Shows how to integrate with LangChain for user interaction systems
python langchain_integration_example.py
```

### Run Tests
```bash
python test_retriever.py
```

The test verifies that the retriever can find top 2 documents by query.

### Inspect Database

View the contents of your database and see how text is split:

```bash
# IMPORTANT: Load data first! (in-memory mode requires loading each session)
python inspect_database.py --use-in-memory --load-data data/raw/raw_materials.jsonl

# With text splitting demonstration
python inspect_database.py --use-in-memory --load-data data/raw/raw_materials.jsonl --show-splitting

# With statistics
python inspect_database.py --use-in-memory --load-data data/raw/raw_materials.jsonl --show-stats --show-splitting

# For Docker mode (data persists, so load once with setup_rag.py)
python inspect_database.py --collection construction_materials
```

**Note**: In in-memory mode, the database is empty by default. You must either:
1. Use `--load-data` flag when inspecting, OR
2. Run `python setup_rag.py` first to load documents


## Project Structure

```
.
├── data/
│   └── raw/
│       └── raw_materials.jsonl    # Input documents
├── src/
│   └── rag/
│       ├── vectore_store.py       # Vector store with Qdrant
│       ├── retriver.py            # KNN retriever
│       └── generator.py            # RAG generator
├── example_no_docker.py           # Basic example (in-memory)
├── langchain_integration_example.py  # LangChain integration example
├── test_retriever.py              # Tests
└── requirements.txt               # Dependencies
```

## Data Format

Documents in `raw_materials.jsonl` should have the following structure:

```json
{
    "url": "https://example.com",
    "error": null,
    "content": "Document text content...",
    "timestamp": 1234567890.0
}
```

## Configuration

You can customize the vector store and retriever:

```python
# Vector store with custom settings
vector_store = VectorStore(
    collection_name="my_collection",
    chunk_size=500,          # Size of text chunks
    chunk_overlap=50,        # Overlap between chunks
    qdrant_host="localhost",
    qdrant_port=6333
)

# Retriever
retriever = Retriever(
    collection_name="my_collection",
    model_name="stsb-roberta-large",  # Embedding model
    qdrant_host="localhost",
    qdrant_port=6333
)
```

## How It Works

1. **Document Loading**: Loads documents from JSONL file
2. **Text Splitting**: Splits documents into chunks using LangChain's `RecursiveCharacterTextSplitter`
   - Default chunk size: 500 characters
   - Default overlap: 50 characters
   - Splits at natural boundaries (paragraphs, sentences)
   - Overlap preserves context across chunk boundaries
3. **Embedding**: Generates embeddings using SentenceTransformer (`stsb-roberta-large`)
   - Each chunk becomes a 1024-dimensional vector
4. **Storage**: Stores embeddings in Qdrant vector database
   - Each chunk stored with metadata (URL, document index, chunk index, original text)
5. **Retrieval**: Searches using KNN with cosine distance
   - Query is embedded using the same model
   - Finds top-k most similar chunks
6. **Generation**: Combines retrieved documents to generate responses

### Text Splitting Details

LangChain's `RecursiveCharacterTextSplitter` works by:

1. **Recursive Strategy**: Tries natural boundaries in order:
   - Double newlines (`\n\n`) - paragraphs
   - Single newlines (`\n`) - lines  
   - Sentence endings (`.`, `!`, `?`) - sentences
   - Spaces - words
   - Characters - last resort

2. **Maintaining Overlap**: Each chunk overlaps with neighbors by 50 characters (default)
   - Preserves context at boundaries
   - Prevents information loss
   - Better search results

3. **Respecting Chunk Size**: Keeps chunks close to 500 characters (default)

**Visual Example:**
```
Text (1000 chars):
┌─────────────────────────┐
│ Chunk 1: 0-500          │
└─────────────────────────┘
         │ 50 char overlap
         ▼
┌─────────────────────────┐
│ Chunk 2: 450-950        │
└─────────────────────────┘
         │ 50 char overlap
         ▼
┌─────────────────────────┐
│ Chunk 3: 900-1000       │
└─────────────────────────┘
```

**Why Overlap Matters:**
- Without overlap: "Бетон М400 имеет прочность" | "393 кгс/см²" (connection lost!)
- With overlap: Both chunks contain "Бетон М400 имеет прочность 393 кгс/см²" (context preserved!)

For detailed explanation, see [TEXT_SPLITTING_EXPLAINED.md](TEXT_SPLITTING_EXPLAINED.md)

## License

MIT
