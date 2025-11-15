# How Text Splitting Works

## Overview

The RAG system uses **LangChain's `RecursiveCharacterTextSplitter`** to split documents into smaller chunks before storing them in the vector database.

## Default Settings

- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters

## How It Works

### 1. Recursive Splitting Strategy

The splitter tries to split at natural boundaries, in this order:

1. **Double newlines** (`\n\n`) - Paragraphs
2. **Single newlines** (`\n`) - Lines
3. **Sentence endings** (`.`, `!`, `?`) - Sentences
4. **Spaces** - Words
5. **Characters** - Last resort

This ensures chunks are split at meaningful boundaries, not in the middle of words or sentences.

### 2. Overlap Mechanism

Each chunk overlaps with the previous and next chunk by 50 characters (default). This means:

```
Original Text (1000 characters)
│
├─ Chunk 1: characters 0-500
│  └─ Contains: chars 0-500
│
├─ Chunk 2: characters 450-950
│  └─ Contains: chars 450-950 (overlaps with Chunk 1 at 450-500)
│
└─ Chunk 3: characters 900-1000
   └─ Contains: chars 900-1000 (overlaps with Chunk 2 at 900-950)
```

### 3. Visual Example

```
Text: "Бетон М400 — это высокопрочный строительный материал..."

┌─────────────────────────────────────────────────────────┐
│ Chunk 1 (500 chars)                                     │
│ Бетон М400 — это высокопрочный строительный материал,  │
│ который широко используется в современном строительстве│
│ ... [continues to char 500]                             │
└─────────────────────────────────────────────────────────┘
         │
         │ Overlap (50 chars)
         ▼
┌─────────────────────────────────────────────────────────┐
│ Chunk 2 (500 chars)                                     │
│ ... [starts at char 450, includes overlap]             │
│ Он обладает отличными характеристиками прочности       │
│ ... [continues to char 950]                             │
└─────────────────────────────────────────────────────────┘
         │
         │ Overlap (50 chars)
         ▼
┌─────────────────────────────────────────────────────────┐
│ Chunk 3 (remaining chars)                              │
│ ... [starts at char 900, includes overlap]              │
│ Бетон М400 применяется для возведения фундаментов...    │
└─────────────────────────────────────────────────────────┘
```

## Why Overlap is Important

1. **Preserves Context**: Information at chunk boundaries isn't lost
2. **Better Search Results**: When searching, you can find relevant information even if it spans multiple chunks
3. **Maintains Meaning**: Sentences and phrases aren't cut in the middle
4. **Redundancy**: Important information appears in multiple chunks, increasing retrieval chances

### Example Without Overlap (Bad):
```
Chunk 1: "...Бетон М400 имеет прочность"
Chunk 2: "393 кгс/см² и морозостойкость F200..."
```
Problem: The connection between "М400" and "393 кгс/см²" is lost!

### Example With Overlap (Good):
```
Chunk 1: "...Бетон М400 имеет прочность 393 кгс/см²..."
Chunk 2: "...прочность 393 кгс/см² и морозостойкость F200..."
```
Solution: Both chunks contain the complete information!

## In Your System

### Step-by-Step Process

1. **Document Loading**: 
   - Reads `raw_materials.jsonl`
   - Extracts `content` field from each document

2. **Text Splitting**:
   - Each document's content is split into chunks
   - Default: ~500 characters per chunk
   - 50 characters overlap between chunks

3. **Embedding**:
   - Each chunk is converted to a 1024-dimensional vector
   - Uses SentenceTransformer model `stsb-roberta-large`

4. **Storage**:
   - Each chunk stored in Qdrant with metadata:
     - `url`: Original document URL
     - `doc_index`: Which document (0, 1, 2, ...)
     - `chunk_index`: Position within document (0, 1, 2, ...)
     - `text`: The actual chunk text
     - `original_content_length`: Total length of original document

5. **Retrieval**:
   - When you search, your query is embedded
   - Qdrant finds the most similar chunks (by cosine distance)
   - Returns chunks with their metadata

## Example from Your Data

If you have a document with 13,294 characters:

```
Document: "Песок | Доставка песка-5 кубов..."
Length: 13,294 characters

Splitting result:
- Chunk 1: chars 0-500
- Chunk 2: chars 450-950 (50 overlap)
- Chunk 3: chars 900-1400 (50 overlap)
- ...
- Chunk N: chars (N*450) to end

Total chunks: ~27 chunks (13,294 / 450 ≈ 27)
```

## Customization

You can customize the splitting:

```python
vector_store = VectorStore(
    chunk_size=1000,      # Larger chunks
    chunk_overlap=100,    # More overlap
    use_in_memory=True
)
```

**Trade-offs:**
- **Larger chunks**: More context, but fewer chunks (less granular search)
- **Smaller chunks**: More granular, but may lose context
- **More overlap**: Better context preservation, but more storage
- **Less overlap**: Less storage, but may lose context at boundaries

## See It in Action

Run the demonstration:

```bash
# Simple demo
python demo_text_splitting.py

# Detailed explanation
python explain_text_splitting.py

# Inspect actual data
python inspect_database.py --use-in-memory --load-data data/raw/raw_materials.jsonl --show-splitting
```

