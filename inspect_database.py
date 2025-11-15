"""
Utility script to inspect the Qdrant database and see how data is split.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
Path = Path  # Keep Path available for type hints

from src.rag.vectore_store import VectorStore
from src.rag.retriver import Retriever


def inspect_collection(vector_store: VectorStore, limit: int = 10):
    """
    Inspect the collection and show sample documents.
    
    Args:
        vector_store: VectorStore instance
        limit: Number of documents to show
    """
    print("=" * 80)
    print("DATABASE INSPECTION")
    print("=" * 80)
    
    # Get collection info
    info = vector_store.get_collection_info()
    print(f"\nCollection: {vector_store.collection_name}")
    print(f"Total points (chunks): {info['points_count']}")
    print(f"Vector dimension: {info['config'].params.vectors.size}")
    print(f"Distance metric: {info['config'].params.vectors.distance}")
    
    # Get sample points
    print(f"\n{'=' * 80}")
    print(f"SAMPLE DOCUMENTS (showing first {limit} chunks)")
    print(f"{'=' * 80}\n")
    
    try:
        # Scroll through collection to get sample points
        scroll_result = vector_store.qdrant_client.scroll(
            collection_name=vector_store.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        points = scroll_result[0]
        
        for i, point in enumerate(points, 1):
            payload = point.payload
            print(f"--- Chunk {i} (ID: {point.id}) ---")
            print(f"URL: {payload.get('url', 'N/A')}")
            print(f"Document Index: {payload.get('doc_index', 'N/A')}")
            print(f"Chunk Index: {payload.get('chunk_index', 'N/A')}")
            print(f"Original Content Length: {payload.get('original_content_length', 'N/A')} chars")
            print(f"Text Preview ({len(payload.get('text', ''))} chars):")
            text = payload.get('text', '')
            if len(text) > 200:
                print(f"  {text[:200]}...")
            else:
                print(f"  {text}")
            print()
    
    except Exception as e:
        print(f"Error retrieving points: {e}")


def show_text_splitting_example(vector_store: VectorStore, sample_text: str = None):
    """
    Demonstrate how LangChain splits text.
    
    Args:
        vector_store: VectorStore instance
        sample_text: Optional sample text to split
    """
    print("=" * 80)
    print("TEXT SPLITTING DEMONSTRATION")
    print("=" * 80)
    
    if sample_text is None:
        # Get a sample document from the database
        try:
            scroll_result = vector_store.qdrant_client.scroll(
                collection_name=vector_store.collection_name,
                limit=1,
                with_payload=True
            )
            if scroll_result[0]:
                # Get original document
                sample_text = scroll_result[0][0].payload.get('text', '')
                if not sample_text:
                    # Try to get a longer sample from a document
                    print("Fetching a longer sample...")
                    # Get all chunks from first document
                    all_points = vector_store.qdrant_client.scroll(
                        collection_name=vector_store.collection_name,
                        limit=1000,
                        with_payload=True
                    )[0]
                    
                    if all_points:
                        first_doc_index = all_points[0].payload.get('doc_index')
                        # Reconstruct original text from chunks
                        chunks = [
                            p.payload.get('text', '')
                            for p in all_points
                            if p.payload.get('doc_index') == first_doc_index
                        ]
                        sample_text = '\n\n'.join(chunks[:3])  # First 3 chunks
        except Exception as e:
            print(f"Could not fetch sample from database: {e}")
            sample_text = """This is a sample text that will be split by LangChain's RecursiveCharacterTextSplitter. 
The splitter will break this text into chunks of a specified size with some overlap between chunks. 
This helps maintain context across chunk boundaries. Each chunk will be embedded separately and stored in the vector database."""
    
    print(f"\nOriginal Text Length: {len(sample_text)} characters")
    print(f"Chunk Size: {vector_store.text_splitter._chunk_size}")
    print(f"Chunk Overlap: {vector_store.text_splitter._chunk_overlap}")
    
    # Split the text
    chunks = vector_store.split_text(sample_text)
    
    print(f"\nNumber of Chunks Created: {len(chunks)}")
    print(f"\n{'=' * 80}")
    print("CHUNKS:")
    print(f"{'=' * 80}\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ({len(chunk)} characters) ---")
        if len(chunk) > 300:
            print(f"{chunk[:150]}...")
            print(f"...{chunk[-150:]}")
        else:
            print(chunk)
        print()
        
        # Show overlap with next chunk
        if i < len(chunks):
            next_chunk = chunks[i]
            overlap = min(len(chunk), len(next_chunk), vector_store.text_splitter._chunk_overlap)
            if overlap > 0:
                chunk_end = chunk[-overlap:] if len(chunk) >= overlap else chunk
                next_start = next_chunk[:overlap] if len(next_chunk) >= overlap else next_chunk
                print(f"  Overlap with next chunk: {overlap} characters")
                print(f"  End of chunk {i}: ...{chunk_end[-50:]}")
                print(f"  Start of chunk {i+1}: {next_start[:50]}...")
                print()


def show_statistics(vector_store: VectorStore):
    """Show statistics about the database."""
    print("=" * 80)
    print("DATABASE STATISTICS")
    print("=" * 80)
    
    try:
        # Get all points
        all_points = vector_store.qdrant_client.scroll(
            collection_name=vector_store.collection_name,
            limit=10000,  # Adjust if you have more
            with_payload=True
        )[0]
        
        if not all_points:
            print("No points found in collection.")
            return
        
        # Group by document
        docs = {}
        for point in all_points:
            doc_idx = point.payload.get('doc_index')
            if doc_idx not in docs:
                docs[doc_idx] = {
                    'url': point.payload.get('url', 'N/A'),
                    'chunks': [],
                    'total_length': 0
                }
            chunks = docs[doc_idx]['chunks']
            chunks.append({
                'chunk_idx': point.payload.get('chunk_index'),
                'text': point.payload.get('text', ''),
                'length': len(point.payload.get('text', ''))
            })
            docs[doc_idx]['total_length'] += len(point.payload.get('text', ''))
        
        print(f"\nTotal Documents: {len(docs)}")
        print(f"Total Chunks: {len(all_points)}")
        print(f"Average Chunks per Document: {len(all_points) / len(docs):.2f}")
        
        # Chunk size statistics
        chunk_lengths = [len(p.payload.get('text', '')) for p in all_points]
        if chunk_lengths:
            print(f"\nChunk Length Statistics:")
            print(f"  Min: {min(chunk_lengths)} characters")
            print(f"  Max: {max(chunk_lengths)} characters")
            print(f"  Average: {sum(chunk_lengths) / len(chunk_lengths):.2f} characters")
        
        # Show top documents by chunk count
        print(f"\nTop 5 Documents by Number of Chunks:")
        sorted_docs = sorted(docs.items(), key=lambda x: len(x[1]['chunks']), reverse=True)
        for i, (doc_idx, doc_info) in enumerate(sorted_docs[:5], 1):
            print(f"  {i}. Document {doc_idx}: {len(doc_info['chunks'])} chunks, "
                  f"{doc_info['total_length']} total chars")
            print(f"     URL: {doc_info['url']}")
    
    except Exception as e:
        print(f"Error calculating statistics: {e}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect Qdrant database and text splitting')
    parser.add_argument('--use-in-memory', action='store_true', 
                       help='Use in-memory Qdrant (no Docker)')
    parser.add_argument('--collection', default='construction_materials',
                       help='Collection name')
    parser.add_argument('--limit', type=int, default=10,
                       help='Number of sample chunks to show')
    parser.add_argument('--show-splitting', action='store_true',
                       help='Show text splitting demonstration')
    parser.add_argument('--show-stats', action='store_true',
                       help='Show database statistics')
    parser.add_argument('--load-data', type=str, default=None,
                       help='Load data from JSONL file before inspecting')
    
    args = parser.parse_args()
    
    # Initialize vector store
    if args.use_in_memory:
        vector_store = VectorStore(
            collection_name=args.collection,
            use_in_memory=True
        )
    else:
        vector_store = VectorStore(
            collection_name=args.collection,
            qdrant_host="localhost",
            qdrant_port=6333
        )
    
    # Load data if requested
    if args.load_data:
        data_path = Path(args.load_data)
        if not data_path.exists():
            print(f"Error: Data file not found at {data_path}")
            return
        print(f"Loading documents from {data_path}...")
        print("This may take a few minutes...")
        try:
            vector_store.add_documents(str(data_path))
            info = vector_store.get_collection_info()
            print(f"✓ Loaded {info['points_count']} chunks\n")
        except Exception as e:
            print(f"Error loading documents: {e}")
            return
    
    # Check if collection is empty
    info = vector_store.get_collection_info()
    if info['points_count'] == 0:
        print("=" * 80)
        print("⚠️  DATABASE IS EMPTY")
        print("=" * 80)
        print("\nNo documents found in the database.")
        print("\nTo load documents, you have two options:")
        print("\n1. Use --load-data flag:")
        print("   python inspect_database.py --use-in-memory --load-data data/raw/raw_materials.jsonl")
        print("\n2. Run setup script first:")
        print("   python setup_rag.py")
        print("\n3. Or run example script:")
        print("   python example_no_docker.py")
        print("\nNote: In in-memory mode, data is lost when the program ends.")
        print("      You need to load data each time you start a new session.")
        return
    
    # Inspect collection
    inspect_collection(vector_store, limit=args.limit)
    
    # Show text splitting if requested
    if args.show_splitting:
        print("\n\n")
        show_text_splitting_example(vector_store)
    
    # Show statistics if requested
    if args.show_stats:
        print("\n\n")
        show_statistics(vector_store)


if __name__ == "__main__":
    main()

