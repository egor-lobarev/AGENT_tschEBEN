"""
Setup script to initialize the RAG system.
Run this first to build the vector store from documents.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.rag.vectore_store import VectorStore


def setup():
    """Set up the vector store with documents."""
    print("Setting up RAG system...")
    print("=" * 60)
    
    # Check if data file exists
    data_path = Path("data/raw/raw_materials.jsonl")
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data file exists.")
        return
    
    print(f"Found data file: {data_path}")
    
    # Initialize vector store
    print("\nInitializing vector store...")
    print("Using in-memory mode (no Docker required)")
    vector_store = VectorStore(
        collection_name="construction_materials",
        use_in_memory=True  # No Docker needed!
    )
    
    # Add documents
    print(f"\nLoading and indexing documents from {data_path}...")
    print("This may take a few minutes depending on the number of documents...")
    
    try:
        vector_store.add_documents(str(data_path))
        
        # Show collection info
        info = vector_store.get_collection_info()
        print(f"\n✓ Successfully indexed {info['points_count']} document chunks")
        print(f"  Collection: {vector_store.collection_name}")
        print("\nSetup complete! You can now use the RAG system.")
        print("\nTry running: python example_rag.py")
        
    except Exception as e:
        print(f"\n✗ Error during setup: {e}")
        print("\nIf you want to use Docker instead of in-memory mode:")
        print("  1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        print("  2. Remove use_in_memory=True from VectorStore initialization")


if __name__ == "__main__":
    setup()

