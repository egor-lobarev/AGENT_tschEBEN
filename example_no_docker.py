"""
Example script using in-memory Qdrant (NO DOCKER REQUIRED).
Perfect for testing and development without installing Docker.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag.vectore_store import VectorStore
from src.rag.retriver import Retriever
from src.rag.generator import RAGGenerator


def main():
    """Main example function - NO DOCKER NEEDED."""
    # Path to data file
    data_path = "data/raw/raw_materials.jsonl"
    
    print("=" * 60)
    print("RAG System Example - NO DOCKER REQUIRED")
    print("Using in-memory Qdrant")
    print("=" * 60)
    
    # Step 1: Initialize and build vector store with in-memory mode
    print("\n1. Building vector store (in-memory)...")
    vector_store = VectorStore(
        collection_name="construction_materials",
        use_in_memory=True  # No Docker needed!
    )
    
    # Add documents to vector store
    print(f"   Loading documents from {data_path}...")
    vector_store.add_documents(data_path)
    
    # Show collection info
    info = vector_store.get_collection_info()
    print(f"   Collection info: {info['points_count']} points stored")
    
    # Step 2: Initialize retriever with shared client (important for in-memory mode!)
    print("\n2. Initializing retriever (in-memory)...")
    retriever = Retriever(
        collection_name="construction_materials",
        qdrant_client=vector_store.qdrant_client  # Share the same client!
    )
    
    # Step 3: Example queries
    print("\n3. Running example queries...")
    queries = [
        "бетон М400 для фундамента",
        "гравийный щебень фракция 20-40",
        "песок карьерный доставка"
    ]
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        results = retriever.retrieve(query, top_k=3)
        print(f"   Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"      {i}. Score: {result['score']:.4f}")
            print(f"         URL: {result['url']}")
            print(f"         Text preview: {result['text'][:100]}...")
    
    # Step 4: Using RAG generator
    print("\n4. Using RAG generator...")
    generator = RAGGenerator(retriever)
    
    query = "бетон М300 характеристики"
    result = generator.generate(query, top_k=2)
    
    print(f"\n   Query: {query}")
    print(f"   Retrieved {len(result['retrieved_documents'])} documents")
    print("\n   Response preview:")
    print(result['response'][:500] + "...")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("Note: Data is stored in memory and will be lost when script ends.")
    print("For persistent storage, use Docker or native Qdrant installation.")
    print("=" * 60)


if __name__ == "__main__":
    main()

