from src.rag.vectore_store import VectorStore
from src.rag.retriver import Retriever
from typing import List, Dict, Any

class CustomRetriever:
    """
    Custom retriever wrapper that integrates with LangChain.
    This makes our Qdrant-based retriever compatible with LangChain chains.
    """
    
    def __init__(self, retriever: Retriever):
        """
        Initialize the custom retriever.
        
        Args:
            retriever: Our Retriever instance
        """
        self.retriever = retriever
    
    def get_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        This method signature is compatible with LangChain's retriever interface.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of document dictionaries with 'page_content' and 'metadata'
        """
        # Use our retriever to get results
        results = self.retriever.retrieve(query, top_k=top_k)
        
        # Convert to LangChain format
        documents = []
        for result in results:
            documents.append({
                'page_content': result['text'],  # LangChain expects 'page_content'
                'metadata': {
                    'url': result['url'],
                    'score': result['score'],
                    'chunk_index': result.get('chunk_index', 0),
                    'doc_index': result.get('doc_index', 0),
                    'timestamp': result.get('timestamp', 0)
                }
            })
        
        return documents
    
    def invoke(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        LangChain-compatible invoke method.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of documents
        """
        return self.get_relevant_documents(query, top_k)
    
def setup_rag_system(use_in_memory: bool = True, data_path: str = "data/raw/raw_materials.jsonl"):
    """
    Set up the RAG system and return components ready for LangChain integration.
    
    Args:
        use_in_memory: Use in-memory Qdrant (no Docker)
        data_path: Path to JSONL data file
        
    Returns:
        Tuple of (vector_store, retriever, custom_retriever)
    """
    # Initialize vector store
    vector_store = VectorStore(
        collection_name="construction_materials",
        use_in_memory=use_in_memory
    )
    
    # Load documents if not already loaded
    info = vector_store.get_collection_info()
    if info['points_count'] == 0:
        print(f"Loading documents from {data_path}...")
        vector_store.add_documents(data_path)
        info = vector_store.get_collection_info()
        print(f"Loaded {info['points_count']} chunks")
    
    # Initialize retriever
    retriever = Retriever(
        collection_name="construction_materials",
        qdrant_client=vector_store.qdrant_client if use_in_memory else None
    )
    
    # Create LangChain-compatible retriever
    custom_retriever = CustomRetriever(retriever)
    
    return vector_store, retriever, custom_retriever