from src.rag.vectore_store import VectorStore
from src.rag.retriver import Retriever
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from pathlib import Path
from config import EMBEDDING_MODEL
from config.config import EMBEDDING_DTYPE

# Default persistent storage path for Qdrant
DEFAULT_QDRANT_STORAGE_PATH = "data/qdrant_storage"

# Shared Qdrant client singleton (for persistent storage)
# This ensures the same instance is reused across bot initializations
_shared_qdrant_client: Optional[QdrantClient] = None
_shared_qdrant_path: Optional[str] = None

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
    
def setup_rag_system(
    use_in_memory: bool = False,
    data_path: str = "data/raw/raw_materials.jsonl",
    qdrant_storage_path: str = DEFAULT_QDRANT_STORAGE_PATH,
    embedding_model: Optional[str] = None,
    embedding_dtype: Optional[str] = None,
):
    """
    Set up the RAG system and return components ready for LangChain integration.
    
    Uses persistent disk storage by default to avoid recomputing embeddings.
    Data persists across bot initializations and process restarts.
    
    Args:
        use_in_memory: If True, use in-memory Qdrant (data lost on restart)
        data_path: Path to JSONL data file
        qdrant_storage_path: Path to store Qdrant data (only used if use_in_memory=False)
        embedding_model: Name of embedding model (SentenceTransformer-compatible).
                         If None, uses EMBEDDING_MODEL from config.
        embedding_dtype: Quantization / dtype for SentenceTransformer weights:
                         "float32" (default), "float16", "float8", "int8", or "int".
                         If None, uses EMBEDDING_DTYPE from config.
        
    Returns:
        Tuple of (vector_store, retriever, custom_retriever)
    """
    global _shared_qdrant_client, _shared_qdrant_path
    
    # Create or reuse shared Qdrant client
    if use_in_memory:
        # In-memory mode (for testing, data lost on restart)
        if _shared_qdrant_client is None:
            print("Creating shared in-memory Qdrant client...")
            _shared_qdrant_client = QdrantClient(":memory:")
        else:
            print("Reusing existing in-memory Qdrant client...")
        qdrant_client = _shared_qdrant_client
    else:
        # Persistent disk storage (default, data persists across restarts)
        if _shared_qdrant_client is None or _shared_qdrant_path != qdrant_storage_path:
            # Ensure storage directory exists
            Path(qdrant_storage_path).mkdir(parents=True, exist_ok=True)
            print(f"Creating persistent Qdrant client at {qdrant_storage_path}...")
            _shared_qdrant_client = QdrantClient(path=qdrant_storage_path)
            _shared_qdrant_path = qdrant_storage_path
        else:
            print(f"Reusing existing persistent Qdrant client at {qdrant_storage_path}...")
        qdrant_client = _shared_qdrant_client
    
    # Resolve embedding model and dtype (config defaults if not provided)
    embedding_model = embedding_model or EMBEDDING_MODEL
    embedding_dtype = embedding_dtype or EMBEDDING_DTYPE

    # Initialize vector store with shared client
    print("Start Vector Store initializating")
    vector_store = VectorStore(
        collection_name="construction_materials",
        model_name=embedding_model,
        embedding_dtype=embedding_dtype,
        use_in_memory=use_in_memory,
        qdrant_client=qdrant_client
    )
    
    # Load documents if not already loaded
    print("Start loading documents")
    info = vector_store.get_collection_info()
    if info['points_count'] == 0:
        print(f"Loading documents from {data_path}...")
        vector_store.add_documents(data_path)
        info = vector_store.get_collection_info()
        print(f"Loaded {info['points_count']} chunks")
    else:
        print(f"Using existing vector store with {info['points_count']} chunks (skipping re-embedding)")
    
    print("Initialize retriever with shared client")
    # Initialize retriever with shared client
    retriever = Retriever(
        collection_name="construction_materials",
        model_name=embedding_model,
        embedding_dtype=embedding_dtype,
        use_in_memory=use_in_memory,
        qdrant_client=qdrant_client
    )
    
    # Create LangChain-compatible retriever
    custom_retriever = CustomRetriever(retriever)
    
    return vector_store, retriever, custom_retriever