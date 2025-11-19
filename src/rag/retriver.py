"""
Retriever module for RAG system.
Handles KNN search using cosine distance in Qdrant.
"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch


class Retriever:
    """Retriever for finding top-k similar documents using KNN."""
    
    def __init__(
        self,
        collection_name: str = "construction_materials",
        model_name: str = "stsb-roberta-large",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        use_in_memory: bool = False,
        qdrant_client: Optional[QdrantClient] = None,
        embedding_dtype: Optional[str] = None,
    ):
        """
        Initialize the retriever.
        
        Args:
            collection_name: Name of the Qdrant collection
            model_name: Name of the SentenceTransformer model
            qdrant_host: Qdrant server host (ignored if use_in_memory=True or qdrant_client provided)
            qdrant_port: Qdrant server port (ignored if use_in_memory=True or qdrant_client provided)
            use_in_memory: If True, use in-memory Qdrant (no Docker/server needed)
            qdrant_client: Optional QdrantClient instance to share (useful for in-memory mode)
            embedding_dtype: Quantization / dtype for SentenceTransformer weights:
                             "float32" (default), "float16", "float8", "int8", or "int"
        """
        self.collection_name = collection_name
        
        # Use provided client, or create new one
        if qdrant_client is not None:
            self.qdrant_client = qdrant_client
        elif use_in_memory:
            self.qdrant_client = QdrantClient(":memory:")
        else:
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Initialize embedding model with optional quantization / dtype
        dtype_key = (embedding_dtype or "float32").lower()
        if dtype_key == "float16":
            torch_dtype = torch.float16
        elif dtype_key in {"float8", "fp8"}:
            torch_dtype = getattr(torch, "float8_e4m3fn", torch.float16)
        elif dtype_key in {"int8", "int"}:
            torch_dtype = torch.int8
        else:
            torch_dtype = torch.float32

        self.embedder = SentenceTransformer(
            model_name,
            model_kwargs={"dtype": torch_dtype}
        )
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        embedding = self.embedder.encode(query, show_progress_bar=False)
        return embedding.tolist()
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar documents using KNN with cosine distance.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of retrieved documents with scores
        """
        # Generate query embedding
        query_vector = self.embed_query(query)
        
        # Search in Qdrant using KNN
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                'id': result.id,
                'score': result.score,  # Cosine similarity score
                'text': result.payload.get('text', ''),
                'url': result.payload.get('url', ''),
                'timestamp': result.payload.get('timestamp', 0),
                'chunk_index': result.payload.get('chunk_index', 0),
                'doc_index': result.payload.get('doc_index', 0)
            })
        
        return results
    
    def retrieve_top_k(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Alias for retrieve method.
        
        Args:
            query: Query text
            k: Number of top results to return
            
        Returns:
            List of retrieved documents with scores
        """
        return self.retrieve(query, top_k=k)

