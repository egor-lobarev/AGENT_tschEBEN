"""
Vector Store module for RAG system.
Handles document loading, text splitting, embedding, and storage in Qdrant.
"""

import json
from typing import List, Dict, Any, Optional

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback for newer langchain versions
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError("Please install langchain or langchain-text-splitters")
from sentence_transformers import SentenceTransformer
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class VectorStore:
    """Vector store using Qdrant for storing document embeddings."""
    
    def __init__(
        self,
        collection_name: str = "construction_materials",
        model_name: str = "stsb-roberta-large",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        use_in_memory: bool = False,
        qdrant_client: Optional[QdrantClient] = None,
        embedding_dtype: Optional[str] = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the Qdrant collection
            model_name: Name of the SentenceTransformer model
            qdrant_host: Qdrant server host (ignored if use_in_memory=True or qdrant_client provided)
            qdrant_port: Qdrant server port (ignored if use_in_memory=True or qdrant_client provided)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
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
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Initialize embedding model with optional quantization / dtype
        dtype_key = (embedding_dtype or "float32").lower()
        # Map human-readable string to torch dtype
        if dtype_key == "float16":
            torch_dtype = torch.float16
        elif dtype_key in {"float8", "fp8"}:
            # Use float8 if available, otherwise fall back to float16
            torch_dtype = getattr(torch, "float8_e4m3fn", torch.float16)
        elif dtype_key in {"int8", "int"}:
            torch_dtype = torch.int8
        else:
            torch_dtype = torch.float32

        self.embedder = SentenceTransformer(
            model_name,
            model_kwargs={"dtype": torch_dtype}
        )
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Create collection if it doesn't exist
        self._create_collection()
    
    def _create_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        try:
            self.qdrant_client.get_collection(self.collection_name)
        except Exception:
            # Collection doesn't exist, create it
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
    
    def load_documents(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """
        Load documents from JSONL file.
        
        Args:
            jsonl_path: Path to JSONL file
            
        Returns:
            List of document dictionaries
        """
        documents = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    # Only process documents with content and no errors
                    if doc.get('content') and not doc.get('error'):
                        documents.append(doc)
        return documents
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using LangChain text splitter.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        return self.text_splitter.split_text(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using SentenceTransformer.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def add_documents(self, jsonl_path: str):
        """
        Load documents from JSONL, split them, embed, and add to Qdrant.
        
        Args:
            jsonl_path: Path to JSONL file
        """
        import hashlib
        
        documents = self.load_documents(jsonl_path)
        
        points = []
        
        for doc_idx, doc in enumerate(documents):
            content = doc.get('content', '')
            if not content:
                continue
            
            # Split text into chunks
            chunks = self.split_text(content)
            
            # Generate embeddings for chunks
            embeddings = self.embed_texts(chunks)
            
            # Create points for Qdrant
            for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create a unique ID based on URL, doc index, and chunk index
                # This prevents duplicates if add_documents is called multiple times
                unique_id_string = f"{doc.get('url', '')}_{doc_idx}_{chunk_idx}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
                point_id = int(hashlib.md5(unique_id_string.encode()).hexdigest()[:15], 16)  # Use first 15 hex chars as int
                
                payload = {
                    'url': doc.get('url', ''),
                    'timestamp': doc.get('timestamp', 0),
                    'chunk_index': chunk_idx,
                    'doc_index': doc_idx,
                    'text': chunk,
                    'original_content_length': len(content)
                }
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                )
        
        # Upload points to Qdrant in batches
        # upsert will update existing points with same ID or add new ones
        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Added {len(points)} chunks from {len(documents)} documents to vector store")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        return {
            'points_count': collection_info.points_count,
            'vectors_count': collection_info.vectors_count,
            'config': collection_info.config
        }
    
    def delete_collection(self):
        """Delete the collection."""
        self.qdrant_client.delete_collection(self.collection_name)
        print(f"Deleted collection: {self.collection_name}")
    
    def get_sample_chunks(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get sample chunks from the collection.
        
        Args:
            limit: Number of chunks to retrieve
            
        Returns:
            List of chunk dictionaries with payload information
        """
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            return [
                {
                    'id': point.id,
                    'payload': point.payload
                }
                for point in scroll_result[0]
            ]
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []
    
    def explain_splitting(self, text: str) -> Dict[str, Any]:
        """
        Explain how a text would be split, showing chunks and overlap.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with splitting information
        """
        chunks = self.split_text(text)
        
        return {
            'original_length': len(text),
            'chunk_size': self.text_splitter._chunk_size,
            'chunk_overlap': self.text_splitter._chunk_overlap,
            'num_chunks': len(chunks),
            'chunks': [
                {
                    'index': i,
                    'text': chunk,
                    'length': len(chunk)
                }
                for i, chunk in enumerate(chunks)
            ],
            'average_chunk_size': sum(len(c) for c in chunks) / len(chunks) if chunks else 0
        }

