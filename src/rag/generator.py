"""
Generator module for RAG system.
Combines retrieved documents with query to generate responses.
"""

from typing import List, Dict, Any
from src.rag.retriver import Retriever


class RAGGenerator:
    """Generator for RAG responses using retrieved documents."""
    
    def __init__(self, retriever: Retriever):
        """
        Initialize the RAG generator.
        
        Args:
            retriever: Retriever instance for document retrieval
        """
        self.retriever = retriever
    
    def generate(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Generate a response using RAG.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary containing query, retrieved documents, and generated response
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
        
        # Combine retrieved documents
        context = "\n\n".join([
            f"[Document {i+1} from {doc['url']}]\n{doc['text']}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Simple response generation (can be extended with LLM)
        response = f"""Based on the retrieved documents, here is the relevant information:

{context}

These documents were retrieved based on your query: "{query}"
"""
        
        return {
            'query': query,
            'retrieved_documents': retrieved_docs,
            'context': context,
            'response': response
        }
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """
        Format the RAG result as a readable string.
        
        Args:
            result: Result dictionary from generate method
            
        Returns:
            Formatted string response
        """
        response_parts = [f"Query: {result['query']}\n"]
        response_parts.append("\nRetrieved Documents:\n")
        
        for i, doc in enumerate(result['retrieved_documents'], 1):
            response_parts.append(f"\n--- Document {i} (Score: {doc['score']:.4f}) ---")
            response_parts.append(f"URL: {doc['url']}")
            response_parts.append(f"Text: {doc['text'][:200]}...")  # First 200 chars
        
        return "\n".join(response_parts)

