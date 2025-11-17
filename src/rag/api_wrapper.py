"""
RAG API wrapper for integration with LangChain chains.
Provides query_rag function as specified in the requirements.

This module uses the project's RAG system:
- src.rag.generator.RAGGenerator - generates responses from retrieved documents
- src.rag.retriver.Retriever - retrieves relevant documents from Qdrant
- src.rag.vectore_store.VectorStore - stores document embeddings

The query_rag() function is used by OrchestratorChain for informational queries.
"""

from typing import Optional
from src.rag.generator import RAGGenerator  # Project's RAG generator
from src.rag.retriver import Retriever  # Project's retriever


# Global RAG generator instance (will be initialized by bot)
_rag_generator: Optional[RAGGenerator] = None


def initialize_rag(retriever: Retriever) -> None:
    """
    Initialize the RAG generator with a retriever.
    
    Args:
        retriever: Retriever instance for document retrieval
    """
    global _rag_generator
    _rag_generator = RAGGenerator(retriever)


def query_rag(question: str, top_k: int = 5) -> str:
    """
    Query the RAG system and return a formatted response.
    This is the API function for Саша's RAG module.
    
    Uses the project's RAG system (src/rag/generator.py, src/rag/retriver.py).
    Called by OrchestratorChain for informational queries.
    
    Args:
        question: User question
        top_k: Number of documents to retrieve
        
    Returns:
        Formatted response string with relevant information
        
    Raises:
        RuntimeError: If RAG system is not initialized
    """
    if _rag_generator is None:
        raise RuntimeError("RAG system not initialized. Call initialize_rag() first.")
    
    # Generate response using project's RAG generator (src/rag/generator.py)
    result = _rag_generator.generate(question, top_k=top_k)
    
    # Extract and format the response
    # For now, return the context from retrieved documents
    # In a full implementation, this could use an LLM to generate a natural response
    if result['retrieved_documents']:
        # Format as a readable response
        context_parts = []
        for i, doc in enumerate(result['retrieved_documents'], 1):
            context_parts.append(f"{doc['text']}")
        
        response = "\n\n".join(context_parts)
        return response
    else:
        return "К сожалению, не удалось найти релевантную информацию по вашему запросу."

