"""
Enhanced RAG API wrapper for integration with LangChain chains.
Provides query_rag function with conversation context support.
"""

from typing import Optional
from src.rag.generator import RAGGenerator
from src.rag.retriver import Retriever


# Global RAG generator instance (will be initialized by bot)
_rag_generator: Optional[RAGGenerator] = None


def initialize_rag(retriever: Retriever, llm=None) -> None:
    """
    Initialize the RAG generator with a retriever and optional LLM.
    
    Args:
        retriever: Retriever instance for document retrieval
        llm: Optional LLM instance for generating natural responses
    """
    global _rag_generator
    _rag_generator = RAGGenerator(retriever, llm=llm)


def query_rag(
    question: str, 
    top_k: int = 5, 
    conversation_context: str = "",
    min_score: float = 0.3
) -> str:
    """
    Query the RAG system and return a formatted response with enhanced prompt engineering.
    
    Args:
        question: User question
        top_k: Number of documents to retrieve
        conversation_context: Optional conversation history for context-aware responses
        min_score: Minimum similarity score for document filtering
        
    Returns:
        Formatted response string with relevant information
        
    Raises:
        RuntimeError: If RAG system is not initialized
    """
    if _rag_generator is None:
        raise RuntimeError("RAG system not initialized. Call initialize_rag() first.")
    
    # Generate response using enhanced RAG generator
    result = _rag_generator.generate(
        query=question,
        top_k=top_k,
        conversation_context=conversation_context,
        min_score=min_score,
        use_llm=True
    )
    
    return result.get('response', 'К сожалению, не удалось сгенерировать ответ.')

