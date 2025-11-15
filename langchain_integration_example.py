"""
Example: Integration with LangChain for user interaction system.

This example shows how to integrate the RAG retriever with LangChain chains
for building conversational systems that work with users.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

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


def example_langchain_usage():
    """
    Example showing how to use the retriever with LangChain.
    """
    print("=" * 80)
    print("LANGCHAIN INTEGRATION EXAMPLE")
    print("=" * 80)
    
    # Setup RAG system
    print("\n1. Setting up RAG system...")
    vector_store, retriever, custom_retriever = setup_rag_system()
    
    # Example 1: Direct retrieval (without LangChain)
    print("\n2. Example: Direct retrieval")
    print("-" * 80)
    query = "бетон М400 характеристики"
    docs = custom_retriever.get_relevant_documents(query, top_k=3)
    
    print(f"Query: '{query}'")
    print(f"Retrieved {len(docs)} documents:\n")
    
    for i, doc in enumerate(docs, 1):
        print(f"Document {i}:")
        print(f"  Score: {doc['metadata']['score']:.4f}")
        print(f"  URL: {doc['metadata']['url']}")
        print(f"  Content preview: {doc['page_content'][:150]}...")
        print()
    
    # Example 2: Using with LangChain chains (pseudo-code)
    print("\n3. Example: Integration with LangChain chains")
    print("-" * 80)
    print("""
    # In your LangChain application, you can use it like this:
    
    from langchain.chains import RetrievalQA
    from langchain.llms import OpenAI  # or any other LLM
    
    # Initialize your LLM
    llm = OpenAI(temperature=0)
    
    # Create a retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=custom_retriever,  # Use our custom retriever
        return_source_documents=True
    )
    
    # Use with user queries
    user_query = "Какие характеристики у бетона М400?"
    result = qa_chain.invoke({"query": user_query})
    
    print(result['result'])  # Generated answer
    print(result['source_documents'])  # Retrieved documents
    """)
    
    # Example 3: Simple RAG pipeline
    print("\n4. Example: Simple RAG pipeline")
    print("-" * 80)
    
    def simple_rag_pipeline(user_query: str, top_k: int = 3) -> str:
        """
        Simple RAG pipeline that retrieves documents and formats a response.
        In a real system, you would use an LLM to generate the final answer.
        """
        # Retrieve relevant documents
        docs = custom_retriever.get_relevant_documents(user_query, top_k=top_k)
        
        # Format context from retrieved documents
        context = "\n\n".join([
            f"[Source {i+1} from {doc['metadata']['url']}]\n{doc['page_content']}"
            for i, doc in enumerate(docs)
        ])
        
        # In a real system, you would pass this context to an LLM
        # For now, we'll just return the formatted context
        response = f"""Based on the retrieved documents, here is relevant information:

{context}

These documents were retrieved based on your query: "{user_query}"
"""
        return response
    
    # Test the pipeline
    user_query = "бетон М300 применение"
    print(f"User Query: '{user_query}'")
    print("\nResponse:")
    response = simple_rag_pipeline(user_query, top_k=2)
    print(response[:500] + "...")
    
    print("\n" + "=" * 80)
    print("INTEGRATION COMPLETE")
    print("=" * 80)
    print("""
Your colleague can now:
1. Use CustomRetriever with LangChain chains
2. Integrate with any LangChain LLM (OpenAI, Anthropic, etc.)
3. Build conversational systems that retrieve from your RAG database
4. Use the retrieved documents as context for LLM generation
""")


def example_conversational_system():
    """
    Example of a simple conversational system using the retriever.
    """
    print("\n" + "=" * 80)
    print("CONVERSATIONAL SYSTEM EXAMPLE")
    print("=" * 80)
    
    # Setup
    _, _, custom_retriever = setup_rag_system()
    
    # Simulate a conversation
    conversation = [
        "бетон М400",
        "какие у него характеристики?",
        "где его можно купить?",
    ]
    
    print("\nSimulated conversation:")
    print("-" * 80)
    
    for i, user_input in enumerate(conversation, 1):
        print(f"\nUser {i}: {user_input}")
        
        # Retrieve relevant documents
        docs = custom_retriever.get_relevant_documents(user_input, top_k=2)
        
        # Format response (in real system, use LLM here)
        print(f"System: Found {len(docs)} relevant documents:")
        for j, doc in enumerate(docs, 1):
            print(f"  {j}. {doc['metadata']['url']} (score: {doc['metadata']['score']:.3f})")
            print(f"     {doc['page_content'][:100]}...")


if __name__ == "__main__":
    # Run examples
    example_langchain_usage()
    example_conversational_system()

