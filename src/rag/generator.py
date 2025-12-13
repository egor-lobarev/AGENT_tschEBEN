"""
Enhanced Generator module for RAG system with prompt engineering.
Combines retrieved documents with query to generate natural responses using LLM.
"""

from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.rag.retriver import Retriever


class RAGGenerator:
    """Enhanced generator for RAG responses using retrieved documents and LLM."""
    
    def __init__(self, retriever: Retriever, llm=None):
        """
        Initialize the RAG generator.
        
        Args:
            retriever: Retriever instance for document retrieval
            llm: Optional LLM instance for generating natural responses (if None, uses simple formatting)
        """
        self.retriever = retriever
        self.llm = llm
        
        # Enhanced prompt template for RAG generation with prompt engineering
        if llm:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """Ты - эксперт-консультант по строительным материалам. 
Твоя задача - отвечать на вопросы пользователей на основе предоставленной документации.

ИНСТРУКЦИИ:
1. Отвечай ТОЛЬКО на основе предоставленных документов. Не придумывай информацию.
2. Если в документах нет ответа на вопрос, честно скажи об этом.
3. Отвечай на русском языке, дружелюбно и профессионально.
4. Структурируй ответ: сначала краткий ответ, затем детали из документов.
5. Если информация из разных документов противоречит друг другу, укажи на это.
6. Используй конкретные данные из документов (марки, характеристики, цены).
7. Если вопрос неясен, попроси уточнить.
8. Не упоминай, что информация взята из документов - просто отвечай естественно.

КОНТЕКСТ РАЗГОВОРА:
{conversation_context}

РЕТРИВИРОВАННЫЕ ДОКУМЕНТЫ:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}

Сгенерируй естественный, информативный ответ на основе предоставленных документов:"""),
                ("human", "{question}")
            ])
            
            # Create chain for LLM-based generation
            self.chain = (
                {
                    "question": RunnablePassthrough(),
                    "context": lambda x: self._format_context(x.get("retrieved_docs", [])),
                    "conversation_context": lambda x: x.get("conversation_context", "")
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
    
    def _filter_documents(self, docs: List[Dict[str, Any]], min_score: float = 0.3) -> List[Dict[str, Any]]:
        """
        Filter retrieved documents by relevance score.
        
        Args:
            docs: List of retrieved documents
            min_score: Minimum similarity score threshold
            
        Returns:
            Filtered list of documents
        """
        # Filter by score (cosine similarity, higher is better)
        filtered = [doc for doc in docs if doc.get('score', 0) >= min_score]
        
        # If no documents pass threshold, return top 2 anyway (might be edge case)
        if not filtered and docs:
            return docs[:2]
        
        return filtered
    
    def _format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "Релевантные документы не найдены."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            doc_text = doc.get('text', '').strip()
            if doc_text:
                # Add source URL if available for transparency
                url_info = f" (Источник: {doc.get('url', 'неизвестен')})" if doc.get('url') else ""
                context_parts.append(f"[Документ {i}{url_info}]\n{doc_text}")
        
        return "\n\n".join(context_parts)
    
    def generate(
        self, 
        query: str, 
        top_k: int = 5, 
        conversation_context: str = "",
        min_score: float = 0.3,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a response using RAG with enhanced prompt engineering.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            conversation_context: Optional conversation history for context
            min_score: Minimum similarity score for document filtering
            use_llm: Whether to use LLM for response generation (if available)
            
        Returns:
            Dictionary containing query, retrieved documents, and generated response
        """
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
        
        # Step 2: Filter documents by relevance score
        filtered_docs = self._filter_documents(retrieved_docs, min_score=min_score)
        
        # Step 3: Format context
        context = self._format_context(filtered_docs)
        
        # Step 4: Generate response
        if self.llm and use_llm and filtered_docs:
            try:
                # Use LLM with prompt engineering for natural response
                response = self.chain.invoke({
                    "question": query,
                    "retrieved_docs": filtered_docs,
                    "conversation_context": conversation_context
                })
            except Exception as e:
                # Fallback to simple formatting if LLM fails
                print(f"Warning: LLM generation failed, using fallback: {e}")
                response = self._generate_fallback_response(query, filtered_docs)
        else:
            print("LLM not used for RAG generation.")
            # Fallback: simple response generation
            response = self._generate_fallback_response(query, filtered_docs)

        return {
            'query': query,
            'retrieved_documents': filtered_docs,
            'context': context,
            'response': response,
            'num_retrieved': len(retrieved_docs),
            'num_filtered': len(filtered_docs)
        }
    
    def _generate_fallback_response(self, query: str, docs: List[Dict[str, Any]]) -> str:
        """
        Generate a fallback response when LLM is not available or fails.
        
        Args:
            query: User query
            docs: Retrieved documents
            
        Returns:
            Formatted response string
        """
        if not docs:
            return "К сожалению, не удалось найти релевантную информацию по вашему запросу. Попробуйте переформулировать вопрос или уточнить детали."
        
        response_parts = [
            f"Вот информация по вашему запросу:\n"
        ]
        
        for i, doc in enumerate(docs, 1):
            doc_text = doc.get('text', '').strip()
            if doc_text:
                response_parts.append(f"{doc_text}")
        
        return "\n\n".join(response_parts)
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """
        Format the RAG result as a readable string.
        
        Args:
            result: Result dictionary from generate method
            
        Returns:
            Formatted string response
        """
        return result.get('response', '')

