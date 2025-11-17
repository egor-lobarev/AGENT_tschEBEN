"""
Classification Chain for determining query type.
Classifies user queries as either informational or order specification.
"""

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class ClassificationChain:
    """Chain for classifying user queries."""
    
    def __init__(self, llm):
        """
        Initialize the classification chain.
        
        Args:
            llm: LangChain LLM instance (e.g., ChatMistralAI)
        """
        self.llm = llm
        
        # Prompt for classification
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты помощник интернет-магазина стройматериалов. 
Твоя задача - определить тип запроса пользователя.

Типы запросов:
1. "informational" - информационный вопрос (например: "Какие характеристики у бетона М300?", "Что такое гравий?", "Как выбрать бетон?")
2. "order_specification" - спецификация заказа (например: "Нужен бетон М300", "Хочу заказать 5 кубов песка", "Мне нужен щебень 20-40")

Определи тип запроса и верни только одно слово: "informational" или "order_specification"."""),
            ("human", "{query}")
        ])
        
        # Create chain
        self.chain = (
            {"query": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def classify(self, query: str) -> Literal["informational", "order_specification"]:
        """
        Classify a user query.
        
        Args:
            query: User query text
            
        Returns:
            Query type: "informational" or "order_specification"
        """
        result = self.chain.invoke(query)
        
        # Clean and normalize the result
        result = result.strip().lower()
        
        # Extract the classification
        if "informational" in result or "информационный" in result.lower():
            return "informational"
        elif "order_specification" in result or "спецификация" in result.lower() or "заказ" in result.lower():
            return "order_specification"
        else:
            # Default: if query contains ordering keywords, it's order_specification
            ordering_keywords = ["нужен", "нужно", "хочу", "заказать", "купить", "мне нужно", "требуется"]
            if any(keyword in query.lower() for keyword in ordering_keywords):
                return "order_specification"
            return "informational"

