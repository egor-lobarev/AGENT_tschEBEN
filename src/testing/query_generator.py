"""
Query generator for creating test queries.
Generates realistic queries that a buyer might ask.
"""

from typing import List, Dict, Any, Optional
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json


class QueryGenerator:
    """Generates test queries for bot evaluation."""
    
    def __init__(
        self,
        mistral_api_key: Optional[str] = None,
        model: str = "mistral-small",  # Use smaller model to avoid rate limits
        temperature: float = 0.7
    ):
        """
        Initialize the query generator.
        
        Args:
            mistral_api_key: Mistral API key
            model: Mistral model to use
            temperature: Temperature for generation
        """
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        if not self.mistral_api_key:
            raise ValueError("Mistral API key not provided.")
        
        self.llm = ChatMistralAI(
            model=model,
            mistral_api_key=self.mistral_api_key,
            temperature=temperature,
            max_retries=3,  # Add retry logic for rate limits
            timeout=60.0  # Increase timeout
        )
        
        self.generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are generating test queries for a construction materials chatbot.
Generate realistic queries that Russian customers might ask when:
1. Looking for information about construction materials (informational queries)
2. Wanting to place an order for construction materials (order specification queries)

For informational queries, customers might ask about:
- Material properties and characteristics
- Usage recommendations
- Technical specifications
- Comparison between materials

For order specification queries, customers might:
- Express intent to buy (e.g., "Хочу заказать бетон М300")
- Provide partial specifications
- Ask about availability

Generate queries in Russian, as they would naturally be asked by customers.
Return a JSON array of query objects, each with:
{{
    "query": "<the actual query text>",
    "type": "informational" or "order_specification",
    "expected_clarification": true/false (for order queries, whether clarification is expected)
}}"""),
            ("human", "Generate {count} diverse test queries for a construction materials chatbot. "
                     "Mix informational and order specification queries. "
                     "Return only the JSON array, no other text.")
        ])
    
    def generate_queries(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Generate test queries.
        
        Args:
            count: Number of queries to generate
            
        Returns:
            List of query dictionaries
        """
        try:
            formatted_prompt = self.generation_prompt.format_messages(count=count)
            result = self.llm.invoke(formatted_prompt)
            queries_text = result.content.strip()
            
            # Parse JSON
            if "```json" in queries_text:
                json_start = queries_text.find("```json") + 7
                json_end = queries_text.find("```", json_start)
                queries_text = queries_text[json_start:json_end].strip()
            elif "```" in queries_text:
                json_start = queries_text.find("```") + 3
                json_end = queries_text.find("```", json_start)
                queries_text = queries_text[json_start:json_end].strip()
            
            queries = json.loads(queries_text)
            
            # Validate structure
            if not isinstance(queries, list):
                queries = [queries]
            
            return queries
            
        except Exception as e:
            # Fallback to predefined queries if generation fails
            print(f"Query generation failed: {e}. Using predefined queries.")
            return self._get_predefined_queries()[:count]
    
    def _get_predefined_queries(self) -> List[Dict[str, Any]]:
        """Fallback predefined queries."""
        return [
            {
                "query": "Какие характеристики у бетона М300?",
                "type": "informational",
                "expected_clarification": False
            },
            {
                "query": "Нужен бетон для фундамента",
                "type": "informational",
                "expected_clarification": False
            },
            {
                "query": "Хочу заказать бетон М300",
                "type": "order_specification",
                "expected_clarification": True
            },
            {
                "query": "Мне нужен песок для строительства",
                "type": "order_specification",
                "expected_clarification": True
            },
            {
                "query": "Какой щебень лучше для дорожного покрытия?",
                "type": "informational",
                "expected_clarification": False
            },
            {
                "query": "Заказать 5 кубов бетона М400",
                "type": "order_specification",
                "expected_clarification": False
            },
            {
                "query": "Какая фракция щебня нужна для бетона?",
                "type": "informational",
                "expected_clarification": False
            },
            {
                "query": "Нужен гравий",
                "type": "order_specification",
                "expected_clarification": True
            },
            {
                "query": "Сравните бетон М300 и М400",
                "type": "informational",
                "expected_clarification": False
            },
            {
                "query": "Хочу купить 10 тонн песка, фракция 0-5",
                "type": "order_specification",
                "expected_clarification": False
            }
        ]

