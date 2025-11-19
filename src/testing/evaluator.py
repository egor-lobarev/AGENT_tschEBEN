"""
Evaluation module for testing bot responses.
Uses a larger Mistral model to score how well answers fit queries.
"""

from typing import Dict, Any, Optional
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json


class ResponseEvaluator:
    """Evaluates bot responses using a larger Mistral model."""
    
    def __init__(
        self,
        mistral_api_key: Optional[str] = None,
        model: str = "mistral-small",  # Use smaller model to avoid rate limits (can use mistral-large-latest for better quality)
        temperature: float = 0.1  # Low temperature for consistent scoring
    ):
        """
        Initialize the response evaluator.
        
        Args:
            mistral_api_key: Mistral API key (if None, reads from MISTRAL_API_KEY env var)
            model: Mistral model to use for evaluation (default: mistral-large-latest)
            temperature: Temperature for evaluation model
        """
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        if not self.mistral_api_key:
            raise ValueError(
                "Mistral API key not provided. "
                "Set MISTRAL_API_KEY environment variable or pass mistral_api_key parameter."
            )
        
        self.evaluator_llm = ChatMistralAI(
            model=model,
            mistral_api_key=self.mistral_api_key,
            temperature=temperature,
            max_retries=3,  # Add retry logic for rate limits
            timeout=60.0  # Increase timeout
        )
        
        # Prompt for evaluating response quality
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator for a construction materials chatbot. 
Your task is to evaluate how well a bot's response answers a user's query.

Evaluate the response on a scale from 0.0 to 1.0 based on:
1. Relevance: Does the response directly address the query? (0.0-0.3)
2. Completeness: Is the response complete and informative? (0.0-0.3)
3. Accuracy: Is the information correct and factual? (0.0-0.2)
4. Clarity: Is the response clear and well-structured? (0.0-0.2)

Return ONLY a JSON object with the following structure:
{{
    "score": <float between 0.0 and 1.0>,
    "relevance": <float 0.0-0.3>,
    "completeness": <float 0.0-0.3>,
    "accuracy": <float 0.0-0.2>,
    "clarity": <float 0.0-0.2>,
    "reasoning": "<brief explanation of the score>"
}}

Be strict but fair. A perfect response gets 1.0, a completely irrelevant response gets 0.0."""),
            ("human", """User Query: {query}

Bot Response: {response}

Query Type: {query_type}
Needs Clarification: {needs_clarification}

Evaluate this response and return the JSON object:""")
        ])
    
    def evaluate(
        self,
        query: str,
        response: str,
        query_type: str = "unknown",
        needs_clarification: bool = False,
        extracted_specs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a bot response for a given query.
        
        Args:
            query: Original user query
            response: Bot's response message
            query_type: Type of query (informational, order_specification)
            needs_clarification: Whether the bot needs clarification
            extracted_specs: Extracted order specifications (if any)
            
        Returns:
            Dictionary with evaluation scores and metadata
        """
        try:
            # Format the prompt
            formatted_prompt = self.evaluation_prompt.format_messages(
                query=query,
                response=response,
                query_type=query_type,
                needs_clarification=str(needs_clarification)
            )
            
            # Get evaluation from LLM
            result = self.evaluator_llm.invoke(formatted_prompt)
            evaluation_text = result.content.strip()
            
            # Try to parse JSON from response
            # Sometimes LLM adds markdown formatting
            if "```json" in evaluation_text:
                json_start = evaluation_text.find("```json") + 7
                json_end = evaluation_text.find("```", json_start)
                evaluation_text = evaluation_text[json_start:json_end].strip()
            elif "```" in evaluation_text:
                json_start = evaluation_text.find("```") + 3
                json_end = evaluation_text.find("```", json_start)
                evaluation_text = evaluation_text[json_start:json_end].strip()
            
            # Parse JSON
            evaluation = json.loads(evaluation_text)
            
            # Validate score is in range
            if not (0.0 <= evaluation.get("score", 0.0) <= 1.0):
                evaluation["score"] = max(0.0, min(1.0, evaluation["score"]))
            
            # Add metadata
            evaluation["query"] = query
            evaluation["response"] = response
            evaluation["query_type"] = query_type
            evaluation["needs_clarification"] = needs_clarification
            
            return evaluation
            
        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            return {
                "score": 0.5,  # Neutral score if evaluation fails
                "relevance": 0.15,
                "completeness": 0.15,
                "accuracy": 0.1,
                "clarity": 0.1,
                "reasoning": f"Failed to parse evaluation: {str(e)}",
                "error": str(e),
                "query": query,
                "response": response,
                "query_type": query_type,
                "needs_clarification": needs_clarification
            }
        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit error
            is_rate_limit = "429" in error_str or "capacity exceeded" in error_str.lower() or "rate limit" in error_str.lower()
            
            if is_rate_limit:
                # Return a neutral score for rate limit errors instead of 0
                return {
                    "score": 0.5,  # Neutral score for rate limit errors
                    "relevance": 0.15,
                    "completeness": 0.15,
                    "accuracy": 0.1,
                    "clarity": 0.1,
                    "reasoning": f"Rate limit error - could not evaluate. Consider using a smaller model (mistral-small or mistral-tiny) or wait before retrying.",
                    "error": error_str,
                    "rate_limit_error": True,
                    "query": query,
                    "response": response,
                    "query_type": query_type,
                    "needs_clarification": needs_clarification
                }
            else:
                # Other errors get score 0
                return {
                    "score": 0.0,
                    "relevance": 0.0,
                    "completeness": 0.0,
                    "accuracy": 0.0,
                    "clarity": 0.0,
                    "reasoning": f"Evaluation error: {error_str}",
                    "error": error_str,
                    "query": query,
                    "response": response,
                    "query_type": query_type,
                    "needs_clarification": needs_clarification
                }

