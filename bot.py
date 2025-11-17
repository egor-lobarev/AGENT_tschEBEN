"""
Main entry point for the Construction Materials Bot.
Initializes all components and provides interface for processing user queries.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from langchain_mistralai import ChatMistralAI
from src.schemas.models import UserQuery, BotResponse
from src.chains.classification import ClassificationChain
from src.chains.extraction import ExtractionChain
from src.chains.clarification import ClarificationChain
from src.chains.orchestrator import OrchestratorChain
from src.rag.api_wrapper import initialize_rag
from setup_rag import setup_rag_system

class ConstructionMaterialsBot:
    """Main bot class for processing user queries."""
    
    def __init__(
        self,
        mistral_api_key: Optional[str] = None,
        use_in_memory: bool = True,
        data_path: str = "data/raw/raw_materials.jsonl"
    ):
        """
        Initialize the Construction Materials Bot.
        
        Args:
            mistral_api_key: Mistral API key (if None, reads from MISTRAL_API_KEY env var)
            use_in_memory: Use in-memory Qdrant (no Docker)
            data_path: Path to JSONL data file for RAG
        """
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        if not self.mistral_api_key:
            raise ValueError(
                "Mistral API key not provided. "
                "Set MISTRAL_API_KEY environment variable or pass mistral_api_key parameter."
            )
        
        print("Initializing Mistral LLM...")
        self.llm = ChatMistralAI(
            model="mistral-large-latest",
            mistral_api_key=self.mistral_api_key,
            temperature=0.3
        )
        
        print("Setting up RAG system...")
        vector_store, retriever, _ = setup_rag_system(
            use_in_memory=use_in_memory,
            data_path=data_path
        )
        
        initialize_rag(retriever)
        
        print("Initializing chains...")
        self.classification_chain = ClassificationChain(self.llm)
        self.extraction_chain = ExtractionChain(self.llm)
        self.clarification_chain = ClarificationChain(self.llm)
        
        self.orchestrator = OrchestratorChain(
            classification_chain=self.classification_chain,
            extraction_chain=self.extraction_chain,
            clarification_chain=self.clarification_chain,
            llm=self.llm
        )
        
        print("Bot initialized successfully!")
    
    def process_query(self, message: str, session_id: str = "default") -> BotResponse:
        """
        Process a user query.
        
        Args:
            message: User message
            session_id: Session identifier for multi-turn dialogue
            
        Returns:
            BotResponse with message and metadata
        """
        user_query = UserQuery(message=message, session_id=session_id)
        return self.orchestrator.process(user_query)
    
    def chat(self, message: str, session_id: str = "default") -> str:
        """
        Simple chat interface that returns only the message.
        
        Args:
            message: User message
            session_id: Session identifier
            
        Returns:
            Bot response message
        """
        response = self.process_query(message, session_id)
        return response.message


def main():
    """Example usage of the bot."""
    print("=" * 80)
    print("Construction Materials Bot")
    print("=" * 80)
    
    try:
        # Initialize bot
        bot = ConstructionMaterialsBot()
        
  
        print("\n" + "=" * 80)
        print("Example Conversation")
        print("=" * 80)
        
        session_id = "user123"
        
        # Turn 1: Informational query
        print("\nUser: Нужен бетон для фундамента")
        response1 = bot.process_query("Нужен бетон для фундамента", session_id=session_id)
        print(f"Bot: {response1.message}")
        print(f"Query Type: {response1.query_type}")
        print(f"Needs Clarification: {response1.needs_clarification}")
        
        # Turn 2: Order specification
        print("\n" + "-" * 80)
        print("\nUser: Хочу заказать бетон М300")
        response2 = bot.process_query("Хочу заказать бетон М300", session_id=session_id)
        print(f"Bot: {response2.message}")
        print(f"Query Type: {response2.query_type}")
        print(f"Needs Clarification: {response2.needs_clarification}")
        if response2.extracted_specs:
            print(f"Extracted Specs: product_type={response2.extracted_specs.product_type}, "
                  f"mark={response2.extracted_specs.characteristics.mark if response2.extracted_specs.characteristics else None}")
        
        if response2.needs_clarification:
            print("\n" + "-" * 80)
            print("\nUser: 5 кубов")
            response3 = bot.process_query("5 кубов", session_id=session_id)
            print(f"Bot: {response3.message}")
            print(f"Query Type: {response3.query_type}")
            print(f"Needs Clarification: {response3.needs_clarification}")
        
        print("\n" + "=" * 80)
        print("Conversation Complete")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

