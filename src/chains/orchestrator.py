"""
Orchestrator Chain that coordinates the entire workflow.
Manages the flow: Classification → RAG/Extraction → Clarification → Products API
"""

from typing import Optional
from langchain_community.chat_message_histories import ChatMessageHistory
from src.schemas.models import UserQuery, BotResponse, OrderSpecs
from src.chains.classification import ClassificationChain
from src.chains.extraction import ExtractionChain
from src.chains.clarification import ClarificationChain
from src.rag.api_wrapper import query_rag  # Uses RAG from this project (src/rag/)
from src.database.products_api import get_products  # Mock API - replace with real implementation


class OrchestratorChain:
    """Main orchestrator chain that coordinates all components."""
    
    def __init__(
        self,
        classification_chain: ClassificationChain,
        extraction_chain: ExtractionChain,
        clarification_chain: ClarificationChain,
        llm
    ):
        """
        Initialize the orchestrator chain.
        
        Args:
            classification_chain: Classification chain instance
            extraction_chain: Extraction chain instance
            clarification_chain: Clarification chain instance
            llm: LLM instance for generating final responses
        """
        self.classification_chain = classification_chain
        self.extraction_chain = extraction_chain
        self.clarification_chain = clarification_chain
        self.llm = llm
        
        # Memory for each session (session_id -> memory)
        self.session_memories: dict[str, ChatMessageHistory] = {}
        
        # Stored specifications for each session (session_id -> OrderSpecs)
        self.session_specs: dict[str, OrderSpecs] = {}
    
    def _get_memory(self, session_id: str) -> ChatMessageHistory:
        """Get or create memory for a session."""
        if session_id not in self.session_memories:
            self.session_memories[session_id] = ChatMessageHistory(
                return_messages=True,
                memory_key="chat_history"
            )
        return self.session_memories[session_id]
    
    def _get_specs(self, session_id: str) -> Optional[OrderSpecs]:
        """Get stored specifications for a session."""
        return self.session_specs.get(session_id)
    
    def _store_specs(self, session_id: str, specs: OrderSpecs) -> None:
        """Store specifications for a session."""
        self.session_specs[session_id] = specs
    
    def process(self, user_query: UserQuery) -> BotResponse:
        """
        Process a user query through the complete workflow.
        
        Args:
            user_query: User query with message and session_id
            
        Returns:
            BotResponse with message and metadata
        """
        query = user_query.message
        session_id = user_query.session_id
        
        # Step 1: Classify the query
        query_type = self.classification_chain.classify(query)
        
        # Step 2: Route based on classification
        if query_type == "informational":
            # Informational query → RAG (uses project's RAG module: src/rag/)
            try:
                rag_response = query_rag(query)  # Calls src.rag.api_wrapper.query_rag()
                return BotResponse(
                    message=rag_response,
                    needs_clarification=False,
                    extracted_specs=None,
                    query_type="informational"
                )
            except Exception as e:
                return BotResponse(
                    message=f"Извините, произошла ошибка при поиске информации: {str(e)}",
                    needs_clarification=False,
                    extracted_specs=None,
                    query_type="informational"
                )
        
        else:  # order_specification
            # Order specification → Extract → Check completeness → Clarify or Get products
            
            # Get existing specs for this session
            existing_specs = self._get_specs(session_id)
            
            # Step 3: Extract specifications
            extracted_specs = self.extraction_chain.extract(query, existing_specs)
            
            # Store updated specs
            self._store_specs(session_id, extracted_specs)
            
            # Step 4: Check completeness
            if not extracted_specs.is_complete():
                # Step 5: Generate clarifying question
                missing_fields = extracted_specs.get_missing_fields()
                clarifying_question = self.clarification_chain.generate_question(
                    extracted_specs, missing_fields
                )
                
                return BotResponse(
                    message=clarifying_question,
                    needs_clarification=True,
                    extracted_specs=extracted_specs,
                    query_type="order_specification"
                )
            else:
                # Step 6: Get products from database
                products = get_products(extracted_specs)
                
                # Step 7: Format response with products
                response_message = self._format_products_response(products, extracted_specs)
                
                # Clear specs after successful order (optional - can keep for follow-up)
                # self.session_specs.pop(session_id, None)
                
                return BotResponse(
                    message=response_message,
                    needs_clarification=False,
                    extracted_specs=extracted_specs,
                    query_type="order_specification"
                )
    
    def _format_products_response(self, products: list[dict], specs: OrderSpecs) -> str:
        """
        Format products list into a readable response message.
        
        Args:
            products: List of product dictionaries
            specs: Order specifications
            
        Returns:
            Formatted response string
        """
        if not products:
            return "К сожалению, товары по вашим параметрам не найдены. Попробуйте изменить параметры заказа."
        
        response_parts = [
            f"Вот предложения по вашему запросу:\n"
        ]
        
        for i, product in enumerate(products, 1):
            product_info = f"{i}. {product['name']}"
            
            if product.get('mark'):
                product_info += f" (марка {product['mark']})"
            if product.get('fraction'):
                product_info += f" (фракция {product['fraction']})"
            
            product_info += f"\n   Цена: {product['price_per_unit']} руб./{product['unit']}"
            
            if product.get('description'):
                product_info += f"\n   {product['description']}"
            
            product_info += "\n"
            response_parts.append(product_info)
        
        response_parts.append("\nДля оформления заказа укажите количество и адрес доставки.")
        
        return "\n".join(response_parts)

