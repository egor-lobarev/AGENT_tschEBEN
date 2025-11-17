"""
Extraction Chain for extracting order specifications from user queries.
Uses structured output with Pydantic models.
"""

from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.schemas.models import OrderSpecs, ProductCharacteristics, DeliveryInfo


class ExtractionChain:
    """Chain for extracting order specifications from user queries."""
    
    def __init__(self, llm):
        """
        Initialize the extraction chain.
        
        Args:
            llm: LangChain LLM instance (e.g., ChatMistralAI)
        """
        self.llm = llm
        
        # Pydantic output parser
        self.output_parser = PydanticOutputParser(pydantic_object=OrderSpecs)
        
        # Prompt for extraction
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты помощник интернет-магазина стройматериалов. 
Твоя задача - извлечь параметры заказа из запроса пользователя.

Извлекай следующую информацию:
1. product_type - тип товара (бетон, песок, гравий, щебень)
2. quantity - количество (объем или вес, например: "5 кубов", "10 тонн", "3 м³")
3. characteristics.mark - марка товара (для бетона: М300, М350, М400 и т.д.)
4. characteristics.fraction - фракция (для щебня, гравия, песка: "20-40", "5-20", "0-5" и т.д.)
5. delivery.address - адрес доставки (если указан)
6. delivery.date - дата доставки (если указана)

Если какая-то информация не указана, оставь поле как None.

Верни результат в формате JSON согласно схеме."""),
            ("human", """Запрос пользователя: {query}

{format_instructions}""")
        ])
        
        # Create chain
        self.chain = (
            {
                "query": RunnablePassthrough(),
                "format_instructions": lambda _: self.output_parser.get_format_instructions()
            }
            | self.prompt
            | self.llm
            | self.output_parser
        )
    
    def extract(self, query: str, existing_specs: Optional[OrderSpecs] = None) -> OrderSpecs:
        """
        Extract order specifications from user query.
        
        Args:
            query: User query text
            existing_specs: Existing specifications from previous turns (to merge)
            
        Returns:
            OrderSpecs object with extracted parameters
        """
        try:
            extracted = self.chain.invoke(query)
            
            # Merge with existing specs if provided
            if existing_specs:
                # Update only non-None fields from extracted specs
                if extracted.product_type:
                    existing_specs.product_type = extracted.product_type
                if extracted.quantity:
                    existing_specs.quantity = extracted.quantity
                
                # Merge characteristics
                if extracted.characteristics:
                    if not existing_specs.characteristics:
                        existing_specs.characteristics = ProductCharacteristics()
                    
                    if extracted.characteristics.mark:
                        existing_specs.characteristics.mark = extracted.characteristics.mark
                    if extracted.characteristics.fraction:
                        existing_specs.characteristics.fraction = extracted.characteristics.fraction
                    if extracted.characteristics.product_type:
                        existing_specs.characteristics.product_type = extracted.characteristics.product_type
                
                # Merge delivery
                if extracted.delivery:
                    if not existing_specs.delivery:
                        existing_specs.delivery = DeliveryInfo()
                    
                    if extracted.delivery.address:
                        existing_specs.delivery.address = extracted.delivery.address
                    if extracted.delivery.date:
                        existing_specs.delivery.date = extracted.delivery.date
                
                return existing_specs
            
            return extracted
        except Exception as e:
            # If extraction fails, return empty specs
            print(f"Extraction error: {e}")
            return OrderSpecs()

