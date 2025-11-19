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
        
        # Prompt for extraction (without existing specs - will be added dynamically)
        self.base_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты помощник интернет-магазина стройматериалов. 
Твоя задача - извлечь параметры заказа из запроса пользователя.

Извлекай следующую информацию:
1. product_type - тип товара (бетон, песок, гравий, щебень)
2. quantity - количество (объем или вес, например: "5 кубов", "10 тонн", "3 м³")
3. characteristics.mark - марка товара (для бетона: М300, М350, М400 и т.д.)
4. characteristics.fraction - фракция (для щебня, гравия, песка: "20-40", "5-20", "0-5" и т.д.)
5. delivery.address - адрес доставки (если указан)
6. delivery.date - дата доставки (если указана)

Если какая-то информация не указана в текущем запросе, оставь поле как None.
Если есть уже известные параметры из предыдущих сообщений, они будут указаны отдельно.

Верни результат в формате JSON согласно схеме."""),
            ("human", """Запрос пользователя: {query}
{existing_specs_context}

{format_instructions}""")
        ])
        
        # Note: Chain will be created dynamically in extract() to include existing_specs
    
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
            # Build context about existing specs
            existing_specs_context = ""
            if existing_specs:
                existing_parts = []
                if existing_specs.product_type:
                    existing_parts.append(f"Тип товара: {existing_specs.product_type}")
                if existing_specs.quantity:
                    existing_parts.append(f"Количество: {existing_specs.quantity}")
                if existing_specs.characteristics:
                    if existing_specs.characteristics.mark:
                        existing_parts.append(f"Марка: {existing_specs.characteristics.mark}")
                    if existing_specs.characteristics.fraction:
                        existing_parts.append(f"Фракция: {existing_specs.characteristics.fraction}")
                if existing_specs.delivery:
                    if existing_specs.delivery.address:
                        existing_parts.append(f"Адрес доставки: {existing_specs.delivery.address}")
                    if existing_specs.delivery.date:
                        existing_parts.append(f"Дата доставки: {existing_specs.delivery.date}")
                
                if existing_parts:
                    existing_specs_context = "\n\nУже известные параметры из предыдущих сообщений:\n" + "\n".join(existing_parts) + "\n\nИзвлеки только НОВУЮ информацию из текущего запроса. Если параметр уже известен и не упоминается в текущем запросе, не включай его в результат (оставь None)."
            
            # Create chain with context
            chain = (
                {
                    "query": RunnablePassthrough(),
                    "existing_specs_context": lambda _: existing_specs_context,
                    "format_instructions": lambda _: self.output_parser.get_format_instructions()
                }
                | self.base_prompt
                | self.llm
                | self.output_parser
            )
            
            extracted = chain.invoke(query)
            
            # Merge with existing specs if provided
            if existing_specs:
                # Create merged specs: use extracted value if not None, otherwise use existing value
                merged_product_type = extracted.product_type if extracted.product_type is not None else existing_specs.product_type
                merged_quantity = extracted.quantity if extracted.quantity is not None else existing_specs.quantity
                
                # Merge characteristics
                merged_characteristics = None
                if extracted.characteristics is not None or existing_specs.characteristics is not None:
                    merged_characteristics = ProductCharacteristics(
                        mark=extracted.characteristics.mark if (extracted.characteristics and extracted.characteristics.mark is not None) 
                            else (existing_specs.characteristics.mark if existing_specs.characteristics else None),
                        fraction=extracted.characteristics.fraction if (extracted.characteristics and extracted.characteristics.fraction is not None)
                            else (existing_specs.characteristics.fraction if existing_specs.characteristics else None),
                        product_type=extracted.characteristics.product_type if (extracted.characteristics and extracted.characteristics.product_type is not None)
                            else (existing_specs.characteristics.product_type if existing_specs.characteristics else None)
                    )
                
                # Merge delivery
                merged_delivery = None
                if extracted.delivery is not None or existing_specs.delivery is not None:
                    merged_delivery = DeliveryInfo(
                        address=extracted.delivery.address if (extracted.delivery and extracted.delivery.address is not None)
                            else (existing_specs.delivery.address if existing_specs.delivery else None),
                        date=extracted.delivery.date if (extracted.delivery and extracted.delivery.date is not None)
                            else (existing_specs.delivery.date if existing_specs.delivery else None)
                    )
                
                # Create new merged OrderSpecs
                merged_specs = OrderSpecs(
                    product_type=merged_product_type,
                    quantity=merged_quantity,
                    characteristics=merged_characteristics,
                    delivery=merged_delivery
                )
                
                return merged_specs
            
            return extracted
        except Exception as e:
            # If extraction fails, return existing specs or empty specs
            print(f"Extraction error: {e}")
            if existing_specs:
                return existing_specs
            return OrderSpecs()

