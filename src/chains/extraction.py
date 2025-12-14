"""
Extraction Chain for extracting order specifications from user queries.
Uses structured output with Pydantic models.
"""

import json
import re
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
3. characteristics.mark - марка товара (для бетона: М300, М350, М400 и т.д.) - ОДНА строка, НЕ массив
4. characteristics.fraction - фракция (для щебня, гравия, песка: "20-40", "5-20", "0-5" и т.д.)
5. delivery.address - адрес доставки (если указан)
6. delivery.date - дата доставки (если указана)

ВАЖНО:
- Верни ТОЛЬКО валидный JSON без комментариев и объяснений
- Не добавляй комментарии в JSON (// или /* */)
- characteristics.mark должен быть строкой, НЕ массивом
- Если информация не указана, используй null (не None)
- Если запрос информационный (например, "какие марки есть?"), верни все поля как null

Верни результат в формате JSON согласно схеме."""),
            ("human", """Запрос пользователя: {query}
{existing_specs_context}

{format_instructions}

Верни ТОЛЬКО валидный JSON, без дополнительных комментариев или объяснений:""")
        ])
        
        # Note: Chain will be created dynamically in extract() to include existing_specs
    
    def _clean_json_response(self, text: str) -> str:
        """
        Extract and clean JSON from LLM response.
        
        Args:
            text: Raw LLM response
            
        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks
        if "```json" in text:
            json_start = text.find("```json") + 7
            json_end = text.find("```", json_start)
            text = text[json_start:json_end].strip()
        elif "```" in text:
            json_start = text.find("```") + 3
            json_end = text.find("```", json_start)
            text = text[json_start:json_end].strip()
        
        # Remove JSON comments (// and /* */)
        # Remove single-line comments
        text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
        # Remove multi-line comments
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        
        # Find JSON object boundaries
        # Look for first { and last }
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace:last_brace + 1]
        
        # Fix common issues: replace None with null, fix array to string for mark
        text = text.replace('None', 'null')
        text = text.replace('True', 'true')
        text = text.replace('False', 'false')
        
        return text.strip()
    
    def extract(
        self, 
        query: str, 
        existing_specs: Optional[OrderSpecs] = None,
        conversation_context: str = ""
    ) -> OrderSpecs:
        """
        Extract order specifications from user query with conversation context.
        
        Args:
            query: User query text
            existing_specs: Existing specifications from previous turns (to merge)
            conversation_context: Optional conversation history for context-aware extraction
            
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
                    existing_specs_context = "\n\nУже известные параметры из предыдущих сообщений:\n" + "\n".join(existing_parts) + "\n\nИзвлеки только НОВУЮ информацию из текущего запроса. Если параметр уже известен и не упоминается в текущем запросе, не включай его в результат (оставь null)."
            
            # Create chain with context
            chain = (
                {
                    "query": RunnablePassthrough(),
                    "existing_specs_context": lambda _: existing_specs_context,
                    "conversation_context": lambda _: conversation_context if conversation_context else "Контекст отсутствует (первое сообщение в диалоге).",
                    "format_instructions": lambda _: self.output_parser.get_format_instructions()
                }
                | self.base_prompt
                | self.llm
            )
            
            # Get raw response
            raw_response = chain.invoke(query)
            response_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
            
            # Clean JSON response
            cleaned_json = self._clean_json_response(response_text)
            
            # Parse JSON manually first to fix any issues
            try:
                parsed_json = json.loads(cleaned_json)
                
                # Fix mark if it's an array - take first element or join
                if isinstance(parsed_json.get('characteristics', {}), dict):
                    mark = parsed_json['characteristics'].get('mark')
                    if isinstance(mark, list):
                        # If it's a list, take the first one or join them
                        parsed_json['characteristics']['mark'] = mark[0] if mark else None
                
                # Convert back to JSON string for Pydantic parser
                cleaned_json = json.dumps(parsed_json, ensure_ascii=False)
            except json.JSONDecodeError:
                # If manual parsing fails, try to fix common issues
                pass
            
            # Parse with Pydantic
            extracted = self.output_parser.parse(cleaned_json)
            
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

