"""
Clarification Chain for generating clarifying questions.
Generates questions to complete order specifications.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.schemas.models import OrderSpecs


class ClarificationChain:
    """Chain for generating clarifying questions."""
    
    def __init__(self, llm):
        """
        Initialize the clarification chain.
        
        Args:
            llm: LangChain LLM instance (e.g., ChatMistralAI)
        """
        self.llm = llm
        
        # Prompt for clarification
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты помощник интернет-магазина стройматериалов. 
Твоя задача - сгенерировать уточняющий вопрос пользователю, чтобы получить недостающую информацию для заказа.

Проанализируй, какие параметры уже есть, а каких не хватает.
Сгенерируй ОДИН естественный, дружелюбный вопрос на русском языке, который поможет получить недостающую информацию.

Будь конкретным и полезным. Например:
- Если нет типа товара: "Какой материал вам нужен? (бетон, песок, щебень, гравий)"
- Если нет количества: "Какое количество вам нужно? (укажите объем или вес)"
- Если нет марки (для бетона): "Какая марка бетона вам нужна? (М300, М350, М400)"
- Если нет фракции (для щебня/гравия): "Какая фракция вам нужна? (например, 20-40, 5-20)"

Верни только вопрос, без дополнительных комментариев."""),
            ("human", """Текущие параметры заказа:
{specs}

Недостающие поля: {missing_fields}

Сгенерируй уточняющий вопрос:""")
        ])
        
        # Create chain
        self.chain = (
            {
                "specs": lambda x: x["specs"],
                "missing_fields": lambda x: x["missing_fields"]
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def generate_question(self, specs: OrderSpecs, missing_fields: list[str]) -> str:
        """
        Generate a clarifying question based on missing fields.
        
        Args:
            specs: Current order specifications
            missing_fields: List of missing field names
            
        Returns:
            Clarifying question string
        """
        # Format specs for display
        specs_str = f"""
Тип товара: {specs.product_type or 'не указан'}
Количество: {specs.quantity or 'не указано'}
Марка: {specs.characteristics.mark if specs.characteristics and specs.characteristics.mark else 'не указана'}
Фракция: {specs.characteristics.fraction if specs.characteristics and specs.characteristics.fraction else 'не указана'}
Адрес доставки: {specs.delivery.address if specs.delivery and specs.delivery.address else 'не указан'}
Дата доставки: {specs.delivery.date if specs.delivery and specs.delivery.date else 'не указана'}
"""
        
        missing_str = ", ".join(missing_fields) if missing_fields else "нет"
        
        result = self.chain.invoke({
            "specs": specs_str,
            "missing_fields": missing_str
        })
        
        return result.strip()

