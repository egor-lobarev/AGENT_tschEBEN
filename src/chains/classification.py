"""
Classification Chain for determining query type.
Classifies user queries as either informational or order specification.
"""

import re
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
        
        # Enhanced prompt for classification with conversation context
        self.prompt = ChatPromptTemplate.from_messages([
                    ("system", """Ты — классификатор сообщений для магазина стройматериалов. Твоя задача — определить намерение (интент) пользователя в ТЕКУЩЕМ сообщении.

        Верни только один из трех классов: "informational", "order_specification" или "other".

        ### ОПИСАНИЕ КЛАССОВ:

        1. "informational" (Поиск знаний/сравнение):
        - Пользователь хочет УЗНАТЬ что-то о стройматериалах, а не купить прямо сейчас.
        - Сравнение товаров ("В чем разница между М300 и М400?", "Что лучше?").
        - Вопросы о свойствах ("Какой бетон крепче?", "Характеристики песка").
        - Вопросы о ценах ("Сколько стоит?", "Почем куб?").
        - Наличие товара ("Есть ли у вас щебень?").
        - Любые вопросы с "как", "почему", "зачем", "что такое" про стройматериалы.

        2. "order_specification" (Намерение купить/уточнение параметров сделки):
        - Пользователь выражает желание КУПИТЬ или ЗАКАЗАТЬ стройматериалы.
        - Указание КОЛИЧЕСТВА ("5 кубов", "10 тонн", "машина песка").
        - Слова действия: "нужен", "надо", "хочу заказать", "везите", "доставка".
        - Указание АДРЕСА ("везите в Мытищи", "доставка на Ленина 5").
        - Прямое утверждение потребности ("Мне нужен бетон М300").

        3. "other" (Не относится к стройматериалам):
        - Вопросы не связанные со стройматериалами или заказами.
        - Общие вопросы о жизни, погоде, политике и т.д.
        - Приветствия, прощания, благодарности (если не в контексте заказа).
        - Вопросы о других товарах/услугах, не связанных со стройматериалами.
        - Любые сообщения, которые НЕ относятся к информационным вопросам о стройматериалах И НЕ являются заказом стройматериалов.

        ### ВАЖНЫЕ ПРАВИЛА (Priority Rules):

        ПРАВИЛО №1 (Марки товаров):
        Упоминание марки (М200, М300, М400) или названия товара (песок, щебень) САМО ПО СЕБЕ — это НЕ заказ.
        - "Чем отличается М300 от М400?" -> informational (Сравнение)
        - "Бетон М300 хороший?" -> informational (Вопрос о качестве)
        - "Мне нужен М300" -> order_specification (Есть слово "нужен")
        - "М300, 5 кубов" -> order_specification (Есть количество)

        ПРАВИЛО №2 (Количество):
        Если есть конкретное числовое количество (кубы, тонны, мешки) -> это ВСЕГДА order_specification.

        ПРАВИЛО №3 (Не относится к стройматериалам):
        Если сообщение не связано со стройматериалами (бетон, песок, щебень, гравий, цемент и т.д.) -> это "other".

        ### ПРИМЕРЫ (Few-shot):
        User: "Какая марка лучше для фундамента?"
        System: informational

        User: "Нужно 10 кубов М300"
        System: order_specification

        User: "Сколько стоит доставка в Химки?"
        System: informational

        User: "Везем в Химки, улица Гоголя 5"
        System: order_specification

        User: "Чем отличаются марки M300, M450, M400"
        System: informational

        User: "Хочу купить песок"
        System: order_specification

        User: "Какая сегодня погода?"
        System: other

        User: "Привет, как дела?"
        System: other

        User: "Где ближайший ресторан?"
        System: other

        User: "Спасибо за помощь"
        System: other

        КОНТЕКСТ ДИАЛОГА (используй только для понимания местоимений, не меняй логику классификации):
        {conversation_context}

        Твой ответ (только одно слово):"""),
                    ("human", "ТЕКУЩИЙ запрос пользователя: {query}")
                ])
        # Create chain
        self.chain = (
            {
                "query": RunnablePassthrough(),
                "conversation_context": lambda x: x.get("conversation_context", "")
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def classify(
        self, 
        query: str, 
        conversation_context: str = ""
    ) -> Literal["informational", "order_specification", "other"]:
        """
        Classify a user query with conversation context.
        
        Args:
            query: User query text
            conversation_context: Optional conversation history for context-aware classification
            
        Returns:
            Query type: "informational", "order_specification", or "other"
        """
        result = self.chain.invoke({
            "query": query,
            "conversation_context": conversation_context if conversation_context else "Контекст отсутствует (первое сообщение в диалоге)."
        })
        
        # Clean and normalize the result
        result = result.strip().lower()
        print("ответ классификационной сетки", result)
        
        # Extract the classification
        if "informational" in result or "информационный" in result.lower():
            return "informational"
        elif "order_specification" in result or "спецификация" in result.lower() or "заказ" in result.lower():
            return "order_specification"
        elif "other" in result or "другое" in result.lower() or "иное" in result.lower():
            return "other"
        else:
            # Enhanced keyword detection with priority for informational questions
            query_lower = query.lower()
            
            # STRONG indicators of informational questions (check these FIRST)
            # Questions about differences, comparisons, explanations
            informational_question_patterns = [
                r'чем\s+отличаются',
                r'какая\s+разница',
                r'что\s+такое',
                r'как\s+выбрать',
                r'как\s+работает',
                r'объясни',
                r'расскажи',
                r'что\s+значит',
                r'в\s+чём\s+разница',
                r'сравни',
            ]
            
            # Check for informational question patterns first
            for pattern in informational_question_patterns:
                if re.search(pattern, query_lower):
                    return "informational"
            
            # Informational keywords (questions)
            informational_keywords = [
                "какие", "что такое", "как выбрать", "сколько стоит", 
                "где", "когда", "какой", "какая", "какое", "чем", "от чего",
                "у вас есть", "есть ли", "можно ли", "расскажи", "объясни",
                "отличаются", "разница", "разница между", "сравни"
            ]
            
            # Order keywords (intent to buy)
            ordering_keywords = [
                "нужен", "нужно", "хочу", "заказать", "купить", 
                "мне нужно", "требуется", "дайте", "пришлите", "доставьте",
                "закажу", "куплю", "возьму"
            ]
            
            # Check for informational keywords
            has_informational_keyword = any(keyword in query_lower for keyword in informational_keywords)
            has_ordering_keyword = any(keyword in query_lower for keyword in ordering_keywords)
            
            # STRONG indicators of order specification (only if NOT an informational question)
            # Quantity indicators (most reliable)
            has_quantity = bool(re.search(r'\d+\s*(куб|тонн|м³|литр|кг)', query_lower))
            
            # Address indicators (very reliable)
            has_address = bool(re.search(r'(на|по адресу|адрес|улиц|город|посёлок|деревня)\s+[а-яё]+', query_lower))
            
            # Fraction indicators (reliable for orders)
            has_fraction = bool(re.search(r'\d+-\d+', query_lower))  # e.g., 20-40
            
            # If it's clearly an informational question, return informational
            if has_informational_keyword and not has_ordering_keyword:
                return "informational"
            
            # If has quantity OR address, it's almost certainly an order
            if has_quantity or has_address:
                return "order_specification"
            
            # If has ordering keyword + (product mention OR fraction), it's an order
            has_product_mention = any(word in query_lower for word in ["бетон", "песок", "щебень", "гравий"])
            if has_ordering_keyword and (has_product_mention or has_fraction):
                return "order_specification"
            
            # If has ordering keywords alone (without question words), it's an order
            if has_ordering_keyword and not has_informational_keyword:
                return "order_specification"
            
            # If has informational keywords, it's informational
            if has_informational_keyword:
                return "informational"
            
            # Check if query is related to construction materials
            construction_materials_keywords = [
                "бетон", "песок", "щебень", "гравий", "цемент", "кирпич",
                "марка", "м300", "м400", "м500", "фракция", "куб", "тонн",
                "стройматериал", "строительный", "материал"
            ]
            
            has_construction_mention = any(keyword in query_lower for keyword in construction_materials_keywords)
            
            # If no construction materials mentioned and no clear intent, it's "other"
            if not has_construction_mention and not has_ordering_keyword and not has_informational_keyword:
                return "other"
            
            # Default to informational for safety (better to answer than to ask for order details)
            return "informational"

