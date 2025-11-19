"""
Simulated buyer agent that can interact with the bot.
Uses a neural network (LLM) to simulate realistic buyer behavior.
"""

from typing import List, Dict, Any, Optional
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
import os


class SimulatedBuyer:
    """Simulates a buyer interacting with the construction materials bot."""
    
    def __init__(
        self,
        mistral_api_key: Optional[str] = None,
        model: str = "mistral-small",  # Use smaller model to avoid rate limits
        temperature: float = 0.8,
        buyer_persona: Optional[str] = None
    ):
        """
        Initialize the simulated buyer.
        
        Args:
            mistral_api_key: Mistral API key
            model: Mistral model to use
            temperature: Temperature for generation
            buyer_persona: Optional persona description (e.g., "experienced builder", "first-time buyer")
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
        
        self.buyer_persona = buyer_persona or "a typical customer looking for construction materials"
        self.conversation_history = ChatMessageHistory()
        
        self.buyer_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are {self.buyer_persona} interacting with a construction materials chatbot.

Your goal is to have a natural conversation. You might:
- Ask questions about materials
- Place orders for construction materials
- Provide additional information when asked
- Respond naturally to the bot's questions

Keep your responses:
- Natural and conversational in Russian
- Relevant to construction materials
- Realistic for a customer interaction
- Brief (1-2 sentences typically)

You are having a conversation with the bot. Respond based on the conversation history and the bot's last message."""),
            ("human", """Conversation so far:
{conversation_history}

Bot's last message: {bot_message}

What would you say next? Respond naturally as a customer:""")
        ])
    
    def start_conversation(self, initial_query: str) -> str:
        """
        Start a conversation with an initial query.
        
        Args:
            initial_query: The buyer's first query
            
        Returns:
            The initial query
        """
        self.conversation_history.clear()
        self.conversation_history.add_user_message(initial_query)
        return initial_query
    
    def respond(self, bot_message: str) -> str:
        """
        Generate a response to the bot's message.
        
        Args:
            bot_message: The bot's response message
            
        Returns:
            The buyer's response
        """
        # Add bot's message to history
        self.conversation_history.add_ai_message(bot_message)
        
        # Format conversation history
        conversation_text = "\n".join([
            f"{'User' if msg.type == 'human' else 'Bot'}: {msg.content}"
            for msg in self.conversation_history.messages
        ])
        
        # Generate buyer response
        try:
            formatted_prompt = self.buyer_prompt.format_messages(
                conversation_history=conversation_text,
                bot_message=bot_message
            )
            
            result = self.llm.invoke(formatted_prompt)
            buyer_response = result.content.strip()
            
            # Add buyer's response to history
            self.conversation_history.add_user_message(buyer_response)
            
            return buyer_response
        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit error
            is_rate_limit = "429" in error_str or "capacity exceeded" in error_str.lower() or "rate limit" in error_str.lower()
            
            if is_rate_limit:
                # Return a simple fallback response for rate limit errors
                fallback_response = "Хорошо, понял. Спасибо за информацию."
                print(f"\n⚠️  Rate limit error in simulated buyer. Using fallback response.")
                print(f"   Consider using a smaller model (mistral-small or mistral-tiny) or wait before retrying.")
                self.conversation_history.add_user_message(fallback_response)
                return fallback_response
            else:
                # Re-raise other errors
                raise
    
    def reset(self):
        """Reset the conversation history."""
        self.conversation_history.clear()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        return [
            {
                "role": "user" if msg.type == "human" else "bot",
                "content": msg.content
            }
            for msg in self.conversation_history.messages
        ]

