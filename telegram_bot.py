"""
Telegram bot using aiogram that interacts with ConstructionMaterialsBot agent.
Использует Bot Token (полученный от @BotFather).
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from bot import ConstructionMaterialsBot

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot wrapper using aiogram."""
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        mistral_api_key: Optional[str] = None,
        use_in_memory: bool = False,
        data_path: str = "data/raw/raw_materials.jsonl",
        qdrant_host: Optional[str] = None,
        qdrant_port: int = 6333
    ):
        """
        Initialize Telegram bot with aiogram.
        
        Args:
            bot_token: Telegram Bot Token (from @BotFather)
            mistral_api_key: Mistral API key
            use_in_memory: Use in-memory Qdrant storage
            data_path: Path to JSONL data file
            qdrant_host: If provided, connect to Qdrant server instead of local storage.
                        Use "localhost" to connect to Docker Qdrant server.
            qdrant_port: Qdrant server port (default: 6333)
        """
        # Получаем Bot Token из переменных окружения
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        
        if not self.bot_token:
            raise ValueError(
                "Telegram Bot Token not provided. "
                "Set TELEGRAM_BOT_TOKEN environment variable "
                "or pass it as parameter. Get it from @BotFather on Telegram."
            )
        
        # Инициализируем aiogram бота и диспетчер
        self.bot = Bot(
            token=self.bot_token,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML)
        )
        self.dp = Dispatcher()
        
        # Инициализируем агента
        logger.info("Initializing ConstructionMaterialsBot agent...")
        try:
            self.agent = ConstructionMaterialsBot(
                mistral_api_key=mistral_api_key,
                use_in_memory=use_in_memory,
                data_path=data_path,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port
            )
            logger.info("Agent initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
        
        # Словарь для хранения session_id для каждого пользователя
        self.user_sessions = {}
        
        # Регистрируем обработчики
        self._register_handlers()
    
    def get_session_id(self, user_id: int) -> str:
        """Получить session_id для пользователя."""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = f"user_{user_id}"
        return self.user_sessions[user_id]
    
    def _register_handlers(self):
        """Регистрирует обработчики команд и сообщений."""
        # Обработчик команды /start
        self.dp.message.register(self.start_command, Command("start"))
        
        # Обработчик команды /help
        self.dp.message.register(self.help_command, Command("help"))
        
        # Обработчик всех текстовых сообщений (кроме команд)
        self.dp.message.register(self.handle_message, F.text)
    
    async def start_command(self, message: Message):
        """Обработчик команды /start."""
        welcome_message = (
            "👋 Привет! Я бот-консультант по строительным материалам.\n\n"
            "Я могу помочь вам:\n"
            "• Найти информацию о строительных материалах\n"
            "• Помочь с оформлением заказа\n"
            "• Ответить на ваши вопросы\n\n"
            "Просто напишите мне ваш вопрос!"
        )
        await message.answer(welcome_message)
    
    async def help_command(self, message: Message):
        """Обработчик команды /help."""
        help_message = (
            "📖 Справка по использованию бота:\n\n"
            "Просто отправьте мне текстовое сообщение с вашим вопросом или запросом.\n\n"
            "Примеры запросов:\n"
            "• 'Нужен бетон для фундамента'\n"
            "• 'Хочу заказать бетон М300'\n"
            "• 'Какие характеристики у бетона М400?'\n\n"
            "Я постараюсь помочь вам с выбором и оформлением заказа!"
        )
        await message.answer(help_message)
    
    async def handle_message(self, message: Message):
        """Обработчик входящих текстовых сообщений."""
        user_id = message.from_user.id
        message_text = message.text or message.caption
        
        if not message_text:
            await message.answer("Пожалуйста, отправьте текстовое сообщение.")
            return
        
        logger.info(f"Received message from user {user_id}: {message_text[:100]}")
        
        # Показываем индикатор печати
        await self.bot.send_chat_action(
            chat_id=message.chat.id,
            action="typing"
        )
        
        try:
            # Получаем session_id для пользователя
            session_id = self.get_session_id(user_id)
            
            # Обрабатываем запрос через агента (в executor, чтобы не блокировать event loop)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.agent.process_query(
                    message=message_text,
                    session_id=session_id
                )
            )
            
            # Отправляем ответ
            await message.answer(response.message)
            
            logger.info(f"Sent response to user {user_id}")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await message.answer(
                "Извините, произошла ошибка при обработке вашего запроса. "
                "Попробуйте позже."
            )
    
    async def start(self):
        """Запустить бота."""
        logger.info("Starting Telegram bot...")
        
        # Запускаем polling
        await self.dp.start_polling(self.bot)
    
    async def stop(self):
        """Остановить бота."""
        logger.info("Stopping Telegram bot...")
        await self.bot.session.close()


async def wait_for_qdrant(host: str, port: int, max_retries: int = 30, delay: float = 2.0) -> bool:
    """
    Ожидает, пока Qdrant сервер станет доступным.
    
    Args:
        host: Хост Qdrant
        port: Порт Qdrant
        max_retries: Максимальное количество попыток
        delay: Задержка между попытками в секундах
        
    Returns:
        True если сервер доступен, False иначе
    """
    import socket
    for attempt in range(max_retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                logger.info(f"Qdrant server at {host}:{port} is ready!")
                return True
        except Exception as e:
            logger.debug(f"Attempt {attempt + 1}/{max_retries}: Qdrant not ready yet: {e}")
        
        if attempt < max_retries - 1:
            await asyncio.sleep(delay)
    
    logger.warning(f"Qdrant server at {host}:{port} is not available after {max_retries} attempts")
    return False


async def main():
    """Главная функция для запуска бота."""
    bot_instance = None
    try:
        # Получаем хост Qdrant из переменной окружения
        # В Docker контейнере используем имя сервиса "qdrant", локально - "localhost" или None
        qdrant_host = os.getenv("QDRANT_HOST")  # Если установлено, используем его
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        # Если QDRANT_HOST не установлен, пытаемся определить автоматически
        if qdrant_host is None:
            try:
                import socket
                # Сначала пробуем подключиться к имени сервиса Docker (для docker-compose)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(('qdrant', qdrant_port))
                sock.close()
                if result == 0:
                    qdrant_host = "qdrant"
                    logger.info("Qdrant Docker service detected, using 'qdrant' host")
                else:
                    # Если не получилось, пробуем localhost (для локального запуска)
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    result = sock.connect_ex(('localhost', qdrant_port))
                    sock.close()
                    if result == 0:
                        qdrant_host = "localhost"
                        logger.info("Qdrant local server detected, using 'localhost' host")
                    else:
                        qdrant_host = None
                        logger.info("Qdrant server not detected, using local storage")
            except Exception as e:
                logger.warning(f"Could not detect Qdrant server: {e}, using local storage")
                qdrant_host = None
        else:
            logger.info(f"Using Qdrant host from environment: {qdrant_host}")
        
        # Если используем Qdrant сервер, ждем пока он станет доступным
        if qdrant_host:
            logger.info(f"Waiting for Qdrant server at {qdrant_host}:{qdrant_port}...")
            await wait_for_qdrant(qdrant_host, qdrant_port)
        
        # Инициализируем бота
        bot_instance = TelegramBot(
            use_in_memory=False,  # Используем постоянное хранилище
            data_path="data/raw/raw_materials.jsonl",
            qdrant_host=qdrant_host,  # Используем сервер, если доступен
            qdrant_port=qdrant_port
        )
        
        logger.info("Bot is ready! Listening for messages...")
        
        # Запускаем бота
        await bot_instance.start()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        if bot_instance:
            try:
                await bot_instance.stop()
            except Exception as e:
                logger.error(f"Error stopping bot: {e}")


if __name__ == "__main__":
    asyncio.run(main())
