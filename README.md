# RAG System for Construction Materials

Система для обработки запросов пользователей интернет-магазина стройматериалов с использованием RAG (Retrieval-Augmented Generation).

## Архитектура проекта

### Основные компоненты

**1. ConstructionMaterialsBot (`bot.py`)**
- Главный оркестратор системы
- Инициализирует все компоненты и обрабатывает запросы пользователей
- Управляет сессиями и контекстом диалога

**2. Цепочки обработки (`src/chains/`)**
- `classification.py` - классификация запросов (информационные / спецификация заказа)
- `extraction.py` - извлечение параметров заказа из запроса
- `clarification.py` - генерация уточняющих вопросов
- `orchestrator.py` - координация всех цепочек

**3. RAG модуль (`src/rag/`)**
- `vectore_store.py` - векторное хранилище на базе Qdrant
- `retriver.py` - семантический поиск похожих документов
- `generator.py` - генерация ответов на основе найденных документов
- `api_wrapper.py` - обертка для интеграции с LangChain

**4. База данных (`src/database/`)**
- `products_api.py` - API для поиска товаров по спецификации
- `db_models.py` - модели данных (SQLAlchemy)
- SQLite база данных (`data/products.db`)

**5. Telegram бот (`telegram_bot.py`)**
- Обертка над ConstructionMaterialsBot для работы в Telegram
- Использует aiogram для обработки сообщений

### Поток обработки запроса

```
Пользователь → Classification → [Информационный?] → RAG → Ответ
                          ↓
                    [Заказ?] → Extraction → [Полная спецификация?]
                                              ↓
                                    [Нет] → Clarification → Уточняющий вопрос
                                              ↓
                                    [Да] → Products API → Список товаров
```

### Технологический стек

- **LLM**: Mistral AI (mistral-tiny)
- **Векторное хранилище**: Qdrant (in-memory или persistent)
- **Embeddings**: SentenceTransformer (stsb-roberta-large)
- **База данных**: SQLite (товары)
- **Фреймворк**: LangChain
- **Telegram**: aiogram

## Запуск проекта

### Требования

- Python 3.10+
- Переменные окружения:
  - `MISTRAL_API_KEY` (обязательно)
  - `TELEGRAM_BOT_TOKEN` (для Telegram бота, получить у @BotFather)

### Вариант 1: Локальный запуск

**1. Установка зависимостей:**
```bash
pip install -r requirements.txt
```

**2. Настройка окружения:**
```bash
cp .env.example .env
# Отредактируйте .env и добавьте MISTRAL_API_KEY
```

**3. Запуск бота:**
```bash
# Консольный режим
python bot.py

# Telegram бот
python telegram_bot.py
```

**Примечание:** При первом запуске система автоматически:
- Создаст векторное хранилище Qdrant (локальное, `data/qdrant_storage`)
- Загрузит документы из `data/raw/raw_materials.jsonl`
- Инициализирует базу данных товаров

### Вариант 2: Docker (рекомендуется для production)

**1. Создайте `.env` файл:**
```bash
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
MISTRAL_API_KEY=your_mistral_api_key
```

**2. Запустите все сервисы:**
```bash
docker-compose up -d
```

**3. Просмотр логов:**
```bash
docker-compose logs -f telegram_bot
```

**4. Остановка:**
```bash
docker-compose down
```

### Структура сервисов в Docker

- **postgres** - PostgreSQL (порт 5432)
- **qdrant** - Qdrant векторное хранилище (порты 6333, 6334)
- **telegram_bot** - Telegram бот приложение

### Настройка Qdrant

Система поддерживает три режима работы с Qdrant:

1. **Локальное хранилище** (по умолчанию) - данные сохраняются в `data/qdrant_storage`
2. **Docker контейнер** - подключение к Qdrant серверу через `localhost:6333`
3. **In-memory** (только для тестирования) - данные теряются при перезапуске

При запуске через `docker-compose` автоматически используется Qdrant из контейнера.

## Структура проекта

```
.
├── bot.py                    # Главный бот (ConstructionMaterialsBot)
├── telegram_bot.py           # Telegram бот
├── setup_rag.py             # Инициализация RAG системы
├── config/
│   └── config.py            # Конфигурация (модели embeddings)
├── data/
│   ├── products.db          # База данных товаров (SQLite)
│   ├── qdrant_storage/      # Локальное хранилище Qdrant
│   └── raw/
│       └── raw_materials.jsonl  # Исходные документы для RAG
├── src/
│   ├── chains/              # LangChain цепочки
│   ├── rag/                 # RAG модуль
│   ├── database/            # Работа с БД товаров
│   └── schemas/             # Pydantic модели
├── docker-compose.yml       # Docker конфигурация
└── Dockerfile               # Образ для Telegram бота
```

## Примеры использования

### Программный интерфейс

```python
from bot import ConstructionMaterialsBot

bot = ConstructionMaterialsBot()
response = bot.process_query("Нужен бетон для фундамента", session_id="user123")
print(response.message)
```

### Telegram бот

После запуска `telegram_bot.py` или через Docker, бот автоматически обрабатывает все входящие сообщения в Telegram.

#### Метрики

Система подсчитывает метрики на каждый ответ бота:

**Individual Response Metrics (per query):**
- **score** (1-10): Overall quality score
- **relevance** (1-10): How well the response addresses the query
- **completeness** (1-10): How complete and informative the response is
- **accuracy** (1-10): How correct and factual the information is
- **clarity** (1-10): How clear and well-structured the response is
- **reasoning**: Text explanation of the evaluation

**Пример использования:**

# Run evaluation with default queries
python test_bot_metrics.py

# Run with specific queries
python test_bot_metrics.py --queries "Нужен бетон М300" "Какие характеристики у бетона?"

# Run with simulated buyer (multi-turn conversations)
python test_bot_metrics.py --simulated-buyer

# Save results to specific file
python test_bot_metrics.py --save-results my_results.json
Results are saved to `test_results/` directory with timestamped filenames.

#### Результат

**Результаты отдельных запросов:**

| Query | Type | Score | Relevance | Completeness | Accuracy | Clarity | Needs Clarification |
|-------|------|-------|-----------|--------------|----------|---------|---------------------|
| "Какие характеристики у бетона М300?" | informational | 8 | 9 | 8 | 9 | 8 | No |
| "Хочу заказать бетон М300" | order_specification | 7 | 8 | 6 | 10 | 8 | Yes |
| "Нужен песок для строительства" | order_specification | 8 | 9 | 8 | 10 | 8 | Yes |

**Средние аггрегированные метрики:**

| Metric | Value |
|--------|-------|
| Total Queries | 3 |
| Average Score | 7.67 |
| Min Score | 7 |
| Max Score | 8 |
| Scores ≥ 7 | 3 |
| Scores ≥ 5 | 3 |
| Scores < 5 | 0 |
| Average Relevance | 8.67 |
| Average Completeness | 7.33 |
| Average Accuracy | 9.67 |
| Average Clarity | 8.0 |

## Лицензия

MIT
