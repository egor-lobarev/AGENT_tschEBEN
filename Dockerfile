FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всего проекта
COPY . .

# Создание директорий для данных
RUN mkdir -p data/qdrant_storage data/raw

# Переменные окружения (можно переопределить через docker-compose)
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Команда по умолчанию (можно переопределить)
CMD ["python", "telegram_bot.py"]
