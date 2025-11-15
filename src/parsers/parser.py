import requests
from bs4 import BeautifulSoup
import time
import json
from pathlib import Path

# Ссылки для обхода
URLS = [
    "https://dostavka-dacha.ru/pesok.php",
    "https://ru.wikipedia.org/wiki/Строительные_материалы",
    "https://ru.wikipedia.org/wiki/Бетон",
    "https://ru.wikibooks.org/wiki/Строительство_и_ремонт/Строительные_материалы/Песок",
    "https://tsk-sistema.ru/company/stati/vidy-stroitelnogo-peska/?ysclid=mi036gw4q4530961850",
    "https://tsk-sistema.ru/company/stati/podushka-fundamenta/",
    "https://tsk-sistema.ru/company/stati/dolomitovyy-shcheben/",
    "https://tsk-sistema.ru/company/stati/mramornyy-shcheben/",
    "https://tsk-sistema.ru/company/stati/izvestnyak/",
    "https://tsk-sistema.ru/company/stati/graviy/",
    "https://tsk-sistema.ru/company/stati/shchebenochno-peschanaya-smes-osobennosti-i-primenenie/",
    "https://tsk-sistema.ru/company/stati/shcheben-dlya-betona/",
    "https://tsk-sistema.ru/company/stati/kvartsevyy-pesok/",
    "https://tsk-sistema.ru/company/stati/shcheben-dlya-fundamenta/",
    "https://tsk-sistema.ru/company/stati/tsement/",
    "https://tsk-sistema.ru/company/stati/vidy-stroitelnogo-peska/",
    "https://tsk-sistema.ru/company/stati/shcheben-vidy/",
    "https://tsk-sistema.ru/company/stati/peschano-graviynye-smesi-i-ikh-primenenie/"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# Определяем путь к директории data/raw относительно корня проекта
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "raw_materials.jsonl"


def extract_text(url):
    """Получает текст со страницы, сохраняя специальные символы-разделители из HTML."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return None, f"Ошибка загрузки: {e}"

    soup = BeautifulSoup(response.text, "html.parser")

    # Убираем скрипты и стили
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Заменяем HTML теги на специальные символы-разделители перед извлечением текста
    # Это позволит сохранить структуру для последующего разбиения текста
    
    # Блочные элементы - заменяем на разделители
    for tag in soup.find_all(["p", "div", "section", "article"]):
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")
    
    # Заголовки
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")
    
    # Переносы строк
    for tag in soup.find_all("br"):
        tag.replace_with("\n")
    
    # Элементы списков
    for tag in soup.find_all("li"):
        tag.insert_before("\n- ")
        tag.insert_after("\n")
    
    # Списки
    for tag in soup.find_all(["ul", "ol"]):
        tag.insert_before("\n")
        tag.insert_after("\n\n")
    
    # Таблицы
    for tag in soup.find_all(["table", "tr"]):
        tag.insert_before("\n")
        tag.insert_after("\n")
    for tag in soup.find_all("td"):
        tag.insert_after(" | ")

    # Извлекаем текст с сохранением разделителей
    text = soup.get_text(separator=" ")
    
    # Убираем множественные пробелы, но сохраняем переносы строк и разделители
    lines = []
    for line in text.split("\n"):
        cleaned_line = " ".join(line.split())
        if cleaned_line:
            lines.append(cleaned_line)
    
    text = "\n".join(lines)
    
    return text, None


def save_to_jsonl(data):
    """Сохраняет одну запись."""
    # Создаём директорию, если её нет
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def main():
    # Создаём директорию, если её нет
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Начинаю сбор → {OUTPUT_FILE}")

    for url in URLS:
        print(f"→ Скачиваю {url}")
        text, error = extract_text(url)

        record = {
            "url": url,
            "error": error,
            "content": text if text else "",
            "timestamp": time.time()
        }

        save_to_jsonl(record)

        # Маленькая задержка чтобы сайт не считал DDOS
        time.sleep(1)

    print("Готово. Файл сохранён:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
