#!/usr/bin/env python3
"""
Скрипт для оценки тональности диалогов из test.jsonl через Ollama.

Формат входного файла (JSONL, по одному диалогу на строку):
    {"sample":["реплика 1","реплика 2", ...]}

Формат выходного файла:
    {"sample":["реплика 1","реплика 2", ...],"sentiment":"positive"}

Установка зависимостей:
    pip install requests tqdm
"""

import json
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm


# ======================= КОНФИГУРАЦИЯ =======================
INPUT_FILE = "test.jsonl"
OUTPUT_FILE = "test_with_sentiment.jsonl"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1:8b"      # можно заменить на "qwen2.5:7b" и т.п.

TEMPERATURE = 0.0
MAX_TOKENS = 20
TIMEOUT = 120
MAX_RETRIES = 3
RETRY_DELAY = 2

# Возобновление: если True — не перезаписывать уже обработанные записи
RESUME = True
# ===========================================================


SYSTEM_PROMPT = """Ты — эксперт по анализу тональности русскоязычных интернет-диалогов.
Твоя задача — определить ОБЩУЮ эмоциональную тональность всего диалога.

═══════════════════════════════════════════
КАТЕГОРИИ ТОНАЛЬНОСТИ
═══════════════════════════════════════════

POSITIVE (позитивная) — выбирай, если в диалоге преобладают:
• доброжелательность, тёплое общение, симпатия;
• юмор, шутки, смех, ирония без злобы;
• благодарность, похвала, комплименты;
• согласие, поддержка, дружеский троллинг;
• радость, восторг, восхищение;
• флирт, кокетство, игривый тон.

NEGATIVE (негативная) — выбирай, если в диалоге преобладают:
• прямые оскорбления, мат в адрес собеседника;
• агрессия, угрозы, унижение;
• злоба, ненависть, презрение;
• обида, упрёки, скандал, ссора;
• уничижительные характеристики («тупой», «дебил», «мразь»);
• сарказм с явной враждебностью.

NEUTRAL (нейтральная) — выбирай, если:
• диалог носит информационный или бытовой характер;
• обсуждаются факты, вопросы, ответы без эмоций;
• тональность смешанная (есть и позитив, и негатив примерно поровну);
• присутствует вежливый спор, дискуссия без оскорблений;
• непонятно, шутит человек или говорит серьёзно;
• диалог слишком короткий и неоднозначный.

═══════════════════════════════════════════
ВАЖНЫЕ ПРАВИЛА
═══════════════════════════════════════════

1. Оценивай ВЕСЬ диалог, а не отдельную реплику.
2. Мат сам по себе НЕ делает диалог негативным. Если мат используется
   для связки слов или в дружеском общении — это neutral или positive.
3. Сарказм и ирония чаще neutral, если нет прямой враждебности.
4. Если в диалоге есть конфликт, но также шутки и примирение —
   скорее neutral.
5. Короткие диалоги (1–2 реплики) без явных эмоций — neutral.
6. Если диалог — обмен вопросами и ответами без эмоций — neutral.
7. Спор с аргументами, но без оскорблений — neutral.
8. Если сомневаешься между positive и neutral — выбирай neutral.
9. Если сомневаешься между negative и neutral — выбирай neutral.

═══════════════════════════════════════════
ПРИМЕРЫ
═══════════════════════════════════════════

Пример 1.
Диалог:
1. Спасибо тебе большое, очень помог!
2. Да не за что, обращайся :)
Ответ: positive

Пример 2.
Диалог:
1. Ты вообще тупой? Как можно было это сделать?
2. Сам такой, идиот. Не лезь.
Ответ: negative

Пример 3.
Диалог:
1. Сколько стоит билет?
2. 500 рублей.
3. А детям скидка есть?
4. Да, до 7 лет бесплатно.
Ответ: neutral

Пример 4.
Диалог:
1. Вчера смотрел новый фильм, зашло на ура!
2. О, тоже хочу глянуть. Что там по сюжету?
3. Да обычный боевик, но спецэффекты огонь.
Ответ: positive

Пример 5.
Диалог:
1. Ты не прав.
2. С чего ты взял?
3. Потому что факты говорят обратное.
4. Ну ок, возможно.
Ответ: neutral

Пример 6.
Диалог:
1. Грустно...
2. А если одинаково стоили бы, то уже веселее?
3. Продам ВАЗ-2112 или обменяю на фару от Порше.
4. Хочешь потрогать мой порше?
Ответ: positive

Пример 7.
Диалог:
1. Ненавижу таких, как ты.
2. Взаимно, мразь.
Ответ: negative

Пример 8.
Диалог:
1. Блин, опять дождь.
2. Ну и ладно, дома посидим.
3. Да, тоже вариант.
Ответ: neutral

═══════════════════════════════════════════
ФОРМАТ ОТВЕТА
═══════════════════════════════════════════

Отвечай РОВНО ОДНИМ словом из списка:
positive
negative
neutral

НЕ пиши пояснений, кавычек, точек, знаков препинания.
Только одно слово — и всё.
"""


def build_user_prompt(dialog: list) -> str:
    """Формирует текст диалога для отправки в модель."""
    lines = [f"{i}. {msg}" for i, msg in enumerate(dialog, 1)]
    return "Диалог:\n" + "\n".join(lines)


def parse_sentiment(raw: str) -> str:
    """Извлекает метку тональности из ответа модели."""
    if not raw:
        return "unknown"
    text = raw.strip().lower()

    # Прямое совпадение по английским меткам
    for label in ("positive", "negative", "neutral"):
        if re.search(rf"\b{label}\b", text):
            return label

    # Русские варианты
    mapping = {
        "позитив": "positive",
        "положит": "positive",
        "негатив": "negative",
        "отрицат": "negative",
        "нейтрал": "neutral",
    }
    for key, val in mapping.items():
        if key in text:
            return val

    return "unknown"


def query_ollama(dialog: list) -> str:
    """Отправляет диалог в Ollama и возвращает ответ модели."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(dialog)},
        ],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "")


def get_sentiment(dialog: list) -> str:
    """Обёртка с ретраями и возвратом метки тональности."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = query_ollama(dialog)
            label = parse_sentiment(raw)
            if label != "unknown":
                return label
            last_error = f"Не удалось распознать ответ: {raw!r}"
        except Exception as e:
            last_error = e
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    print(f"\n[!] Не удалось получить оценку: {last_error}")
    return "error"


def load_jsonl(path: Path) -> list:
    """Читает JSONL-файл с диалогами."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[!] Строка {line_num}: ошибка JSON — {e}")
                continue
            if "sample" not in data or not isinstance(data["sample"], list):
                print(f"[!] Строка {line_num}: нет ключа 'sample' или он не список")
                continue
            records.append(data)
    return records


def append_record(path: Path, record: dict) -> None:
    """Дописывает одну запись в выходной JSONL-файл."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def check_ollama() -> bool:
    """Проверяет, что Ollama запущен и нужная модель доступна."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"[i] Доступные модели: {models}")
        base = OLLAMA_MODEL.split(":")[0]
        if not any(base in m for m in models):
            print(f"[!] Модель '{OLLAMA_MODEL}' не найдена.")
            print(f"    Загрузите её командой: ollama pull {OLLAMA_MODEL}")
            return False
        return True
    except Exception as e:
        print(f"[!] Ollama недоступен: {e}")
        print("    Убедитесь, что сервер запущен (команда: ollama serve).")
        return False


def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        print(f"[!] Файл {INPUT_FILE} не найден.")
        return

    if not check_ollama():
        return

    records = load_jsonl(input_path)
    print(f"[i] Загружено диалогов: {len(records)}")

    # Проверяем, сколько уже обработано (если RESUME = True)
    already_done = 0
    if RESUME and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            already_done = sum(1 for line in f if line.strip())
        print(f"[i] Уже обработано ранее: {already_done}")
    else:
        output_path.write_text("", encoding="utf-8")

    if already_done >= len(records):
        print("[i] Все диалоги уже обработаны. Нечего делать.")
        return

    # Обрабатываем оставшиеся
    todo = records[already_done:]
    errors = 0

    for record in tqdm(todo, desc="Оценка тональности", initial=already_done,
                       total=len(records)):
        sentiment = get_sentiment(record["sample"])
        if sentiment in ("error", "unknown"):
            errors += 1
        # Создаём новую запись, сохраняя исходные поля и добавляя sentiment
        out_record = dict(record)
        out_record["sentiment"] = sentiment
        append_record(output_path, out_record)

    print(f"\n[i] Готово. Обработано: {len(todo)}, ошибок: {errors}")
    print(f"[i] Результат: {output_path}")


if __name__ == "__main__":
    main()