import csv
import time
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# НАСТРОЙКИ
# ============================================================

# Какую модель использовать.
# Варианты:
#   "facebook/nllb-200-distilled-600M"  — лёгкая (~2.4 GB), быстрая, качество базовое
#   "facebook/nllb-200-distilled-1.3B"  — средняя, качество выше
#   "facebook/nllb-200-3.3B"            — тяжёлая (~13 GB), лучшее качество
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Языковые коды NLLB (FLORES-200)
SOURCE_LANG = "eng_Latn"   # английский
TARGET_LANG = "rus_Cyrl"   # русский

# Пути к файлам
DATA_ROOT   = Path(__file__).resolve().parent
INPUT_FILE  = DATA_ROOT / "book_dataset_15k.csv"
OUTPUT_FILE = DATA_ROOT / "book_dataset_15k_ru.csv"

# Сколько строк перевести (None — все)
LIMIT = None

# Размер батча для перевода (сколько текстов обрабатывать за раз)
TRANSLATE_BATCH_SIZE = 32

# Максимальная длина текста в токенах (обрезается)
MAX_LENGTH = 512

# Сколько строк смотреть для статистики
LOG_EVERY = 10

# ============================================================


def escape_field(text: str) -> str:
    """Экранирует текст для CSV-поля в кавычках."""
    text = text.replace("\r", " ").replace("\n", " ").strip()
    text = text.replace('"', '""')
    return f'"{text}"'


def load_model(model_name: str, source_lang: str, target_lang: str):
    """
    Загружает модель и токенизатор NLLB-200.
    При первом запуске модель скачается с Hugging Face (~2.4 GB).
    """
    print(f"[INFO] Загружаю модель: {model_name}")
    print("[INFO] Первый запуск может занять несколько минут (скачивание модели)...")

    # Выбираем устройство: GPU, если доступен
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Устройство: {device}")

    # Тип данных: на GPU можно float16 для скорости, на CPU — float32
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    # Устанавливаем исходный язык для токенизатора
    tokenizer.src_lang = source_lang

    # Получаем ID токена для целевого языка
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

    model.eval()
    print("[INFO] Модель загружена.")
    return model, tokenizer, device, forced_bos_token_id


def translate_batch(model, tokenizer, device, forced_bos_token_id, texts):
    """
    Переводит список текстов батчем.
    Возвращает список переводов той же длины.
    """
    if not texts:
        return []

    # Токенизируем все тексты сразу
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Генерируем перевод
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=MAX_LENGTH,
            num_beams=1,               # 1 = greedy, быстрее; 5 = beam search, качественнее
            no_repeat_ngram_size=3,    # защита от зацикливания [citation:3]
            repetition_penalty=1.2,    # штраф за повторы [citation:3]
        )

    # Декодируем обратно в текст
    translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return translated


def count_translated_rows(output_file: Path) -> int:
    """Считает, сколько строк уже переведено (для возобновления)."""
    if not output_file.exists():
        return 0
    count = 0
    with open(output_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for _ in reader:
            count += 1
    return count


def translate_csv(input_file: Path, output_file: Path, limit=None):
    if not input_file.exists():
        raise FileNotFoundError(f"Файл не найден: {input_file}")

    # 1. Загружаем модель
    model, tokenizer, device, forced_bos = load_model(
        MODEL_NAME, SOURCE_LANG, TARGET_LANG
    )

    # 2. Читаем CSV
    rows = []
    with open(input_file, "r", encoding="utf-8", newline="") as fin:
        reader = csv.reader(fin)
        for row in reader:
            if len(row) < 2:
                continue
            rows.append((row[0].strip(), row[1]))

    if limit is not None:
        rows = rows[:limit]

    print(f"[INFO] Всего строк в датасете: {len(rows)}")

    labels = [r[0] for r in rows]
    texts  = [r[1] for r in rows]
    total = len(texts)

    # 3. Проверяем, сколько уже переведено (возобновление)
    already_done = count_translated_rows(output_file)
    if already_done > 0:
        print(f"[INFO] Найдено уже переведённых строк: {already_done}")
        if already_done >= total:
            print("[INFO] Всё уже переведено. Выход.")
            return
        print(f"[INFO] Продолжаю с строки {already_done}...")

    # 4. Открываем файл в режиме дозаписи (или создаём новый)
    mode = "a" if already_done > 0 else "w"
    fout = open(output_file, mode, encoding="utf-8", newline="")
    writer = csv.writer(fout, quoting=csv.QUOTE_ALL)

    # 5. Переводим батчами, начиная с already_done
    try:
        for start in range(already_done, total, TRANSLATE_BATCH_SIZE):
            chunk = texts[start:start + TRANSLATE_BATCH_SIZE]

            try:
                translated_chunk = translate_batch(
                    model, tokenizer, device, forced_bos, chunk
                )
            except Exception as e:
                print(f"[ERROR] Батч {start}-{start+len(chunk)}: {e}")
                translated_chunk = chunk   # фолбэк — оригиналы

            # Сразу пишем батч в файл и сбрасываем буфер на диск
            for label, tr in zip(labels[start:start + len(chunk)], translated_chunk[:len(chunk)]):

                if tr is None:
                    tr = ""
                tr = str(tr).replace("\r", " ").replace("\n", " ").strip()
                writer.writerow([label, tr])

            fout.flush()          # сбрасываем буфер Python
            # os.fsync(fout.fileno())  # (опционально) жёсткий сброс на диск

            processed = min(start + TRANSLATE_BATCH_SIZE, total)
            if processed % LOG_EVERY == 0 or processed == total:
                print(f"... переведено {processed}/{total}")

    except KeyboardInterrupt:
        print("\n[INFO] Прервано пользователем. Прогресс сохранён.")
    finally:
        fout.close()

    print(f"[INFO] Готово. Файл: {output_file}")


if __name__ == "__main__":
    translate_csv(INPUT_FILE, OUTPUT_FILE, limit=LIMIT)