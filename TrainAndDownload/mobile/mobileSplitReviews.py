import csv
import random
from pathlib import Path
from collections import Counter

# ============================================================
# ЗАРАНЕЕ ЗАДАННЫЕ ВХОДНЫЕ ДАННЫЕ
# ============================================================

# Папка на Рабочем столе с исходным датасетом
DESKTOP   = Path(r"D:\Desktop")
DATA_ROOT = DESKTOP / "mobile"

# Входной файл с колонками Review,Rating
INPUT_FILES = [
    DATA_ROOT / "data.csv",
]

# Куда сохранять итоговый сбалансированный датасет
BASE_DIR    = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "phone_dataset_15k.csv"

# Сколько отзывов взять в КАЖДОМ классе (POSITIVE / NEGATIVE / NEUTRAL).
# Итоговый размер датасета = N_PER_CLASS * 3.
# None -> взять по минимуму из доступного.
N_PER_CLASS = 5000

# Пороги классификации по Rating:
#   rating > POS_THRESHOLD  -> POSITIVE
#   rating < NEG_THRESHOLD  -> NEGATIVE
#   иначе                   -> NEUTRAL
POS_THRESHOLD = 4.0
NEG_THRESHOLD = 3.0

# Максимальная длина отзыва в символах.
# Обрезка идёт по границе предложения: отзыв укорачивается
# до ближайшего конца предложения, не превышающего лимит.
MAX_REVIEW_LEN = 500

# Минимальная длина отзыва (после обрезки).
# Слишком короткие отзывы отбрасываются.
MIN_REVIEW_LEN = 20

SHUFFLE = True
SEED    = 42

LABEL_POSITIVE = "POSITIVE"
LABEL_NEGATIVE = "NEGATIVE"
LABEL_NEUTRAL  = "NEUTRAL"
# ============================================================


def escape_field(text: str) -> str:
    """Экранирует текст для CSV-поля в кавычках."""
    text = text.replace("\r", " ").replace("\n", " ").strip()
    text = text.replace('"', '""')
    return f'"{text}"'


def detect_delimiter(path: Path) -> str:
    """Определяет разделитель: .tsv -> таб, иначе запятая."""
    return "\t" if path.suffix.lower() == ".tsv" else ","


def score_to_label(score: float) -> str:
    if score > POS_THRESHOLD:
        return LABEL_POSITIVE
    if score < NEG_THRESHOLD:
        return LABEL_NEGATIVE
    return LABEL_NEUTRAL


def truncate_by_sentence(text: str, max_len: int = MAX_REVIEW_LEN) -> str:
    """
    Обрезает текст до max_len символов так, чтобы не рвать предложение.
    """
    text = text.replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text

    window = text[:max_len]

    # 1. Ищем последний конец предложения в пределах лимита.
    sentence_ends = ".!?…"
    last_end = -1
    for i, ch in enumerate(window):
        if ch in sentence_ends:
            last_end = i

    if last_end != -1:
        truncated = window[:last_end + 1].rstrip()
        if truncated and truncated[-1] not in sentence_ends + '")»':
            truncated += "."
        return truncated

    # 2. Если конца предложения нет — режем по последнему пробелу.
    last_space = window.rfind(" ")
    if last_space > 0:
        return window[:last_space].rstrip() + "."

    # 3. Совсем плохой случай — режем жёстко.
    return window.rstrip() + "."


def read_reviews_by_class(input_files):
    """
    Читает все входные файлы с колонками Review, Rating
    и раскладывает тексты по трём спискам в соответствии с меткой класса.
    Текст обрезается до MAX_REVIEW_LEN символов по границе предложения.
    Слишком короткие отзывы (< MIN_REVIEW_LEN) отбрасываются.
    """
    buckets = {
        LABEL_POSITIVE: [],
        LABEL_NEGATIVE: [],
        LABEL_NEUTRAL:  [],
    }

    for path in input_files:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        delim = detect_delimiter(path)
        print(f"[INFO] Читаю {path} (delimiter={repr(delim)})")

        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter=delim)
            try:
                header = next(reader)
            except StopIteration:
                continue

            header_norm = [h.strip().lower() for h in header]

            def find_col(*names):
                for n in names:
                    if n in header_norm:
                        return header_norm.index(n)
                return None

            # Ищем колонки: Review и Rating
            idx_text  = find_col("review", "text", "review/text", "review_text")
            idx_score = find_col("rating", "review/score", "score")

            if idx_score is None or idx_text is None:
                print(f"[WARN] В {path} не найдены нужные колонки. "
                      f"Найдены: {header_norm}")
                continue

            for row in reader:
                if len(row) <= max(idx_score, idx_text):
                    continue

                raw_score = row[idx_score].strip()
                text      = row[idx_text].strip()

                if not text or not raw_score:
                    continue

                try:
                    score = float(raw_score)
                except ValueError:
                    continue

                # Обрезаем текст по границе предложения
                text = truncate_by_sentence(text, MAX_REVIEW_LEN)

                # Отбрасываем слишком короткие
                if len(text) < MIN_REVIEW_LEN:
                    continue

                label = score_to_label(score)
                buckets[label].append(text)

    return buckets


def build_balanced_dataset(input_files, output_file,
                           n_per_class=None,
                           shuffle=True, seed=42):
    random.seed(seed)

    buckets = read_reviews_by_class(input_files)
    print("Доступно по классам до балансировки: "
          f"POSITIVE={len(buckets[LABEL_POSITIVE])}, "
          f"NEGATIVE={len(buckets[LABEL_NEGATIVE])}, "
          f"NEUTRAL={len(buckets[LABEL_NEUTRAL])}")

    available = min(len(v) for v in buckets.values())
    if n_per_class is None:
        target = available
    else:
        target = min(n_per_class, available)

    if target == 0:
        raise RuntimeError("Недостаточно данных: как минимум один класс пуст.")

    if n_per_class is not None and n_per_class > available:
        print(f"[WARN] Запрошено {n_per_class} на класс, "
              f"но в самом маленьком классе только {available}. "
              f"Беру по {target}.")

    samples = []
    for label, texts in buckets.items():
        chosen = random.sample(texts, target)
        samples.extend((label, t) for t in chosen)

    if shuffle:
        random.shuffle(samples)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for label, text in samples:
            fout.write(f"{label},{escape_field(text)}\n")

    counts = Counter(lbl for lbl, _ in samples)
    print(f"Готово. Записано {len(samples)} записей в {output_file}")
    print(f"Распределение: POSITIVE={counts[LABEL_POSITIVE]}, "
          f"NEGATIVE={counts[LABEL_NEGATIVE]}, "
          f"NEUTRAL={counts[LABEL_NEUTRAL]}")


if __name__ == "__main__":
    build_balanced_dataset(
        input_files=INPUT_FILES,
        output_file=OUTPUT_FILE,
        n_per_class=N_PER_CLASS,
        shuffle=SHUFFLE,
        seed=SEED,
    )