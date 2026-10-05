import os
import random
from pathlib import Path

# ============================================================
# ЗАРАНЕЕ ЗАДАННЫЕ ВХОДНЫЕ ДАННЫЕ
# ============================================================

# Папка на Рабочем столе, где лежат подпапки с отзывами
DATA_ROOT = Path(r"D:\Desktop\dataset")

POSITIVE_DIR = DATA_ROOT / "pos"
NEGATIVE_DIR = DATA_ROOT / "neg"
NEUTRAL_DIR  = DATA_ROOT / "neu"

BASE_DIR    = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "movie_dataset_1k.csv"

# Сколько отзывов взять из каждой папки. None -> взять все.
N_POSITIVE = 5000
N_NEGATIVE = 5000
N_NEUTRAL  = 5000

SHUFFLE = True
SEED    = 42

LABEL_POSITIVE = "POSITIVE"
LABEL_NEGATIVE = "NEGATIVE"
LABEL_NEUTRAL  = "NEUTRAL"
# ============================================================


def collect_files(folder: Path):
    """Рекурсивно собирает все .txt файлы в папке."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Папка не найдена: {folder}")
    return [p for p in folder.rglob("*.txt") if p.is_file()]


def escape_field(text: str) -> str:
    """Экранирует текст для CSV-поля в кавычках."""
    text = text.replace("\r", " ").replace("\n", " ").strip()
    text = text.replace('"', '""')
    return f'"{text}"'


def build_dataset(pos_dir, neg_dir, neu_dir, output_file,
                  n_pos=None, n_neg=None, n_neu=None,
                  shuffle=True, seed=42):
    random.seed(seed)

    pos_files = collect_files(pos_dir)
    neg_files = collect_files(neg_dir)
    neu_files = collect_files(neu_dir)

    print(f"Найдено: POSITIVE={len(pos_files)}, "
          f"NEGATIVE={len(neg_files)}, NEUTRAL={len(neu_files)}")

    def sample(files, n):
        if n is None or n >= len(files):
            return files[:]
        return random.sample(files, n)

    pos_files = sample(pos_files, n_pos)
    neg_files = sample(neg_files, n_neg)
    neu_files = sample(neu_files, n_neu)

    print(f"Отобрано: POSITIVE={len(pos_files)}, "
          f"NEGATIVE={len(neg_files)}, NEUTRAL={len(neu_files)}")

    samples = (
        [(p, LABEL_POSITIVE) for p in pos_files] +
        [(p, LABEL_NEGATIVE) for p in neg_files] +
        [(p, LABEL_NEUTRAL)  for p in neu_files]
    )

    if shuffle:
        random.shuffle(samples)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for path, label in samples:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fin:
                    text = fin.read().strip()
            except Exception as e:
                print(f"[WARN] Не удалось прочитать {path}: {e}")
                skipped += 1
                continue

            if not text:
                skipped += 1
                continue

            fout.write(f"{label},{escape_field(text)}\n")
            written += 1

    print(f"Готово. Записано {written} записей в {output_file}. Пропущено: {skipped}")


if __name__ == "__main__":
    build_dataset(
        pos_dir=POSITIVE_DIR,
        neg_dir=NEGATIVE_DIR,
        neu_dir=NEUTRAL_DIR,
        output_file=OUTPUT_FILE,
        n_pos=N_POSITIVE,
        n_neg=N_NEGATIVE,
        n_neu=N_NEUTRAL,
        shuffle=SHUFFLE,
        seed=SEED,
    )