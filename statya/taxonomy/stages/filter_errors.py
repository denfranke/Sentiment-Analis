#!/usr/bin/env python3
"""
Отбор только ошибочно предсказанных текстов.

Читает CSV с колонками (минимум):
    label, text, ensemble_sentiment, actual_sentiment, ...

Оставляет строки, где ensemble_sentiment != actual_sentiment.
Сохраняет результат в новый CSV с теми же колонками
плюс добавленными: error_direction, ensemble_confidence,
agreement, culprit_model.
"""

import csv
import os
import sys
from collections import Counter


# --- Настройки путей ---
INPUT_PATH = r"D:\Documents\Visual Studio Code\MainFolder\Diplom\statya\taxonomy\data.csv"
OUTPUT_PATH = r"D:\Documents\Visual Studio Code\MainFolder\Diplom\statya\taxonomy\errors_only.csv"


# --- Соответствие строк-значений тональности внутренним классам ---
SENTIMENT_ALIASES = {
    "negative": "negative", "neg": "negative", "негативный": "negative",
    "neutral": "neutral",  "neu": "neutral",  "нейтральный": "neutral",
    "positive": "positive", "pos": "positive", "позитивный": "positive",
}

MODEL_COLUMNS = [
    "vader", "flair", "rubert", "roberta", "distilbert",
    "logistic_regression", "svm", "random_forest",
]


def normalize(value: str) -> str:
    if value is None:
        return "unknown"
    return SENTIMENT_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def ensemble_confidence(row: dict) -> float:
    """Доля моделей, согласных с решением сумматора."""
    ens = normalize(row.get("ensemble_sentiment", ""))
    if not ens or ens == "unknown":
        return 0.0
    agree = 0
    total = 0
    for m in MODEL_COLUMNS:
        s = normalize(row.get(f"{m}_sentiment", ""))
        if s == "unknown":
            continue
        total += 1
        if s == ens:
            agree += 1
    return agree / total if total else 0.0


def agreement_label(row: dict) -> str:
    """Согласованность моделей: unanimous / majority / split."""
    counts = Counter()
    for m in MODEL_COLUMNS:
        s = normalize(row.get(f"{m}_sentiment", ""))
        if s != "unknown":
            counts[s] += 1
    if not counts:
        return "unknown"
    total = sum(counts.values())
    top = max(counts.values())
    if top == total:
        return "unanimous"
    if top > total / 2:
        return "majority"
    return "split"


def culprit_model(row: dict) -> str:
    """
    Модель-«виновник»: предсказала класс сумматора,
    но не класс пользователя. Берётся первая по порядку MODEL_COLUMNS.
    """
    ens = normalize(row.get("ensemble_sentiment", ""))
    actual = normalize(row.get("actual_sentiment", ""))
    for m in MODEL_COLUMNS:
        s = normalize(row.get(f"{m}_sentiment", ""))
        if s == ens and s != actual:
            return m
    return ""


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Файл не найден: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("CSV пуст или не содержит заголовков.", file=sys.stderr)
            sys.exit(1)

        original_fields = list(reader.fieldnames)
        extra_fields = [
            "error_direction",
            "ensemble_confidence",
            "agreement",
            "culprit_model",
        ]
        out_fields = original_fields + [
            f for f in extra_fields if f not in original_fields
        ]

        kept_rows = []
        total = 0
        errors_count = 0
        direction_counter = Counter()

        for row in reader:
            total += 1
            ens = normalize(row.get("ensemble_sentiment", ""))
            actual = normalize(row.get("actual_sentiment", ""))

            if not ens or not actual or ens == actual:
                continue

            errors_count += 1
            direction = f"{actual}->{ens}"
            direction_counter[direction] += 1

            row["error_direction"] = direction
            row["ensemble_confidence"] = f"{ensemble_confidence(row):.3f}"
            row["agreement"] = agreement_label(row)
            row["culprit_model"] = culprit_model(row)
            kept_rows.append(row)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"Всего строк прочитано:   {total}")
    print(f"Ошибочных предсказаний:  {errors_count} "
          f"({errors_count / total * 100:.1f}%)")
    print(f"Сохранено в:             {OUTPUT_PATH}")

    if direction_counter:
        print("\nРаспределение по направлениям ошибки:")
        for d, n in direction_counter.most_common():
            print(f"  {d:30s} {n:4d}")


if __name__ == "__main__":
    main()