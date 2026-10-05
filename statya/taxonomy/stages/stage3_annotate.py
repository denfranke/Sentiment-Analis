"""Этап 3. Интерактивная ручная разметка ошибок."""

import csv
import os
from typing import List, Optional

from models import (
    Case, ANNOTATION_CATEGORIES, CATEGORY_TO_CAUSE,
    CATEGORY_TO_DIVERGENCE, CATEGORY_TO_FIXABLE,
    MODEL_COLUMNS,
)

PROGRESS_FILE = "output/stage3_progress.csv"


def _load_progress() -> dict:
    """Загружает уже размеченные кейсы (чтобы можно было прервать и продолжить)."""
    if not os.path.exists(PROGRESS_FILE):
        return {}
    progress = {}
    with open(PROGRESS_FILE, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            progress[int(row["row_id"])] = row
    return progress


def _append_progress(c: Case):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    exists = os.path.exists(PROGRESS_FILE)
    with open(PROGRESS_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "row_id", "annotation_id", "annotation_name",
                "divergence_level", "cause", "fixable", "comment",
            ])
        writer.writerow([
            c.row_id, c.annotation_id, c.annotation_name,
            c.divergence_level, c.cause, c.fixable, c.annotation_comment or "",
        ])


def _culprit(c: Case) -> Optional[str]:
    """
    Модель-«виновник»: та, чей класс отличается от пользовательского,
    но совпадает с предсказанием сумматора.
    Если таких несколько — выбирается первая по порядку MODEL_COLUMNS.
    """
    for m in MODEL_COLUMNS:
        if (c.model_sentiments.get(m) == c.ensemble_sentiment
                and c.model_sentiments.get(m) != c.actual_sentiment):
            return m
    return None


def _print_case(c: Case, idx: int, total: int):
    print("\n" + "=" * 78)
    print(f"Кейс {idx}/{total}  (row_id={c.row_id})")
    print("=" * 78)
    print(f"Текст: {c.text[:600]}")
    if c.translated_text:
        print(f"Перевод: {c.translated_text[:300]}")
    print(f"\nПользователь: {c.actual_sentiment}  (рейтинг {c.label})")
    print(f"Сумматор:     {c.ensemble_sentiment}  "
          f"(уверенность {c.ensemble_confidence:.2f}, согласие: {c.agreement})")
    print(f"Направление:  {c.error_direction}")
    print(f"\nПредсказания моделей:")
    for m in MODEL_COLUMNS:
        s = c.model_sentiments.get(m)
        conf = c.model_confidence.get(m, 0)
        mark = "  <- виновник?" if _culprit(c) == m else ""
        print(f"  {m:25s} {s:10s} conf={conf:.2f}{mark}")


def _choose_category() -> Optional[int]:
    print("\nВыберите категорию ошибки:")
    for k, name in ANNOTATION_CATEGORIES.items():
        print(f"  {k}. {name}")
    print("  s. пропустить (skip)")
    print("  q. сохранить и выйти")
    while True:
        ans = input("Ваш выбор: ").strip().lower()
        if ans == "q":
            return -1
        if ans == "s":
            return None
        if ans.isdigit() and int(ans) in ANNOTATION_CATEGORIES:
            return int(ans)
        print("Некорректный ввод, попробуйте снова.")


def run(errors: List[Case], start_from: int = 0):
    progress = _load_progress()
    already = {int(k) for k in progress.keys()}

    # Восстанавливаем ранее размеченные
    for c in errors:
        if c.row_id in progress:
            row = progress[c.row_id]
            c.annotation_id = int(row["annotation_id"])
            c.annotation_name = row["annotation_name"]
            c.divergence_level = row["divergence_level"]
            c.cause = row["cause"]
            c.fixable = float(row["fixable"])
            c.annotation_comment = row["comment"]
            c.culprit_model = _culprit(c)

    todo = [c for c in errors if c.row_id not in already]
    print(f"\n=== Этап 3. Ручная разметка ===")
    print(f"Всего ошибочных кейсов:  {len(errors)}")
    print(f"Уже размечено:           {len(already)}")
    print(f"Осталось:                {len(todo)}")
    if not todo:
        return errors

    total = len(errors)
    for i, c in enumerate(todo, start=1):
        _print_case(c, i, len(todo))
        ans = _choose_category()
        if ans == -1:
            print("Сохранение и выход...")
            break
        if ans is None:
            continue

        c.annotation_id = ans
        c.annotation_name = ANNOTATION_CATEGORIES[ans]
        c.divergence_level = CATEGORY_TO_DIVERGENCE[ans]
        c.cause = CATEGORY_TO_CAUSE[ans]
        c.fixable = CATEGORY_TO_FIXABLE[ans]
        c.culprit_model = _culprit(c)
        c.annotation_comment = input("Комментарий (необязательно): ").strip()
        _append_progress(c)

    return errors