"""Этап 1. Отбор ошибочных кейсов."""

from typing import List, Tuple

from models import Case, MODEL_COLUMNS


def run(cases: List[Case]) -> Tuple[List[Case], List[Case]]:
    """
    Возвращает (errors, disputes):
      errors   — все кейсы, где ensemble_sentiment != actual_sentiment
      disputes — подгруппа, где модели расходились более чем на 1 класс
    """
    errors: List[Case] = []
    disputes: List[Case] = []

    for c in cases:
        if c.ensemble_sentiment == c.actual_sentiment:
            continue
        errors.append(c)

        # «Спорный» — если модели предсказали разные классы,
        # и разница между крайними больше одного класса
        classes_seen = set(c.model_sentiments.values())
        if len(classes_seen) >= 3:
            disputes.append(c)

    # Определяем направление ошибки и уверенность сумматора
    for c in errors:
        c.error_direction = f"{c.actual_sentiment}->{c.ensemble_sentiment}"
        c.ensemble_confidence = _ensemble_confidence(c)

    return errors, disputes


def _ensemble_confidence(c: Case) -> float:
    """
    Оценка уверенности сумматора: доля моделей, согласных с его решением.
    Если у вас есть ensemble_score — можно использовать его.
    """
    if not c.model_sentiments:
        return 0.0
    agree = sum(
        1 for m in MODEL_COLUMNS
        if c.model_sentiments.get(m) == c.ensemble_sentiment
    )
    return agree / len(MODEL_COLUMNS)


def print_summary(errors, disputes, total):
    print(f"\n=== Этап 1. Отбор ошибочных кейсов ===")
    print(f"Всего наблюдений:      {total}")
    print(f"Ошибочных кейсов:      {len(errors)} ({len(errors)/total*100:.1f}%)")
    print(f"Из них «спорных»:      {len(disputes)}")

    directions = {}
    for c in errors:
        directions[c.error_direction] = directions.get(c.error_direction, 0) + 1
    print("\nРаспределение по направлениям ошибки:")
    for d, n in sorted(directions.items(), key=lambda x: -x[1]):
        print(f"  {d:30s} {n:4d}")