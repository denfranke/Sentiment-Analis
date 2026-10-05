"""Утилиты ввода-вывода."""

import csv
import json
import os
from typing import List, Dict

from models import (
    Case, MODEL_COLUMNS, SENTIMENT_TO_CLASS,
)


def _to_class(value: str) -> str:
    if value is None:
        return "unknown"
    v = str(value).strip().lower()
    return SENTIMENT_TO_CLASS.get(v, v)


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "да")


def read_cases(path: str) -> List[Case]:
    """Читает CSV-файл с отзывами и предсказаниями моделей."""
    cases: List[Case] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            actual = _to_class(row.get("actual_sentiment", ""))
            ens = _to_class(row.get("ensemble_sentiment", ""))

            model_sentiments = {}
            model_scores = {}
            model_conf = {}
            for m in MODEL_COLUMNS:
                model_sentiments[m] = _to_class(row.get(f"{m}_sentiment", ""))
                model_scores[m] = _to_float(row.get(f"{m}_score", 0))
                model_conf[m] = _to_float(row.get(f"{m}_confidence", 0))

            case = Case(
                row_id=i,
                label=int(_to_float(row.get("label", 0))),
                text=row.get("text", ""),
                actual_sentiment=actual,
                ensemble_sentiment=ens,
                model_sentiments=model_sentiments,
                model_scores=model_scores,
                model_confidence=model_conf,
                translated_text=row.get("translated_text", ""),
                translation_ok=_to_bool(row.get("translation_ok", "true")),
            )
            cases.append(case)
    return cases


def write_cases(cases: List[Case], path: str):
    """Сохраняет список кейсов в CSV."""
    if not cases:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(cases[0].to_dict().keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in cases:
            writer.writerow(c.to_dict())


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)