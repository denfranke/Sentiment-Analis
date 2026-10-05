"""Модели данных для таксономии ошибок."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict


# Соответствие sentiment-строк внутренним классам
SENTIMENT_TO_CLASS = {
    "negative": "negative",
    "neg": "negative",
    "нейтральный": "neutral",
    "neutral": "neutral",
    "neu": "neutral",
    "positive": "positive",
    "pos": "positive",
    "позитивный": "positive",
    "негативный": "negative",
}


# Категории ручной разметки (этап 3)
ANNOTATION_CATEGORIES = {
    1: "Сарказм / ирония",
    2: "Имплицитная оценка",
    3: "Амбивалентность",
    4: "Отрицание и область действия",
    5: "Сленг / жаргон",
    6: "Короткий текст",
    7: "Ошибка перевода",
    8: "Шум разметки",
    9: "Прочие",
}


# Методологические причины (этап 4)
CATEGORY_TO_CAUSE = {
    1: "Контекстная перегрузка",
    2: "Лексическая слепота",
    3: "Смещение класса",
    4: "Контекстная перегрузка",
    5: "Лексическая слепота",
    6: "Недостаток контекста",
    7: "Потеря при переводе",
    8: "Дефект разметки",
    9: "Прочие",
}


# Уровень расхождения (этап 4)
CATEGORY_TO_DIVERGENCE = {
    1: "model-model",
    2: "model-user",
    3: "model-user",
    4: "model-model",
    5: "model-model",
    6: "model-user",
    7: "model-model",
    8: "model-user",
    9: "unknown",
}


# Устранимость по категории (экспертная оценка, этап 5)
CATEGORY_TO_FIXABLE = {
    1: 0.60,
    2: 0.30,
    3: 0.45,
    4: 0.70,
    5: 0.50,
    6: 0.15,
    7: 0.80,
    8: 0.00,
    9: 0.10,
}


MODEL_COLUMNS = [
    "vader", "flair", "rubert", "roberta", "distilbert",
    "logistic_regression", "svm", "random_forest",
]


@dataclass
class Case:
    """Один наблюдение из датасета."""
    row_id: int
    label: int
    text: str
    actual_sentiment: str
    ensemble_sentiment: str

    # Предсказания моделей
    model_sentiments: Dict[str, str] = field(default_factory=dict)
    model_scores: Dict[str, float] = field(default_factory=dict)
    model_confidence: Dict[str, float] = field(default_factory=dict)

    translated_text: str = ""
    translation_ok: bool = True

    # Заполняется на этапах 1–5
    error_direction: Optional[str] = None       # напр. "positive->negative"
    ensemble_confidence: Optional[float] = None
    agreement: Optional[str] = None             # единогласие/большинство/раскол
    cluster: Optional[str] = None

    annotation_id: Optional[int] = None
    annotation_name: Optional[str] = None
    divergence_level: Optional[str] = None
    cause: Optional[str] = None
    fixable: Optional[float] = None
    annotation_comment: Optional[str] = None

    # Модель-«виновник»
    culprit_model: Optional[str] = None

    def to_dict(self):
        d = asdict(self)
        # Разворачиваем словари моделей для CSV
        for m in MODEL_COLUMNS:
            d[f"{m}_sentiment"] = self.model_sentiments.get(m)
            d[f"{m}_score"] = self.model_scores.get(m)
            d[f"{m}_confidence"] = self.model_confidence.get(m)
        d.pop("model_sentiments")
        d.pop("model_scores")
        d.pop("model_confidence")
        return d