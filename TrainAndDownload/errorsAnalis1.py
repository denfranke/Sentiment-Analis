# auto_error_classifier.py
# -*- coding: utf-8 -*-
"""
Автоматическая предразметка ошибок по эвристикам.
Дополнительно формирует errors_detailed.csv с разбором каждой ошибки:
- тип ошибки (sarcasm / mixed / ...)
- уточнение
- какие модели ошиблись / угадали
"""

import pandas as pd
import numpy as np
import re
import os

# ---------- Словари-маркеры ----------
MIXED_MARKERS = ['но ', 'однако', 'зато', 'хотя', 'несмотря на', 'с одной стороны']
SARCASM_MARKERS = [
    'ну конечно', 'ну спасибо', 'спасибо большое', 'обязательно',
    'как же', 'конечно же', 'ясное дело', 'аж', 'целых',
]
SLANG_MARKERS = [
    'кринж', 'топ', 'огонь', 'жесть', 'имба', 'пушка', 'бомба',
    'ужас', 'капец', 'жестко', 'ни о чем', 'дно', 'трэш', 'треш',
]
NEGATION_MARKERS = ['не ', 'ни ', 'без ', 'нет ']
DOMAIN_MARKERS = [
    'доставка', 'курьер', 'приложение', 'заказ', 'поддержка',
    'возврат', 'оплата', 'пункт выдачи', 'пвз',
]
POSITIVE_WORDS = ['хорош', 'отличн', 'супер', 'класс', 'нрав', 'люблю', 'доволен', 'рекоменд']
NEGATIVE_WORDS = ['плох', 'ужасн', 'отврат', 'ненавиж', 'разочарован', 'не рекоменд', 'кошмар']

MODELS = ['vader', 'flair', 'rubert', 'roberta', 'distilbert',
          'logistic_regression', 'svm', 'random_forest']

# Русские названия моделей для читаемости в отчёте
MODEL_RU = {
    'vader': 'VADER',
    'flair': 'Flair',
    'rubert': 'RuBERT',
    'roberta': 'RoBERTa',
    'distilbert': 'DistilBERT',
    'logistic_regression': 'LogReg',
    'svm': 'SVM',
    'random_forest': 'RandomForest',
}


def norm_label(x):
    """Нормализация метки к Positive/Negative/Neutral."""
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if 'pos' in s:
        return 'Positive'
    if 'neg' in s:
        return 'Negative'
    if 'neu' in s:
        return 'Neutral'
    return None


def get_model_status(row):
    """
    Возвращает (models_wrong, models_correct) — списки русских имён моделей.
    Модель считается ошибившейся, если её sentiment != actual_norm.
    """
    actual = row['actual_norm']
    wrong, correct = [], []

    for m in MODELS:
        col = f'{m}_sentiment'
        if col not in row or pd.isna(row[col]):
            continue
        pred = norm_label(row[col])
        if pred is None:
            continue
        name = MODEL_RU.get(m, m)
        if pred == actual:
            correct.append(name)
        else:
            wrong.append(name)

    return ', '.join(wrong), ', '.join(correct)


def explain_error(row):
    """
    Возвращает (category, detail, confidence) — тип ошибки, текстовое уточнение, уверенность.
    Логика повторяет auto_classify, но с детализацией.
    """
    text = str(row['text']).lower()
    actual = row['actual_norm']
    ensemble = row['ensemble_norm']
    n_words = len(text.split())

    scores = {}
    details = {}

    # 1. Короткий текст
    if n_words < 5:
        scores['short_text'] = 0.9
        details['short_text'] = f'Очень короткий текст ({n_words} слов) — модели не хватает контекста'
    elif n_words < 8:
        scores['short_text'] = 0.5
        details['short_text'] = f'Короткий текст ({n_words} слов) — контекста мало'

    # 2. Шум разметки: модели единогласно, а label против
    model_preds = []
    for m in MODELS:
        col = f'{m}_sentiment'
        if col in row and pd.notna(row[col]):
            model_preds.append(str(row[col]).lower())

    if model_preds:
        pos_count = sum('pos' in p for p in model_preds)
        neg_count = sum('neg' in p for p in model_preds)
        n = len(model_preds)

        if actual == 'Negative' and pos_count / n >= 0.75:
            scores['label_noise'] = 0.85
            details['label_noise'] = (
                f'{pos_count}/{n} моделей дали Positive, а label = Negative — '
                f'похоже на шум разметки или ироничный отзыв'
            )
        elif actual == 'Positive' and neg_count / n >= 0.75:
            scores['label_noise'] = 0.85
            details['label_noise'] = (
                f'{neg_count}/{n} моделей дали Negative, а label = Positive — '
                f'похоже на шум разметки или смешанный отзыв'
            )

    # 3. Смешанная тональность
    mixed_found = [m for m in MIXED_MARKERS if m in text]
    if mixed_found:
        has_pos = any(w in text for w in POSITIVE_WORDS)
        has_neg = any(w in text for w in NEGATIVE_WORDS)
        if has_pos and has_neg:
            scores['mixed'] = 0.85
            details['mixed'] = (
                f'Смешанная тональность: маркеры {mixed_found}, '
                f'есть и позитив, и негатив'
            )
        elif has_pos or has_neg:
            scores['mixed'] = 0.5
            details['mixed'] = (
                f'Есть маркер противопоставления {mixed_found}, '
                f'но только одна полярность'
            )

    # 4. Сарказм
    sarc_found = [m for m in SARCASM_MARKERS if m in text]
    if sarc_found:
        has_pos = any(w in text for w in POSITIVE_WORDS)
        if has_pos and actual == 'Negative':
            scores['sarcasm'] = 0.7
            details['sarcasm'] = (
                f'Маркеры сарказма {sarc_found} + позитивные слова, '
                f'но label = Negative'
            )
        elif has_pos:
            scores['sarcasm'] = 0.4
            details['sarcasm'] = (
                f'Маркеры сарказма {sarc_found} + позитивные слова'
            )

    # 5. Сленг
    slang_found = [m for m in SLANG_MARKERS if m in text]
    if slang_found:
        scores['slang'] = 0.6
        details['slang'] = f'Сленговые маркеры: {slang_found}'

    # 6. Отрицание
    neg_found = [m for m in NEGATION_MARKERS if m in text]
    if neg_found:
        has_pos = any(w in text for w in POSITIVE_WORDS)
        if has_pos:
            scores['negation'] = 0.5
            details['negation'] = (
                f'Отрицание {neg_found} + позитивные слова — '
                f'возможна инверсия тональности'
            )

    # 7. Имплицитная оценка
    if ensemble == 'Neutral' and actual in ('Positive', 'Negative'):
        if n_words < 10:
            scores['implicit'] = 0.55
            details['implicit'] = (
                f'Ансамбль дал Neutral, а label = {actual} — '
                f'оценка выражена имплицитно, текст короткий'
            )

    # 8. Домен
    dom_found = [m for m in DOMAIN_MARKERS if m in text]
    if dom_found:
        if not scores:
            scores['domain'] = 0.3
            details['domain'] = f'Доменные слова: {dom_found}'

    # 9. Проблемы перевода
    if 'translation_ok' in row and pd.notna(row['translation_ok']):
        ok = str(row['translation_ok']).strip().lower() in ('true', '1', 'yes')
        if not ok:
            scores['translation_issue'] = 0.8
            details['translation_issue'] = 'translation_ok = False — перевод мог исказить смысл'

    if not scores:
        return 'other', 'Не подошла ни одна эвристика — требуется ручной разбор', 0.0

    best_cat = max(scores.items(), key=lambda x: x[1])[0]
    return best_cat, details.get(best_cat, ''), scores[best_cat]


def auto_annotate(input_csv, output_csv, high_conf_threshold=0.7):
    """
    Автоматически размечает ошибки.
    Формирует:
    - errors_auto.csv     — ошибки с высокой уверенностью эвристики
    - errors_manual.csv   — ошибки, требующие ручной проверки
    - errors_detailed.csv — детальный разбор КАЖДОЙ ошибки
                            (тип, уточнение, кто ошибся, кто угадал)
    """
    df = pd.read_csv(input_csv, encoding='utf-8-sig')

    # Нормализация
    df['actual_norm'] = df['label'].apply(norm_label)
    df['ensemble_norm'] = df['ensemble_sentiment'].apply(norm_label)

    # Только ошибки ансамбля
    df = df[df['actual_norm'].notna() & (df['ensemble_norm'] != df['actual_norm'])].copy()

    print(f"Ошибок для разметки: {len(df)}")

    if len(df) == 0:
        print("[!] Ошибок не найдено — файлы не создаются.")
        return None, None

    # ---------- Автоклассификация ----------
    results = df.apply(explain_error, axis=1, result_type='expand')
    df['auto_category'] = results[0]
    df['error_detail'] = results[1]
    df['auto_confidence'] = results[2]

    # ---------- Кто ошибся / кто угадал ----------
    model_status = df.apply(get_model_status, axis=1, result_type='expand')
    df['models_wrong'] = model_status[0]
    df['models_correct'] = model_status[1]
    df['n_wrong'] = df['models_wrong'].apply(
        lambda s: len(s.split(', ')) if s else 0
    )
    df['n_correct'] = df['models_correct'].apply(
        lambda s: len(s.split(', ')) if s else 0
    )

    # ---------- Разделение по уверенности ----------
    high_conf = df[df['auto_confidence'] >= high_conf_threshold].copy()
    low_conf = df[df['auto_confidence'] < high_conf_threshold].copy()

    print(f"\nАвтоматически размечено (conf >= {high_conf_threshold}): {len(high_conf)}")
    print(f"Требуют ручной проверки: {len(low_conf)}")

    print(f"\nРаспределение типов ошибок:")
    print(df['auto_category'].value_counts().to_string())

    # ---------- Сохранение ----------
    high_conf.to_csv(output_csv.replace('.csv', '_auto.csv'),
                     index=False, encoding='utf-8-sig')
    low_conf.to_csv(output_csv.replace('.csv', '_manual.csv'),
                    index=False, encoding='utf-8-sig')

    # ---------- Детальный файл ----------
    detailed_cols = [
        'text', 'actual_norm', 'ensemble_norm',
        'auto_category', 'error_detail', 'auto_confidence',
        'models_wrong', 'n_wrong',
        'models_correct', 'n_correct',
    ]
    # добавляем остальные колонки, если они есть
    extra_cols = [c for c in ['translated_text', 'translation_ok',
                              'vader_sentiment', 'flair_sentiment',
                              'rubert_sentiment', 'roberta_sentiment',
                              'distilbert_sentiment',
                              'logistic_regression_sentiment',
                              'svm_sentiment', 'random_forest_sentiment']
                  if c in df.columns]
    detailed = df[detailed_cols + extra_cols].copy()
    detailed = detailed.sort_values(
        by=['auto_category', 'n_wrong'],
        ascending=[True, False]
    )
    detailed_path = output_csv.replace('.csv', '_detailed.csv')
    detailed.to_csv(detailed_path, index=False, encoding='utf-8-sig')

    # ---------- Краткая сводка по моделям ----------
    print("\n" + "="*60)
    print("КТО ЧАЩЕ ВСЕГО ОШИБАЕТСЯ (по всем ошибкам ансамбля)")
    print("="*60)
    wrong_counter = {m: 0 for m in MODEL_RU.values()}
    total = len(df)
    for s in df['models_wrong']:
        if not s:
            continue
        for name in s.split(', '):
            wrong_counter[name] = wrong_counter.get(name, 0) + 1
    for name, cnt in sorted(wrong_counter.items(), key=lambda x: -x[1]):
        if cnt > 0:
            print(f"  {name:15}: {cnt:4} ({cnt/total*100:5.1f}% от всех ошибок)")

    print(f"\n[OK] Авторазмеченные: {output_csv.replace('.csv', '_auto.csv')}")
    print(f"[OK] Для ручной проверки: {output_csv.replace('.csv', '_manual.csv')}")
    print(f"[OK] Детальный разбор: {detailed_path}")

    return high_conf, low_conf


if __name__ == "__main__":
    input_csv = "answer/отзывы(результат работы).csv"
    output_csv = "answer/errors_auto.csv"

    if os.path.exists(input_csv):
        auto_annotate(input_csv, output_csv)
    else:
        print(f"[X] Файл не найден: {input_csv}")