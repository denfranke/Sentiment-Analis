# auto_error_analysis.py
# -*- coding: utf-8 -*-
"""
Автоматический анализ ошибок без ручной разметки.
Анализ ведётся по ОРИГИНАЛЬНОМУ РУССКОМУ тексту (колонка text).
Формат входного файла:
label,text,...,ensemble_sentiment,actual_sentiment,is_correct_*,translated_text,translation_ok
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

MODELS = ['vader', 'flair', 'rubert', 'roberta', 'distilbert',
          'logistic_regression', 'svm', 'random_forest']

# Русские стоп-слова (расширенный список)
RUSSIAN_STOPWORDS = [
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
    'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
    'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
    'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже',
    'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был',
    'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там',
    'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть',
    'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам',
    'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет',
    'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой',
    'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем',
    'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда',
    'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после',
    'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них',
    'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо',
    'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя',
    'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между',
    'это', 'всё', 'ещё', 'который', 'которая', 'которые', 'весь', 'вся',
    'также', 'такие', 'такое', 'эта', 'эти', 'бы', 'же', 'ли', 'ведь',
]


def norm_label(x):
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


def load_data(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # Истинная метка — из колонки label
    df['actual_norm'] = df['label'].apply(norm_label)
    # Предсказание ансамбля — из ensemble_sentiment
    df['ensemble_norm'] = df['ensemble_sentiment'].apply(norm_label)

    df = df[df['actual_norm'].notna()].copy()
    df['is_correct'] = df['ensemble_norm'] == df['actual_norm']

    # Анализ длины — по РУССКОМУ тексту
    df['text_ru'] = df['text'].fillna('').astype(str)
    df['len_chars'] = df['text_ru'].str.len()
    df['len_words'] = df['text_ru'].str.split().str.len()

    # Для TF-IDF — русский текст (без fallback на перевод)
    df['text_for_tfidf'] = df['text_ru']

    return df


def plot_confusion_matrix(df, output_dir='answer'):
    """Матрица ошибок ансамбля"""
    labels = ['Negative', 'Neutral', 'Positive']

    cm = confusion_matrix(df['actual_norm'], df['ensemble_norm'], labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar_kws={'label': 'Доля'})
    ax.set_xlabel('Предсказано', fontsize=12)
    ax.set_ylabel('Фактически', fontsize=12)
    ax.set_title('Матрица ошибок ансамбля (русский текст)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {output_dir}/confusion_matrix.png")


def plot_accuracy_by_length(df, output_dir='answer'):
    """Точность vs длина РУССКОГО текста (в словах)"""
    df['len_bin'] = pd.cut(df['len_words'],
                            bins=[0, 3, 7, 15, 30, 100, 10000],
                            labels=['1-3', '4-7', '8-15', '16-30', '31-100', '100+'])

    acc = df.groupby('len_bin', observed=True)['is_correct'].agg(['mean', 'count'])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(acc.index.astype(str), acc['mean'], color='#4ECDC4',
                  edgecolor='black')

    for bar, (idx, row) in zip(bars, acc.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{row["mean"]:.1%}\n(n={int(row["count"])})',
                ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Длина русского текста (слов)', fontsize=12)
    ax.set_ylabel('Точность ансамбля', fontsize=12)
    ax.set_title('Точность vs длина русского текста',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=df['is_correct'].mean(), color='red', linestyle='--',
               label=f'Средняя: {df["is_correct"].mean():.1%}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_by_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {output_dir}/accuracy_by_length.png")


def plot_accuracy_by_translation(df, output_dir='answer'):
    """Точность vs качество перевода (translation_ok) — вспомогательный график"""
    if 'translation_ok' not in df.columns:
        return

    df['translation_ok_bool'] = df['translation_ok'].astype(str).str.strip().str.lower().isin(
        ['true', '1', 'yes']
    )

    acc = df.groupby('translation_ok_bool')['is_correct'].agg(['mean', 'count'])

    if len(acc) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Перевод OK', 'Проблемы перевода']
    colors = ['#96CEB4', '#FF6B6B']
    bars = ax.bar(labels, acc['mean'].values, color=colors, edgecolor='black')

    for bar, (idx, row) in zip(bars, acc.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{row["mean"]:.1%}\n(n={int(row["count"])})',
                ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Точность ансамбля', fontsize=12)
    ax.set_title('Точность vs качество перевода (влияние на русский текст)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_by_translation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {output_dir}/accuracy_by_translation.png")


def plot_model_agreement(df, output_dir='answer'):
    """Точность vs согласие моделей"""
    model_cols = [f'{m}_sentiment' for m in MODELS if f'{m}_sentiment' in df.columns]

    from collections import Counter

    def count_agreement(row):
        preds = [str(row[c]).lower() for c in model_cols if pd.notna(row[c])]
        if not preds:
            return 0, 0
        most_common = Counter(preds).most_common(1)[0][1]
        return most_common, len(preds)

    agreement = df.apply(count_agreement, axis=1, result_type='expand')
    df['n_agree'] = agreement[0]
    df['n_models'] = agreement[1]
    df['agreement_ratio'] = df['n_agree'] / df['n_models'].replace(0, np.nan)

    df['agreement_bin'] = pd.cut(df['agreement_ratio'],
                                  bins=[0, 0.5, 0.75, 0.99, 1.01],
                                  labels=['<50%', '50-75%', '75-99%', '100%'])

    acc = df.groupby('agreement_bin', observed=True)['is_correct'].agg(['mean', 'count'])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(acc.index.astype(str), acc['mean'], color='#96CEB4',
                  edgecolor='black')

    for bar, (idx, row) in zip(bars, acc.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{row["mean"]:.1%}\n(n={int(row["count"])})',
                ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Согласие моделей', fontsize=12)
    ax.set_ylabel('Точность ансамбля', fontsize=12)
    ax.set_title('Точность vs согласие моделей (на русском тексте)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_by_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {output_dir}/accuracy_by_agreement.png")


def plot_model_correlation(df, output_dir='answer'):
    """Корреляция между моделями"""
    score_cols = [f'{m}_score' for m in MODELS if f'{m}_score' in df.columns]

    if len(score_cols) < 2:
        return

    corr = df[score_cols].corr()
    corr.columns = [c.replace('_score', '').replace('_', ' ').title() for c in corr.columns]
    corr.index = corr.columns

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title('Корреляция предсказаний моделей', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {output_dir}/model_correlation.png")


def top_words_in_errors(df, output_dir='answer', top_n=25):
    """
    Топ-слова в ОШИБОЧНЫХ РУССКИХ текстах.
    Сравниваем TF-IDF ошибок и правильных — показываем слова-триггеры ошибок.
    """
    errors = df[~df['is_correct']]
    correct = df[df['is_correct']]

    if len(errors) < 10:
        print("[WARNING] Слишком мало ошибок для анализа слов")
        return

    # Русские стоп-слова + пользовательские
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words=RUSSIAN_STOPWORDS,
    )

    try:
        X_err = vectorizer.fit_transform(errors['text_for_tfidf'])
        feature_names = vectorizer.get_feature_names_out()
        scores_err = X_err.mean(axis=0).A1

        # TF-IDF для правильных — чтобы вычесть "общие" слова
        if len(correct) >= 10:
            X_cor = vectorizer.transform(correct['text_for_tfidf'])
            scores_cor = X_cor.mean(axis=0).A1
        else:
            scores_cor = np.zeros_like(scores_err)

        # Разница: слова, которые чаще встречаются в ошибках
        diff = scores_err - scores_cor
        top_idx = diff.argsort()[::-1][:top_n]
        top_words = [(feature_names[i], diff[i]) for i in top_idx if diff[i] > 0]

        if not top_words:
            print("[WARNING] Не удалось выделить слова-триггеры ошибок")
            return

    except Exception as e:
        print(f"[WARNING] TF-IDF ошибка: {e}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    words, values = zip(*top_words)
    ax.barh(list(words)[::-1], list(values)[::-1], color='#FF6B6B')
    ax.set_xlabel('Разница TF-IDF (ошибки − правильные)', fontsize=12)
    ax.set_title(f'Топ-{len(top_words)} слов-триггеров ошибок (русский текст)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_words_in_errors.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {output_dir}/top_words_in_errors.png")

    # Дополнительно: сохраняем таблицу
    words_df = pd.DataFrame(top_words, columns=['word', 'tfidf_diff'])
    words_df.to_csv(f'{output_dir}/top_error_words.csv',
                    index=False, encoding='utf-8-sig')
    print(f"[OK] {output_dir}/top_error_words.csv")


def plot_errors_by_class(df, output_dir='answer'):
    """Доля ошибок по классам (label)"""
    labels = ['Negative', 'Neutral', 'Positive']
    stats = []
    for lab in labels:
        sub = df[df['actual_norm'] == lab]
        if len(sub) > 0:
            stats.append({
                'class': lab,
                'error_rate': 1 - sub['is_correct'].mean(),
                'count': len(sub)
            })
    if not stats:
        return
    stats_df = pd.DataFrame(stats)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(stats_df['class'], stats_df['error_rate'],
                  color=['#FF6B6B', '#FFE194', '#4ECDC4'], edgecolor='black')

    for bar, (_, row) in zip(bars, stats_df.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{row["error_rate"]:.1%}\n(n={int(row["count"])})',
                ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Доля ошибок', fontsize=12)
    ax.set_title('Доля ошибок по классам (label)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(0.5, stats_df['error_rate'].max() * 1.3))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/errors_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {output_dir}/errors_by_class.png")


def plot_error_pairs(df, output_dir='answer'):
    """
    Heatmap: какие пары (истина → предсказание) дают больше всего ошибок.
    Помогает понять, куда "съезжает" ансамбль на русском.
    """
    errors = df[~df['is_correct']]
    if len(errors) < 5:
        return

    pairs = errors.groupby(['actual_norm', 'ensemble_norm']).size().reset_index(name='count')
    pairs = pairs.sort_values('count', ascending=False)

    pivot = errors.pivot_table(
        index='actual_norm',
        columns='ensemble_norm',
        values='text',
        aggfunc='count',
        fill_value=0
    )
    # Убираем диагональ (правильные)
    for lab in pivot.index:
        if lab in pivot.columns:
            pivot.loc[lab, lab] = 0

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='Reds', ax=ax,
                cbar_kws={'label': 'Число ошибок'})
    ax.set_xlabel('Предсказано ансамблем', fontsize=12)
    ax.set_ylabel('Истинная метка (label)', fontsize=12)
    ax.set_title('Матрица ошибок (без диагонали) — русский текст',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_pairs.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {output_dir}/error_pairs.png")


def main():
    csv_path = "answer/отзывы(результат работы).csv"

    if not os.path.exists(csv_path):
        print(f"[X] Файл не найден: {csv_path}")
        return

    os.makedirs('answer', exist_ok=True)
    df = load_data(csv_path)

    print(f"Загружено: {len(df)} строк")
    print(f"Ошибок ансамбля: {(~df['is_correct']).sum()} "
          f"({(~df['is_correct']).mean()*100:.1f}%)")
    print(f"Средняя длина русского текста: {df['len_words'].mean():.1f} слов")

    # Основные графики
    plot_confusion_matrix(df)
    plot_error_pairs(df)
    plot_accuracy_by_length(df)
    plot_errors_by_class(df)
    plot_accuracy_by_translation(df)
    plot_model_agreement(df)
    plot_model_correlation(df)
    top_words_in_errors(df)

    # Сводная таблица
    print("\n" + "="*60)
    print("СВОДКА ДЛЯ СТАТЬИ (анализ русского текста)")
    print("="*60)
    print(f"Всего отзывов: {len(df)}")
    print(f"Точность ансамбля: {df['is_correct'].mean():.1%}")

    print("\nПо классам (label):")
    for label in ['Negative', 'Neutral', 'Positive']:
        subset = df[df['actual_norm'] == label]
        if len(subset) > 0:
            acc = subset['is_correct'].mean()
            print(f"  {label:10}: {acc:.1%} (n={len(subset)})")

    print("\nПо длине русского текста:")
    for bin_label in ['1-3', '4-7', '8-15', '16-30', '31-100', '100+']:
        subset = df[df['len_bin'] == bin_label]
        if len(subset) > 0:
            acc = subset['is_correct'].mean()
            print(f"  {bin_label:8} слов: {acc:.1%} (n={len(subset)})")

    # Топ частых ошибок
    print("\nТоп-5 пар ошибок (истина → предсказание):")
    errors = df[~df['is_correct']]
    if len(errors) > 0:
        pairs = (errors.groupby(['actual_norm', 'ensemble_norm'])
                 .size().reset_index(name='count')
                 .sort_values('count', ascending=False).head(5))
        for _, r in pairs.iterrows():
            print(f"  {r['actual_norm']:10} → {r['ensemble_norm']:10}: {r['count']}")


if __name__ == "__main__":
    main()