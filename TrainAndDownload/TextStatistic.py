#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ датасета label,text (POSITIVE / NEGATIVE / NEUTRAL).

Вход:  data/15k_ozon.csv
Выход: answer/review_analisis/*.png
"""

import re
import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

warnings.filterwarnings('ignore')

# ═════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═════════════════════════════════════════════════════════════
SCRIPT_DIR   = Path(__file__).resolve().parent          # ...\Diplom\TrainAndDownload
PROJECT_ROOT = SCRIPT_DIR.parent                        # ...\Diplom

INPUT_FILE  = PROJECT_ROOT / "data" / "15k_ozon_updated.csv"
OUTPUT_DIR  = PROJECT_ROOT / "answer" / "review_analisis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_ROWS = 1000     # None — весь файл

LABEL_ORDER  = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
LABEL_COLORS = {'NEGATIVE': '#e74c3c', 'NEUTRAL': '#f39c12', 'POSITIVE': '#2ecc71'}
LABEL_TO_RATING = {'NEGATIVE': 1, 'NEUTRAL': 3, 'POSITIVE': 5}

# Стиль
plt.style.use('default')
sns.set_palette("husl")


# ═════════════════════════════════════════════════════════════
# Загрузка
# ═════════════════════════════════════════════════════════════
def load_data(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path.resolve()}")
    df = pd.read_csv(path, nrows=max_rows)
    # Нормализуем метки
    df['label'] = df['label'].astype(str).str.strip().str.upper()
    # Отбрасываем строки с неизвестными метками
    df = df[df['label'].isin(LABEL_ORDER)].reset_index(drop=True)
    # Числовой эквивалент для корреляций
    df['rating'] = df['label'].map(LABEL_TO_RATING)
    return df


# ═════════════════════════════════════════════════════════════
# Текстовые признаки
# ═════════════════════════════════════════════════════════════
POSITIVE_EMOJIS = ['👍', '😊', '😍', '❤️', '👌', '🔥', '💯']
NEGATIVE_EMOJIS = ['👎', '😠', '😡', '💢', '🤬', '😞']

def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df['text'].astype(str)

    df['text_length']       = text.str.len()
    df['word_count']        = text.str.split().str.len()
    df['sentence_count']    = text.apply(lambda x: len(re.findall(r'[.!?]+', x)))
    df['exclamation_count'] = text.str.count('!')
    df['question_count']    = text.str.count(r'\?')
    df['capital_letters']   = text.apply(lambda x: sum(1 for c in x if c.isupper()))
    df['capital_ratio']     = df['capital_letters'] / df['text_length'].replace(0, 1)

    df['positive_emojis']   = text.apply(lambda x: sum(x.count(e) for e in POSITIVE_EMOJIS))
    df['negative_emojis']   = text.apply(lambda x: sum(x.count(e) for e in NEGATIVE_EMOJIS))

    # Средняя длина предложения
    df['avg_sentence_length'] = df['text_length'] / df['sentence_count'].replace(0, 1)
    return df


def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\d+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ═════════════════════════════════════════════════════════════
# Печать статистики
# ═════════════════════════════════════════════════════════════
def print_stats(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("ПЕРВИЧНЫЙ ОБЗОР ДАННЫХ")
    print("=" * 60)
    print(f"Размер датасета: {df.shape}")
    print(f"Колонки: {list(df.columns)}")
    print("\nТипы данных:")
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("РАСПРЕДЕЛЕНИЕ ТОНАЛЬНОСТИ")
    print("=" * 60)
    counts = df['label'].value_counts().reindex(LABEL_ORDER, fill_value=0)
    pct    = (counts / len(df) * 100).round(2)
    for lbl in LABEL_ORDER:
        print(f"  {lbl:10}: {counts[lbl]:5} ({pct[lbl]:5.2f}%)")

    print("\n" + "=" * 60)
    print("ТЕКСТОВЫЙ АНАЛИЗ")
    print("=" * 60)
    print(f"Средняя длина текста: {df['text_length'].mean():.0f} символов")
    print(f"Медианная длина: {df['text_length'].median():.0f} символов")
    print(f"Мин/Макс длина: {df['text_length'].min()} / {df['text_length'].max()}")
    print(f"Среднее количество слов: {df['word_count'].mean():.1f}")
    print(f"Медианное количество слов: {df['word_count'].median():.0f}")
    print(f"Среднее количество предложений: {df['sentence_count'].mean():.2f}")
    print(f"Среднее количество восклицательных знаков: {df['exclamation_count'].mean():.2f}")
    print(f"Среднее количество вопросительных знаков: {df['question_count'].mean():.2f}")
    print(f"Средняя доля заглавных букв: {df['capital_ratio'].mean():.3f}")

    print("\n" + "=" * 60)
    print("КОРРЕЛЯЦИИ")
    print("=" * 60)
    corr_len  = df['text_length'].corr(df['rating'])
    corr_word = df['word_count'].corr(df['rating'])
    print(f"Длина текста ↔ тональность (1/3/5): {corr_len:.3f}")
    print(f"Количество слов ↔ тональность: {corr_word:.3f}")

    print("\n" + "=" * 60)
    print("СМАЙЛИКИ И ДУБЛИКАТЫ")
    print("=" * 60)
    print(f"Позитивных смайликов: {df['positive_emojis'].sum()}")
    print(f"Негативных смайликов: {df['negative_emojis'].sum()}")
    dups = df['text'].duplicated(keep=False).sum()
    print(f"Дубликатов: {dups} ({dups/len(df)*100:.1f}%)")


# ═════════════════════════════════════════════════════════════
# Дашборд №1: базовые графики
# ═════════════════════════════════════════════════════════════
def plot_main_dashboard(df: pd.DataFrame, save_path: Path) -> None:
    fig = plt.figure(figsize=(16, 12))

    # 1. Распределение классов
    ax = fig.add_subplot(3, 3, 1)
    counts = df['label'].value_counts().reindex(LABEL_ORDER, fill_value=0)
    bars = ax.bar(LABEL_ORDER, counts.values,
                  color=[LABEL_COLORS[l] for l in LABEL_ORDER],
                  edgecolor='black')
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f'{v}', ha='center', va='bottom', fontweight='bold')
    ax.set_title('Распределение тональности')
    ax.set_ylabel('Количество отзывов')

    # 2. Круговая диаграмма
    ax = fig.add_subplot(3, 3, 2)
    counts.plot(kind='pie', autopct='%1.1f%%',
                colors=[LABEL_COLORS[l] for l in LABEL_ORDER], ax=ax)
    ax.set_ylabel('')
    ax.set_title('Доля каждого класса')

    # 3. Средняя длина текста по классам
    ax = fig.add_subplot(3, 3, 3)
    mean_len = df.groupby('label')['text_length'].mean().reindex(LABEL_ORDER)
    mean_len.plot(kind='barh',
                  color=[LABEL_COLORS[l] for l in LABEL_ORDER], ax=ax)
    ax.set_xlabel('Средняя длина (символы)')
    ax.set_title('Средняя длина текста по тональности')

    # 4. Гистограмма длины
    ax = fig.add_subplot(3, 3, 4)
    df['text_length'].hist(bins=30, color='purple', edgecolor='black', alpha=0.7, ax=ax)
    ax.axvline(df['text_length'].mean(), color='red', linestyle='--',
               label=f'Среднее: {df["text_length"].mean():.0f}')
    ax.set_xlabel('Длина текста')
    ax.set_ylabel('Частота')
    ax.set_title('Распределение длины отзывов')
    ax.legend()

    # 5. Boxplot длины по классам
    ax = fig.add_subplot(3, 3, 5)
    sns.boxplot(x='label', y='text_length', data=df,
                order=LABEL_ORDER, palette=LABEL_COLORS, ax=ax)
    ax.set_xlabel('Тональность')
    ax.set_ylabel('Длина текста')
    ax.set_title('Длина текста по тональности')

    # 6. Scatter: длина vs числовой эквивалент
    ax = fig.add_subplot(3, 3, 6)
    jitter = np.random.normal(0, 0.1, size=len(df))
    colors_scatter = df['label'].map(LABEL_COLORS)
    ax.scatter(df['text_length'], df['rating'] + jitter,
               c=colors_scatter, alpha=0.5,
               edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Длина текста')
    ax.set_ylabel('Тональность')
    ax.set_yticks([1, 3, 5])
    ax.set_yticklabels(['NEGATIVE', 'NEUTRAL', 'POSITIVE'])
    ax.set_title('Зависимость тональности от длины')

    # 7. Корреляционная матрица
    ax = fig.add_subplot(3, 3, 7)
    corr = df[['rating', 'text_length', 'word_count', 'sentence_count']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Корреляционная матрица')

    # 8. Stacked bar: тональность по категориям длины
    ax = fig.add_subplot(3, 3, 8)
    df['length_category'] = pd.cut(df['text_length'],
                                   bins=[0, 50, 100, 200, 500, 1000, 10000],
                                   labels=['<50', '50-100', '100-200',
                                           '200-500', '500-1000', '>1000'])
    pivot = pd.crosstab(df['length_category'], df['label']).reindex(columns=LABEL_ORDER, fill_value=0)
    pivot.plot(kind='bar', stacked=True,
               color=[LABEL_COLORS[l] for l in LABEL_ORDER], ax=ax)
    ax.set_xlabel('Длина текста (символы)')
    ax.set_ylabel('Количество')
    ax.set_title('Тональность по категориям длины')
    plt.setp(ax.get_xticklabels(), rotation=45)

    # 9. Кумулятивное распределение длины по классам
    ax = fig.add_subplot(3, 3, 9)
    for lbl in LABEL_ORDER:
        data = df[df['label'] == lbl]['text_length'].sort_values()
        if len(data) > 0:
            ax.plot(data.values,
                    np.arange(1, len(data) + 1) / len(data),
                    label=lbl, color=LABEL_COLORS[lbl], linewidth=2)
    ax.set_xlabel('Длина текста')
    ax.set_ylabel('Накопленная доля')
    ax.set_title('Кумулятивное распределение длины')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {save_path.name} сохранён")


# ═════════════════════════════════════════════════════════════
# Дашборд №2: дополнительные графики
# ═════════════════════════════════════════════════════════════
def plot_additional_dashboard(df: pd.DataFrame, save_path: Path) -> None:
    fig = plt.figure(figsize=(15, 15))

    # 1. Радар характеристик по классам
    ax = fig.add_subplot(3, 3, 1, projection='polar')
    metrics = ['length', 'words', 'sentences', 'exclamations', 'questions']
    cols    = ['text_length', 'word_count', 'sentence_count',
               'exclamation_count', 'question_count']
    groups = df.groupby('label')[cols].mean().reindex(LABEL_ORDER)
    # Нормализация 0..1
    norm = (groups - groups.min()) / (groups.max() - groups.min()).replace(0, 1)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for lbl in LABEL_ORDER:
        if lbl in norm.index:
            vals = norm.loc[lbl].tolist() + [norm.loc[lbl].tolist()[0]]
            ax.plot(angles, vals, 'o-', linewidth=2,
                    label=lbl, color=LABEL_COLORS[lbl])
            ax.fill(angles, vals, alpha=0.1, color=LABEL_COLORS[lbl])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title('Радар характеристик по тональности')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # 2. Сравнение характеристик по тональности
    ax = fig.add_subplot(3, 3, 2)
    stats = df.groupby('label')[['text_length', 'word_count', 'sentence_count']].mean().reindex(LABEL_ORDER)
    stats.plot(kind='bar', ax=ax,
               color=['#3498db', '#e67e22', '#9b59b6'])
    ax.set_xlabel('')
    ax.set_ylabel('Среднее значение')
    ax.set_title('Сравнение характеристик текста')
    ax.legend(loc='upper right')
    plt.setp(ax.get_xticklabels(), rotation=0)

    # 3. Гистограмма средней длины предложения
    ax = fig.add_subplot(3, 3, 3)
    df['avg_sentence_length'].hist(bins=30, color='teal',
                                   edgecolor='black', alpha=0.7, ax=ax)
    ax.axvline(df['avg_sentence_length'].mean(), color='red', linestyle='--',
               label=f'Среднее: {df["avg_sentence_length"].mean():.1f}')
    ax.set_xlabel('Средняя длина предложения')
    ax.set_ylabel('Частота')
    ax.set_title('Распределение длины предложений')
    ax.legend()

    # 4. Scatter с размером точек = число предложений
    ax = fig.add_subplot(3, 3, 4)
    sc = ax.scatter(df['text_length'], df['rating'],
                    c=df['word_count'], s=df['sentence_count'].clip(1, 20) * 20,
                    alpha=0.6, cmap='viridis',
                    edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Длина текста')
    ax.set_ylabel('Тональность')
    ax.set_yticks([1, 3, 5])
    ax.set_yticklabels(['NEGATIVE', 'NEUTRAL', 'POSITIVE'])
    ax.set_title('Длина × тональность × слова')
    plt.colorbar(sc, ax=ax, label='Количество слов')

    # 5. Heatmap длины vs тональность
    ax = fig.add_subplot(3, 3, 5)
    df['length_bin'] = pd.cut(df['text_length'], bins=10)
    heatmap = pd.crosstab(df['length_bin'].astype(str), df['label']).reindex(columns=LABEL_ORDER, fill_value=0)
    sns.heatmap(heatmap, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('Длина (категории)')
    ax.set_title('Распределение длины и тональности')

    # 6. Смайлики по классам
    ax = fig.add_subplot(3, 3, 6)
    emoji_df = df.groupby('label')[['positive_emojis', 'negative_emojis']].sum().reindex(LABEL_ORDER)
    emoji_df.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
    ax.set_xlabel('Тональность')
    ax.set_ylabel('Количество смайликов')
    ax.set_title('Использование смайликов по классам')
    ax.legend(['Позитивные', 'Негативные'])
    plt.setp(ax.get_xticklabels(), rotation=0)

    # 7. Облака слов NEGATIVE vs POSITIVE
    ax = fig.add_subplot(3, 3, 7)
    try:
        text_neg = ' '.join(df[df['label'] == 'NEGATIVE']['clean_text'].tolist())
        text_pos = ' '.join(df[df['label'] == 'POSITIVE']['clean_text'].tolist())
        if text_neg and text_pos:
            wc_neg = WordCloud(width=400, height=200, background_color='white',
                              colormap='Reds', max_words=80).generate(text_neg)
            wc_pos = WordCloud(width=400, height=200, background_color='white',
                              colormap='Greens', max_words=80).generate(text_pos)
            ax.imshow(np.hstack([wc_neg.to_array(), wc_pos.to_array()]),
                      interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Слева: NEGATIVE, справа: POSITIVE')
        else:
            ax.text(0.5, 0.5, 'Недостаточно данных', ha='center', va='center')
            ax.axis('off')
    except Exception as e:
        ax.text(0.5, 0.5, f'Ошибка облака: {e}', ha='center', va='center')
        ax.axis('off')

    # 8. KDE длины по классам
    ax = fig.add_subplot(3, 3, 8)
    for lbl in LABEL_ORDER:
        data = df[df['label'] == lbl]['text_length']
        if len(data) > 1:
            sns.kdeplot(data=data, label=lbl,
                        ax=ax, fill=True, color=LABEL_COLORS[lbl])
    ax.set_xlabel('Длина текста')
    ax.set_ylabel('Плотность')
    ax.set_title('Распределение длины по тональности')
    ax.legend()

    # 9. Boxplot восклицаний по классам
    ax = fig.add_subplot(3, 3, 9)
    sns.boxplot(x='label', y='exclamation_count', data=df,
                order=LABEL_ORDER, palette=LABEL_COLORS, ax=ax)
    ax.set_xlabel('Тональность')
    ax.set_ylabel('Количество «!»')
    ax.set_title('Восклицания по тональности')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {save_path.name} сохранён")


# ═════════════════════════════════════════════════════════════
# Облако слов по всем отзывам
# ═════════════════════════════════════════════════════════════
def plot_wordcloud(df: pd.DataFrame, save_path: Path) -> None:
    try:
        text_all = ' '.join(df['clean_text'].tolist())
        wc = WordCloud(width=1200, height=600, background_color='white',
                       colormap='viridis', max_words=120).generate(text_all)
        plt.figure(figsize=(14, 7))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('Облако слов из всех отзывов', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] {save_path.name} сохранён")
    except Exception as e:
        print(f"[warn] Не удалось создать облако слов: {e}")


# ═════════════════════════════════════════════════════════════
# Топ-слова по классам
# ═════════════════════════════════════════════════════════════
def print_top_words(df: pd.DataFrame, n: int = 10) -> None:
    print("\n" + "=" * 60)
    print("ТОП-СЛОВА ПО КЛАССАМ")
    print("=" * 60)
    for lbl in LABEL_ORDER:
        words = ' '.join(df[df['label'] == lbl]['clean_text']).split()
        if not words:
            print(f"\n{lbl}: нет данных")
            continue
        print(f"\n{lbl} — топ-{n}:")
        for word, count in Counter(words).most_common(n):
            print(f"  {word}: {count}")


# ═════════════════════════════════════════════════════════════
# Выводы
# ═════════════════════════════════════════════════════════════
def print_conclusions(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("КЛЮЧЕВЫЕ ВЫВОДЫ")
    print("=" * 60)
    counts = df['label'].value_counts(normalize=True) * 100
    neg = counts.get('NEGATIVE', 0)
    pos = counts.get('POSITIVE', 0)
    neu = counts.get('NEUTRAL', 0)

    print(f"\nВсего отзывов: {len(df)}")
    print(f"POSITIVE: {pos:.1f}%")
    print(f"NEUTRAL:  {neu:.1f}%")
    print(f"NEGATIVE: {neg:.1f}%")

    if neg > 50:
        print(f"\n⚠️ Доминируют негативные отзывы ({neg:.1f}%)")
    elif pos > 50:
        print(f"\n✅ Доминируют позитивные отзывы ({pos:.1f}%)")
    else:
        print(f"\n⚖️ Сбалансированное распределение классов")

    mean_len = df.groupby('label')['text_length'].mean()
    if 'NEGATIVE' in mean_len and 'POSITIVE' in mean_len:
        if mean_len['NEGATIVE'] > mean_len['POSITIVE']:
            print(f"📏 Негативные отзывы в среднем длиннее "
                  f"({mean_len['NEGATIVE']:.0f} vs {mean_len['POSITIVE']:.0f} симв.)")
        else:
            print(f"📏 Позитивные отзывы в среднем длиннее "
                  f"({mean_len['POSITIVE']:.0f} vs {mean_len['NEGATIVE']:.0f} симв.)")


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def main():
    print(f"Загрузка: {INPUT_FILE}")
    df = load_data(INPUT_FILE, MAX_ROWS)
    print(f"Загружено строк: {len(df)}")

    df = add_text_features(df)
    df['clean_text'] = df['text'].apply(clean_text)

    print_stats(df)
    print_top_words(df, n=10)

    plot_main_dashboard(df,       OUTPUT_DIR / "reviews_analysis.png")
    plot_additional_dashboard(df, OUTPUT_DIR / "additional_analysis.png")
    plot_wordcloud(df,            OUTPUT_DIR / "wordcloud.png")

    print_conclusions(df)

    print(f"\nВсе результаты в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()