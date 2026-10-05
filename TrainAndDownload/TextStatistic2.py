import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля
plt.style.use('default')
sns.set_palette("husl")

# ============================================================
# НАСТРОЙКА ПУТЕЙ
# ============================================================
INPUT_FILE = 'data/15k_phone.csv'
OUTPUT_DIR = 'statya'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"📂 Входной файл: {os.path.abspath(INPUT_FILE)}")
print(f"💾 Папка результатов: {os.path.abspath(OUTPUT_DIR)}")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ Файл не найден: {os.path.abspath(INPUT_FILE)}")

# ============================================================
# ПАСТЕЛЬНАЯ ПАЛИТРА ДЛЯ ТОНАЛЬНОСТЕЙ
# ============================================================
colors_map = {
    'POSITIVE': '#66BB6A',   # мягкий зелёный
    'NEGATIVE': '#EF5350',   # мягкий красный
    'NEUTRAL':  '#BDBDBD',   # мягкий серый
}
# Цвета для круговой диаграммы (чуть насыщеннее, но не яркие)
pie_colors_map = {
    'POSITIVE': '#A5D6A7',
    'NEGATIVE': '#FFCDD2',
    'NEUTRAL':  '#E0E0E0',
}
# Цвета для облаков слов
wordcloud_colors = {
    'POSITIVE': 'Greens',
    'NEGATIVE': 'Reds',
    'NEUTRAL':  'Blues',
    'ALL':      'viridis',
}

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================
df = pd.read_csv(INPUT_FILE, encoding='utf-8', nrows=1000)

print("="*60)
print("ПЕРВИЧНЫЙ ОБЗОР ДАННЫХ")
print("="*60)
print(f"Исходные колонки: {list(df.columns)}")
print(f"Размер датасета: {df.shape}")

rename_map = {}
for col in df.columns:
    low = col.lower().strip()
    if low in ('sentiment', 'label', 'target', 'class'):
        rename_map[col] = 'label'
    elif low in ('review', 'text', 'comment', 'content'):
        rename_map[col] = 'text'
df = df.rename(columns=rename_map)

if 'label' not in df.columns or 'text' not in df.columns:
    if df.shape[1] == 2:
        df.columns = ['label', 'text']
    else:
        raise ValueError(f"Не удалось определить колонки: {list(df.columns)}")

df = df[['label', 'text']].copy()
df['label'] = df['label'].astype(str).str.strip().str.upper()
label_alias = {
    'POS': 'POSITIVE', 'NEG': 'NEGATIVE', 'NEU': 'NEUTRAL',
    '1': 'POSITIVE', '0': 'NEGATIVE',
}
df['label'] = df['label'].replace(label_alias)
df = df.dropna(subset=['text']).reset_index(drop=True)
df['text'] = df['text'].astype(str)

print(f"\nИтоговые колонки: {list(df.columns)}")
print(f"Типы данных:\n{df.dtypes}")
print("\nПервые 3 строки (обрезано):")
print(df.head(3).to_string(max_colwidth=80))

# ============================================================
# СТАТИСТИКА ПО ТОНАЛЬНОСТИ
# ============================================================
print("\n" + "="*60)
print("СТАТИСТИКА ПО ТОНАЛЬНОСТИ")
print("="*60)

label_distribution = df['label'].value_counts()
print("\nРаспределение тональности:")
print(label_distribution)

label_percentage = (df['label'].value_counts(normalize=True) * 100).round(2)
print("\nРаспределение тональности в процентах:")
print(label_percentage)

print("\nПропуски в данных:")
print(df.isnull().sum())

# ============================================================
# ТЕКСТОВЫЙ АНАЛИЗ
# ============================================================
print("\n" + "="*60)
print("ТЕКСТОВЫЙ АНАЛИЗ")
print("="*60)

df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['sentence_count'] = df['text'].apply(lambda x: len(re.findall(r'[.!?]+', x)))
df['exclamation_count'] = df['text'].str.count('!')
df['question_count'] = df['text'].str.count(r'\?')
df['capital_letters'] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()))
df['capital_ratio'] = df['capital_letters'] / df['text_length'].replace(0, 1)
df['avg_sentence_length'] = df['text_length'] / df['sentence_count'].replace(0, 1)

positive_emojis = ['👍', '😊', '😍', '❤️', '👌', '🔥', '🎉', '😀', '😃', '🙂']
negative_emojis = ['👎', '😠', '😡', '💢', '🤬', '😞', '😢', '😤', '🙁', '☹️']

df['positive_emojis'] = df['text'].apply(
    lambda x: sum(x.count(e) for e in positive_emojis))
df['negative_emojis'] = df['text'].apply(
    lambda x: sum(x.count(e) for e in negative_emojis))

print(f"Средняя длина текста: {df['text_length'].mean():.0f} символов")
print(f"Медианная длина текста: {df['text_length'].median():.0f} символов")
print(f"Минимальная длина: {df['text_length'].min()} символов")
print(f"Максимальная длина: {df['text_length'].max()} символов")
print(f"\nСреднее количество слов: {df['word_count'].mean():.1f}")
print(f"Медианное количество слов: {df['word_count'].median():.0f}")
print(f"\nСреднее количество предложений: {df['sentence_count'].mean():.2f}")
print(f"Среднее количество '!': {df['exclamation_count'].mean():.2f}")
print(f"Среднее количество '?': {df['question_count'].mean():.2f}")
print(f"Средняя доля заглавных букв: {df['capital_ratio'].mean():.3f}")
print(f"\nВсего позитивных смайликов: {df['positive_emojis'].sum()}")
print(f"Всего негативных смайликов: {df['negative_emojis'].sum()}")

# ============================================================
# СРАВНЕНИЕ МЕТРИК ПО ТОНАЛЬНОСТИ
# ============================================================
print("\n" + "="*60)
print("СРАВНЕНИЕ МЕТРИК ПО ТОНАЛЬНОСТИ")
print("="*60)

sentiment_stats = df.groupby('label')[
    ['text_length', 'word_count', 'sentence_count',
     'exclamation_count', 'question_count', 'capital_ratio',
     'positive_emojis', 'negative_emojis']
].mean().round(2)
print(sentiment_stats)

labels_order = [l for l in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
                if l in df['label'].unique()]

# ============================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ СОХРАНЕНИЯ
# ============================================================
def save_plot(fig, filename):
    """Сохранить отдельный график."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {filename}")

print("\n" + "="*60)
print("СОХРАНЕНИЕ ГРАФИКОВ ПО ОТДЕЛЬНОСТИ")
print("="*60)

# ============================================================
# ГРАФИК 1: Распределение тональности (bar)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
label_counts = df['label'].value_counts().reindex(labels_order)
bars = ax.bar(label_counts.index, label_counts.values,
              color=[colors_map.get(x, '#BDBDBD') for x in label_counts.index],
              edgecolor='gray', linewidth=1.2)
ax.set_xlabel('Тональность')
ax.set_ylabel('Количество отзывов')
ax.set_title('Распределение тональности')
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}',
            ha='center', va='bottom', fontsize=10)
save_plot(fig, '01_distribution_bar.png')

# ============================================================
# ГРАФИК 2: Круговая диаграмма тональности
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))
label_counts.plot(kind='pie', autopct='%1.1f%%',
                  colors=[pie_colors_map.get(x, '#E0E0E0')
                          for x in label_counts.index],
                  ax=ax, wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                  textprops={'fontsize': 11})
ax.set_ylabel('')
ax.set_title('Доля каждой тональности')
save_plot(fig, '02_distribution_pie.png')

# ============================================================
# ГРАФИК 3: Средняя длина текста по тональности
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
avg_len = df.groupby('label')['text_length'].mean().reindex(labels_order)
avg_len.plot(kind='bar',
             color=[colors_map.get(x, '#BDBDBD') for x in avg_len.index],
             edgecolor='gray', linewidth=1.2, ax=ax)
ax.set_xlabel('Тональность')
ax.set_ylabel('Средняя длина (символы)')
ax.set_title('Средняя длина текста по тональности')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for i, v in enumerate(avg_len.values):
    ax.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=10)
save_plot(fig, '03_avg_length.png')

# ============================================================
# ГРАФИК 4: Распределение длины текстов
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for label in labels_order:
    subset = df[df['label'] == label]['text_length']
    ax.hist(subset, bins=30, alpha=0.6, label=label,
            color=colors_map.get(label, '#BDBDBD'), edgecolor='gray')
ax.set_xlabel('Длина текста (символы)')
ax.set_ylabel('Частота')
ax.set_title('Распределение длины текстов')
ax.legend()
save_plot(fig, '04_length_hist.png')

# ============================================================
# ГРАФИК 5: Box plot длины
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='label', y='text_length', data=df, ax=ax,
            order=labels_order, palette=colors_map,
            hue='label', legend=False)
ax.set_xlabel('Тональность')
ax.set_ylabel('Длина текста')
ax.set_title('Box plot длины текста по тональности')
save_plot(fig, '05_length_boxplot.png')

# ============================================================
# ГРАФИК 6: Scatter длина vs слова
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for label in labels_order:
    subset = df[df['label'] == label]
    ax.scatter(subset['text_length'], subset['word_count'],
               alpha=0.4, s=8, label=label,
               c=colors_map.get(label, '#BDBDBD'), edgecolors='none')
ax.set_xlabel('Длина текста')
ax.set_ylabel('Количество слов')
ax.set_title('Длина vs Количество слов')
ax.legend()
save_plot(fig, '06_scatter_length_words.png')

# ============================================================
# ГРАФИК 7: Тепловая карта корреляций
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7))
corr_matrix = df[['text_length', 'word_count', 'sentence_count',
                  'exclamation_count', 'question_count', 'capital_ratio']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            ax=ax, fmt='.2f', annot_kws={'size': 9})
ax.set_title('Корреляционная матрица')
save_plot(fig, '07_correlation_heatmap.png')

# ============================================================
# ГРАФИК 8: Сравнение метрик (норм.)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
metrics_for_compare = ['word_count', 'sentence_count',
                       'exclamation_count', 'question_count']
stats_norm = df.groupby('label')[metrics_for_compare].mean().reindex(labels_order)
stats_norm = (stats_norm - stats_norm.min()) / \
             (stats_norm.max() - stats_norm.min()).replace(0, 1)
stats_norm.plot(kind='bar', ax=ax, colormap='Set3', edgecolor='gray')
ax.set_xlabel('Тональность')
ax.set_ylabel('Нормализованное значение')
ax.set_title('Сравнение метрик по тональности (норм.)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(loc='upper right', fontsize=9)
save_plot(fig, '08_metrics_comparison_norm.png')

# ============================================================
# ГРАФИК 9: Средняя длина предложения
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
avg_sent = df.groupby('label')['avg_sentence_length'].mean().reindex(labels_order)
avg_sent.plot(kind='bar',
              color=[colors_map.get(x, '#BDBDBD') for x in avg_sent.index],
              edgecolor='gray', linewidth=1.2, ax=ax)
ax.set_xlabel('Тональность')
ax.set_ylabel('Средняя длина предложения')
ax.set_title('Средняя длина предложения по тональности')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
save_plot(fig, '09_avg_sentence_length.png')

# ============================================================
# ГРАФИК 10: Распределение "!"
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for label in labels_order:
    subset = df[df['label'] == label]['exclamation_count']
    ax.hist(subset, bins=15, alpha=0.6, label=label,
            color=colors_map.get(label, '#BDBDBD'), edgecolor='gray')
ax.set_xlabel('Количество "!"')
ax.set_ylabel('Частота')
ax.set_title('Распределение восклицательных знаков')
ax.legend()
save_plot(fig, '10_exclamation_hist.png')

# ============================================================
# ГРАФИК 11: Распределение доли заглавных
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for label in labels_order:
    subset = df[df['label'] == label]['capital_ratio']
    ax.hist(subset, bins=20, alpha=0.6, label=label,
            color=colors_map.get(label, '#BDBDBD'), edgecolor='gray')
ax.set_xlabel('Доля заглавных букв')
ax.set_ylabel('Частота')
ax.set_title('Распределение доли заглавных букв')
ax.legend()
save_plot(fig, '11_capital_ratio_hist.png')

# ============================================================
# ГРАФИК 12: Violin plot длины
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x='label', y='text_length', data=df, ax=ax,
               order=labels_order, palette=colors_map,
               hue='label', legend=False, inner='quartile')
ax.set_xlabel('Тональность')
ax.set_ylabel('Длина текста')
ax.set_title('Violin plot длины текста')
save_plot(fig, '12_length_violin.png')

# ============================================================
# ГРАФИК 13: Радарная диаграмма
# ============================================================
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='polar')
metrics_radar = ['length', 'words', 'sentences', 'exclamations', 'questions']
rating_groups = df.groupby('label')[
    ['text_length', 'word_count', 'sentence_count',
     'exclamation_count', 'question_count']
].mean().reindex(labels_order)
normalized = (rating_groups - rating_groups.min()) / \
             (rating_groups.max() - rating_groups.min()).replace(0, 1)
angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
angles += angles[:1]
for label in labels_order:
    if label in normalized.index:
        values = normalized.loc[label].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=label,
                color=colors_map.get(label, '#BDBDBD'))
        ax.fill(angles, values, alpha=0.15,
                color=colors_map.get(label, '#BDBDBD'))
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_radar)
ax.set_title('Радар характеристик текста по тональности')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
save_plot(fig, '13_radar_metrics.png')

# ============================================================
# ГРАФИК 14: Распределение средней длины предложений
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
df['avg_sentence_length'].hist(bins=30, color='#90CAF9',
                                edgecolor='gray', alpha=0.8, ax=ax)
ax.axvline(df['avg_sentence_length'].mean(), color='#EF5350',
           linestyle='--',
           label=f'Среднее: {df["avg_sentence_length"].mean():.1f}')
ax.set_xlabel('Средняя длина предложения')
ax.set_ylabel('Частота')
ax.set_title('Распределение средней длины предложений')
ax.legend()
save_plot(fig, '14_avg_sentence_distribution.png')

# ============================================================
# ГРАФИК 15: Scatter с цветом по предложениям
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['text_length'], df['word_count'],
                     c=df['sentence_count'], s=12,
                     alpha=0.5, cmap='viridis', edgecolors='none')
ax.set_xlabel('Длина текста')
ax.set_ylabel('Количество слов')
ax.set_title('Длина vs Слова (цвет = предложения)')
plt.colorbar(scatter, ax=ax, label='Предложения')
save_plot(fig, '15_scatter_color_sentences.png')

# ============================================================
# ГРАФИК 16: Тепловая карта длины и тональности
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7))
df['length_bin'] = pd.cut(df['text_length'], bins=10)
heatmap_2d = pd.crosstab(df['length_bin'].astype(str), df['label'])
heatmap_2d = heatmap_2d.loc[(heatmap_2d.sum(axis=1) > 0)]
sns.heatmap(heatmap_2d, annot=True, fmt='d', cmap='Blues', ax=ax,
            annot_kws={'size': 8})
ax.set_xlabel('Тональность')
ax.set_ylabel('Длина (категории)')
ax.set_title('Распределение длины и тональности')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, fontsize=8)
save_plot(fig, '16_length_sentiment_heatmap.png')

# ============================================================
# ГРАФИК 17: Сравнение характеристик текста
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
stats_plot = df.groupby('label')[
    ['text_length', 'word_count', 'sentence_count']
].mean().reindex(labels_order)
stats_plot.plot(kind='bar', ax=ax, colormap='Pastel1',
                edgecolor='gray')
ax.set_xlabel('Тональность')
ax.set_ylabel('Среднее значение')
ax.set_title('Характеристики текста по тональности')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(loc='upper right', fontsize=9)
save_plot(fig, '17_text_characteristics.png')

# ============================================================
# ГРАФИК 18: KDE длины по тональности
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for label in labels_order:
    data = df[df['label'] == label]['text_length']
    if not data.empty:
        sns.kdeplot(data=data, label=label, ax=ax,
                    fill=True, alpha=0.3,
                    color=colors_map.get(label, '#BDBDBD'))
ax.set_xlabel('Длина текста')
ax.set_ylabel('Плотность')
ax.set_title('KDE длины текста по тональности')
ax.legend()
save_plot(fig, '18_length_kde.png')

# ============================================================
# ГРАФИК 19: Смайлики по тональности
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))
emoji_by_label = df.groupby('label')[
    ['positive_emojis', 'negative_emojis']
].sum().reindex(labels_order)
emoji_by_label.plot(kind='bar', ax=ax,
                    color=['#A5D6A7', '#FFCDD2'],
                    edgecolor='gray')
ax.set_xlabel('Тональность')
ax.set_ylabel('Количество смайликов')
ax.set_title('Смайлики по тональности')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(['Позитивные', 'Негативные'])
save_plot(fig, '19_emojis_by_sentiment.png')

# ============================================================
# ГРАФИК 20: Кумулятивное распределение
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for label in labels_order:
    subset = df[df['label'] == label]['text_length'].sort_values()
    if not subset.empty:
        cum = np.arange(1, len(subset)+1) / len(subset) * 100
        ax.plot(subset.values, cum, label=label, linewidth=2,
                color=colors_map.get(label, '#BDBDBD'))
ax.set_xlabel('Длина текста')
ax.set_ylabel('Накопленный %')
ax.set_title('Кумулятивное распределение длины')
ax.legend()
ax.grid(True, alpha=0.3)
save_plot(fig, '20_cumulative_length.png')


# ============================================================
# ГРАФИК 21: Распределение средней длины отзывов
# (аналог Рисунка 2.22)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))
ax.hist(df['text_length'], bins=40, color='#80CBC4',
        edgecolor='black', alpha=0.85)
ax.axvline(df['text_length'].mean(), color='red',
           linestyle='--', linewidth=2,
           label=f'Среднее: {df["text_length"].mean():.1f}')
ax.set_xlabel('Средняя длина отзыва')
ax.set_ylabel('Частота')
ax.set_title('Распределение средней длины отзывов')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
save_plot(fig, '21_avg_review_length_distribution.png')

# ============================================================
# ГРАФИК 22: Зависимость тональности от длины текста
# (адаптация Рисунка 2.23 — вместо рейтинга тональность)
# ============================================================
# Преобразуем тональность в числа: POSITIVE=1, NEUTRAL=0, NEGATIVE=-1
sentiment_to_num = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
df['sentiment_num'] = df['label'].map(sentiment_to_num)

fig, ax = plt.subplots(figsize=(11, 7))

# Jitter по оси Y, чтобы точки не сливались
np.random.seed(42)
jitter = np.random.normal(0, 0.06, size=len(df))

# Рисуем точки по каждой тональности своим цветом
for label in labels_order:
    subset = df[df['label'] == label]
    jitter_subset = np.random.normal(0, 0.06, size=len(subset))
    ax.scatter(subset['text_length'],
               subset['sentiment_num'] + jitter_subset,
               alpha=0.4, s=15,
               c=colors_map.get(label, '#BDBDBD'),
               edgecolors='black', linewidth=0.3,
               label=label)

ax.set_yticks([-1, 0, 1])
ax.set_yticklabels(['NEGATIVE', 'NEUTRAL', 'POSITIVE'])
ax.set_xlabel('Длина текста')
ax.set_ylabel('Тональность')
ax.set_title('Зависимость тональности от длины текста')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.25)
save_plot(fig, '22_sentiment_vs_length.png')

# ============================================================
# ЧАСТОТНЫЙ АНАЛИЗ
# ============================================================
print("\n" + "="*60)
print("ЧАСТОТНЫЙ АНАЛИЗ СЛОВ")
print("="*60)

stop_words = set([
    'и','в','во','не','что','он','на','я','с','со','как','а','то','все','она',
    'так','его','но','да','ты','к','у','же','вы','за','бы','по','только','ее',
    'мне','было','вот','от','меня','еще','нет','о','из','ему','теперь','когда',
    'даже','ну','вдруг','ли','если','уже','или','ни','быть','был','него','до',
    'вас','нибудь','опять','уж','вам','ведь','там','потом','себя','ничего','ей',
    'может','они','тут','где','есть','надо','ней','для','мы','тебя','их','чем',
    'была','сам','чтоб','без','будто','чего','раз','тоже','себе','под','будет',
    'ж','тогда','кто','этот','того','потому','этого','какой','совсем','ним',
    'здесь','этом','один','почти','мой','тем','чтобы','нее','были','куда',
    'зачем','всех','никогда','можно','при','наконец','два','об','другой','хоть',
    'после','над','больше','тот','через','эти','нас','про','всего','них',
    'какая','много','разве','три','эту','моя','впрочем','хорошо','свою','этой',
    'перед','иногда','лучше','чуть','том','нельзя','такой','им','более',
    'всегда','конечно','всю','между','это','эта','эти','весь','вся','сама',
    'этих','этому','этим','этими','который','которая','которое','которые',
    'которого','которой','которых','которым','которыми','всё','свой','своя',
    'своё','свои','своего','своей','своих','своим','своими','та','те','ту',
    'того','тому','тем','теми','тех','the','a','an','and','or','of','to','is',
    'it','in','this','that','was','for','with','as','on','at','by','be','are',
    'but','not','you','have','from','they','his','her','she','he','we','i'
])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_top_words(texts, n=15):
    all_words = ' '.join(texts).split()
    filtered = [w for w in all_words if w not in stop_words and len(w) > 2]
    return Counter(filtered).most_common(n)

df['clean_text'] = df['text'].apply(clean_text)

for label in labels_order:
    print(f"\nТоп-15 слов в отзывах [{label}]:")
    for word, count in get_top_words(df[df['label'] == label]['clean_text']):
        print(f"  {word}: {count}")

# ============================================================
# ТОП-20 СЛОВ ПО КЛАССАМ — отдельные графики
# ============================================================
print("\n" + "="*60)
print("ТОП-20 СЛОВ ПО ТОНАЛЬНОСТИ (отдельные графики)")
print("="*60)

for i, label in enumerate(labels_order, 21):
    fig, ax = plt.subplots(figsize=(9, 8))
    top = get_top_words(df[df['label'] == label]['clean_text'], 20)
    if top:
        words, counts = zip(*top)
        ax.barh(range(len(words)), counts,
                color=colors_map.get(label, '#BDBDBD'),
                edgecolor='gray')
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Частота')
        ax.set_title(f'Топ-20 слов: {label}')
    save_plot(fig, f'{i}_top20_words_{label.lower()}.png')

# ============================================================
# ОБЛАКА СЛОВ — ОБЩЕЕ И ПО ТОНАЛЬНОСТЯМ
# ============================================================
print("\n" + "="*60)
print("ОБЛАКА СЛОВ")
print("="*60)

# --- Общее облако слов ---
all_text = ' '.join(df['clean_text'])
if all_text.strip():
    fig, ax = plt.subplots(figsize=(14, 8))
    try:
        wc_all = WordCloud(width=1400, height=800,
                           background_color='white',
                           colormap='viridis',
                           max_words=200,
                           stopwords=stop_words,
                           collocations=False).generate(all_text)
        ax.imshow(wc_all, interpolation='bilinear')
        ax.set_title('Общее облако слов (все отзывы)', fontsize=16)
        ax.axis('off')
    except Exception as e:
        ax.text(0.5, 0.5, f'Ошибка: {e}', ha='center', va='center')
        ax.axis('off')
    save_plot(fig, '30_wordcloud_ALL.png')

# --- Облака по тональностям ---
for i, label in enumerate(labels_order, 31):
    text = ' '.join(df[df['label'] == label]['clean_text'])
    if not text.strip():
        continue
    fig, ax = plt.subplots(figsize=(12, 7))
    try:
        wc = WordCloud(width=1200, height=700,
                       background_color='white',
                       colormap=wordcloud_colors.get(label, 'viridis'),
                       max_words=150,
                       stopwords=stop_words,
                       collocations=False).generate(text)
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(f'Облако слов: {label}', fontsize=16)
        ax.axis('off')
    except Exception as e:
        ax.text(0.5, 0.5, f'Ошибка: {e}', ha='center', va='center')
        ax.axis('off')
    save_plot(fig, f'{i}_wordcloud_{label.lower()}.png')

# ============================================================
# ВЫВОДЫ
# ============================================================
print("\n" + "="*60)
print("КЛЮЧЕВЫЕ ВЫВОДЫ")
print("="*60)

total = len(df)
print(f"\n📊 Всего отзывов: {total}")
for label in labels_order:
    pct = (df['label'] == label).mean() * 100
    print(f"  {label}: {pct:.1f}%")

print(f"\n📏 Средняя длина отзыва: {df['text_length'].mean():.0f} символов")
print(f"📝 Среднее количество слов: {df['word_count'].mean():.1f}")

longest_idx = df['text_length'].idxmax()
shortest_idx = df['text_length'].idxmin()
print(f"\n📖 Самый длинный отзыв: {df.loc[longest_idx, 'text_length']} символов "
      f"(тональность: {df.loc[longest_idx, 'label']})")
print(f"📄 Самый короткий отзыв: {df.loc[shortest_idx, 'text_length']} символов "
      f"(тональность: {df.loc[shortest_idx, 'label']})")

print("\n" + "-"*40)
print("СРАВНЕНИЕ КЛАССОВ:")
print("-"*40)
for label in labels_order:
    subset = df[df['label'] == label]
    print(f"\n{label}:")
    print(f"  Средняя длина: {subset['text_length'].mean():.0f} символов")
    print(f"  Среднее слов: {subset['word_count'].mean():.1f}")
    print(f"  Среднее предложений: {subset['sentence_count'].mean():.1f}")
    print(f"  Среднее '!': {subset['exclamation_count'].mean():.2f}")
    print(f"  Средняя доля заглавных: {subset['capital_ratio'].mean():.3f}")
    print(f"  Позитивных смайликов: {subset['positive_emojis'].sum()}")
    print(f"  Негативных смайликов: {subset['negative_emojis'].sum()}")

duplicates = df['text'].duplicated(keep=False)
print(f"\n🔁 Дубликатов текстов: {duplicates.sum()} "
      f"({(duplicates.sum()/len(df)*100):.1f}%)")

processed_path = os.path.join(OUTPUT_DIR, 'processed_reviews.csv')
df.to_csv(processed_path, index=False, encoding='utf-8')
print(f"\n💾 Обработанный датасет сохранён: {processed_path}")

# ============================================================
# ИТОГОВАЯ СВОДКА
# ============================================================
print("\n" + "="*60)
print("СОЗДАННЫЕ ФАЙЛЫ")
print("="*60)
files = sorted(os.listdir(OUTPUT_DIR))
for f in files:
    full = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(full) / 1024
    print(f"  📄 {f} ({size_kb:.1f} КБ)")
print(f"\n✅ Всего файлов: {len(files)}")


