import os
import json
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
INPUT_FILE = 'data/old1/test_with_sentiment_1500.jsonl'     # JSONL: {"sample": [...], "sentiment": "..."}
OUTPUT_DIR = 'dialog_analysis'
NROWS = 15000                         # анализируем только первые 15 000 диалогов

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"📂 Входной файл: {os.path.abspath(INPUT_FILE)}")
print(f"💾 Папка результатов: {os.path.abspath(OUTPUT_DIR)}")
print(f"🔢 Ограничение строк (nrows): {NROWS}")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ Файл не найден: {os.path.abspath(INPUT_FILE)}")

# ============================================================
# СООТВЕТСТВИЕ МЕТОК ТОНАЛЬНОСТИ
# ============================================================
LABEL_MAP = {
    '0': 'NEUTRAL', '1': 'POSITIVE', '2': 'NEGATIVE',
    'NEUTRAL': 'NEUTRAL', 'POSITIVE': 'POSITIVE', 'NEGATIVE': 'NEGATIVE',
    'NEU': 'NEUTRAL', 'POS': 'POSITIVE', 'NEG': 'NEGATIVE',
}

colors_map = {
    'POSITIVE': '#66BB6A',
    'NEGATIVE': '#EF5350',
    'NEUTRAL':  '#BDBDBD',
}
pie_colors_map = {
    'POSITIVE': '#A5D6A7',
    'NEGATIVE': '#FFCDD2',
    'NEUTRAL':  '#E0E0E0',
}
wordcloud_colors = {
    'POSITIVE': 'Greens',
    'NEGATIVE': 'Reds',
    'NEUTRAL':  'Blues',
    'ALL':      'viridis',
}

# ============================================================
# ЗАГРУЗКА ДИАЛОГОВ (JSONL)
# ============================================================
def load_dialogs(path, nrows=None):
    """Читает JSONL. Возвращает список словарей."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if nrows is not None and i >= nrows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # ожидаем поля 'sample' (список реплик) и 'sentiment'
            sample = obj.get('sample') or obj.get('dialog') or obj.get('messages')
            sentiment = obj.get('sentiment') or obj.get('label')
            if sample is None or sentiment is None:
                continue
            # нормализуем реплики в строки
            turns = []
            for t in sample:
                if isinstance(t, dict):
                    # на случай {"role": "...", "text": "..."}
                    t = t.get('text') or t.get('content') or ''
                turns.append(str(t))
            records.append({
                'turns': turns,
                'sentiment_raw': str(sentiment).strip(),
            })
    return records

print("\n" + "=" * 60)
print("ЗАГРУЗКА ДИАЛОГОВ")
print("=" * 60)

raw = load_dialogs(INPUT_FILE, nrows=NROWS)
print(f"Загружено диалогов: {len(raw)}")

if not raw:
    raise ValueError("Не удалось загрузить ни одного диалога. Проверьте формат файла.")

# ============================================================
# ПОДГОТОВКА DATAFRAME
# ============================================================
df = pd.DataFrame(raw)
df['label'] = df['sentiment_raw'].str.upper().map(LABEL_MAP).fillna(
    df['sentiment_raw'].str.upper()
)
df = df.dropna(subset=['turns']).reset_index(drop=True)
df = df[df['turns'].apply(lambda x: len(x) > 0)].reset_index(drop=True)

print(f"Диалогов после фильтрации: {len(df)}")
print("Распределение тональности:")
print(df['label'].value_counts())

# ============================================================
# ПРИЗНАКИ УРОВНЯ ДИАЛОГА
# ============================================================
def dialog_features(turns):
    n_turns = len(turns)
    turn_lengths = [len(t) for t in turns]
    word_counts = [len(t.split()) for t in turns]
    q_turns = sum(1 for t in turns if '?' in t)
    e_turns = sum(1 for t in turns if '!' in t)
    q_marks = sum(t.count('?') for t in turns)
    e_marks = sum(t.count('!') for t in turns)
    ellipsis = sum(t.count('...') + t.count('…') for t in turns)
    capital = sum(1 for t in turns for c in t if c.isupper())
    total_len = sum(turn_lengths)
    pos_emojis = ['👍', '😊', '😍', '❤️', '👌', '🔥', '🎉', '😀', '😃', '🙂']
    neg_emojis = ['👎', '😠', '😡', '💢', '🤬', '😞', '😢', '😤', '🙁', '☹️']
    pos_e = sum(sum(t.count(e) for e in pos_emojis) for t in turns)
    neg_e = sum(sum(t.count(e) for e in neg_emojis) for t in turns)
    return pd.Series({
        'turn_count': n_turns,
        'dialog_length': total_len,
        'avg_turn_length': total_len / n_turns if n_turns else 0,
        'max_turn_length': max(turn_lengths) if turn_lengths else 0,
        'min_turn_length': min(turn_lengths) if turn_lengths else 0,
        'avg_words_per_turn': np.mean(word_counts) if word_counts else 0,
        'question_turns': q_turns,
        'exclamation_turns': e_turns,
        'question_turns_ratio': q_turns / n_turns if n_turns else 0,
        'exclamation_turns_ratio': e_turns / n_turns if n_turns else 0,
        'question_marks': q_marks,
        'exclamation_marks': e_marks,
        'ellipsis_count': ellipsis,
        'capital_ratio': capital / total_len if total_len else 0,
        'positive_emojis': pos_e,
        'negative_emojis': neg_e,
    })

feat = df['turns'].apply(dialog_features)
df = pd.concat([df, feat], axis=1)

print("\n" + "=" * 60)
print("ОБЗОР ДАННЫХ")
print("=" * 60)
print(f"Колонки: {list(df.columns)}")
print(df.head(3)[['turn_count', 'dialog_length', 'avg_turn_length',
                  'question_turns_ratio', 'label']].to_string())

print("\nСтатистика по метрикам диалога:")
print(df[['turn_count', 'dialog_length', 'avg_turn_length',
          'avg_words_per_turn', 'question_turns_ratio',
          'exclamation_turns_ratio', 'capital_ratio']].describe().round(2))

print("\nСредние метрики по тональности:")
print(df.groupby('label')[
    ['turn_count', 'dialog_length', 'avg_turn_length', 'avg_words_per_turn',
     'question_turns_ratio', 'exclamation_turns_ratio', 'capital_ratio']
].mean().round(2))

labels_order = [l for l in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
                if l in df['label'].unique()]

# ============================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ СОХРАНЕНИЯ
# ============================================================
def save_plot(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {filename}")

print("\n" + "=" * 60)
print("ПОСТРОЕНИЕ ГРАФИКОВ")
print("=" * 60)

# ============================================================
# ГРАФИК 1: Распределение тональности (bar)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
label_counts = df['label'].value_counts().reindex(labels_order)
bars = ax.bar(label_counts.index, label_counts.values,
              color=[colors_map.get(x, '#BDBDBD') for x in label_counts.index],
              edgecolor='gray', linewidth=1.2)
ax.set_xlabel('Тональность диалога')
ax.set_ylabel('Количество диалогов')
ax.set_title('Распределение тональности диалогов')
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., h, f'{int(h)}',
            ha='center', va='bottom', fontsize=10)
save_plot(fig, '01_sentiment_bar.png')

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
ax.set_title('Доля тональностей диалогов')
save_plot(fig, '02_sentiment_pie.png')

# ============================================================
# ГРАФИК 3: Распределение числа реплик в диалоге
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for label in labels_order:
    subset = df[df['label'] == label]['turn_count']
    ax.hist(subset, bins=30, alpha=0.6, label=label,
            color=colors_map.get(label, '#BDBDBD'), edgecolor='gray')
ax.set_xlabel('Число реплик в диалоге')
ax.set_ylabel('Частота')
ax.set_title('Распределение числа реплик по тональности')
ax.legend()
save_plot(fig, '03_turn_count_hist.png')

# ============================================================
# ГРАФИК 4: Распределение длины диалога (символы)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for label in labels_order:
    subset = df[df['label'] == label]['dialog_length']
    ax.hist(subset, bins=40, alpha=0.6, label=label,
            color=colors_map.get(label, '#BDBDBD'), edgecolor='gray')
ax.set_xlabel('Длина диалога (символы)')
ax.set_ylabel('Частота')
ax.set_title('Распределение длины диалога по тональности')
ax.legend()
save_plot(fig, '04_dialog_length_hist.png')

# ============================================================
# ГРАФИК 5: Box plot длины диалога
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='label', y='dialog_length', data=df, ax=ax,
            order=labels_order, palette=colors_map,
            hue='label', legend=False)
ax.set_xlabel('Тональность')
ax.set_ylabel('Длина диалога')
ax.set_title('Box plot длины диалога по тональности')
save_plot(fig, '05_dialog_length_boxplot.png')

# ============================================================
# ГРАФИК 6: Violin plot числа реплик
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x='label', y='turn_count', data=df, ax=ax,
               order=labels_order, palette=colors_map,
               hue='label', legend=False, inner='quartile')
ax.set_xlabel('Тональность')
ax.set_ylabel('Число реплик')
ax.set_title('Violin plot числа реплик по тональности')
save_plot(fig, '06_turn_count_violin.png')

# ============================================================
# ГРАФИК 7: Средняя длина реплики по тональности
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
avg_turn = df.groupby('label')['avg_turn_length'].mean().reindex(labels_order)
avg_turn.plot(kind='bar',
              color=[colors_map.get(x, '#BDBDBD') for x in avg_turn.index],
              edgecolor='gray', linewidth=1.2, ax=ax)
ax.set_xlabel('Тональность')
ax.set_ylabel('Средняя длина реплики (символы)')
ax.set_title('Средняя длина реплики по тональности')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for i, v in enumerate(avg_turn.values):
    ax.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=10)
save_plot(fig, '07_avg_turn_length.png')

# ============================================================
# ГРАФИК 8: Доля "вопросных" реплик по тональности
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
q_ratio = df.groupby('label')['question_turns_ratio'].mean().reindex(labels_order)
q_ratio.plot(kind='bar',
             color=[colors_map.get(x, '#BDBDBD') for x in q_ratio.index],
             edgecolor='gray', linewidth=1.2, ax=ax)
ax.set_xlabel('Тональность')
ax.set_ylabel('Доля реплик с "?"')
ax.set_title('Доля вопросных реплик по тональности')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for i, v in enumerate(q_ratio.values):
    ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
save_plot(fig, '08_question_ratio.png')

# ============================================================
# ГРАФИК 9: Доля "восклицательных" реплик по тональности
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
e_ratio = df.groupby('label')['exclamation_turns_ratio'].mean().reindex(labels_order)
e_ratio.plot(kind='bar',
             color=[colors_map.get(x, '#BDBDBD') for x in e_ratio.index],
             edgecolor='gray', linewidth=1.2, ax=ax)
ax.set_xlabel('Тональность')
ax.set_ylabel('Доля реплик с "!"')
ax.set_title('Доля восклицательных реплик по тональности')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for i, v in enumerate(e_ratio.values):
    ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
save_plot(fig, '09_exclamation_ratio.png')

# ============================================================
# ГРАФИК 10: Scatter число реплик vs длина диалога
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for label in labels_order:
    subset = df[df['label'] == label]
    ax.scatter(subset['turn_count'], subset['dialog_length'],
               alpha=0.4, s=10, label=label,
               c=colors_map.get(label, '#BDBDBD'), edgecolors='none')
ax.set_xlabel('Число реплик')
ax.set_ylabel('Длина диалога')
ax.set_title('Число реплик vs Длина диалога')
ax.legend()
save_plot(fig, '10_scatter_turns_length.png')

# ============================================================
# ГРАФИК 11: Тепловая карта корреляций метрик
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7))
corr_cols = ['turn_count', 'dialog_length', 'avg_turn_length',
             'avg_words_per_turn', 'question_turns_ratio',
             'exclamation_turns_ratio', 'ellipsis_count', 'capital_ratio']
corr_matrix = df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            ax=ax, fmt='.2f', annot_kws={'size': 8})
ax.set_title('Корреляционная матрица метрик диалога')
save_plot(fig, '11_correlation_heatmap.png')

# ============================================================
# ГРАФИК 12: KDE длины диалога по тональности
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for label in labels_order:
    data = df[df['label'] == label]['dialog_length']
    if not data.empty:
        sns.kdeplot(data=data, label=label, ax=ax, fill=True, alpha=0.3,
                    color=colors_map.get(label, '#BDBDBD'))
ax.set_xlabel('Длина диалога')
ax.set_ylabel('Плотность')
ax.set_title('KDE длины диалога по тональности')
ax.legend()
save_plot(fig, '12_dialog_length_kde.png')

# ============================================================
# ГРАФИК 13: Радар метрик по тональности
# ============================================================
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='polar')
radar_cols = ['turn_count', 'dialog_length', 'avg_turn_length',
              'question_turns_ratio', 'exclamation_turns_ratio']
radar_names = ['Реплики', 'Длина', 'Ср. реплика', 'Вопросы', 'Восклицания']
rating = df.groupby('label')[radar_cols].mean().reindex(labels_order)
norm = (rating - rating.min()) / (rating.max() - rating.min()).replace(0, 1)
angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
angles += angles[:1]
for label in labels_order:
    if label in norm.index:
        vals = norm.loc[label].tolist() + [norm.loc[label].tolist()[0]]
        ax.plot(angles, vals, 'o-', linewidth=2, label=label,
                color=colors_map.get(label, '#BDBDBD'))
        ax.fill(angles, vals, alpha=0.15,
                color=colors_map.get(label, '#BDBDBD'))
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_names)
ax.set_title('Радар метрик диалога по тональности')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
save_plot(fig, '13_radar_metrics.png')

# ============================================================
# ГРАФИК 14: Кумулятивное распределение длины диалога
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for label in labels_order:
    subset = df[df['label'] == label]['dialog_length'].sort_values()
    if not subset.empty:
        cum = np.arange(1, len(subset) + 1) / len(subset) * 100
        ax.plot(subset.values, cum, label=label, linewidth=2,
                color=colors_map.get(label, '#BDBDBD'))
ax.set_xlabel('Длина диалога')
ax.set_ylabel('Накопленный %')
ax.set_title('Кумулятивное распределение длины диалога')
ax.legend()
ax.grid(True, alpha=0.3)
save_plot(fig, '14_cumulative_length.png')

# ============================================================
# ГРАФИК 15: Распределение многоточий ("...") по тональности
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for label in labels_order:
    subset = df[df['label'] == label]['ellipsis_count']
    ax.hist(subset, bins=15, alpha=0.6, label=label,
            color=colors_map.get(label, '#BDBDBD'), edgecolor='gray')
ax.set_xlabel('Количество "..." в диалоге')
ax.set_ylabel('Частота')
ax.set_title('Распределение многоточий по тональности')
ax.legend()
save_plot(fig, '15_ellipsis_hist.png')

# ============================================================
# ГРАФИК 16: Смайлики по тональности
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))
emoji_by_label = df.groupby('label')[
    ['positive_emojis', 'negative_emojis']
].sum().reindex(labels_order)
emoji_by_label.plot(kind='bar', ax=ax,
                    color=['#A5D6A7', '#FFCDD2'], edgecolor='gray')
ax.set_xlabel('Тональность')
ax.set_ylabel('Количество смайликов')
ax.set_title('Смайлики по тональности')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(['Позитивные', 'Негативные'])
save_plot(fig, '16_emojis_by_sentiment.png')

# ============================================================
# ГРАФИК 17: Средняя длина первой и последней реплики
# ============================================================
df['first_turn_len'] = df['turns'].apply(lambda x: len(x[0]))
df['last_turn_len'] = df['turns'].apply(lambda x: len(x[-1]))
fig, ax = plt.subplots(figsize=(9, 6))
comparison = df.groupby('label')[['first_turn_len', 'last_turn_len']].mean().reindex(labels_order)
comparison.plot(kind='bar', ax=ax, colormap='Pastel2', edgecolor='gray')
ax.set_xlabel('Тональность')
ax.set_ylabel('Средняя длина (символы)')
ax.set_title('Первая vs последняя реплика по тональности')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(['Первая реплика', 'Последняя реплика'])
save_plot(fig, '17_first_last_turn.png')

# ============================================================
# ЧАСТОТНЫЙ АНАЛИЗ
# ============================================================
print("\n" + "=" * 60)
print("ЧАСТОТНЫЙ АНАЛИЗ СЛОВ В РЕПЛИКАХ")
print("=" * 60)

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

df['clean_text'] = df['turns'].apply(lambda turns: ' '.join(clean_text(t) for t in turns))

for label in labels_order:
    print(f"\nТоп-15 слов в диалогах [{label}]:")
    for word, count in get_top_words(df[df['label'] == label]['clean_text']):
        print(f"  {word}: {count}")

# ============================================================
# ТОП-20 СЛОВ ПО КЛАССАМ
# ============================================================
print("\n" + "=" * 60)
print("ТОП-20 СЛОВ ПО ТОНАЛЬНОСТИ (отдельные графики)")
print("=" * 60)

for i, label in enumerate(labels_order, 21):
    fig, ax = plt.subplots(figsize=(9, 8))
    top = get_top_words(df[df['label'] == label]['clean_text'], 20)
    if top:
        words, counts = zip(*top)
        ax.barh(range(len(words)), counts,
                color=colors_map.get(label, '#BDBDBD'), edgecolor='gray')
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Частота')
        ax.set_title(f'Топ-20 слов: {label}')
    save_plot(fig, f'{i}_top20_words_{label.lower()}.png')

# ============================================================
# ОБЛАКА СЛОВ
# ============================================================
print("\n" + "=" * 60)
print("ОБЛАКА СЛОВ")
print("=" * 60)

all_text = ' '.join(df['clean_text'])
if all_text.strip():
    fig, ax = plt.subplots(figsize=(14, 8))
    try:
        wc_all = WordCloud(width=1400, height=800, background_color='white',
                           colormap='viridis', max_words=200,
                           stopwords=stop_words, collocations=False).generate(all_text)
        ax.imshow(wc_all, interpolation='bilinear')
        ax.set_title('Общее облако слов (все диалоги)', fontsize=16)
        ax.axis('off')
    except Exception as e:
        ax.text(0.5, 0.5, f'Ошибка: {e}', ha='center', va='center')
        ax.axis('off')
    save_plot(fig, '30_wordcloud_ALL.png')

for i, label in enumerate(labels_order, 31):
    text = ' '.join(df[df['label'] == label]['clean_text'])
    if not text.strip():
        continue
    fig, ax = plt.subplots(figsize=(12, 7))
    try:
        wc = WordCloud(width=1200, height=700, background_color='white',
                       colormap=wordcloud_colors.get(label, 'viridis'),
                       max_words=150, stopwords=stop_words,
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
print("\n" + "=" * 60)
print("КЛЮЧЕВЫЕ ВЫВОДЫ")
print("=" * 60)

total = len(df)
print(f"\n📊 Всего диалогов: {total}")
for label in labels_order:
    pct = (df['label'] == label).mean() * 100
    print(f"  {label}: {pct:.1f}%")

print(f"\n💬 Среднее число реплик: {df['turn_count'].mean():.1f}")
print(f"📏 Средняя длина диалога: {df['dialog_length'].mean():.0f} символов")
print(f"✂️ Средняя длина реплики: {df['avg_turn_length'].mean():.0f} символов")
print(f"❓ Средняя доля вопросных реплик: {df['question_turns_ratio'].mean():.2f}")
print(f"❗ Средняя доля восклицательных реплик: {df['exclamation_turns_ratio'].mean():.2f}")
print(f"… Среднее число многоточий: {df['ellipsis_count'].mean():.2f}")

print("\n" + "-" * 40)
print("СРАВНЕНИЕ КЛАССОВ:")
print("-" * 40)
for label in labels_order:
    subset = df[df['label'] == label]
    print(f"\n{label}:")
    print(f"  Среднее реплик: {subset['turn_count'].mean():.1f}")
    print(f"  Средняя длина диалога: {subset['dialog_length'].mean():.0f}")
    print(f"  Средняя длина реплики: {subset['avg_turn_length'].mean():.0f}")
    print(f"  Доля вопросных реплик: {subset['question_turns_ratio'].mean():.2f}")
    print(f"  Доля восклицательных: {subset['exclamation_turns_ratio'].mean():.2f}")
    print(f"  Многоточий (всего): {subset['ellipsis_count'].sum()}")

# ============================================================
# СОХРАНЕНИЕ ОБРАБОТАННЫХ ДАННЫХ
# ============================================================
processed = df[['label', 'turn_count', 'dialog_length', 'avg_turn_length',
                'avg_words_per_turn', 'question_turns', 'question_turns_ratio',
                'exclamation_turns', 'exclamation_turns_ratio',
                'ellipsis_count', 'capital_ratio',
                'positive_emojis', 'negative_emojis']].copy()
processed['turns_joined'] = df['turns'].apply(lambda x: ' ||| '.join(x))
processed_path = os.path.join(OUTPUT_DIR, 'processed_dialogs.csv')
processed.to_csv(processed_path, index=False, encoding='utf-8')
print(f"\n💾 Обработанный датасет сохранён: {processed_path}")

# ============================================================
# ИТОГОВАЯ СВОДКА
# ============================================================
print("\n" + "=" * 60)
print("СОЗДАННЫЕ ФАЙЛЫ")
print("=" * 60)
files = sorted(os.listdir(OUTPUT_DIR))
for f in files:
    full = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(full) / 1024
    print(f"  📄 {f} ({size_kb:.1f} КБ)")
print(f"\n✅ Всего файлов: {len(files)}")