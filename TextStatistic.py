import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля для графиков
plt.style.use('default')
sns.set_palette("husl")

# Загружаем данные
df = pd.read_csv('data/отзывы.csv',nrows=1000)

print("="*50)
print("ПЕРВИЧНЫЙ ОБЗОР ДАННЫХ")
print("="*50)

# Основная информация о датасете
print(f"Размер датасета: {df.shape}")
print(f"Колонки: {list(df.columns)}")
print(f"\nТипы данных:")
print(df.dtypes)

print("\nПервые 5 строк:")
print(df.head())

print("\n" + "="*50)
print("СТАТИСТИКА ПО РЕЙТИНГАМ")
print("="*50)

# Базовая статистика по рейтингам
print(f"Средний рейтинг: {df['rating'].mean():.2f}")
print(f"Медианный рейтинг: {df['rating'].median():.2f}")
print(f"Мода рейтинга: {df['rating'].mode()[0]}")
print(f"Стандартное отклонение: {df['rating'].std():.2f}")
print(f"Минимальный рейтинг: {df['rating'].min()}")
print(f"Максимальный рейтинг: {df['rating'].max()}")

print("\nРаспределение рейтингов:")
rating_distribution = df['rating'].value_counts().sort_index()
print(rating_distribution)

print("\nРаспределение рейтингов в процентах:")
rating_percentage = (df['rating'].value_counts(normalize=True) * 100).sort_index()
print(rating_percentage.round(2))

print("\n" + "="*50)
print("СТАТИСТИКА ПО АВТОРАМ")
print("="*50)

# Анализ авторов
print(f"Уникальных авторов: {df['author'].nunique()}")
print(f"Всего отзывов: {len(df)}")

print("\nТоп-5 самых активных авторов:")
top_authors = df['author'].value_counts().head()
print(top_authors)

# Средний рейтинг по авторам (только для авторов с несколькими отзывами)
authors_with_multiple = df.groupby('author').filter(lambda x: len(x) > 1)
if not authors_with_multiple.empty:
    print("\nСредний рейтинг по авторам (минимум 2 отзыва):")
    author_avg_rating = authors_with_multiple.groupby('author')['rating'].mean().round(2)
    print(author_avg_rating.head())

print("\n" + "="*50)
print("ТЕКСТОВЫЙ АНАЛИЗ")
print("="*50)

# Длина текстов
df['text_length'] = df['text'].astype(str).str.len()
df['word_count'] = df['text'].astype(str).str.split().str.len()

print(f"Средняя длина текста: {df['text_length'].mean():.0f} символов")
print(f"Медианная длина текста: {df['text_length'].median():.0f} символов")
print(f"Минимальная длина: {df['text_length'].min()} символов")
print(f"Максимальная длина: {df['text_length'].max()} символов")

print(f"\nСреднее количество слов: {df['word_count'].mean():.1f}")
print(f"Медианное количество слов: {df['word_count'].median():.0f}")

print("\n" + "="*50)
print("КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("="*50)

# Корреляция между длиной текста и рейтингом
correlation = df['text_length'].corr(df['rating'])
print(f"Корреляция между длиной текста и рейтингом: {correlation:.3f}")

print("\n" + "="*50)
print("ВИЗУАЛИЗАЦИЯ ДАННЫХ")
print("="*50)

# Создаем фигуру с несколькими графиками
fig = plt.figure(figsize=(16, 12))

# 1. Распределение рейтингов
ax1 = plt.subplot(3, 3, 1)
rating_counts = df['rating'].value_counts().sort_index()
bars = ax1.bar(rating_counts.index, rating_counts.values, color='skyblue', edgecolor='navy')
ax1.set_xlabel('Рейтинг')
ax1.set_ylabel('Количество отзывов')
ax1.set_title('Распределение рейтингов')
ax1.set_xticks([1, 2, 3, 4, 5])
# Добавляем значения на столбцы
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

# 2. Круговая диаграмма рейтингов
ax2 = plt.subplot(3, 3, 2)
colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
df['rating'].value_counts().sort_index().plot(
    kind='pie', autopct='%1.1f%%', colors=colors, ax=ax2
)
ax2.set_ylabel('')
ax2.set_title('Доля каждого рейтинга')

# 3. Средний рейтинг по авторам (топ-10)
ax3 = plt.subplot(3, 3, 3)
author_ratings = df.groupby('author')['rating'].mean().sort_values(ascending=False).head(10)
author_ratings.plot(kind='barh', color='coral', ax=ax3)
ax3.set_xlabel('Средний рейтинг')
ax3.set_title('Топ-10 авторов по среднему рейтингу')

# 4. Распределение длины текстов
ax4 = plt.subplot(3, 3, 4)
df['text_length'].hist(bins=20, color='purple', edgecolor='black', alpha=0.7, ax=ax4)
ax4.set_xlabel('Длина текста (символы)')
ax4.set_ylabel('Частота')
ax4.set_title('Распределение длины отзывов')

# 5. Box plot рейтингов по длине текста
ax5 = plt.subplot(3, 3, 5)
# Создаем категории длины текста
df['length_category'] = pd.cut(df['text_length'], 
                                bins=[0, 50, 100, 200, 500, 1000],
                                labels=['Очень короткие', 'Короткие', 'Средние', 'Длинные', 'Очень длинные'])
# Убираем NaN для boxplot
df_box = df.dropna(subset=['length_category'])
sns.boxplot(x='length_category', y='rating', data=df_box, ax=ax5)
ax5.set_xlabel('Категория длины')
ax5.set_ylabel('Рейтинг')
ax5.set_title('Рейтинги по категориям длины')
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

# 6. Scatter plot длина vs рейтинг
ax6 = plt.subplot(3, 3, 6)
# Добавляем небольшой шум для лучшей визуализации
jitter = np.random.normal(0, 0.1, size=len(df))
ax6.scatter(df['text_length'], df['rating'] + jitter, alpha=0.5, c='green', edgecolors='black', linewidth=0.5)
ax6.set_xlabel('Длина текста')
ax6.set_ylabel('Рейтинг')
ax6.set_title('Зависимость рейтинга от длины текста')
ax6.set_yticks([1, 2, 3, 4, 5])

# 7. Тепловая карта корреляций
ax7 = plt.subplot(3, 3, 7)
corr_matrix = df[['rating', 'text_length', 'word_count']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax7)
ax7.set_title('Корреляционная матрица')

# 8. Количество отзывов по рейтингам и авторам
ax8 = plt.subplot(3, 3, 8)
pivot_table = pd.crosstab(df['author'].head(20), df['rating'])
pivot_table.plot(kind='bar', stacked=True, ax=ax8, colormap='viridis')
ax8.set_xlabel('Автор')
ax8.set_ylabel('Количество отзывов')
ax8.set_title('Распределение рейтингов по авторам (топ-20)')
ax8.legend(title='Рейтинг')
plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 9. Cumulative распределение рейтингов
ax9 = plt.subplot(3, 3, 9)
rating_cumsum = df['rating'].value_counts().sort_index().cumsum()
ax9.plot(rating_cumsum.index, rating_cumsum.values, marker='o', linewidth=2, markersize=8)
ax9.set_xlabel('Рейтинг')
ax9.set_ylabel('Накопленная частота')
ax9.set_title('Кумулятивное распределение')
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('answer/review_analisis/reviews_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nГрафики сохранены в файл 'reviews_analysis.png'")

print("\n" + "="*50)
print("ДЕТАЛЬНЫЙ АНАЛИЗ ТЕКСТОВ")
print("="*50)

# Функция для очистки текста
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Применяем очистку
df['clean_text'] = df['text'].apply(clean_text)

# Анализ часто встречающихся слов по рейтингам
print("\nТоп-10 слов в позитивных отзывах (рейтинг 4-5):")
positive_words = ' '.join(df[df['rating'] >= 4]['clean_text']).split()
positive_counter = Counter(positive_words)
for word, count in positive_counter.most_common(10):
    print(f"  {word}: {count}")

print("\nТоп-10 слов в негативных отзывах (рейтинг 1-2):")
negative_words = ' '.join(df[df['rating'] <= 2]['clean_text']).split()
negative_counter = Counter(negative_words)
for word, count in negative_counter.most_common(10):
    print(f"  {word}: {count}")

# Облако слов для всех отзывов
try:
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap='viridis',
                         max_words=100).generate(' '.join(df['clean_text']))
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Облако слов из всех отзывов', fontsize=16)
    plt.tight_layout()
    plt.savefig('answer/review_analisis/wordcloud.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nОблако слов сохранено в 'wordcloud.png'")
except:
    print("Не удалось создать облако слов (возможно, недостаточно данных)")

print("\n" + "="*50)
print("ВЫВОДЫ И РЕКОМЕНДАЦИИ")
print("="*50)

# Анализ тональности на основе ключевых слов
positive_keywords = ['удобный', 'вовремя', 'спасибо', 'отлично', 'хорошо', 'нравится', '👍']
negative_keywords = ['плохо', 'ужас', 'глюк', 'проблем', 'опоздание', 'обман', '👎']

def check_sentiment(text):
    text = str(text).lower()
    pos_count = sum(1 for word in positive_keywords if word in text)
    neg_count = sum(1 for word in negative_keywords if word in text)
    if pos_count > neg_count:
        return 'Позитивный'
    elif neg_count > pos_count:
        return 'Негативный'
    else:
        return 'Нейтральный'

df['sentiment'] = df['text'].apply(check_sentiment)

print("\nРаспределение тональности (по ключевым словам):")
sentiment_dist = df['sentiment'].value_counts()
print(sentiment_dist)
print(f"\nПроцентное соотношение:")
sentiment_percent = df['sentiment'].value_counts(normalize=True) * 100
print(sentiment_percent.round(1))

# Основные выводы
print("\n" + "-"*30)
print("КЛЮЧЕВЫЕ ВЫВОДЫ:")
print("-"*30)

# Вывод 1: Общая оценка
avg_rating = df['rating'].mean()
if avg_rating >= 4:
    print(f"✅ Общая оценка высокая ({avg_rating:.2f}/5)")
elif avg_rating >= 3:
    print(f"⚠️ Общая оценка средняя ({avg_rating:.2f}/5)")
else:
    print(f"❌ Общая оценка низкая ({avg_rating:.2f}/5)")

# Вывод 2: Проблемные места
negative_percent = (df['rating'] <= 2).mean() * 100
if negative_percent > 30:
    print(f"⚠️ Высокий процент негативных отзывов ({negative_percent:.1f}%)")
    
# Вывод 3: Активность пользователей
if df['author'].nunique() < len(df) * 0.7:
    print(f"📊 Много повторяющихся авторов, возможно, нужна проверка на накрутки")

print(f"\n📝 Всего проанализировано отзывов: {len(df)}")
print(f"👥 Уникальных авторов: {df['author'].nunique()}")







df['text_length'] = df['text'].astype(str).str.len()
df['word_count'] = df['text'].astype(str).str.split().str.len()
df['sentence_count'] = df['text'].astype(str).apply(lambda x: len(re.findall(r'[.!?]+', x)))
df['exclamation_count'] = df['text'].astype(str).str.count('!')
df['question_count'] = df['text'].astype(str).str.count(r'\?')
df['capital_letters'] = df['text'].astype(str).apply(lambda x: sum(1 for c in x if c.isupper()))
df['capital_ratio'] = df['capital_letters'] / df['text_length'].replace(0, 1)

print("="*60)
print("ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ ДЛЯ АНАЛИЗА")
print("="*60)

# Создаем фигуру с сеткой 3x3 (9 графиков)
fig = plt.figure(figsize=(15, 15))

# 1. Радарная диаграмма характеристик текста по рейтингам
ax1 = plt.subplot(3, 3, 1, projection='polar')
# Подготавливаем данные для радара
metrics = ['length', 'words', 'sentences', 'exclamations', 'questions']
rating_groups = df.groupby('rating')[['text_length', 'word_count', 'sentence_count', 
                                      'exclamation_count', 'question_count']].mean()

# Нормализуем данные
normalized_data = (rating_groups - rating_groups.min()) / (rating_groups.max() - rating_groups.min())

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

for rating in [1, 2, 3, 4, 5]:
    if rating in normalized_data.index:
        values = normalized_data.loc[rating].tolist()
        values += values[:1]
        ax1.plot(angles, values, 'o-', linewidth=2, label=f'Рейтинг {rating}')
        ax1.fill(angles, values, alpha=0.1)

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(metrics)
ax1.set_title('Радар характеристик текста по рейтингам')
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))


# 3. Распределение средней длины предложений
ax3 = plt.subplot(3, 3, 3)
df['avg_sentence_length'] = df['text_length'] / df['sentence_count'].replace(0, 1)
df['avg_sentence_length'].hist(bins=30, color='teal', edgecolor='black', alpha=0.7, ax=ax3)
ax3.set_xlabel('Средняя длина предложения')
ax3.set_ylabel('Частота')
ax3.set_title('Распределение средней длины предложений')
ax3.axvline(df['avg_sentence_length'].mean(), color='red', linestyle='--', 
            label=f'Среднее: {df["avg_sentence_length"].mean():.1f}')
ax3.legend()

# 4. Диаграмма рассеяния (длина vs рейтинг)
ax4 = plt.subplot(3, 3, 4)
scatter = ax4.scatter(df['text_length'], df['rating'], 
                      c=df['word_count'], s=df['sentence_count']*20, 
                      alpha=0.6, cmap='viridis', edgecolors='black', linewidth=0.5)
ax4.set_xlabel('Длина текста')
ax4.set_ylabel('Рейтинг')
ax4.set_title('Длина vs Рейтинг')
plt.colorbar(scatter, ax=ax4, label='Количество слов')

# 5. Тепловая карта распределения длины и рейтинга
ax5 = plt.subplot(3, 3, 5)
# Создаем категории для тепловой карты
df['length_bin'] = pd.cut(df['text_length'], bins=10)
df['rating_bin'] = pd.cut(df['rating'], bins=5, labels=[1,2,3,4,5])

heatmap_2d = pd.crosstab(df['length_bin'].astype(str), df['rating_bin'])
sns.heatmap(heatmap_2d, annot=True, fmt='d', cmap='Blues', ax=ax5)
ax5.set_xlabel(' ')
ax5.set_ylabel('Длина текста (категории)')
ax5.set_title('Распределение длины и рейтинга')

# 6. Сравнение характеристик текста по тональности
ax6 = plt.subplot(3, 3, 2)
df['sentiment'] = pd.cut(df['rating'], bins=[0, 2, 3, 5], labels=['Негативные', 'Нейтральные', 'Позитивные'])

sentiment_stats = df.groupby('sentiment')[['text_length', 'word_count', 'sentence_count']].mean()
sentiment_stats.plot(kind='bar', ax=ax6, colormap='Set3')
ax6.set_xlabel(' ')
ax6.set_ylabel('Среднее значение')
ax6.set_title('Сравнение характеристик текста по тональности')
ax6.legend(loc='upper right')
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=0)

# 7. Сравнение облаков слов (рейтинг 1 vs 5)
ax7 = plt.subplot(3, 3, 7)
try:
    # Берем отзывы с рейтингом 1 и 5 для сравнения
    text_1 = ' '.join(df[df['rating'] == 1]['text'].astype(str).tolist())
    text_5 = ' '.join(df[df['rating'] == 5]['text'].astype(str).tolist())
    
    # Создаем два облака слов рядом
    if text_1 and text_5:
        wordcloud_1 = WordCloud(width=400, height=200, background_color='white', colormap='Reds').generate(text_1)
        wordcloud_5 = WordCloud(width=400, height=200, background_color='white', colormap='Greens').generate(text_5)
        
        ax7.imshow(np.hstack([wordcloud_1.to_array(), wordcloud_5.to_array()]), interpolation='bilinear')
        ax7.axis('off')
        ax7.set_title('Слева: рейтинг 1, Справа: рейтинг 5')
    else:
        ax7.text(0.5, 0.5, 'Недостаточно данных\nдля облака слов', ha='center', va='center')
        ax7.axis('off')
except:
    ax7.text(0.5, 0.5, 'Ошибка создания облака слов', ha='center', va='center')
    ax7.axis('off')

# 8. Распределение длины по рейтингам (KDE)
ax8 = plt.subplot(3, 3, 8)
for i, rating in enumerate([1, 2, 3, 4, 5], 1):
    if rating in df['rating'].values:
        data = df[df['rating'] == rating]['text_length']
        if not data.empty:
            sns.kdeplot(data=data, label=f'Рейтинг {rating}', ax=ax8, shade=True)
ax8.set_xlabel('Длина текста')
ax8.set_ylabel('Плотность')
ax8.set_title('Распределение длины по рейтингам')
ax8.legend()

# 9. Использование смайликов по рейтингам
ax9 = plt.subplot(3, 3, 9)
# Подсчет смайликов
positive_emojis = ['👍', '😊', '😍', '❤️', '👌']
negative_emojis = ['👎', '😠', '😡', '💢', '🤬']

df['positive_emojis'] = df['text'].astype(str).apply(lambda x: sum(x.count(emoji) for emoji in positive_emojis))
df['negative_emojis'] = df['text'].astype(str).apply(lambda x: sum(x.count(emoji) for emoji in negative_emojis))

emoji_by_rating = df.groupby('rating')[['positive_emojis', 'negative_emojis']].sum()
emoji_by_rating.plot(kind='bar', ax=ax9, color=['green', 'red'])
ax9.set_xlabel('Рейтинг')
ax9.set_ylabel('Количество смайликов')
ax9.set_title('Использование смайликов по рейтингам')
ax9.legend(['Позитивные', 'Негативные'])
ax9.set_xticklabels(ax9.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('answer/review_analisis/additional_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Дополнительные графики сохранены в 'additional_analysis.png'")

print("\n" + "="*60)
print("СТАТИСТИКА ПО НОВЫМ МЕТРИКАМ")
print("="*60)

print(f"Среднее количество предложений: {df['sentence_count'].mean():.2f}")
print(f"Среднее количество восклицательных знаков: {df['exclamation_count'].mean():.2f}")
print(f"Среднее количество вопросительных знаков: {df['question_count'].mean():.2f}")
print(f"Средняя доля заглавных букв: {df['capital_ratio'].mean():.3f}")

print("\nСтатистика по использованию смайликов:")
print(f"Всего позитивных смайликов: {df['positive_emojis'].sum()}")
print(f"Всего негативных смайликов: {df['negative_emojis'].sum()}")

print("\nАнализ дубликатов:")
duplicates = df['text'].duplicated(keep=False)
print(f"Количество дубликатов текстов: {duplicates.sum()}")
if duplicates.sum() > 0:
    print(f"Процент дубликатов: {(duplicates.sum()/len(df)*100):.1f}%")