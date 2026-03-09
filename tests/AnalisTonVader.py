"""
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Загрузка данных
df = pd.read_csv('отзывы.csv')
df['text'].fillna('', inplace=True)  # Обработка пропусков

# Инициализация анализатора
analyzer = SentimentIntensityAnalyzer()

# Функция для определения тональности
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Расчёт оценок
df['compound'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df['sentiment'] = df['text'].apply(get_sentiment)
 
# Сохранение результатов
df.to_csv('sentiment_analysis_results.csv', index=False)

print(df[['text', 'compound', 'sentiment']].head())
"""






import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import asyncio
from googletrans import Translator

async def translate_text(text):
    translator = Translator()
    try:
        result = await translator.translate(text, src='ru', dest='en')
        return result.text
    except Exception as e:
        return text

# Загрузка лексикона VADER
nltk.download('vader_lexicon')

# Инициализация анализатора
sia = SentimentIntensityAnalyzer()

def interpret_sentiment_scores(sentiment_scores):
    if sentiment_scores['compound'] >= 0.05:
        return "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def get_actual_sentiment(rating):
    if pd.isna(rating):
        return None
    rating = int(rating)
    if rating in [1, 2]:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    elif rating in [4, 5]:
        return "Positive"
    else:
        return None

async def analyze_sentiment_from_csv(input_file, output_file, summary_file, text_column='text', rating_column='rating'):
    # Чтение CSV
    df = pd.read_csv(input_file, nrows=1000)
    
    # Проверка столбцов
    if text_column not in df.columns:
        raise ValueError(f"Столбец '{text_column}' не найден в CSV-файле.")
    if rating_column not in df.columns:
        print(f"Предупреждение: столбец '{rating_column}' не найден. Статистика совпадений не будет рассчитана.")
        use_rating = False
    else:
        use_rating = True

    sentiment_results = []
    translation_tasks = []

    for text in df[text_column]:
        if pd.isna(text):
            sentiment_results.append({
                'neg': 0.0,
                'neu': 0.0,
                'pos': 0.0,
                'compound': 0.0,
                'sentiment_label': 'Neutral',
                'actual_sentiment': None,
                'is_correct': None
            })
        else:
            task = asyncio.create_task(translate_text(text))
            translation_tasks.append(task)

    translated_texts = await asyncio.gather(*translation_tasks)

    for idx, translated_text in enumerate(translated_texts):
        scores = sia.polarity_scores(translated_text)
        predicted = interpret_sentiment_scores(scores)

        # Если есть rating, получаем actual_sentiment
        actual = None
        is_correct = None
        if use_rating:
            rating = df.iloc[idx][rating_column]
            actual = get_actual_sentiment(rating)
            is_correct = (predicted == actual) if actual is not None else None


        sentiment_results.append({
            'neg': scores['neg'],
            'neu': scores['neu'],
            'pos': scores['pos'],
            'compound': scores['compound'],
            'sentiment_label': predicted,
            'actual_sentiment': actual,
            'is_correct': is_correct
        })

    results_df = pd.DataFrame(sentiment_results)
    final_df = pd.concat([df, results_df], axis=1)

    # Сохранение результатов
    final_df.to_csv(output_file, index=False)
    print(f"Анализ завершён. Результаты сохранены в '{output_file}'.")

    # Сводная статистика по тональности
    counts = final_df['sentiment_label'].value_counts()
    total = len(final_df)
    print("\n--- Сводная статистика по предсказанной тональности ---")
    print(f"Всего текстов обработано: {total}")
    print(f"Positive: {counts.get('Positive', 0)} ({counts.get('Positive', 0)/total:.1%})")
    print(f"Negative: {counts.get('Negative', 0)} ({counts.get('Negative', 0)/total:.1%})")
    print(f"Neutral:  {counts.get('Neutral', 0)} ({counts.get('Neutral', 0)/total:.1%})")

    # Статистика совпадений (если есть rating)
    if use_rating:
        correct_series = final_df['is_correct'].dropna()  # убираем None
        correct_count = correct_series.sum()
        accuracy = correct_count / len(correct_series) if len(correct_series) > 0 else 0

        print("\n--- Статистика совпадений с rating ---")
        print(f"Количество текстов с корректным rating: {len(correct_series)}")
        print(f"Совпадения: {int(correct_count)}")
        print(f"Точность (accuracy): {accuracy:.1%}")

        # Детально по классам
        for label in ['Positive', 'Negative', 'Neutral']:
            subset = final_df[final_df['actual_sentiment'] == label]
            if len(subset) == 0:
                continue
            correct_in_class = subset['is_correct'].sum()
            acc_in_class = correct_in_class / len(subset)
            print(f"{label}: {int(correct_in_class)}/{len(subset)} ({acc_in_class:.1%})")

        # Сохранение сводки совпадений
        if summary_file:
            summary_data = pd.DataFrame({
                'metric': [
                    'total_with_rating',
                    'correct_predictions',
                    'accuracy',
                    'positive_count',
                    'positive_accuracy',
                    'negative_count',
                    'negative_accuracy',
                    'neutral_count',
                    'neutral_accuracy'
                ],
                'value': [
                    len(correct_series),
                    int(correct_count),
                    accuracy,
                    len(final_df[final_df['actual_sentiment'] == 'Positive']),
                    (final_df[final_df['actual_sentiment'] == 'Positive']['is_correct'].sum() / 
                     len(final_df[final_df['actual_sentiment'] == 'Positive'])) if len(final_df[final_df['actual_sentiment'] == 'Positive']) > 0 else 0,
                    len(final_df[final_df['actual_sentiment'] == 'Negative']),
                    (final_df[final_df['actual_sentiment'] == 'Negative']['is_correct'].sum() /
                     len(final_df[final_df['actual_sentiment'] == 'Negative'])) if len(final_df[final_df['actual_sentiment'] == 'Negative']) > 0 else 0,
                    len(final_df[final_df['actual_sentiment'] == 'Neutral']),
                    (final_df[final_df['actual_sentiment'] == 'Neutral']['is_correct'].sum() /
                     len(final_df[final_df['actual_sentiment'] == 'Neutral'])) if len(final_df[final_df['actual_sentiment'] == 'Neutral']) > 0 else 0
                ]
            })
        summary_data.to_csv(summary_file, index=False)
        print(f"Сводка сохранена в '{summary_file}'.")

# Пример использования
if __name__ == "__main__":
    input_csv = "отзывы.csv"      # Путь к входному файлу
    output_csv = "отзывы(результат работы vader).csv"  # Путь к выходному файлу
    text_col = "text"                   # Имя столбца с текстами
    summary_csv = "sentiment_summary.csv"  # Путь к файлу со сводкой (опционально)
    
    asyncio.run(analyze_sentiment_from_csv(input_csv, output_csv, summary_csv, text_col))