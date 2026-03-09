"""
import nltk
nltk.download('vader_lexicon')  # Словарь для анализа тональности
nltk.download('punkt')         # Токенизация текста
from nltk.sentiment import SentimentIntensityAnalyzer

# Инициализация анализатора
sia = SentimentIntensityAnalyzer()

# Текст для анализа
text = "This product is just great! I am very satisfied."

# Получение оценки тональности
scores = sia.polarity_scores(text)
print(scores)
"""




"""
from googletrans import Translator
import asyncio

async def translate_text():
    text = "Привет, ты очень плохой человек"
    translator = Translator()
    result = await translator.translate(text, src='ru', dest='en')
    print(result.text)

# Запуск асинхронной функции
asyncio.run(translate_text())
"""



# from flair.models import SequenceTagger
# from flair.data import Sentence

# # Загрузка модели для NER (русский язык)
# tagger = SequenceTagger.load("flair/ner-news-cyrillic")  # или "flair/ner-russian"

# # Текст для анализа
# text = "Путин встретился с Меркель в Берлине 15 января 2023 года."

# # Создание предложения
# sentence = Sentence(text)

# # Предсказание сущностей
# tagger.predict(sentence)

# # Вывод результатов
# for entity in sentence.get_spans('ner'):
#     print(f"Текст: {entity.text}, Метка: {entity.tag}, Уверенность: {entity.score:.4f}")





# from flair.nn import Classifier
# from flair.data import Sentence

# # Загружаем модель для анализа тональности (по умолчанию английская, но работает с контекстом)
# classifier = Classifier.load('sentiment')

# # Создаем предложение на русском
# sentence = Sentence('нейтрально')

# # Предсказываем тональность
# classifier.predict(sentence)

# # Выводим результат
# print(sentence)





# from dostoevsky.tokenization import RegexTokenizer
# from dostoevsky.models import FastTextSocialNetworkModel

# tokenizer = RegexTokenizer()
# model = FastTextSocialNetworkModel(tokenizer=tokenizer)

# messages = [
#     'Ты меня бесишь, всё испортил!',
#     'Спасибо, отличный фильм, очень трогательно.',
#     'Ну такое, пластиковая игра, неинтересно.'
# ]

# results = model.predict(messages, k=2)

# for message, sentiment in zip(messages, results):
#     print(message, '->', sentiment)




from transformers import pipeline

# Модель для анализа тональности на русском
sentiment_pipeline = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

result = sentiment_pipeline("я люблю тебя!!")
print(result)





# from deeppavlov import build_model, configs

# # Модель для анализа тональности
# sentiment_model = build_model(configs.classifiers.sentiment_rubert, download=True)
# result = sentiment_model(['Отличный сервис!'])





# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.pipeline import Pipeline
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Создаем пример данных с тремя классами
# texts = [
#     # Позитивные (2)
#     "Это отличный фильм, мне очень понравилось",
#     "Хороший сюжет и красивая картинка",
#     "Замечательная работа режиссера, шедевр",
#     "Интересно, захватывающе, рекомендую",
#     "Прекрасный актерский состав, браво",
    
#     # Негативные (1)
#     "Ужасная игра актеров, полное разочарование",
#     "Скучный и затянутый фильм, не советую",
#     "Плохой сценарий, слабая игра",
#     "Не понравилось, зря потратил время",
#     "Отвратительная операторская работа",
    
#     # Нейтральные (0)
#     "Обычный фильм, ничего особенного",
#     "Средненько, можно посмотреть один раз",
#     "Неплохо, но и не хорошо",
#     "Нормальный фильм для одного просмотра",
#     "Стандартный боевик, без изысков"
# ]

# # Метки: 2 - позитивный, 1 - негативный, 0 - нейтральный
# labels = [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


# # Создаем DataFrame
# df = pd.DataFrame({'text': texts, 'sentiment': labels})

# # Разделяем на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(
#     df['text'], df['sentiment'], test_size=0.2, random_state=42
# )

# # Создаем пайплайн
# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
#     ('classifier', LogisticRegression(random_state=42))
# ])

# # Обучаем модель
# pipeline.fit(X_train, y_train)

# # Предсказания
# y_pred = pipeline.predict(X_test)

# # Оценка качества
# print(classification_report(y_test, y_pred))

# # Функция для предсказания тональности нового текста с нейтральной оценкой
# def predict_sentiment(text):
#     prediction = pipeline.predict([text])[0]
#     probability = pipeline.predict_proba([text])[0]
    
#     # Определяем тональность с учетом нейтрального класса
#     if prediction == 2:
#         sentiment = "Позитивный"
#     elif prediction == 1:
#         sentiment = "Негативный"
#     else:
#         sentiment = "Нейтральный"
    
#     confidence = max(probability)
    
#     # Получаем вероятности для каждого класса
#     prob_dict = {
#         "позитивный": probability[2] if len(probability) > 2 else 0,
#         "негативный": probability[1] if len(probability) > 1 else 0,
#         "нейтральный": probability[0] if len(probability) > 0 else 0
#     }
    
#     # Формируем детальный отчет
#     result = f"Тональность: {sentiment} (уверенность: {confidence:.2f})\n"
#     result += f"Детальная оценка:\n"
#     result += f"  Позитивный: {prob_dict['позитивный']:.2f}\n"
#     result += f"  Негативный: {prob_dict['негативный']:.2f}\n"
#     result += f"  Нейтральный: {prob_dict['нейтральный']:.2f}"
    
#     return result

# # Функция для визуализации предсказания
# def visualize_prediction(text):
#     prediction = pipeline.predict([text])[0]
#     probability = pipeline.predict_proba([text])[0]
    
#     # Создаем DataFrame для визуализации
#     classes = ['Нейтральный', 'Негативный', 'Позитивный']
#     probs = probability
    
#     plt.figure(figsize=(8, 4))
#     colors = ['gray', 'red', 'green']
#     bars = plt.bar(classes, probs, color=colors)
    
#     # Добавляем значения на столбцы
#     for bar, prob in zip(bars, probs):
#         plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
#                 f'{prob:.2f}', ha='center', va='bottom')
    
#     plt.title(f'Оценка тональности текста: "{text[:50]}..."')
#     plt.ylabel('Вероятность')
#     plt.ylim(0, 1)
#     plt.grid(axis='y', alpha=0.3)
#     plt.show()



# for text in texts:
#     print(f"\nТекст: {text}")
#     print(predict_sentiment(text))
