import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import asyncio
from googletrans import Translator
from flair.models import TextClassifier
from flair.data import Sentence
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from tqdm import tqdm
import time
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os
import string
import pymorphy2
from nltk.corpus import stopwords

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Инициализация анализаторов
sia = SentimentIntensityAnalyzer()
# flair_sentiment = TextClassifier.load('sentiment')
flair_sentiment = TextClassifier.load('data/rusentiment-flair-model/final-model.pt')


# Инициализация pymorphy2 для лемматизации русского текста
try:
    morph = pymorphy2.MorphAnalyzer()
    logger.info("Pymorphy2 инициализирован")
except Exception as e:
    logger.error(f"Ошибка инициализации pymorphy2: {e}")
    morph = None

# Загрузка стоп-слов
russian_stopwords = set(stopwords.words('russian'))
english_stopwords = set(stopwords.words('english'))

logger.info("Загрузка моделей трансформеров...")

# Модель для русского языка (RuBERT)
try:
    # ru_model_name = "blanchefort/rubert-base-cased-sentiment"
    ru_model_name = "seara/rubert-tiny2-russian-sentiment"
    ru_tokenizer = AutoTokenizer.from_pretrained(ru_model_name)
    ru_model = AutoModelForSequenceClassification.from_pretrained(ru_model_name)
    ru_sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=ru_model,
        tokenizer=ru_tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    logger.info(f"RuBERT модель загружена. Используется {'GPU' if torch.cuda.is_available() else 'CPU'}")
except Exception as e:
    logger.error(f"Ошибка загрузки RuBERT: {e}")
    ru_sentiment_pipeline = None

# Модель для английского языка (roberta)
try:
    # en_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    en_model_name = "clapAI/roberta-large-multilingual-sentiment"
    en_tokenizer = AutoTokenizer.from_pretrained(en_model_name)
    en_model = AutoModelForSequenceClassification.from_pretrained(en_model_name)
    en_sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=en_model,
        tokenizer=en_tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    logger.info(f"RoBERTa модель загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки RoBERTa: {e}")
    en_sentiment_pipeline = None

#  DistilBERT 
try:
    light_ru_sentiment = pipeline(
        "sentiment-analysis",
        # model='Geotrend/distilbert-base-ru-cased', 
        # tokenizer='Geotrend/distilbert-base-ru-cased',
        model='cointegrated/rubert-tiny2',
        tokenizer='cointegrated/rubert-tiny2',
        device=0 if torch.cuda.is_available() else -1
    )
    logger.info("DistilBERT (Russian sentiment) модель загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки русской DistilBERT: {e}")
    # Альтернативная модель
    try:
        light_ru_sentiment = pipeline(
            "sentiment-analysis",
            model='cointegrated/rubert-tiny2',
            tokenizer='cointegrated/rubert-tiny2',
            device=0 if torch.cuda.is_available() else -1
        )
        logger.info("RuBERT-tiny2 модель загружена как альтернатива DistilBERT")
    except Exception as e2:
        logger.error(f"Ошибка загрузки альтернативной модели: {e2}")
        light_ru_sentiment = None

# Класс для предобработки текстов перед использованием моделей scikit-learn
class TextPreprocessor:
    
    def __init__(self, language='russian', use_lemmatization=True, use_stopwords=True):
        self.language = language
        self.use_lemmatization = use_lemmatization
        self.use_stopwords = use_stopwords
        self.morph = morph if language == 'russian' else None
        
        # Таблица для удаления пунктуации
        self.punctuation_table = str.maketrans('', '', string.punctuation)
        
        # Выбор стоп-слов
        if use_stopwords:
            self.stopwords = russian_stopwords if language == 'russian' else english_stopwords
        else:
            self.stopwords = set()
    
    def clean_text(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        text = text.lower()
        
        # Замена распространенных URL-маркеров на пустую строку
        url_indicators = ['http://', 'https://', 'www.', '.com', '.ru', '.org', '.net']
        for indicator in url_indicators:
            text = text.replace(indicator, '')
        
        # Удаление email адресов (простейшая проверка на наличие @)
        if '@' in text:
            words = text.split()
            filtered_words = []
            for word in words:
                if '@' not in word:
                    filtered_words.append(word)
            text = ' '.join(filtered_words)
        
        # Удаление пунктуации
        text = text.translate(self.punctuation_table)
        
        # Удаление чисел (заменяем цифры на пробелы)
        digits = '0123456789'
        for digit in digits:
            text = text.replace(digit, ' ')
        
        # Удаление лишних пробелов
        text = ' '.join(text.split())
        
        return text
    
    def lemmatize_russian(self, text):
        if self.morph is None:
            return text
        
        words = text.split()
        lemmatized_words = []
        
        for word in words:
            # Пропускаем короткие слова
            if len(word) < 3:
                lemmatized_words.append(word)
                continue
            
            try:
                # Лемматизация
                parsed = self.morph.parse(word)[0]
                lemmatized = parsed.normal_form
                lemmatized_words.append(lemmatized)
            except:
                lemmatized_words.append(word)
        
        return ' '.join(lemmatized_words)
    
    def remove_stopwords(self, text):
        if not self.use_stopwords:
            return text
        
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords and len(word) > 2]
        return ' '.join(filtered_words)
    
    def preprocess(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Базовая очистка
        text = self.clean_text(text)
        
        if not text:
            return ""
        
        # Удаление стоп-слов
        text = self.remove_stopwords(text)
        
        # Лемматизация (только для русского)
        if self.use_lemmatization and self.language == 'russian':
            text = self.lemmatize_russian(text)
        
        return text
    
    def preprocess_batch(self, texts):
        return [self.preprocess(text) for text in texts]


class SklearnSentimentAnalyzer:
    
    def __init__(self, model_type='logistic_regression', use_preprocessing=True):
        """
        Инициализация анализатора
        
        Parameters:
        model_type: str - тип модели ('logistic_regression', 'svm', 'random_forest')
        use_preprocessing: bool - использовать предобработку текста
        """
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.classifier = None
        self.use_preprocessing = use_preprocessing
        self.preprocessor = TextPreprocessor(
            language='russian',
            use_lemmatization=True,
            use_stopwords=True
        ) if use_preprocessing else None
        
        self.is_trained = False
        self.label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.reverse_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
    def load_model(self, filepath):
        """
        Загрузка предобученной модели из .joblib файла
        
        Parameters:
        filepath: str - путь к файлу модели .joblib
        
        Returns:
        bool: успешность загрузки
        """
        try:
            # Загрузка модели из .joblib файла
            model_data = joblib.load(filepath)
            
            # Проверяем тип загруженных данных
            if isinstance(model_data, dict):
                # Модель сохранена как словарь с компонентами
                if 'vectorizer' in model_data and 'classifier' in model_data:
                    self.vectorizer = model_data['vectorizer']
                    self.classifier = model_data['classifier']
                    # Создаем pipeline для удобства
                    from sklearn.pipeline import Pipeline
                    self.model = Pipeline([
                        ('vectorizer', self.vectorizer),
                        ('classifier', self.classifier)
                    ])
                    logger.info(f"Модель {self.model_type} загружена из словаря с vectorizer и classifier")
                elif 'pipeline' in model_data:
                    self.model = model_data['pipeline']
                    logger.info(f"Модель {self.model_type} загружена из словаря с pipeline")
                elif 'model' in model_data:
                    self.model = model_data['model']
                    logger.info(f"Модель {self.model_type} загружена из словаря с model")
                else:
                    # Пробуем использовать все данные как модель
                    self.model = model_data
                    logger.info(f"Модель {self.model_type} загружена как словарь")
            elif hasattr(model_data, 'predict'):
                # Непосредственно модель или pipeline
                self.model = model_data
                logger.info(f"Модель {self.model_type} загружена как объект с методом predict")
            else:
                logger.warning(f"Неизвестный формат модели {self.model_type}")
                return False
            
            self.is_trained = True
            logger.info(f"Модель {self.model_type} успешно загружена из {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {self.model_type} из {filepath}: {e}")
            return False
    
    def predict(self, text):
        """
        Предсказание тональности для одного текста с предобработкой
        
        Returns:
        tuple: (score, sentiment_label, confidence)
        """
        if not self.is_trained or self.model is None:
            logger.warning(f"Модель {self.model_type} не загружена. Возвращаю нейтральный результат.")
            return 3, "Neutral", 0.0
        
        try:
            # Предобработка текста
            if self.use_preprocessing and self.preprocessor:
                processed_text = self.preprocessor.preprocess(text)
            else:
                processed_text = text
            
            if not processed_text:
                return 3, "Neutral", 0.0
            
            # Получение предсказания
            pred_class = self.model.predict([processed_text])[0]
            
            # Обработка разных форматов предсказаний
            if isinstance(pred_class, (int, np.integer)):
                class_idx = pred_class
            elif isinstance(pred_class, str):
                # Прямое предсказание строковой метки
                if pred_class in ['Positive', 'Negative', 'Neutral']:
                    sentiment_label = pred_class
                    if sentiment_label == "Positive":
                        score = 5
                    elif sentiment_label == "Negative":
                        score = 1
                    else:
                        score = 3
                    return score, sentiment_label, 1.0
                else:
                    # Преобразование строки в индекс
                    pred_lower = pred_class.lower()
                    if pred_lower in self.label_mapping:
                        class_idx = self.label_mapping[pred_lower]
                    else:
                        logger.warning(f"Неизвестная метка {pred_class}, использую Neutral")
                        return 3, "Neutral", 0.0
            else:
                class_idx = int(pred_class)
            
            # Получение вероятностей (если модель поддерживает)
            confidence = 0.0
            if hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba([processed_text])[0]
                    if len(proba) > class_idx:
                        confidence = proba[class_idx]
                except:
                    pass
            
            # Преобразование в нужный формат
            sentiment_label = self.reverse_mapping.get(class_idx, "Neutral")
            
            # Преобразование в score (1,3,5)
            if sentiment_label == "Positive":
                score = 5
            elif sentiment_label == "Negative":
                score = 1
            else:
                score = 3
            
            return score, sentiment_label, confidence
            
        except Exception as e:
            logger.error(f"Ошибка в предсказании модели {self.model_type}: {e}")
            return 3, "Neutral", 0.0


def load_sklearn_models(model_dir='models'):
    """
    Загрузка всех предобученных scikit-learn моделей из .joblib файлов в указанной директории
    
    Parameters:
    model_dir: str - директория с моделями .joblib
    
    Returns:
    dict: словарь загруженных моделей
    """
    logger.info(f"Загрузка scikit-learn моделей из директории '{model_dir}'...")
    
    models = {}
    
    # Проверяем существование директории
    if not os.path.exists(model_dir):
        logger.warning(f"Директория '{model_dir}' не существует")
        return models
    
    # Ищем все .joblib файлы в директории
    joblib_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    
    if not joblib_files:
        logger.warning(f"В директории '{model_dir}' не найдено .joblib файлов")
        return models
    
    logger.info(f"Найдено {len(joblib_files)} .joblib файлов")
    
    # Маппинг имен файлов на типы моделей
    model_name_mapping = {
        'logistic_regression_model': 'logistic_regression',
        'random_forest_model': 'random_forest',
        'svm_linear_model': 'svm',
    }
    
    for filename in joblib_files:
        filepath = os.path.join(model_dir, filename)
        
        # Определяем тип модели по имени файла
        model_type = 'unknown'
        file_lower = filename.lower()
        
        for key, value in model_name_mapping.items():
            if key in file_lower:
                model_type = value
                break
        
        # Создаем экземпляр анализатора
        model = SklearnSentimentAnalyzer(model_type, use_preprocessing=True)
        
        # Загружаем модель
        if model.load_model(filepath):
            # Если модель с таким типом уже загружена, добавляем с суффиксом
            if model_type in models:
                base_type = model_type
                counter = 1
                while f"{base_type}_{counter}" in models:
                    counter += 1
                model_key = f"{base_type}_{counter}"
                logger.info(f"Модель с типом {model_type} уже существует. Сохраняю как {model_key}")
            else:
                model_key = model_type
            
            models[model_key] = model
    
    logger.info(f"Загружено {len(models)} моделей из {len(joblib_files)} .joblib файлов")
    return models

def get_sklearn_sentiment(text, model_name='logistic_regression', models_dict=None):
    """
    Получение тональности с помощью загруженной scikit-learn модели
    
    Parameters:
    text: str - текст для анализа
    model_name: str - название модели ('logistic_regression', 'svm', 'random_forest')
    models_dict: dict - словарь загруженных моделей
    
    Returns:
    tuple: (score, sentiment_label, confidence)
    """
    if models_dict is None:
        logger.warning("Словарь моделей не предоставлен. Возвращаю нейтральный результат.")
        return 3, "Neutral", 0.0
    
    if model_name not in models_dict:
        logger.warning(f"Модель {model_name} не найдена в загруженных моделях. Использую logistic_regression если доступна")
        # Пробуем использовать logistic_regression как запасной вариант
        if 'logistic_regression' in models_dict:
            model_name = 'logistic_regression'
        else:
            # Если нет ни одной модели, берем первую доступную
            available_models = list(models_dict.keys())
            if available_models:
                model_name = available_models[0]
                logger.info(f"Использую модель {model_name} как запасной вариант")
            else:
                logger.warning("Нет доступных моделей для анализа")
                return 3, "Neutral", 0.0
    
    model = models_dict[model_name]
    return model.predict(text)


# Интерпретация scores от VADER
def interpret_sentiment_scores(sentiment_scores):
    if sentiment_scores['compound'] >= 0.15: #0.15
        return "Positive"
    elif sentiment_scores['compound'] <= -0.15: #-0.15
        return "Negative"
    else:
        return "Neutral"
    
def get_actual_sentiment(rating):
    if pd.isna(rating):
        return None
    
    if isinstance(rating, str):
        rating = rating.strip().upper()
        if rating in ['NEGATIVE', 'NEUTRAL', 'POSITIVE']:
            return rating.capitalize()
        try:
            rating = int(float(rating))
        except:
            return None
    else:
        rating = int(rating)
        if rating in [1, 2]:
            return "Negative"
        elif rating == 3:
            return "Neutral"
        elif rating in [4, 5]:
            return "Positive"
    
    return None

def get_flair_score(text):
    """
    Анализирует тональность текста с помощью Flair.
    Возвращает числовую оценку (1, 3, 5) и метку тональности.
    """
    try:
        sentence = Sentence(text)
        flair_sentiment.predict(sentence)
        label = sentence.labels[0].value
        confidence = sentence.labels[0].score

        if label == 'POSITIVE':
            score = 5
            sentiment_label = "Positive"
        elif label == 'NEGATIVE':
            score = 1
            sentiment_label = "Negative"
        else:
            score = 3
            sentiment_label = "Neutral"

        return score, sentiment_label, confidence
    except Exception as e:
        logger.error(f"Ошибка в Flair анализе: {e}")
        return 3, "Neutral", 0.0

def get_transformer_sentiment_ru(text):
    """
    Анализирует тональность русского текста с помощью RuBERT
    """
    if ru_sentiment_pipeline is None:
        return None, None, "Neutral", 0.0
    
    try:
        # Обрезаем слишком длинные тексты
        if len(text) > 600:
            text = text[:600]
        
        result = ru_sentiment_pipeline(text)[0]
        label = result['label']
        confidence = result['score']
        
        # Преобразование меток модели в наш формат
        if label in ['POSITIVE', 'positive', 'LABEL_1', '1']:
            sentiment_label = "Positive"
            score = 5
        elif label in ['NEGATIVE', 'negative', 'LABEL_0', '0']:
            sentiment_label = "Negative"
            score = 1
        else:
            sentiment_label = "Neutral"
            score = 3
            
        return label, score, sentiment_label, confidence
    except Exception as e:
        logger.error(f"Ошибка в RuBERT анализе: {e}")
        return None, 3, "Neutral", 0.0

# Исправленная функция get_transformer_sentiment_en (для английских моделей)
def get_transformer_sentiment_en(text, model_type='roberta'):
    """
    Анализирует тональность английского текста с помощью 'roberta'
    """
    if model_type != 'roberta':
        logger.warning(f"Модель {model_type} не поддерживается для английского, использую roberta")
    
    if en_sentiment_pipeline is None:
        return None, 3, "Neutral", 0.0
    
    try:
        # Обрезаем слишком длинные тексты
        if len(text) > 600:
            text = text[:600]
        
        result = en_sentiment_pipeline(text)[0]
        label = result['label']
        confidence = result['score']
        
        # Преобразование меток модели в наш формат
        if label in ['LABEL_2', 'POSITIVE', 'positive'] or 'positive' in label.lower():
            sentiment_label = "Positive"
            score = 5
        elif label in ['LABEL_0', 'NEGATIVE', 'negative'] or 'negative' in label.lower():
            sentiment_label = "Negative"
            score = 1
        else:
            sentiment_label = "Neutral"
            score = 3
            
        return label, score, sentiment_label, confidence
    except Exception as e:
        logger.error(f"Ошибка в RoBERTa анализе: {e}")
        return None, 3, "Neutral", 0.0

def get_transformer_sentiment_ru_light(text):
    """
    Анализирует тональность русского текста с помощью легкой DistilBERT модели
    Исправлено: использование русской модели для анализа оригинального текста
    """
    # Инициализация легкой русской модели (если еще не создана)
    if not hasattr(get_transformer_sentiment_ru_light, 'light_ru_pipeline'):
        try:
            # Используем русскую DistilBERT модель для sentiment analysis
            get_transformer_sentiment_ru_light.light_ru_pipeline = pipeline(
                "sentiment-analysis",
                model='Geotrend/distilbert-base-ru-cased',  # Русская DistilBERT
                tokenizer='Geotrend/distilbert-base-ru-cased',
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("DistilBERT (Russian) модель загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки русской DistilBERT: {e}")
            # Альтернативная русская модель, если первая не загрузилась
            try:
                get_transformer_sentiment_ru_light.light_ru_pipeline = pipeline(
                    "sentiment-analysis",
                    model='cointegrated/rubert-tiny2',
                    tokenizer='cointegrated/rubert-tiny2',
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("RuBERT-tiny2 модель загружена как альтернатива")
            except Exception as e2:
                logger.error(f"Ошибка загрузки альтернативной модели: {e2}")
                get_transformer_sentiment_ru_light.light_ru_pipeline = None
    
    if get_transformer_sentiment_ru_light.light_ru_pipeline is None:
        return None, 3, "Neutral", 0.0
    
    try:
        # Обрезаем слишком длинные тексты
        if len(text) > 512:
            text = text[:512]
        
        result = get_transformer_sentiment_ru_light.light_ru_pipeline(text)[0]
        label = result['label']
        confidence = result['score']
        
        # Преобразование меток модели в наш формат
        label_lower = label.lower()
        
        # Обработка различных форматов меток
        if 'positive' in label_lower or 'pos' in label_lower or label in ['LABEL_1', '1']:
            sentiment_label = "Positive"
            score = 5
        elif 'negative' in label_lower or 'neg' in label_lower or label in ['LABEL_0', '0']:
            sentiment_label = "Negative"
            score = 1
        elif 'neutral' in label_lower:
            sentiment_label = "Neutral"
            score = 3
        else: 
            # Если метка не распознана, считаем нейтральной
            sentiment_label = "Neutral"
            score = 3
            
        return label, score, sentiment_label, confidence
        
    except Exception as e:
        logger.error(f"Ошибка в русской DistilBERT анализе: {e}")
        return None, 3, "Neutral", 0.0

class TextTranslator:
    """
    Класс для управления переводами текстов
    """
    def __init__(self, max_concurrent_translations=10):
        self.translator = Translator()
        self.max_concurrent = max_concurrent_translations
        self.semaphore = asyncio.Semaphore(max_concurrent_translations)
        logger.info(f"Инициализирован переводчик с максимум {max_concurrent_translations} параллельных переводов")
    
    async def translate_single(self, text):
        """
        Перевод одного текста с использованием семафора для ограничения параллельных запросов
        """
        if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
            return None
        
        async with self.semaphore:
            try:
                # Добавляем небольшую задержку для предотвращения перегрузки API
                await asyncio.sleep(0.05)
                result = await self.translator.translate(text, src='ru', dest='en')
                return result.text
            except Exception as e:
                logger.error(f"Ошибка перевода: {e}")
                return text  # Возвращаем оригинал в случае ошибки
    
    async def translate_batch(self, texts, batch_size=50):
        """
        Пакетный перевод текстов с контролем параллельности
        
        Parameters:
        texts: list - список текстов для перевода
        batch_size: int - размер пакета для последовательной обработки
        
        Returns:
        list: список переведенных текстов
        """
        translated_texts = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        logger.info(f"Начинается перевод {len(texts)} текстов, разбито на {total_batches} пакетов по {batch_size}")
        
        for batch_idx in range(0, len(texts), batch_size):
            batch = texts[batch_idx:batch_idx + batch_size]
            logger.info(f"Обработка пакета {batch_idx // batch_size + 1}/{total_batches} ({len(batch)} текстов)")
            
            # Создаем задачи для текущего пакета
            tasks = []
            for text in batch:
                if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
                    tasks.append(None)
                else:
                    tasks.append(asyncio.create_task(self.translate_single(text)))
            
            # Ждем завершения всех задач в пакете
            batch_results = []
            for task in tasks:
                if task is None:
                    batch_results.append(None)
                else:
                    try:
                        result = await task
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Ошибка при получении результата перевода: {e}")
                        batch_results.append(None)
            
            translated_texts.extend(batch_results)
            
            # Небольшая задержка между пакетами
            if batch_idx + batch_size < len(texts):
                await asyncio.sleep(0.5)
        
        logger.info(f"Перевод завершен. Успешно переведено: {sum(1 for t in translated_texts if t is not None and t != '')} из {len(texts)}")
        return translated_texts
    
    def translate_sync(self, text):
        """
        Синхронный перевод для случаев, когда async не используется
        """
        try:
            if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
                return None
            result = self.translator.translate(text, src='ru', dest='en')
            return result.text
        except Exception as e:
            logger.error(f"Ошибка синхронного перевода: {e}")
            return text


async def analyze_sentiment_from_csv(input_file, output_file, summary_file, stats_file, 
                                     text_column='text', rating_column='label',
                                     models_dir='models'):
    logger.info(f"Начинаем анализ файла: {input_file}")
    start_time = time.time()
    
    # Чтение CSV
    try:
        df = pd.read_csv(input_file, nrows=10)
        logger.info(f"Загружено {len(df)} строк отзывов")
    except Exception as e:
        logger.error(f"Ошибка чтения файла: {e}")
        return

    # Проверка столбца с текстом
    if text_column not in df.columns:
        raise ValueError(f"Столбец '{text_column}' не найден в CSV-файле.")
    
    # Проверка столбца с оценкой
    if rating_column not in df.columns:
        logger.warning(f"Столбец '{rating_column}' не найден. Статистика совпадений не будет рассчитана.")
        use_rating = False
    else:
        use_rating = True
    
    # Загрузка предобученных scikit-learn моделей из .joblib файлов
    sklearn_models = load_sklearn_models(models_dir)
    
    if not sklearn_models:
        logger.warning("Не удалось загрузить ни одной scikit-learn модели. Будет использован только ensemble из трансформеров.")

    # Получаем тексты для перевода
    original_texts = df[text_column].tolist()
    
    # Инициализация переводчика
    translator = TextTranslator(max_concurrent_translations=10)
    
    # Перевод всех текстов (асинхронно)
    logger.info("Начало перевода текстов...")
    translation_start = time.time()
    translated_texts = await translator.translate_batch(original_texts, batch_size=50)
    translation_time = time.time() - translation_start
    logger.info(f"Перевод завершен за {translation_time:.2f} секунд")

    # Анализ тональности
    logger.info("Анализ тональности с помощью всех моделей...")
    sentiment_results = []
    
    for idx, (original_text, translated_text) in enumerate(tqdm(zip(original_texts, translated_texts), desc="Анализ", total=len(original_texts))):
        if pd.isna(original_text) or original_text.strip() == "":
            result_row = create_empty_result_row()
        else:
            # VADER анализирует переведенный текст (английский)
            if translated_text:
                scores = sia.polarity_scores(translated_text)
                vader_predicted = interpret_sentiment_scores(scores)
            else:
                scores = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
                vader_predicted = 'Neutral'

            # Flair анализирует оригинальный текст (русский)
            flair_score, flair_predicted, flair_confidence = get_flair_score(original_text)
            
            # RuBERT анализирует оригинальный текст (русский)
            rubert_raw, rubert_score, rubert_predicted, rubert_confidence = get_transformer_sentiment_ru(original_text)

            # RoBERTa анализирует переведенный текст (английский)
            roberta_raw, roberta_score, roberta_predicted, roberta_confidence = get_transformer_sentiment_en(
                translated_text if translated_text else "", 'roberta'
            )
            
            # DistilBERT анализирует оригинальный текст (русский) - легкая русская модель
            distilbert_raw, distilbert_score, distilbert_predicted, distilbert_confidence = get_transformer_sentiment_ru_light(original_text)
                   
            # scikit-learn модели (русский) - используем загруженные модели из .joblib
            sklearn_results = {}
            for model_name in sklearn_models.keys():
                score, sentiment, confidence = get_sklearn_sentiment(original_text, model_name, sklearn_models)
                sklearn_results[model_name] = {
                    'score': score,
                    'sentiment': sentiment,
                    'confidence': confidence
                }

            # Если нет загруженных моделей, создаем пустые результаты
            if not sklearn_models:
                for model_name in ['logistic_regression', 'svm', 'random_forest']:
                    sklearn_results[model_name] = {
                        'score': 3,
                        'sentiment': 'Neutral',
                        'confidence': 0.0
                    }

            # Если есть rating, получаем actual_sentiment
            actual = None
            if use_rating:
                rating = df.iloc[idx][rating_column]
                actual = get_actual_sentiment(rating)

            # Формирование строки результатов
            result_row = create_result_row(
                scores, vader_predicted,
                flair_score, flair_predicted, flair_confidence,
                rubert_raw, rubert_score, rubert_predicted, rubert_confidence,
                roberta_raw, roberta_score, roberta_predicted, roberta_confidence,
                distilbert_raw, distilbert_score, distilbert_predicted, distilbert_confidence,
                sklearn_results,
                actual, translated_text
            )

        sentiment_results.append(result_row)

    # Создание итогового DataFrame
    results_df = pd.DataFrame(sentiment_results)
    final_df = pd.concat([df, results_df], axis=1)

    # Сохранение результатов
    try:
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"Анализ завершён. Результаты сохранены в '{output_file}'.")
    except Exception as e:
        logger.error(f"Ошибка сохранения результатов: {e}")

    # Вывод статистики в файл
    print_statistics_to_file(final_df, use_rating, stats_file)
    
    # Сохранение сводки
    if use_rating:
        # Динамически создаем список моделей на основе загруженных
        models = [
            ('VADER', 'is_correct_vader'),
            ('Flair', 'is_correct_flair'),
            ('RuBERT', 'is_correct_rubert'),
            ('RoBERTa', 'is_correct_roberta'),
            ('DistilBERT', 'is_correct_distilbert')
        ]
        
        # Добавляем загруженные scikit-learn модели
        for model_name in sklearn_models.keys():
            col_name = f'is_correct_{model_name}'
            if col_name in final_df.columns:
                models.append((model_name.replace('_', ' ').title(), col_name))
        
        # Добавляем Ensemble
        models.append(('Ensemble', 'is_correct_ensemble'))
        
        save_summary(final_df, summary_file, models)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Общее время выполнения: {elapsed_time:.2f} секунд")
    logger.info(f"Время перевода: {translation_time:.2f} секунд ({translation_time/elapsed_time:.1%} от общего времени)")
    logger.info(f"Статистика сохранена в файл: {stats_file}")


def create_empty_result_row():
    result = {
        # VADER результаты
        'vader_neg': 0.0, 'vader_neu': 0.0, 'vader_pos': 0.0,
        'vader_compound': 0.0, 'vader_sentiment': 'Neutral', 'vader_score': 3,
        
        # Flair результаты
        'flair_score': 3, 'flair_sentiment': 'Neutral', 'flair_confidence': 0.0,
        
        # RuBERT результаты
        'rubert_raw_label': None, 'rubert_score': 3,
        'rubert_sentiment': 'Neutral', 'rubert_confidence': 0.0,
        
        # RoBERTa результаты
        'roberta_raw_label': None, 'roberta_score': 3,
        'roberta_sentiment': 'Neutral', 'roberta_confidence': 0.0,
        
        # DistilBERT результаты
        'distilbert_raw_label': None, 'distilbert_score': 3,
        'distilbert_sentiment': 'Neutral', 'distilbert_confidence': 0.0,
        
        # Logistic Regression
        'logistic_regression_score': 3, 'logistic_regression_sentiment': 'Neutral',
        'logistic_regression_confidence': 0.0,
        
        # SVM
        'svm_score': 3, 'svm_sentiment': 'Neutral', 'svm_confidence': 0.0,
        
        # Random Forest
        'random_forest_score': 3, 'random_forest_sentiment': 'Neutral',
        'random_forest_confidence': 0.0,
        
        # Ensemble
        'ensemble_score_raw': 3.0, 'ensemble_score': 3, 'ensemble_sentiment': 'Neutral',
        
        # Actual и метрики
        'actual_sentiment': None,
        'is_correct_vader': None, 'is_correct_flair': None, 'is_correct_rubert': None,
        'is_correct_roberta': None, 'is_correct_distilbert': None,
        'is_correct_logistic_regression': None, 'is_correct_svm': None,
        'is_correct_random_forest': None, 'is_correct_ensemble': None,
        'translated_text': None
    }
    
    return result

def create_result_row(scores, vader_predicted,
                     flair_score, flair_predicted, flair_confidence,
                     rubert_raw, rubert_score, rubert_predicted, rubert_confidence,
                     roberta_raw, roberta_score, roberta_predicted, roberta_confidence,
                     distilbert_raw, distilbert_score, distilbert_predicted, distilbert_confidence,
                     sklearn_results, actual, translated_text):
    
    # Преобразование compound score в vader_score (1, 3, 5)
    if scores['compound'] >= 0.33:
        vader_score = 5
    elif scores['compound'] <= -0.33:
        vader_score = 1
    else:
        vader_score = 3
    
    # Получаем scores от scikit-learn моделей с значениями по умолчанию
    logistic_score = 3
    svm_score = 3
    random_forest_score = 3
    
    logistic_sentiment = "Neutral"
    svm_sentiment = "Neutral"
    random_forest_sentiment = "Neutral"
    
    logistic_confidence = 0.0
    svm_confidence = 0.0
    random_forest_confidence = 0.0
    
    # Извлекаем значения из sklearn_results, если они есть
    for model_name, model_result in sklearn_results.items():
        if 'logistic' in model_name.lower():
            logistic_score = model_result.get('score', 3)
            logistic_sentiment = model_result.get('sentiment', 'Neutral')
            logistic_confidence = model_result.get('confidence', 0.0)
        elif 'svm' in model_name.lower():
            svm_score = model_result.get('score', 3)
            svm_sentiment = model_result.get('sentiment', 'Neutral')
            svm_confidence = model_result.get('confidence', 0.0)
        elif 'random' in model_name.lower() or 'forest' in model_name.lower():
            random_forest_score = model_result.get('score', 3)
            random_forest_sentiment = model_result.get('sentiment', 'Neutral')
            random_forest_confidence = model_result.get('confidence', 0.0)
    
    # Сбор всех 8 scores для ensemble
    all_scores = [
        vader_score,           # 1. VADER
        flair_score,           # 2. Flair
        rubert_score,          # 3. RuBERT
        roberta_score,         # 4. RoBERTa
        distilbert_score,      # 5. DistilBERT
        logistic_score,        # 6. Logistic Regression
        svm_score,             # 7. SVM
        random_forest_score    # 8. Random Forest
    ]
    
    # Фильтруем None значения и вычисляем ensemble_score
    valid_scores = [s for s in all_scores if s is not None]
    
    if valid_scores:
        ensemble_score_raw = sum(valid_scores) / len(valid_scores)
        ensemble_score_rounded = round(ensemble_score_raw)
    else:
        ensemble_score_raw = 3.0
        ensemble_score_rounded = 3
    
    # Определение ensemble_sentiment на основе ensemble_score_rounded
    if ensemble_score_rounded >= 4:
        ensemble_sentiment = "Positive"
    elif ensemble_score_rounded <= 2:
        ensemble_sentiment = "Negative"
    else:
        ensemble_sentiment = "Neutral"
    
    # Вычисление метрик корректности
    actual_sentiment = actual
    
    is_correct_vader = (vader_predicted == actual_sentiment) if actual_sentiment is not None else None
    is_correct_flair = (flair_predicted == actual_sentiment) if actual_sentiment is not None else None
    is_correct_rubert = (rubert_predicted == actual_sentiment) if actual_sentiment is not None else None
    is_correct_roberta = (roberta_predicted == actual_sentiment) if actual_sentiment is not None else None
    is_correct_distilbert = (distilbert_predicted == actual_sentiment) if actual_sentiment is not None else None
    is_correct_logistic = (logistic_sentiment == actual_sentiment) if actual_sentiment is not None else None
    is_correct_svm = (svm_sentiment == actual_sentiment) if actual_sentiment is not None else None
    is_correct_random_forest = (random_forest_sentiment == actual_sentiment) if actual_sentiment is not None else None
    is_correct_ensemble = (ensemble_sentiment == actual_sentiment) if actual_sentiment is not None else None
    
    result = {
        # VADER результаты
        'vader_neg': scores['neg'], 'vader_neu': scores['neu'],
        'vader_pos': scores['pos'], 'vader_compound': scores['compound'],
        'vader_sentiment': vader_predicted, 'vader_score': vader_score,
        
        # Flair результаты
        'flair_score': flair_score, 'flair_sentiment': flair_predicted,
        'flair_confidence': flair_confidence,
        
        # RuBERT результаты
        'rubert_raw_label': rubert_raw, 'rubert_score': rubert_score,
        'rubert_sentiment': rubert_predicted, 'rubert_confidence': rubert_confidence,
        
        # RoBERTa результаты
        'roberta_raw_label': roberta_raw, 'roberta_score': roberta_score,
        'roberta_sentiment': roberta_predicted, 'roberta_confidence': roberta_confidence,
        
        # DistilBERT результаты
        'distilbert_raw_label': distilbert_raw, 'distilbert_score': distilbert_score,
        'distilbert_sentiment': distilbert_predicted, 'distilbert_confidence': distilbert_confidence,
        
        # Logistic Regression результаты
        'logistic_regression_score': logistic_score,
        'logistic_regression_sentiment': logistic_sentiment,
        'logistic_regression_confidence': logistic_confidence,
        
        # SVM результаты
        'svm_score': svm_score,
        'svm_sentiment': svm_sentiment,
        'svm_confidence': svm_confidence,
        
        # Random Forest результаты
        'random_forest_score': random_forest_score,
        'random_forest_sentiment': random_forest_sentiment,
        'random_forest_confidence': random_forest_confidence,
        
        # Ensemble результаты
        'ensemble_score_raw': ensemble_score_raw,
        'ensemble_score': ensemble_score_rounded,
        'ensemble_sentiment': ensemble_sentiment,
        
        # Actual и метрики
        'actual_sentiment': actual_sentiment,
        'is_correct_vader': is_correct_vader,
        'is_correct_flair': is_correct_flair,
        'is_correct_rubert': is_correct_rubert,
        'is_correct_roberta': is_correct_roberta,
        'is_correct_distilbert': is_correct_distilbert,
        'is_correct_logistic_regression': is_correct_logistic,
        'is_correct_svm': is_correct_svm,
        'is_correct_random_forest': is_correct_random_forest,
        'is_correct_ensemble': is_correct_ensemble,
        'translated_text': translated_text
    }
    
    return result

def print_statistics_to_file(final_df, use_rating, stats_file):
    """Вывод статистики анализа в файл"""
    total = len(final_df)
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("\n" + "="*70 + "\n")
        f.write("СТАТИСТИКА АНАЛИЗА ТОНАЛЬНОСТИ\n")
        f.write("="*70 + "\n")
        
        # Список всех моделей для вывода
        models_list = [
            ('VADER', 'vader_sentiment', 'vader_score', 'английского перевода'),
            ('Flair', 'flair_sentiment', 'flair_score', 'русского оригинала'),
            ('RuBERT', 'rubert_sentiment', 'rubert_score', 'русского оригинала'),
            ('RoBERTa', 'roberta_sentiment', 'roberta_score', 'английского перевода'),
            ('DistilBERT', 'distilbert_sentiment', 'distilbert_score', 'русского оригинала'),
            ('Logistic Regression', 'logistic_regression_sentiment', 'logistic_regression_score', 'русского оригинала'),
            ('SVM', 'svm_sentiment', 'svm_score', 'русского оригинала'),
            ('Random Forest', 'random_forest_sentiment', 'random_forest_score', 'русского оригинала'),
            ('ENSEMBLE', 'ensemble_sentiment', 'ensemble_score', 'усредненный результат')
        ]
        
        for name, sent_col, score_col, lang in models_list:
            if sent_col in final_df.columns:
                counts = final_df[sent_col].value_counts()
                f.write(f"\n--- {name} (анализ {lang}) ---\n")
                f.write(f"Всего текстов: {total}\n")
                f.write(f"Positive: {counts.get('Positive', 0)} ({counts.get('Positive', 0)/total:.1%})\n")
                f.write(f"Negative: {counts.get('Negative', 0)} ({counts.get('Negative', 0)/total:.1%})\n")
                f.write(f"Neutral:  {counts.get('Neutral', 0)} ({counts.get('Neutral', 0)/total:.1%})\n")
                if score_col in final_df.columns:
                    f.write(f"Средний {name.split()[0]} score: {final_df[score_col].mean():.2f}\n")
        
        # Статистика совпадений (если есть rating)
        if use_rating:
            f.write("\n" + "="*70 + "\n")
            f.write("СТАТИСТИКА СОВПАДЕНИЙ С РЕАЛЬНЫМИ ОЦЕНКАМИ\n")
            f.write("="*70 + "\n")
            
            # Список моделей для проверки корректности
            correct_models = [
                ('VADER', 'is_correct_vader'),
                ('Flair', 'is_correct_flair'),
                ('RuBERT', 'is_correct_rubert'),
                ('RoBERTa', 'is_correct_roberta'),
                ('DistilBERT', 'is_correct_distilbert'),
                ('Logistic Regression', 'is_correct_logistic_regression'),
                ('SVM', 'is_correct_svm'),
                ('Random Forest', 'is_correct_random_forest'),
                ('Ensemble', 'is_correct_ensemble')
            ]
            
            for model_name, col_name in correct_models:
                if col_name in final_df.columns:
                    correct_series = final_df[col_name].dropna()
                    if len(correct_series) > 0:
                        correct_count = correct_series.sum()
                        accuracy = correct_count / len(correct_series)
                        f.write(f"\n{model_name}:\n")
                        f.write(f"  Совпадения: {int(correct_count)}/{len(correct_series)}\n")
                        f.write(f"  Точность: {accuracy:.1%}\n")
            
            # Детальная статистика по классам
            f.write("\n--- Детальная статистика по классам ---\n")
            for label in ['Positive', 'Negative', 'Neutral']:
                subset = final_df[final_df['actual_sentiment'] == label]
                if len(subset) == 0:
                    continue
                
                f.write(f"\n{label}: всего {len(subset)}\n")
                
                for model_name, col_name in correct_models:
                    if col_name in subset.columns:
                        correct = subset[col_name].sum()
                        if len(subset) > 0:
                            acc = correct / len(subset)
                            f.write(f"  {model_name:20}: {int(correct)}/{len(subset)} ({acc:.1%})\n")

def save_summary(final_df, summary_file, models):
    try:
        summary_data = []
        
        # Общая статистика
        summary_data.append(['total_with_rating', len(final_df[final_df['actual_sentiment'].notna()])])
        
        # Статистика по каждой модели
        for model_name, col_name in models:
            if col_name in final_df.columns:
                correct_series = final_df[col_name].dropna()
                if len(correct_series) > 0:
                    summary_data.append([f'correct_predictions_{model_name.lower().replace(" ", "_")}', 
                                        int(correct_series.sum())])
                    summary_data.append([f'accuracy_{model_name.lower().replace(" ", "_")}', 
                                        correct_series.sum() / len(correct_series)])
        
        # Статистика по классам для каждой модели
        for label in ['Positive', 'Negative', 'Neutral']:
            subset = final_df[final_df['actual_sentiment'] == label]
            summary_data.append([f'{label.lower()}_count', len(subset)])
            
            for model_name, col_name in models:
                if col_name in subset.columns and len(subset) > 0:
                    correct = subset[col_name].sum()
                    accuracy = correct / len(subset) if len(subset) > 0 else 0
                    summary_data.append([f'{label.lower()}_accuracy_{model_name.lower().replace(" ", "_")}', 
                                        accuracy])
        
        # Добавляем средние scores
        score_columns = [
            ('vader_score', 'vader'),
            ('flair_score', 'flair'),
            ('rubert_score', 'rubert'),
            ('roberta_score', 'roberta'),
            ('distilbert_score', 'distilbert'),
            ('ensemble_score', 'ensemble')
        ]
        
        # Добавляем scores от scikit-learn моделей
        sklearn_score_cols = [col for col in final_df.columns if col.endswith('_score') and col not in
                             ['vader_score', 'flair_score', 'rubert_score', 'roberta_score', 
                              'distilbert_score', 'ensemble_score']]
        
        for col_name in sklearn_score_cols:
            prefix = col_name.replace('_score', '')
            score_columns.append((col_name, prefix))
        
        for col_name, prefix in score_columns:
            if col_name in final_df.columns:
                summary_data.append([f'avg_{prefix}_score', final_df[col_name].mean()])
        
        summary_df = pd.DataFrame(summary_data, columns=['metric', 'value'])
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        logger.info(f"Сводка сохранена в '{summary_file}'.")
    except Exception as e:
        logger.error(f"Ошибка сохранения сводки: {e}")


if __name__ == "__main__":
    
    input_csv = "data/отзывы.csv"             
    output_csv = "answer/отзывы(результат работы).csv"  
    text_col = "text"                     
    rating_col = "rating"                   
    summary_csv = "answer/sentiment_summary.csv" # файл со сводкой
    stats_file = "answer/sentiment_statistics.txt" # файл с логами
    models_dir = "models"  # директория с предобученными моделями .joblib

    try:
        asyncio.run(analyze_sentiment_from_csv(
            input_csv, output_csv, summary_csv, stats_file, 
            text_col, rating_col, models_dir
        ))
    except KeyboardInterrupt:
        logger.info("Анализ прерван пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")