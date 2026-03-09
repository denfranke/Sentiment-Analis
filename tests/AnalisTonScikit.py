import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Замена pymorphy2 на pymorphy3
import pymorphy3

# Загрузка ресурсов NLTK (при первом запуске)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')


class SklearnSentimentAnalyzer:
    """
    Класс для анализа тональности русских текстов с использованием scikit-learn моделей
    """
    
    def __init__(self, model_type='logistic_regression', use_preprocessing=True):
        """
        Инициализация анализатора тональности
        
        Args:
            model_type: Тип модели ('logistic_regression', 'svm', 'random_forest')
            use_preprocessing: Использовать ли предобработку текста
        """
        self.model_type = model_type
        self.use_preprocessing = use_preprocessing
        self.pipeline = None
        self.vectorizer = None
        self.model = None
        self.label_mapping = None  # Словарь для преобразования меток
        self.target_names = None   # Список названий классов для отчетов
        
        # Использование pymorphy3 вместо pymorphy2
        self.morph = None
        self.stop_words = None
        
        if use_preprocessing:
            try:
                self.morph = pymorphy3.MorphAnalyzer()
                self.stop_words = set(stopwords.words('russian'))
                print("Морфологический анализатор (pymorphy3) и стоп-слова загружены")
            except Exception as e:
                print(f"Предупреждение: Не удалось загрузить pymorphy3: {e}")
                print("Предобработка будет работать в ограниченном режиме")
                self.morph = None
                self.stop_words = set()
        
        # Список символов для удаления
        self.punctuation_chars = set(string.punctuation + '0123456789')
        self.url_indicators = ['http', 'https', 'www', '.com', '.ru', '.org', '.net']
        self.mention_indicators = ['@']
        self.hashtag_indicators = ['#']
    
    def preprocess_text(self, text):
        """
        Предобработка русского текста
        
        Args:
            text: Исходный текст
            
        Returns:
            Очищенный и лемматизированный текст
        """
        if not isinstance(text, str):
            return ""
        
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление URL, упоминаний, хештегов
        words = text.split()
        filtered_words = []
        for word in words:
            # Пропускаем URL
            if any(indicator in word for indicator in self.url_indicators):
                continue
            # Пропускаем @упоминания
            if any(word.startswith(indicator) for indicator in self.mention_indicators):
                continue
            # Пропускаем #хештеги
            if any(word.startswith(indicator) for indicator in self.hashtag_indicators):
                continue
            filtered_words.append(word)
        text = ' '.join(filtered_words)
        
        # Удаление пунктуации и цифр
        text = ''.join(char for char in text if char not in self.punctuation_chars)
        
        # Нормализация пробелов
        text = ' '.join(text.split())
        
        # Лемматизация через pymorphy3
        if self.morph is not None and text.strip():
            try:
                # Токенизация
                try:
                    tokens = word_tokenize(text, language='russian')
                except:
                    tokens = text.split()
                
                lemmatized_tokens = []
                for token in tokens:
                    if token and token not in self.stop_words and len(token) > 2:
                        try:
                            lemma = self.morph.parse(token)[0].normal_form
                            lemmatized_tokens.append(lemma)
                        except:
                            if len(token) > 2:
                                lemmatized_tokens.append(token)
                
                return ' '.join(lemmatized_tokens)
            except Exception as e:
                return text
        else:
            return text
    
    def create_pipeline(self, **kwargs):
        """Создание пайплайна с TF-IDF векторизатором и выбранной моделью"""
        # Настройки TF-IDF
        tfidf_params = {
            'max_features': kwargs.get('max_features', 10000),
            'ngram_range': kwargs.get('ngram_range', (1, 2)),
            'min_df': kwargs.get('min_df', 5),
            'max_df': kwargs.get('max_df', 0.8),
            'sublinear_tf': True
        }
        
        self.vectorizer = TfidfVectorizer(**tfidf_params)
        
        # Выбор модели
        if self.model_type == 'logistic_regression':
            model_params = {
                'C': kwargs.get('C', 1.0),
                'max_iter': kwargs.get('max_iter', 1000),
                'random_state': kwargs.get('random_state', 42),
                'class_weight': kwargs.get('class_weight', 'balanced')
            }
            self.model = LogisticRegression(**model_params)
            
        elif self.model_type == 'svm':
            use_linear = kwargs.get('use_linear_svm', True)
            if use_linear:
                model_params = {
                    'C': kwargs.get('C', 1.0),
                    'max_iter': kwargs.get('max_iter', 1000),
                    'random_state': kwargs.get('random_state', 42),
                    'class_weight': kwargs.get('class_weight', 'balanced'),
                    'dual': False
                }
                self.model = LinearSVC(**model_params)
            else:
                model_params = {
                    'C': kwargs.get('C', 1.0),
                    'kernel': kwargs.get('kernel', 'rbf'),
                    'gamma': kwargs.get('gamma', 'scale'),
                    'random_state': kwargs.get('random_state', 42),
                    'class_weight': kwargs.get('class_weight', 'balanced')
                }
                self.model = SVC(**model_params)
                
        elif self.model_type == 'random_forest':
            model_params = {
                'n_estimators': kwargs.get('n_estimators', 100),
                'max_depth': kwargs.get('max_depth', None),
                'min_samples_split': kwargs.get('min_samples_split', 2),
                'min_samples_leaf': kwargs.get('min_samples_leaf', 1),
                'random_state': kwargs.get('random_state', 42),
                'class_weight': kwargs.get('class_weight', 'balanced'),
                'n_jobs': kwargs.get('n_jobs', -1)
            }
            self.model = RandomForestClassifier(**model_params)
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {self.model_type}")
        
        # Создание пайплайна
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.model)
        ])
        
        return self.pipeline
    
    def train(self, X_train, y_train, **kwargs):
        """Обучение модели"""
        if self.pipeline is None:
            self.create_pipeline(**kwargs)
        
        if self.use_preprocessing:
            print("Применение предобработки к обучающим текстам...")
            X_train = [self.preprocess_text(text) for text in X_train]
        
        self.pipeline.fit(X_train, y_train)
        return self
    
    def predict(self, texts, return_labels=True):
        """Предсказание тональности"""
        if self.pipeline is None:
            raise ValueError("Модель не обучена. Сначала вызовите train().")
        
        if isinstance(texts, str):
            texts = [texts]
        
        if self.use_preprocessing:
            texts = [self.preprocess_text(text) for text in texts]
        
        predictions = self.pipeline.predict(texts)
        
        if return_labels and self.label_mapping:
            # Создаем обратное отображение
            inverse_mapping = {v: k for k, v in self.label_mapping.items()}
            predictions = [inverse_mapping[p] for p in predictions]
        
        return predictions[0] if len(predictions) == 1 else predictions
    
    def save(self, model_dir='models', name=None):
        """Сохранение модели"""
        os.makedirs(model_dir, exist_ok=True)
        
        if name is None:
            name = self.model_type
        
        model_path = os.path.join(model_dir, f"{name}_model.joblib")
        joblib.dump({
            'pipeline': self.pipeline,
            'label_mapping': self.label_mapping,
            'target_names': self.target_names,
            'model_type': self.model_type
        }, model_path)
        print(f"Модель сохранена в {model_path}")
        
        return model_path
    
    def load(self, model_path):
        """Загрузка модели"""
        data = joblib.load(model_path)
        self.pipeline = data['pipeline']
        self.label_mapping = data['label_mapping']
        self.target_names = data.get('target_names', None)
        self.model_type = data['model_type']
        self.vectorizer = self.pipeline.named_steps['tfidf']
        self.model = self.pipeline.named_steps['classifier']
        print(f"Модель загружена из {model_path}")
        return self


def prepare_labels(y):
    """
    Подготовка меток для обучения
    
    Args:
        y: Исходные метки (строковые)
        
    Returns:
        tuple: (числовые метки, mapping словарь, список названий классов)
    """
    unique_labels = np.unique(y)
    
    # Сортируем для консистентности
    unique_labels = sorted(unique_labels)
    
    # Создаем отображение строковых меток в числа
    mapping = {label: i for i, label in enumerate(unique_labels)}
    y_numeric = np.array([mapping[label] for label in y])
    
    # Список названий классов для отчетов
    target_names = list(unique_labels)
    
    print(f"Найдены метки: {unique_labels}")
    print(f"Преобразованы в числа: {mapping}")
    print(f"Названия классов для отчетов: {target_names}")
    
    return y_numeric, mapping, target_names


def train_sklearn_models_from_dataframe(df, text_column, rating_column, model_dir='models', 
                                         test_size=0.2, random_state=42, use_grid_search=False,
                                         sample_size=None):
    """
    Обучение нескольких моделей scikit-learn для анализа тональности
    
    Args:
        df: DataFrame с данными
        text_column: Название колонки с текстами
        rating_column: Название колонки с метками тональности
        model_dir: Директория для сохранения моделей
        test_size: Размер тестовой выборки
        random_state: Random state для воспроизводимости
        use_grid_search: Использовать ли GridSearchCV для подбора параметров
        sample_size: Если указано, использует только sample_size записей (для тестирования)
        
    Returns:
        Словарь с обученными моделями и результатами
    """
    X = df[text_column].values
    y = df[rating_column].values
    
    # Опционально: использовать только часть данных для тестирования
    if sample_size and sample_size < len(X):
        print(f"\nИспользуем только {sample_size} записей для тестирования...")
        indices = np.random.choice(len(X), sample_size, replace=False)
        X = X[indices]
        y = y[indices]
    
    unique_labels = np.unique(y)
    print(f"\nУникальные метки в данных: {unique_labels}")
    
    if len(unique_labels) < 2:
        raise ValueError(f"Ошибка: найдено только {len(unique_labels)} класс(ов).")
    
    # Преобразование меток в числовой формат
    y_numeric, label_mapping, target_names = prepare_labels(y)
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_numeric, test_size=test_size, random_state=random_state, stratify=y_numeric
    )
    
    print(f"\nРазмер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")
    
    # Статистика по классам
    print("\nРаспределение классов в обучающей выборке:")
    for i, name in enumerate(target_names):
        count = np.sum(y_train == i)
        print(f"  {name}: {count} ({count/len(y_train)*100:.1f}%)")
    
    trained_models = {}
    results = {}
    
    sklearn_models = {
        'logistic_regression': SklearnSentimentAnalyzer('logistic_regression', use_preprocessing=True),
        'svm': SklearnSentimentAnalyzer('svm', use_preprocessing=True),
        'random_forest': SklearnSentimentAnalyzer('random_forest', use_preprocessing=True)
    }
    
    for model_name, analyzer in sklearn_models.items():
        print(f"\n{'='*60}")
        print(f"Обучение модели: {model_name}")
        print('='*60)
        
        # Сохраняем метаданные
        analyzer.label_mapping = label_mapping
        analyzer.target_names = target_names
        
        if use_grid_search:
            # Подбор гиперпараметров
            if model_name == 'logistic_regression':
                param_grid = {
                    'classifier__C': [0.1, 1.0, 10.0],
                    'tfidf__max_features': [5000, 10000]
                }
            elif model_name == 'svm':
                param_grid = {
                    'classifier__C': [0.1, 1.0, 10.0],
                    'tfidf__ngram_range': [(1, 1), (1, 2)]
                }
            elif model_name == 'random_forest':
                param_grid = {
                    'classifier__n_estimators': [50, 100],
                    'classifier__max_depth': [None, 20],
                    'tfidf__max_features': [5000, 10000]
                }
            
            pipeline = analyzer.create_pipeline()
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=3, 
                scoring='f1_weighted', n_jobs=-1, verbose=1
            )
            
            print("Предобработка текстов для GridSearch...")
            X_train_processed = [analyzer.preprocess_text(text) for text in X_train]
            grid_search.fit(X_train_processed, y_train)
            
            analyzer.pipeline = grid_search.best_estimator_
            analyzer.vectorizer = analyzer.pipeline.named_steps['tfidf']
            analyzer.model = analyzer.pipeline.named_steps['classifier']
            
            print(f"Лучшие параметры: {grid_search.best_params_}")
            print(f"Лучшая CV оценка: {grid_search.best_score_:.4f}")
        else:
            analyzer.train(X_train, y_train)
        
        # Оценка на тестовой выборке
        print("\nОценка на тестовой выборке...")
        if analyzer.use_preprocessing:
            X_test_processed = [analyzer.preprocess_text(text) for text in X_test]
        else:
            X_test_processed = X_test
        
        y_pred = analyzer.pipeline.predict(X_test_processed)
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\nРезультаты на тестовой выборке:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score (weighted): {f1:.4f}")
        
        # Classification report с правильными названиями классов
        print("\nClassification Report:")
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=target_names,  # Используем строковые названия
            digits=4
        )
        print(report)
        
        # Сохранение модели
        model_path = analyzer.save(model_dir, name=model_name)
        
        trained_models[model_name] = analyzer
        results[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'model_path': model_path,
            'predictions': y_pred,
            'true_labels': y_test,
            'label_mapping': label_mapping,
            'target_names': target_names,
            'report': report
        }
    
    # Сравнение моделей
    print(f"\n{'='*60}")
    print("СРАВНЕНИЕ МОДЕЛЕЙ:")
    print('='*60)
    for model_name, metrics in results.items():
        print(f"{model_name:20} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")
    
    return trained_models, results


def load_and_prepare_data(file_path, text_column, rating_column, max_samples=None):
    """
    Загрузка и подготовка данных для обучения
    
    Args:
        file_path: Путь к файлу с данными
        text_column: Колонка с текстами
        rating_column: Колонка с метками тональности
        max_samples: Максимальное количество записей для загрузки (для тестирования)
        
    Returns:
        Подготовленный DataFrame
    """
    print(f"Загрузка данных из {file_path}...")
    
    # Определение формата файла
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        # Пробуем прочитать как CSV по умолчанию
        try:
            df = pd.read_csv(file_path)
        except:
            raise ValueError("Неподдерживаемый формат файла")
    
    # Удаление пустых значений
    initial_len = len(df)
    df = df.dropna(subset=[text_column, rating_column])
    
    # Опционально: ограничить количество записей
    if max_samples and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=42)
        print(f"Используем {max_samples} записей (случайная выборка)")
    
    print(f"Загружено {len(df)} записей (удалено {initial_len - len(df)} пустых)")
    print(f"Колонки: {list(df.columns)}")
    
    # Вывод информации о метках
    unique_labels = df[rating_column].unique()
    print(f"Уникальные метки: {unique_labels}")
    
    # Статистика по меткам
    print("\nРаспределение меток:")
    for label in unique_labels:
        count = len(df[df[rating_column] == label])
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    return df


def test_predictions(models, test_texts):
    """
    Тестирование моделей на примерах
    
    Args:
        models: Словарь обученных моделей
        test_texts: Список тестовых текстов
    """
    print(f"\n{'='*60}")
    print("ТЕСТОВЫЕ ПРЕДСКАЗАНИЯ")
    print('='*60)
    
    for model_name, model in models.items():
        print(f"\nМодель: {model_name}")
        print("-" * 50)
        for i, text in enumerate(test_texts, 1):
            sentiment = model.predict(text)
            print(f"{i}. {text[:70]}...")
            print(f"   → Тональность: {sentiment}")
            print()


if __name__ == "__main__":
    print("="*60)
    print("КЛАССИФИКАТОР ТОНАЛЬНОСТИ РУССКИХ ТЕКСТОВ")
    print("="*60)
    
    # Загрузка данных (используем максимум 10000 записей для быстрого тестирования)
    file_path = 'data/sentiment_dataset_merged.csv'
    
    try:
        df = load_and_prepare_data(
            file_path, 
            'text', 
            'label',
            max_samples=10000  # Ограничиваем для быстрого тестирования
        )
        
        # Обучение моделей
        models, results = train_sklearn_models_from_dataframe(
            df=df,
            text_column='text',
            rating_column='label',
            model_dir='models',
            use_grid_search=False,  # True для подбора параметров (медленнее)
            sample_size=None  # Можно указать число для тестирования
        )
        
        # Тестовые предсказания
        test_texts = [
            "Этот фильм просто замечательный, очень понравился!",
            "Ужасный фильм, потраченное время зря",
            "Нормальный фильм, но могло быть и лучше",
            "Отличная книга, рекомендую всем!",
            "Разочарован, ожидал большего"
        ]
        
        test_predictions(models, test_texts)
        
    except FileNotFoundError:
        print(f"\nОшибка: Файл {file_path} не найден!")
        print("Убедитесь, что файл существует в указанной директории.")
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()