#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Программа для обучения моделей анализа тональности русских текстов
с использованием scikit-learn и pymorphy3
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
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
import pymorphy3
import warnings
warnings.filterwarnings('ignore')

# ==================== ПАРАМЕТРЫ КОНФИГУРАЦИИ ====================
# Измените эти параметры для настройки обучения

# Параметры данных
DATA_CONFIG = {
    # 'file_path': 'data/sentiment_dataset_merged_2.csv',  # Путь к файлу с данными
    'file_path': 'data/sentiment_dataset_merged_3_shuffled.csv',
    'text_column': 'text',                              # Колонка с текстами
    'label_column': 'label',                             # Колонка с метками
    'encoding': 'utf-8',                                 # Кодировка файла
    'max_samples': None,                                 # Максимум записей (None = все)
    'test_size': 0.2,                                    # Размер тестовой выборки
    'random_state': 42,                                  # Random state
    'sample_random': False ,                              # Случайная выборка
    'nrows':100000
}

# Параметры обучения
TRAINING_CONFIG = {
    'output_dir': 'models',                              # Директория для сохранения
    'use_grid_search': False,                            # Использовать GridSearch
    'models_to_train': [                                 # Модели для обучения
        'logistic_regression',
        'svm_linear',  # Отдельно линейный SVM
        # 'svm_rbf',     # Отдельно RBF SVM        
        'random_forest'
    ]
}

# Параметры TF-IDF векторизации
TFIDF_CONFIG = {
    'max_features': 10000,                               # Максимум признаков
    'ngram_range': (1, 2),                               # Диапазон n-грамм
    'min_df': 5,                                          # Минимальная частота документа
    'max_df': 0.8,                                        # Максимальная частота документа
    'sublinear_tf': True                                  # Логарифмическое масштабирование TF
}

# Параметры моделей
# Замените секцию MODELS_CONFIG для SVM на:

MODELS_CONFIG = {
    'logistic_regression': {
        'C': 1.0,
        'max_iter': 1000,
        'solver': 'lbfgs',
        'class_weight': 'balanced'
    },
    'svm_linear': {  # Отдельная модель для линейного SVM
        'C': 1.0,
        'loss': 'squared_hinge',  # Для LinearSVC
        'class_weight': 'balanced',
        'random_state': 42,
        'max_iter': 2000,
        'dual': False  # Для больших данных
    },
    'svm_rbf': {  # Отдельная модель для RBF SVM
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'class_weight': 'balanced',
        'probability': False,  # Отключаем для скорости
        'random_state': 42,
        'cache_size': 1000  # Кэш для ускорения
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': 42
    }
}


# Параметры GridSearch (используются если use_grid_search = True)
GRID_SEARCH_CONFIG = {
    'cv_folds': 3,
    'scoring': 'f1_weighted',
    'n_jobs': -1,
    'verbose': 1,
    
    # Сетки параметров для каждой модели
    'logistic_regression_params': {
        'tfidf__max_features': [5000, 10000, 15000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__solver': ['lbfgs', 'liblinear']
    },
    'svm_params': {                                       # Объединенные параметры для SVM
        'tfidf__max_features': [5000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__kernel': ['linear', 'rbf'],          # Выбор между linear и rbf
        'classifier__gamma': ['scale', 'auto', 0.1, 1.0]  # Для RBF ядра
    },
    'random_forest_params': {
        'tfidf__max_features': [5000, 10000],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 20, 30],
        'classifier__min_samples_split': [2, 5]
    }
}

# ==================== КОНЕЦ ПАРАМЕТРОВ ====================


# Загрузка ресурсов NLTK
def download_nltk_resources():
    """Загрузка необходимых ресурсов NLTK"""
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt_tab', 'punkt_tab')
    ]
    
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Загрузка ресурса NLTK: {resource_name}")
            nltk.download(resource_name)


class TextPreprocessor:
    """
    Класс для предобработки русских текстов
    """
    
    def __init__(self):
        """Инициализация препроцессора"""
        self.morph = None
        self.stop_words = None
        self._initialize_resources()
        
        # Список символов для удаления
        self.punctuation_chars = set(string.punctuation + '0123456789')
        self.url_indicators = ['http', 'https', 'www', '.com', '.ru', '.org', '.net']
        self.mention_indicators = ['@']
        self.hashtag_indicators = ['#']
    
    def _initialize_resources(self):
        """Инициализация морфологического анализатора и стоп-слов"""
        try:
            self.morph = pymorphy3.MorphAnalyzer()
            self.stop_words = set(stopwords.words('russian'))
            print("✓ Морфологический анализатор (pymorphy3) и стоп-слова загружены")
        except Exception as e:
            print(f"⚠ Предупреждение: Не удалось загрузить pymorphy3: {e}")
            print("  Предобработка будет работать в ограниченном режиме")
            self.morph = None
            self.stop_words = set()
    
    def preprocess(self, text):
        """
        Предобработка русского текста
        
        Args:
            text: Исходный текст
            
        Returns:
            Очищенный и лемматизированный текст
        """
        if not isinstance(text, str) or not text.strip():
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
    
    def preprocess_batch(self, texts, verbose=True):
        """
        Пакетная предобработка текстов
        
        Args:
            texts: Список текстов
            verbose: Выводить прогресс
            
        Returns:
            Список обработанных текстов
        """
        processed = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if verbose and i > 0 and i % 1000 == 0:
                print(f"  Обработано {i}/{total} текстов")
            processed.append(self.preprocess(text))
        
        if verbose:
            print(f"  Обработано {total}/{total} текстов")
        
        return processed


class SklearnModelTrainer:
    """
    Класс для обучения моделей scikit-learn
    """
    
    def __init__(self, output_dir='models', random_state=42):
        """
        Инициализация тренера моделей
        
        Args:
            output_dir: Директория для сохранения моделей
            random_state: Random state для воспроизводимости
        """
        self.output_dir = output_dir
        self.random_state = random_state
        self.preprocessor = TextPreprocessor()
        self.models = {}
        self.results = {}
        
        # Создание директории для моделей
        os.makedirs(output_dir, exist_ok=True)
        
        # Создание директории для результатов
        self.results_dir = os.path.join(output_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def prepare_labels(self, y):
        """
        Подготовка меток для обучения
        
        Args:
            y: Исходные метки
            
        Returns:
            tuple: (числовые метки, mapping словарь, список названий классов)
        """
        unique_labels = np.unique(y)
        unique_labels = sorted(unique_labels)  # Сортируем для консистентности
        
        # Создаем отображение строковых меток в числа
        mapping = {label: i for i, label in enumerate(unique_labels)}
        y_numeric = np.array([mapping[label] for label in y])
        
        # Список названий классов для отчетов
        target_names = list(unique_labels)
        
        print(f"\nНайдены метки: {unique_labels}")
        print(f"Преобразованы в числа: {mapping}")
        
        return y_numeric, mapping, target_names
    
    def create_model(self, model_type):
        """
        Создание модели и пайплайна на основе конфигурации
        """
        # Настройки TF-IDF из глобальной конфигурации
        vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
        
        # Выбор модели из глобальной конфигурации
        if model_type == 'logistic_regression':
            params = MODELS_CONFIG['logistic_regression'].copy()
            model = LogisticRegression(**params)
            
        elif model_type == 'svm_linear':
            # Используем LinearSVC для линейного SVM (быстрее)
            from sklearn.svm import LinearSVC
            params = MODELS_CONFIG['svm_linear'].copy()
            model = LinearSVC(**params)
            
        elif model_type == 'svm_rbf':
            # Используем SVC с RBF ядром
            params = MODELS_CONFIG['svm_rbf'].copy()
            model = SVC(**params)
            
        elif model_type == 'random_forest':
            params = MODELS_CONFIG['random_forest'].copy()
            model = RandomForestClassifier(**params)
            
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
        
        # Создание пайплайна
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('classifier', model)
        ])
        
        return pipeline
    
    def train_multiple_models(self, X_train, y_train, X_test, y_test, 
                            label_mapping, target_names):
        """
        Обучение нескольких моделей и их сравнение
        
        Args:
            X_train: Обучающие тексты
            y_train: Обучающие метки
            X_test: Тестовые тексты
            y_test: Тестовые метки
            label_mapping: Отображение меток
            target_names: Названия классов
            
        Returns:
            Словарь с обученными моделями и результатами
        """
        models_to_train = TRAINING_CONFIG['models_to_train']
        use_grid_search = TRAINING_CONFIG['use_grid_search']
        
        # Предобработка всех текстов сразу
        print("\nПредобработка всех текстов...")
        print("Обучающая выборка:")
        X_train_processed = self.preprocessor.preprocess_batch(X_train)
        print("Тестовая выборка:")
        X_test_processed = self.preprocessor.preprocess_batch(X_test)
        
        for model_type in models_to_train:
            print(f"\n{'='*60}")
            print(f"Обучение модели: {model_type}")
            print('='*60)
            
            # Создание модели
            if use_grid_search:
                # Получение параметров для GridSearch
                param_grid_key = f"{model_type}_params"
                if param_grid_key in GRID_SEARCH_CONFIG:
                    param_grid = GRID_SEARCH_CONFIG[param_grid_key]
                    
                    pipeline = self.create_model(model_type)
                    grid_search = GridSearchCV(
                        pipeline, 
                        param_grid, 
                        cv=GRID_SEARCH_CONFIG['cv_folds'],
                        scoring=GRID_SEARCH_CONFIG['scoring'],
                        n_jobs=GRID_SEARCH_CONFIG['n_jobs'],
                        verbose=GRID_SEARCH_CONFIG['verbose']
                    )
                    grid_search.fit(X_train_processed, y_train)
                    
                    pipeline = grid_search.best_estimator_
                    print(f"Лучшие параметры: {grid_search.best_params_}")
                    print(f"Лучшая CV оценка: {grid_search.best_score_:.4f}")
                else:
                    print(f"Предупреждение: Нет параметров GridSearch для {model_type}")
                    pipeline = self.create_model(model_type)
                    pipeline.fit(X_train_processed, y_train)
            else:
                # Обучение с параметрами по умолчанию
                pipeline = self.create_model(model_type)
                pipeline.fit(X_train_processed, y_train)
            
            # Оценка на тестовой выборке
            y_pred = pipeline.predict(X_test_processed)
            
            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"\nРезультаты на тестовой выборке:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-score (weighted): {f1:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            report = classification_report(
                y_test, y_pred, 
                target_names=target_names,
                digits=4
            )
            print(report)
            
            # Сохранение модели
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{model_type}_model.joblib"
            model_path = os.path.join(self.output_dir, model_filename)
            
            model_data = {
                'pipeline': pipeline,
                'label_mapping': label_mapping,
                'target_names': target_names,
                'model_type': model_type,
                'train_date': timestamp,
                'metrics': {
                    'accuracy': accuracy,
                    'f1_score': f1
                }
            }
            
            joblib.dump(model_data, model_path)
            print(f"Модель сохранена: {model_path}")
            
            # Сохранение результатов
            self.models[model_type] = pipeline
            self.results[model_type] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'model_path': model_path,
                'report': report,
                'predictions': y_pred
            }
        
        return self.models, self.results
    
    def save_results(self, filename=None):
        """
        Сохранение результатов сравнения моделей
        
        Args:
            filename: Имя файла для сохранения
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"
        
        results_path = os.path.join(self.results_dir, filename)
        
        # Подготовка данных для сохранения
        results_data = {}
        for model_name, metrics in self.results.items():
            results_data[model_name] = {
                'accuracy': float(metrics['accuracy']),
                'f1_score': float(metrics['f1_score']),
                'model_path': metrics['model_path']
            }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nРезультаты сохранены в {results_path}")
        
        # Создание текстового отчета
        report_path = os.path.join(self.results_dir, f"report_{timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("ОТЧЕТ ОБ ОБУЧЕНИИ МОДЕЛЕЙ\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Дата обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Параметры данных:\n")
            f.write(f"  Файл: {DATA_CONFIG['file_path']}\n")
            f.write(f"  Колонка текста: {DATA_CONFIG['text_column']}\n")
            f.write(f"  Колонка меток: {DATA_CONFIG['label_column']}\n")
            f.write(f"  Тестовая выборка: {DATA_CONFIG['test_size']*100}%\n\n")
            
            f.write("СРАВНЕНИЕ МОДЕЛЕЙ:\n")
            f.write("-"*40 + "\n")
            for model_name, metrics in self.results.items():
                f.write(f"{model_name:20} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}\n")
            
            f.write("\n\nДЕТАЛЬНЫЕ ОТЧЕТЫ:\n")
            f.write("="*60 + "\n")
            for model_name, metrics in self.results.items():
                f.write(f"\nМодель: {model_name}\n")
                f.write("-"*40 + "\n")
                f.write(metrics['report'])
                f.write("\n")
        
        print(f"Отчет сохранен в {report_path}")
        
        return results_path, report_path
    
    def print_summary(self):
        """Вывод сводки по обученным моделям"""
        print(f"\n{'='*60}")
        print("СВОДКА ПО ОБУЧЕННЫМ МОДЕЛЯМ")
        print('='*60)
        
        # Сортировка по F1-score
        sorted_models = sorted(
            self.results.items(), 
            key=lambda x: x[1]['f1_score'], 
            reverse=True
        )
        
        print(f"\n{'Модель':20} | {'Accuracy':10} | {'F1-score':10} | {'Рейтинг'}")
        print('-'*60)
        
        for i, (model_name, metrics) in enumerate(sorted_models, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"{model_name:20} | {metrics['accuracy']:.4f}     | {metrics['f1_score']:.4f}     | {medal}")
        
        # Лучшая модель
        best_model = sorted_models[0][0]
        best_metrics = sorted_models[0][1]
        
        print(f"\n🏆 Лучшая модель: {best_model}")
        print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"   F1-score: {best_metrics['f1_score']:.4f}")
        print(f"   Путь: {best_metrics['model_path']}")

def load_data():
    """
    Загрузка данных из файла согласно конфигурации
    
    Returns:
        DataFrame с данными
    """
    file_path = DATA_CONFIG['file_path']
    text_column = DATA_CONFIG['text_column']
    label_column = DATA_CONFIG['label_column']
    encoding = DATA_CONFIG['encoding']
    max_samples = DATA_CONFIG['max_samples']
    sample_random = DATA_CONFIG['sample_random']
    nrows = DATA_CONFIG['nrows']

    
    print(f"\nЗагрузка данных из {file_path}...")
    
    if nrows != -1:
        df = pd.read_csv(file_path, encoding=encoding, nrows=nrows)
    else:
        df = pd.read_csv(file_path, encoding=encoding)
    
    # Проверка наличия колонок
    if text_column not in df.columns:
        raise ValueError(f"Колонка '{text_column}' не найдена. Доступные колонки: {list(df.columns)}")
    
    if label_column not in df.columns:
        raise ValueError(f"Колонка '{label_column}' не найдена. Доступные колонки: {list(df.columns)}")
    
    # Удаление пустых значений
    initial_len = len(df)
    df = df.dropna(subset=[text_column, label_column])
    
    # Удаление пустых строк
    df = df[df[text_column].astype(str).str.strip() != '']
    
    print(f"Загружено {len(df)} записей из {initial_len} (удалено {initial_len - len(df)} пустых)")
    
    # Опционально: ограничение количества записей
    if max_samples and max_samples < len(df):
        if sample_random:
            df = df.sample(n=max_samples, random_state=DATA_CONFIG['random_state'])
            print(f"Используем случайную выборку из {max_samples} записей")
        else:
            df = df.head(max_samples)
            print(f"Используем первые {max_samples} записей")
    
    # Статистика по меткам
    unique_labels = df[label_column].unique()
    print(f"\nУникальные метки: {unique_labels}")
    
    print("\nРаспределение меток:")
    for label in unique_labels:
        count = len(df[df[label_column] == label])
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    return df


def main():
    """Основная функция программы"""
    
    print("="*60)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ АНАЛИЗА ТОНАЛЬНОСТИ")
    print("="*60)
    print(f"Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nТЕКУЩАЯ КОНФИГУРАЦИЯ:")
    print(f"  Файл данных: {DATA_CONFIG['file_path']}")
    print(f"  Колонка текста: {DATA_CONFIG['text_column']}")
    print(f"  Колонка меток: {DATA_CONFIG['label_column']}")
    print(f"  Тестовая выборка: {DATA_CONFIG['test_size']*100}%")
    print(f"  GridSearch: {'включен' if TRAINING_CONFIG['use_grid_search'] else 'выключен'}")
    print(f"  Модели: {', '.join(TRAINING_CONFIG['models_to_train'])}")
    
    # Загрузка ресурсов NLTK
    download_nltk_resources()
    
    try:
        # Загрузка данных
        df = load_data()
        
        # Проверка наличия данных
        if len(df) == 0:
            print("Ошибка: Нет данных для обучения")
            return
        
        # Подготовка данных
        X = df[DATA_CONFIG['text_column']].astype(str).values
        y = df[DATA_CONFIG['label_column']].values
        
        # Проверка количества классов
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise ValueError(f"Ошибка: найдено только {len(unique_labels)} класс(ов). Нужно минимум 2 класса.")
        
        # Создание тренера
        trainer = SklearnModelTrainer(
            output_dir=TRAINING_CONFIG['output_dir'],
            random_state=DATA_CONFIG['random_state']
        )
        
        # Преобразование меток в числовой формат
        y_numeric, label_mapping, target_names = trainer.prepare_labels(y)
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_numeric, 
            test_size=DATA_CONFIG['test_size'], 
            random_state=DATA_CONFIG['random_state'], 
            stratify=y_numeric
        )
        
        print(f"\nРазмер обучающей выборки: {len(X_train)}")
        print(f"Размер тестовой выборки: {len(X_test)}")
        
        # Обучение моделей
        models, results = trainer.train_multiple_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            label_mapping=label_mapping,
            target_names=target_names
        )
        
        # Сохранение результатов
        trainer.save_results()
        
        # Вывод сводки
        trainer.print_summary()
        
        print(f"\n✅ Обучение завершено успешно!")
        print(f"   Модели сохранены в директории: {TRAINING_CONFIG['output_dir']}")
        
    except FileNotFoundError:
        print(f"\n❌ Ошибка: Файл {DATA_CONFIG['file_path']} не найден!")
        print("   Убедитесь, что файл существует и путь указан правильно.")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
