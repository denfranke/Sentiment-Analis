#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Программа для обучения моделей анализа тональности русских текстов
с использованием scikit-learn и pymorphy3.

Модели: logistic_regression, svm_linear, random_forest
"""

import os
import re
import json
import logging
import warnings
from datetime import datetime, timezone
from functools import lru_cache
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import joblib
import nltk
import pymorphy3

from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')


# ==================== ЛОГИРОВАНИЕ ====================

def setup_logging(log_dir: str = 'logs') -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'train_{ts}.log')

    logger = logging.getLogger('sentiment')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = setup_logging()


# ==================== КОНФИГУРАЦИЯ ====================

DATA_CONFIG = {
    'file_path': 'data/sentiment_dataset_merged_3_shuffled.csv',
    'text_column': 'text',
    'label_column': 'label',
    'encoding': 'utf-8',
    'max_samples': None,
    'test_size': 0.2,
    'random_state': 42,
    'sample_random': False,
    'nrows': 500000,
}

TRAINING_CONFIG = {
    'output_dir': 'models',
    'use_grid_search': False,
    'use_cv_eval': True,
    'cv_folds': 3,
    'build_ensemble': False,       # ensemble отключён — оставляем только 3 модели
    'models_to_train': [
        'logistic_regression',
        'svm_linear',
        'random_forest',
    ],
}

# Общий TF-IDF (для линейных моделей)
TFIDF_CONFIG = {
    'max_features': 50000,
    'ngram_range': (1, 2),
    'min_df': 3,
    'max_df': 0.9,
    'sublinear_tf': True,
    'use_idf': True,
    'smooth_idf': True,
    'token_pattern': r'\w{2,}',
}

# Отдельный TF-IDF для RandomForest — меньше признаков, только униграммы.
# Деревья плохо работают с очень разреженными матрицами: 50k признаков
# превращают обучение в часы и ухудшают качество из-за переобучения на шум.
TFIDF_CONFIG_RF = {
    'max_features': 5000,
    'ngram_range': (1, 1),
    'min_df': 5,
    'max_df': 0.9,
    'sublinear_tf': True,
    'use_idf': True,
    'smooth_idf': True,
    'token_pattern': r'\w{2,}',
}

MODELS_CONFIG = {
    'logistic_regression': {
        'C': 5.0,
        'max_iter': 2000,
        'solver': 'lbfgs',
        'class_weight': 'balanced',
        'n_jobs': -1,
    },
    'svm_linear': {
        'C': 1.0,
        'loss': 'squared_hinge',
        'class_weight': 'balanced',
        'random_state': 42,
        'max_iter': 5000,
        'dual': 'auto',
        'tol': 1e-3,
    },
    'random_forest': {
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'class_weight': 'balanced_subsample',
        'n_jobs': -1,
        'random_state': 42,
        'verbose': 0,
    },
}

GRID_SEARCH_CONFIG = {
    'cv_folds': 3,
    'scoring': 'f1_macro',
    'n_jobs': -1,
    'verbose': 2,
    'use_randomized': True,
    'n_iter': 15,

    'logistic_regression_params': {
        'tfidf__max_features': [30000, 50000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__min_df': [2, 3, 5],
        'classifier__C': [0.5, 1.0, 5.0, 10.0],
    },
    'svm_linear_params': {
        'tfidf__max_features': [30000, 50000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'classifier__C': [0.1, 0.5, 1.0, 5.0],
    },
    'random_forest_params': {
        # Для RF — маленький TF-IDF, иначе обучение будет часами
        'tfidf__max_features': [3000, 5000, 10000],
        'tfidf__ngram_range': [(1, 1)],
        'classifier__n_estimators': [200, 300, 500],
        'classifier__max_depth': [None, 50, 100],
        'classifier__min_samples_leaf': [1, 2, 5],
    },
}


# ==================== NLTK ====================

def download_nltk_resources():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            log.info(f"Загрузка ресурса NLTK: {resource_name}")
            try:
                nltk.download(resource_name, quiet=True)
            except Exception as e:
                log.warning(f"Не удалось загрузить {resource_name}: {e}")


# ==================== ПРЕПРОЦЕССИНГ ====================

_URL_RE = re.compile(r'https?://\S+|www\.\S+')
_MENTION_RE = re.compile(r'@\w+')
_HASHTAG_RE = re.compile(r'#\w+')
_PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)
_DIGIT_RE = re.compile(r'\b\d+\b')
_SPACE_RE = re.compile(r'\s+')


@lru_cache(maxsize=200_000)
def _cached_lemma(morph, word):
    return morph.parse(word)[0].normal_form


def _preprocess_with(morph, stop_words, text):
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = _URL_RE.sub(' ', text)
    text = _MENTION_RE.sub(' ', text)
    text = _HASHTAG_RE.sub(' ', text)
    text = _DIGIT_RE.sub(' ', text)
    text = _PUNCT_RE.sub(' ', text)
    text = _SPACE_RE.sub(' ', text).strip()

    tokens = []
    for token in text.split():
        if len(token) <= 1 or token in stop_words:
            continue
        tokens.append(_cached_lemma(morph, token))
    return ' '.join(tokens)


def _lemmatize_worker(args):
    morph, stop_words, text = args
    return _preprocess_with(morph, stop_words, text)


class TextPreprocessor:
    def __init__(self, use_multiprocessing=True, n_jobs=None):
        self.morph = pymorphy3.MorphAnalyzer()
        self.stop_words = set(stopwords.words('russian'))
        self.stop_words -= {'не', 'ни', 'нет', 'без', 'никогда', 'ничего'}

        self.use_multiprocessing = use_multiprocessing
        self.n_jobs = n_jobs or max(1, cpu_count() - 1)

    def preprocess(self, text):
        return _preprocess_with(self.morph, self.stop_words, text)

    def preprocess_batch(self, texts, verbose=True):
        texts = list(texts)
        if self.use_multiprocessing and len(texts) > 5000:
            try:
                with Pool(self.n_jobs) as pool:
                    args = [(self.morph, self.stop_words, t) for t in texts]
                    return pool.map(_lemmatize_worker, args, chunksize=500)
            except Exception as e:
                log.warning(f"Multiprocessing не удалось ({e}), fallback")
        return [self.preprocess(t) for t in texts]


class PreprocessorTransformer(BaseEstimator, TransformerMixin):
    """Трансформер для встраивания в Pipeline — сохраняется вместе с моделью."""

    def __init__(self, use_multiprocessing=True, n_jobs=None):
        self.use_multiprocessing = use_multiprocessing
        self.n_jobs = n_jobs
        self._preprocessor = None

    def fit(self, X, y=None):
        self._preprocessor = TextPreprocessor(
            use_multiprocessing=self.use_multiprocessing,
            n_jobs=self.n_jobs,
        )
        return self

    def transform(self, X):
        if self._preprocessor is None:
            self._preprocessor = TextPreprocessor(
                use_multiprocessing=self.use_multiprocessing,
                n_jobs=self.n_jobs,
            )
        return self._preprocessor.preprocess_batch(list(X), verbose=False)


# ==================== ТРЕНЕР ====================

class SklearnModelTrainer:

    def __init__(self, output_dir='models', random_state=42):
        self.output_dir = output_dir
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_params = {}

        os.makedirs(output_dir, exist_ok=True)
        self.results_dir = os.path.join(output_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

    def prepare_labels(self, y):
        unique_labels = sorted(np.unique(y))
        mapping = {label: i for i, label in enumerate(unique_labels)}
        y_numeric = np.array([mapping[label] for label in y])
        target_names = list(unique_labels)

        log.info(f"Найдены метки: {unique_labels}")
        log.info(f"Преобразованы в числа: {mapping}")
        return y_numeric, mapping, target_names

    # ---------- factory ----------

    def _get_tfidf_config(self, model_type):
        """Для RandomForest — уменьшенный TF-IDF."""
        if model_type == 'random_forest':
            return TFIDF_CONFIG_RF
        return TFIDF_CONFIG

    def _get_classifier(self, model_type):
        if model_type == 'logistic_regression':
            return LogisticRegression(**MODELS_CONFIG['logistic_regression'])
        if model_type == 'svm_linear':
            return LinearSVC(**MODELS_CONFIG['svm_linear'])
        if model_type == 'random_forest':
            return RandomForestClassifier(**MODELS_CONFIG['random_forest'])
        raise ValueError(f"Неподдерживаемый тип модели: {model_type}")

    def create_model(self, model_type):
        """Pipeline: preprocess -> tfidf -> classifier (для сохранения)."""
        vectorizer = TfidfVectorizer(**self._get_tfidf_config(model_type))
        clf = self._get_classifier(model_type)

        return Pipeline([
            ('preprocess', PreprocessorTransformer()),
            ('tfidf', vectorizer),
            ('classifier', clf),
        ])

    # ---------- training ----------

    def train_multiple_models(self, X_train, y_train, X_test, y_test,
                              label_mapping, target_names):
        models_to_train = TRAINING_CONFIG['models_to_train']
        use_grid_search = TRAINING_CONFIG['use_grid_search']
        use_cv_eval = TRAINING_CONFIG.get('use_cv_eval', True)
        cv_folds = TRAINING_CONFIG.get('cv_folds', 3)

        # Один раз предобрабатываем тексты — ускоряет CV и GridSearch
        log.info("\nПредобработка текстов (один раз)...")
        pre = TextPreprocessor()
        X_train_p = pre.preprocess_batch(X_train)
        X_test_p = pre.preprocess_batch(X_test)
        log.info(f"  Готово: train={len(X_train_p)}, test={len(X_test_p)}")

        for model_type in models_to_train:
            log.info("\n" + "=" * 60)
            log.info(f"Обучение модели: {model_type}")
            log.info("=" * 60)

            vectorizer = TfidfVectorizer(**self._get_tfidf_config(model_type))
            clf = self._get_classifier(model_type)

            pipeline = Pipeline([
                ('tfidf', vectorizer),
                ('classifier', clf),
            ])

            best_params = None

            # --- Grid / Randomized Search ---
            if use_grid_search:
                key = f"{model_type}_params"
                if key in GRID_SEARCH_CONFIG:
                    param_grid = GRID_SEARCH_CONFIG[key]
                    cv = GRID_SEARCH_CONFIG['cv_folds']
                    scoring = GRID_SEARCH_CONFIG['scoring']

                    if GRID_SEARCH_CONFIG.get('use_randomized', True):
                        search = RandomizedSearchCV(
                            pipeline,
                            param_distributions=param_grid,
                            n_iter=GRID_SEARCH_CONFIG['n_iter'],
                            cv=cv, scoring=scoring,
                            n_jobs=GRID_SEARCH_CONFIG['n_jobs'],
                            verbose=GRID_SEARCH_CONFIG['verbose'],
                            random_state=self.random_state,
                            refit=True,
                        )
                    else:
                        search = GridSearchCV(
                            pipeline, param_grid=param_grid,
                            cv=cv, scoring=scoring,
                            n_jobs=GRID_SEARCH_CONFIG['n_jobs'],
                            verbose=GRID_SEARCH_CONFIG['verbose'],
                            refit=True,
                        )
                    search.fit(X_train_p, y_train)
                    pipeline = search.best_estimator_
                    best_params = search.best_params_
                    log.info(f"Лучшие параметры: {best_params}")
                    log.info(f"Лучшая CV-оценка ({scoring}): {search.best_score_:.4f}")
                else:
                    log.warning(f"Нет параметров поиска для {model_type}")
                    pipeline.fit(X_train_p, y_train)
            else:
                pipeline.fit(X_train_p, y_train)

            # --- Кросс-валидация (честная оценка) ---
            cv_mean = cv_std = None
            if use_cv_eval:
                try:
                    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                          random_state=self.random_state)
                    cv_scores = cross_val_score(
                        clone(pipeline), X_train_p, y_train,
                        cv=skf, scoring='f1_macro',
                        n_jobs=1, verbose=0,
                    )
                    cv_mean = float(cv_scores.mean())
                    cv_std = float(cv_scores.std())
                    log.info(f"CV f1_macro: {cv_mean:.4f} ± {cv_std:.4f}")
                except Exception as e:
                    log.warning(f"CV не удалось: {e}")

            # --- Тест ---
            y_pred = pipeline.predict(X_test_p)
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')

            log.info("\nРезультаты на тестовой выборке:")
            log.info(f"  Accuracy:      {accuracy:.4f}")
            log.info(f"  F1 (macro):    {f1_macro:.4f}")
            log.info(f"  F1 (weighted): {f1_weighted:.4f}")

            report = classification_report(
                y_test, y_pred, target_names=target_names, digits=4,
            )
            log.info("\nClassification Report:\n" + report)

            # --- Сохранение (с препроцессором внутри) ---
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.output_dir, f"{model_type}_model.joblib")

            full_pipeline = Pipeline([
                ('preprocess', PreprocessorTransformer()),
                ('tfidf', pipeline.named_steps['tfidf']),
                ('classifier', pipeline.named_steps['classifier']),
            ])
            full_pipeline.named_steps['preprocess'].fit(X_train[:5])

            joblib.dump({
                'pipeline': full_pipeline,
                'label_mapping': label_mapping,
                'target_names': target_names,
                'model_type': model_type,
                'train_date': timestamp,
                'best_params': best_params,
                'metrics': {
                    'accuracy': float(accuracy),
                    'f1_macro': float(f1_macro),
                    'f1_weighted': float(f1_weighted),
                    'cv_f1_macro_mean': cv_mean,
                    'cv_f1_macro_std': cv_std,
                },
            }, model_path)
            log.info(f"Модель сохранена: {model_path}")

            self.models[model_type] = pipeline
            self.results[model_type] = {
                'accuracy': float(accuracy),
                'f1_macro': float(f1_macro),
                'f1_weighted': float(f1_weighted),
                'cv_f1_macro_mean': cv_mean,
                'cv_f1_macro_std': cv_std,
                'model_path': model_path,
                'report': report,
                'best_params': best_params,
                'predictions': y_pred,
            }
            self.best_params[model_type] = best_params

        return self.models, self.results

    # ---------- reports ----------

    def save_results(self, filename=None):
        if filename is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"results_{ts}.json"

        results_path = os.path.join(self.results_dir, filename)

        results_data = {}
        for name, m in self.results.items():
            results_data[name] = {
                'accuracy': m['accuracy'],
                'f1_macro': m['f1_macro'],
                'f1_weighted': m['f1_weighted'],
                'cv_f1_macro_mean': m['cv_f1_macro_mean'],
                'cv_f1_macro_std': m['cv_f1_macro_std'],
                'model_path': m['model_path'],
                'best_params': m['best_params'],
            }

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2, default=str)

        log.info(f"\nРезультаты сохранены в {results_path}")

        report_path = os.path.join(self.results_dir, f"report_{ts}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ОТЧЕТ ОБ ОБУЧЕНИИ МОДЕЛЕЙ\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Дата: {datetime.now(timezone.utc).isoformat()}\n\n")
            f.write("Параметры данных:\n")
            f.write(f"  Файл: {DATA_CONFIG['file_path']}\n")
            f.write(f"  Колонка текста: {DATA_CONFIG['text_column']}\n")
            f.write(f"  Колонка меток: {DATA_CONFIG['label_column']}\n")
            f.write(f"  Тест: {DATA_CONFIG['test_size']*100}%\n\n")

            f.write("СРАВНЕНИЕ МОДЕЛЕЙ:\n")
            f.write("-" * 60 + "\n")
            for name, m in self.results.items():
                f.write(
                    f"{name:20} | Acc: {m['accuracy']:.4f} | "
                    f"F1_macro: {m['f1_macro']:.4f} | "
                    f"F1_weighted: {m['f1_weighted']:.4f} | "
                    f"CV: {m['cv_f1_macro_mean'] if m['cv_f1_macro_mean'] is not None else '—'}\n"
                )

            f.write("\n\nДЕТАЛЬНЫЕ ОТЧЕТЫ:\n" + "=" * 60 + "\n")
            for name, m in self.results.items():
                f.write(f"\nМодель: {name}\n")
                f.write("-" * 40 + "\n")
                if m.get('best_params'):
                    f.write(f"Лучшие параметры: {m['best_params']}\n")
                f.write(m['report'])
                f.write("\n")

        log.info(f"Отчет сохранен в {report_path}")
        return results_path, report_path

    def print_summary(self):
        log.info("\n" + "=" * 60)
        log.info("СВОДКА ПО ОБУЧЕННЫМ МОДЕЛЯМ")
        log.info("=" * 60)

        sorted_models = sorted(
            self.results.items(),
            key=lambda x: x[1]['f1_macro'],
            reverse=True,
        )

        log.info(f"\n{'Модель':20} | {'Accuracy':10} | {'F1_macro':10} | {'CV F1_macro':12} | Рейтинг")
        log.info("-" * 75)
        for i, (name, m) in enumerate(sorted_models, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            cv = m['cv_f1_macro_mean']
            cv_str = f"{cv:.4f}" if cv is not None else "—"
            log.info(f"{name:20} | {m['accuracy']:.4f}     | {m['f1_macro']:.4f}     | {cv_str:12} | {medal}")

        best_name, best_m = sorted_models[0]
        log.info(f"\n🏆 Лучшая модель: {best_name}")
        log.info(f"   Accuracy:   {best_m['accuracy']:.4f}")
        log.info(f"   F1 (macro): {best_m['f1_macro']:.4f}")
        log.info(f"   Путь:       {best_m['model_path']}")


# ==================== ДАННЫЕ ====================

def load_data():
    file_path = DATA_CONFIG['file_path']
    text_column = DATA_CONFIG['text_column']
    label_column = DATA_CONFIG['label_column']
    encoding = DATA_CONFIG['encoding']
    max_samples = DATA_CONFIG['max_samples']
    sample_random = DATA_CONFIG['sample_random']
    nrows = DATA_CONFIG['nrows']

    log.info(f"\nЗагрузка данных из {file_path}...")

    if nrows is not None and nrows != -1:
        df = pd.read_csv(file_path, encoding=encoding, nrows=nrows)
    else:
        df = pd.read_csv(file_path, encoding=encoding)

    if text_column not in df.columns:
        raise ValueError(f"Колонка '{text_column}' не найдена. Доступные: {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"Колонка '{label_column}' не найдена. Доступные: {list(df.columns)}")

    initial_len = len(df)
    df = df.dropna(subset=[text_column, label_column])
    df = df[df[text_column].astype(str).str.strip() != '']
    log.info(f"Загружено {len(df)} записей из {initial_len} "
             f"(удалено {initial_len - len(df)} пустых)")

    if max_samples and max_samples < len(df):
        if sample_random:
            df = df.sample(n=max_samples, random_state=DATA_CONFIG['random_state'])
            log.info(f"Случайная выборка: {max_samples}")
        else:
            df = df.head(max_samples)
            log.info(f"Первые {max_samples} записей")

    unique_labels = df[label_column].unique()
    log.info(f"\nУникальные метки: {unique_labels}")
    log.info("\nРаспределение меток:")
    for label in unique_labels:
        count = len(df[df[label_column] == label])
        log.info(f"  {label}: {count} ({count/len(df)*100:.1f}%)")

    return df


# ==================== MAIN ====================

def main():
    log.info("=" * 60)
    log.info("ОБУЧЕНИЕ МОДЕЛЕЙ АНАЛИЗА ТОНАЛЬНОСТИ")
    log.info("=" * 60)
    log.info(f"Дата запуска: {datetime.now(timezone.utc).isoformat()}")
    log.info("\nТЕКУЩАЯ КОНФИГУРАЦИЯ:")
    log.info(f"  Файл данных:     {DATA_CONFIG['file_path']}")
    log.info(f"  Колонка текста:  {DATA_CONFIG['text_column']}")
    log.info(f"  Колонка меток:   {DATA_CONFIG['label_column']}")
    log.info(f"  Тест:            {DATA_CONFIG['test_size']*100}%")
    log.info(f"  GridSearch:      {'вкл' if TRAINING_CONFIG['use_grid_search'] else 'выкл'}")
    log.info(f"  CV-оценка:       {'вкл' if TRAINING_CONFIG['use_cv_eval'] else 'выкл'}")
    log.info(f"  Модели:          {', '.join(TRAINING_CONFIG['models_to_train'])}")

    download_nltk_resources()

    try:
        df = load_data()
        if len(df) == 0:
            log.error("Нет данных для обучения")
            return

        X = df[DATA_CONFIG['text_column']].astype(str).values
        y = df[DATA_CONFIG['label_column']].values

        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise ValueError(f"Найдено только {len(unique_labels)} класс(ов). Нужно ≥ 2.")

        trainer = SklearnModelTrainer(
            output_dir=TRAINING_CONFIG['output_dir'],
            random_state=DATA_CONFIG['random_state'],
        )

        y_numeric, label_mapping, target_names = trainer.prepare_labels(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_numeric,
            test_size=DATA_CONFIG['test_size'],
            random_state=DATA_CONFIG['random_state'],
            stratify=y_numeric,
            shuffle=True,
        )

        log.info(f"\nРазмер обучающей выборки: {len(X_train)}")
        log.info(f"Размер тестовой выборки:  {len(X_test)}")

        try:
            weights = compute_class_weight(
                'balanced', classes=np.unique(y_train), y=y_train
            )
            log.info(f"Веса классов: {dict(zip(np.unique(y_train), weights))}")
        except Exception:
            pass

        trainer.train_multiple_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            label_mapping=label_mapping,
            target_names=target_names,
        )

        trainer.save_results()
        trainer.print_summary()

        log.info("\n✅ Обучение завершено успешно!")
        log.info(f"   Модели: {TRAINING_CONFIG['output_dir']}")

    except FileNotFoundError:
        log.error(f"\n❌ Файл {DATA_CONFIG['file_path']} не найден!")
    except Exception as e:
        log.error(f"\n❌ Ошибка: {e}")
        import traceback
        log.error(traceback.format_exc())


if __name__ == "__main__":
    main()