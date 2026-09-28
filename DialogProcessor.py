#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль обработки диалогов для анализа тональности.

Поддерживает:
- Загрузку диалогов из JSON, JSONL, NDJSON, CSV
- Конвертацию диалогов в CSV для дальнейшего анализа
- Анализ тональности каждой реплики и диалога в целом
- Создание визуализаций по результатам анализа
"""

import json
import os
import sys
import io
import glob
import pandas as pd
import numpy as np
from datetime import datetime

# Настройка кодировки для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class DialogLoader:
    """Загрузчик диалогов из различных форматов: JSON, JSONL, NDJSON, CSV"""
    
    # Приоритетные ключи для поиска сообщений в JSON-объекте
    MESSAGE_KEYS = [
        'messages', 'message', 'dialogs', 'dialog',
        'conversation', 'conversations', 'text', 'texts',
        'replies', 'utterances', 'turns', 'sample', 'data', 'items'
    ]
    
    # Служебные ключи (не используются как ID)
    SERVICE_KEYS = [
        'dialog_id', 'id', 'name', 'label', 'rating',
        'score', 'sentiment', 'date', 'time', 'timestamp'
    ]

    # Ключи, в которых может лежать эталонная тональность
    SENTIMENT_KEYS = [
        'sentiment', 'label', 'rating', 'target',
        'actual_sentiment', 'gold', 'ground_truth', 'class'
    ]

    @staticmethod
    def _normalize_sentiment(value):
        """
        Приводит значение эталонной тональности к одному из:
        'Positive', 'Neutral', 'Negative' или None.
        """
        if value is None:
            return None

        # Числовые оценки 1..5 (как в отзывах)
        if isinstance(value, (int, float)):
            try:
                v = float(value)
            except (TypeError, ValueError):
                return None
            if v >= 4:
                return 'Positive'
            if v <= 2:
                return 'Negative'
            if v == 3:
                return 'Neutral'
            return None

        # Строковые метки
        s = str(value).strip().lower()
        if not s:
            return None

        mapping = {
            'positive': 'Positive', 'pos': 'Positive', 'p': 'Positive',
            'положит': 'Positive', 'позитив': 'Positive',
            'negative': 'Negative', 'neg': 'Negative', 'n': 'Negative',
            'отрицат': 'Negative', 'негатив': 'Negative',
            'neutral': 'Neutral', 'neu': 'Neutral',
            'нейтрал': 'Neutral',
        }
        for key, label in mapping.items():
            if key in s:
                return label

        return None
    
    @staticmethod
    def load(file_path, dialog_id_column='dialog_id', message_column='message'):
        """
        Универсальный загрузчик — определяет формат по расширению файла.
        
        Поддерживает: .json, .jsonl, .ndjson, .csv
        
        Parameters:
        file_path: str - путь к файлу
        dialog_id_column: str - имя колонки с ID диалога (для CSV)
        message_column: str - имя колонки с сообщением (для CSV)
        
        Returns:
        list: список словарей с ключами 'dialog_id', 'messages', 'full_text', 'num_messages'
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        file_lower = file_path.lower()
        
        if file_lower.endswith(('.jsonl', '.ndjson')):
            return DialogLoader.load_from_jsonl(file_path)
        elif file_lower.endswith('.json'):
            return DialogLoader.load_from_json(file_path)
        elif file_lower.endswith('.csv'):
            return DialogLoader.load_from_csv(file_path, dialog_id_column, message_column)
        else:
            # Определяем формат по содержимому
            return DialogLoader._detect_and_load(file_path)
    
    @staticmethod
    def _detect_and_load(file_path):
        """Определяет формат файла по содержимому"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    lines.append(line.strip())
            
            # Проверяем, является ли файл JSONL
            # Если первые 2+ строки — валидные JSON-объекты, это JSONL
            if len(lines) >= 2:
                valid_json_lines = 0
                for line in lines:
                    if line:
                        try:
                            json.loads(line)
                            valid_json_lines += 1
                        except json.JSONDecodeError:
                            break
                
                if valid_json_lines >= 2:
                    print(f"[INFO] Формат определён как JSONL")
                    return DialogLoader.load_from_jsonl(file_path)
            
            # Пробуем как обычный JSON
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                print(f"[INFO] Формат определён как JSON")
                return DialogLoader.load_from_json(file_path)
            except json.JSONDecodeError:
                pass
            
            # Пробуем как CSV
            try:
                pd.read_csv(file_path, nrows=5)
                print(f"[INFO] Формат определён как CSV")
                return DialogLoader.load_from_csv(file_path)
            except Exception:
                pass
            
            raise ValueError(f"Не удалось определить формат файла: {file_path}")
        
        except Exception as e:
            raise ValueError(f"Ошибка определения формата файла: {e}")
            
    @staticmethod
    def load_from_jsonl(file_path):
        """
        Загружает диалоги из JSONL / NDJSON файла.
        Каждая строка — отдельный JSON-объект с диалогом.
        
        Поддерживаемые форматы строк:
        1. {"sample": ["реплика1", "реплика2", ...]}
        2. {"dialog_id": "...", "messages": [...]}
        3. {"id": "...", "text": ["..."]}
        4. {"любой_ключ": ["реплика1", "реплика2"]}
        5. ["реплика1", "реплика2", ...]
        6. {"dialog_id": "...", "messages": [{"text": "..."}, ...]}
        
        Returns:
        list: список диалогов
        """
        dialogs = []
        total_lines = 0
        skipped_lines = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                total_lines += 1
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Строка {line_num}: ошибка парсинга JSON: {e}")
                    skipped_lines += 1
                    continue
                
                dialog = DialogLoader._parse_jsonl_line(data, line_num)
                if dialog:
                    dialogs.append(dialog)
                else:
                    skipped_lines += 1
                    print(f"[WARNING] Строка {line_num}: не удалось извлечь диалог")
        
        print(f"[OK] Загружено {len(dialogs)} диалогов из {file_path}")
        if skipped_lines > 0:
            print(f"[INFO] Пропущено строк: {skipped_lines}")
        
        return dialogs
    
    @staticmethod
    def _parse_jsonl_line(data, line_num):
        """
        Парсит одну строку JSONL и возвращает диалог или None.
        
        Returns:
        dict или None
        """
        
        # Случай 1: это список реплик
        if isinstance(data, list):
            messages = DialogLoader._extract_messages_from_list(data)
            if messages:
                return {
                    'dialog_id': f"dialog_{line_num}",
                    'messages': messages,
                    'full_text': ' '.join(messages),
                    'num_messages': len(messages),
                    'actual_sentiment': None
                }
            return None
        
        # Случай 2: это словарь
        if isinstance(data, dict):
            return DialogLoader._parse_dict_dialog(data, line_num)
        
        # Случай 3: строка
        if isinstance(data, str):
            return {
                'dialog_id': f"dialog_{line_num}",
                'messages': [data],
                'full_text': data,
                'num_messages': 1,
                'actual_sentiment': None
            }
        
        return None
    
    @staticmethod
    def _parse_dict_dialog(data, line_num):
        """Парсит словарь и извлекает диалог + эталонную тональность."""

        # --- Извлекаем dialog_id ---
        dialog_id = None
        for key in ['dialog_id', 'id', 'name', 'uid']:
            if key in data and isinstance(data[key], (str, int)):
                dialog_id = str(data[key])
                break

        # --- Извлекаем эталонную тональность ---
        actual_sentiment = None
        for key in DialogLoader.SENTIMENT_KEYS:
            if key in data:
                actual_sentiment = DialogLoader._normalize_sentiment(data[key])
                if actual_sentiment is not None:
                    break

        # --- Ищем сообщения по приоритетным ключам ---
        messages = None
        found_key = None

        for key in DialogLoader.MESSAGE_KEYS:
            if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                messages = data[key]
                found_key = key
                break

        # Если не нашли по приоритетным, ищем любой подходящий список
        if messages is None:
            for key, value in data.items():
                if key in DialogLoader.SERVICE_KEYS or key in DialogLoader.SENTIMENT_KEYS:
                    continue
                if isinstance(value, list) and len(value) > 0:
                    if all(isinstance(item, (str, dict)) for item in value):
                        messages = value
                        found_key = key
                        break

        if messages:
            parsed_messages = DialogLoader._extract_messages_from_list(messages)
            if parsed_messages:
                if dialog_id is None:
                    dialog_id = f"{found_key}_{line_num}" if found_key else f"dialog_{line_num}"
                return {
                    'dialog_id': str(dialog_id),
                    'messages': parsed_messages,
                    'full_text': ' '.join(parsed_messages),
                    'num_messages': len(parsed_messages),
                    'actual_sentiment': actual_sentiment, 
                }

        # Одиночный текст
        if 'text' in data and isinstance(data['text'], str):
            if dialog_id is None:
                dialog_id = f"dialog_{line_num}"
            return {
                'dialog_id': str(dialog_id),
                'messages': [data['text']],
                'full_text': data['text'],
                'num_messages': 1,
                'actual_sentiment': actual_sentiment,
            }

        return None
    
    @staticmethod
    def _extract_messages_from_list(messages_list):
        """
        Извлекает текстовые сообщения из списка.
        Поддерживает список строк и список словарей.
        """
        parsed = []
        
        for msg in messages_list:
            if msg is None:
                continue
            
            if isinstance(msg, str):
                if msg.strip():
                    parsed.append(msg)
            elif isinstance(msg, dict):
                # Ищем текст в словаре
                text = None
                for key in ['text', 'message', 'content', 'utterance', 
                           'reply', 'msg', 'sentence', 'body']:
                    if key in msg and isinstance(msg[key], str):
                        text = msg[key]
                        break
                
                if text is None:
                    # Если не нашли по ключам, берём первое строковое значение
                    for v in msg.values():
                        if isinstance(v, str) and v.strip():
                            text = v
                            break
                
                if text and text.strip():
                    parsed.append(text)
            elif isinstance(msg, (int, float)):
                parsed.append(str(msg))
        
        return parsed
         
    @staticmethod
    def load_from_csv(file_path, dialog_id_column='dialog_id', message_column='message'):
        """
        Загружает диалоги из CSV, где каждая строка — реплика.
        
        Если указанные колонки не найдены, пытается определить автоматически.
        
        Returns:
        list: список диалогов
        """
        # Пробуем разные кодировки
        df = None
        for encoding in ['utf-8-sig', 'utf-8', 'cp1251']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except Exception:
                continue
        
        if df is None:
            raise ValueError(f"Не удалось прочитать CSV файл: {file_path}")
        
        dialogs = []
        columns = df.columns.tolist()
        
        # Автоопределение колонок
        if dialog_id_column not in columns:
            dialog_id_column = None
            for col in columns:
                if 'dialog' in col.lower() and 'id' in col.lower():
                    dialog_id_column = col
                    break
                if col.lower() in ['id', 'dialog', 'conversation_id', 'chat_id']:
                    dialog_id_column = col
                    break
        
        if message_column not in columns:
            message_column = None
            for col in columns:
                if col.lower() in ['message', 'text', 'content', 'utterance', 'reply']:
                    message_column = col
                    break
        
        # Если нашли колонку с ID диалога — группируем
        if dialog_id_column and message_column:
            print(f"[INFO] Группировка по колонкам: '{dialog_id_column}' и '{message_column}'")
            
            for dialog_id, group in df.groupby(dialog_id_column):
                messages = group[message_column].dropna().astype(str).tolist()
                messages = [m for m in messages if m.strip()]
                
                if messages:
                    dialogs.append({
                        'dialog_id': str(dialog_id),
                        'messages': messages,
                        'full_text': ' '.join(messages),
                        'num_messages': len(messages),
                        'actual_sentiment': None
                    })
        
        # Иначе — каждая строка отдельный диалог
        else:
            # Ищем колонку с текстом
            if message_column is None:
                message_column = columns[0] if columns else None
            
            if message_column:
                print(f"[INFO] Каждая строка — отдельный диалог (колонка: '{message_column}')")
                
                for idx, row in df.iterrows():
                    text = str(row[message_column]) if pd.notna(row[message_column]) else ""
                    if text.strip():
                        dialogs.append({
                            'dialog_id': f"dialog_{idx}",
                            'messages': [text],
                            'full_text': text,
                            'num_messages': 1,
                            'actual_sentiment': None
                        })
        
        print(f"[OK] Загружено {len(dialogs)} диалогов из {file_path}")
        return dialogs


class DialogToCSVConverter:
    """Конвертер диалогов в CSV для анализа"""
    
    def __init__(self, include_individual_messages=True):
        """
        Parameters:
        include_individual_messages: bool - включать ли отдельные реплики как строки
        """
        self.include_individual_messages = include_individual_messages
    
    def convert(self, dialogs, output_file=None):
        """
        Конвертирует диалоги в DataFrame/CSV.
        
        Returns:
        pd.DataFrame: DataFrame с колонками:
            - text: текст реплики или всего диалога
            - dialog_id: ID диалога
            - message_index: индекс реплики (-1 для полного диалога)
            - is_dialog: True для полного диалога
            - num_messages_in_dialog: количество реплик в диалоге
        """
        rows = []
        
        for dialog in dialogs:
            dialog_id = dialog['dialog_id']
            messages = dialog['messages']
            actual_sent = dialog.get('actual_sentiment')
            
            if self.include_individual_messages:
                # Каждая реплика как отдельная строка
                for msg_idx, message in enumerate(messages):
                    text = str(message).strip()
                    if text:
                        rows.append({
                            'text': text,
                            'dialog_id': dialog_id,
                            'message_index': msg_idx,
                            'is_dialog': False,
                            'num_messages_in_dialog': len(messages),
                            'actual_sentiment': actual_sent
                        })
            
            # Полный текст диалога как отдельная строка
            full_text = dialog['full_text'].strip()
            if full_text:
                rows.append({
                    'text': full_text,
                    'dialog_id': dialog_id,
                    'message_index': -1,  # -1 означает весь диалог
                    'is_dialog': True,
                    'num_messages_in_dialog': len(messages),
                    'actual_sentiment': actual_sent
                })
        
        df = pd.DataFrame(rows)
        
        if output_file:
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"[OK] Диалоги сохранены в {output_file}")
        
        return df
    
    def convert_to_simple_csv(self, dialogs, output_file=None):
        """
        Конвертирует диалоги в простой CSV с одним столбцом text.
        Каждая строка — полный текст диалога.
        """
        rows = []
        for dialog in dialogs:
            rows.append({
                'text': dialog['full_text'],
                'dialog_id': dialog['dialog_id'],
                'num_messages': dialog['num_messages']
            })
        
        df = pd.DataFrame(rows)
        
        if output_file:
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"[OK] Диалоги сохранены в {output_file}")
        
        return df



class DialogAnalyzer:
    """Анализатор диалогов с учётом контекста"""
    
    def __init__(self, sentiment_function):
        """
        Parameters:
        sentiment_function: callable - функция анализа тональности текста.
                           Должна возвращать (score, sentiment_label, confidence)
        """
        self.sentiment_function = sentiment_function
    
    def analyze_dialog(self, messages):
        """
        Анализирует диалог по репликам и возвращает агрегированный результат.
        
        Returns:
        dict: результаты анализа
        """
        results = {
            'message_sentiments': [],
            'message_scores': [],
            'overall_sentiment': 'Neutral',
            'overall_score': 3,
            'sentiment_trend': 'stable',
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0
        }
        
        for message in messages:
            if not message or not str(message).strip():
                continue
            
            score, sentiment, confidence = self.sentiment_function(str(message))
            
            results['message_sentiments'].append(sentiment)
            results['message_scores'].append(score)
            
            if sentiment == 'Positive':
                results['positive_count'] += 1
            elif sentiment == 'Negative':
                results['negative_count'] += 1
            else:
                results['neutral_count'] += 1
        
        if results['message_scores']:
            # Средний score
            avg_score = np.mean(results['message_scores'])
            results['overall_score'] = round(avg_score)
            
            if avg_score >= 3.5:
                results['overall_sentiment'] = 'Positive'
            elif avg_score <= 2.5:
                results['overall_sentiment'] = 'Negative'
            else:
                results['overall_sentiment'] = 'Neutral'
            
            # Определяем тренд
            if len(results['message_scores']) >= 2:
                mid = len(results['message_scores']) // 2
                first_half = np.mean(results['message_scores'][:mid])
                second_half = np.mean(results['message_scores'][mid:])
                
                if second_half - first_half > 0.5:
                    results['sentiment_trend'] = 'improving'
                elif first_half - second_half > 0.5:
                    results['sentiment_trend'] = 'worsening'
                else:
                    results['sentiment_trend'] = 'stable'
        
        return results


def prepare_dialogs_for_analysis(input_file, output_dir='answer', 
                                  include_individual_messages=True):
    """
    Подготавливает диалоги для анализа — конвертирует в CSV.
    
    Поддерживает .json, .jsonl, .ndjson, .csv
    
    Parameters:
    input_file: str - путь к файлу с диалогами
    output_dir: str - директория для сохранения
    include_individual_messages: bool - включать ли отдельные реплики
    
    Returns:
    str: путь к созданному CSV файлу
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Используем универсальный загрузчик
    dialogs = DialogLoader.load(input_file)
    
    if not dialogs:
        raise ValueError("Не удалось загрузить диалоги")
    
    # Конвертируем в CSV
    converter = DialogToCSVConverter(include_individual_messages)
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{base_name}_dialogs_{timestamp}.csv")
    
    df = converter.convert(dialogs, output_file)
    
    print(f"\n[INFO] Статистика диалогов:")
    print(f"  Всего диалогов: {len(dialogs)}")
    print(f"  Всего строк в CSV: {len(df)}")
    if dialogs:
        avg_len = np.mean([d['num_messages'] for d in dialogs])
        max_len = max(d['num_messages'] for d in dialogs)
        min_len = min(d['num_messages'] for d in dialogs)
        print(f"  Средняя длина диалога: {avg_len:.1f} реплик")
        print(f"  Мин/Макс длина: {min_len}/{max_len} реплик")
    
    return output_file

def analyze_dialogs_with_sentiment(input_file, output_dir='answer'):
    """
    Полный анализ диалогов с использованием всех моделей из AnalisTon.
    
    Поддерживает .json, .jsonl, .ndjson, .csv
    
    Parameters:
    input_file: str - путь к файлу с диалогами
    output_dir: str - директория для результатов
    
    Returns:
    dict: результаты анализа с путями к файлам
    """
    import asyncio
    
    # Импортируем всё необходимое из AnalisTon
    try:
        from AnalisTon import (
            sia, flair_sentiment, ru_sentiment_pipeline, en_sentiment_pipeline,
            light_ru_sentiment, load_sklearn_models, get_sklearn_sentiment,
            get_flair_score, get_transformer_sentiment_ru, 
            get_transformer_sentiment_en, get_transformer_sentiment_ru_light,
            interpret_sentiment_scores, TextTranslator
        )
    except ImportError as e:
        print(f"[X] Ошибка импорта из AnalisTon: {e}")
        print("[INFO] Убедитесь, что AnalisTon.py находится в той же директории")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем диалоги (универсальный загрузчик)
    print(f"\n[INFO] Загрузка диалогов из {input_file}")
    dialogs = DialogLoader.load(input_file)
    
    if not dialogs:
        print("[X] Не удалось загрузить диалоги")
        return None
    
    # Загружаем sklearn модели
    print(f"\n[INFO] Загрузка scikit-learn моделей...")
    sklearn_models = load_sklearn_models('models')
    if sklearn_models:
        print(f"[OK] Загружено {len(sklearn_models)} sklearn моделей: {list(sklearn_models.keys())}")
    else:
        print("[WARNING] sklearn модели не найдены, будут использованы только трансформеры")
    
    # Инициализируем переводчик
    translator = TextTranslator(max_concurrent_translations=10)
    
    # Собираем все тексты для перевода
    all_texts = []
    text_mapping = []  # (dialog_idx, msg_idx)
    
    for d_idx, dialog in enumerate(dialogs):
        for m_idx, message in enumerate(dialog['messages']):
            all_texts.append(str(message))
            text_mapping.append((d_idx, m_idx))
    
    # Переводим все тексты
    print(f"\n[INFO] Перевод {len(all_texts)} реплик (RU -> EN)...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        translated_texts = loop.run_until_complete(
            translator.translate_batch(all_texts, batch_size=50)
        )
    finally:
        loop.close()
    
    # Создаём маппинг переводов
    translation_map = {}
    for (d_idx, m_idx), translated in zip(text_mapping, translated_texts):
        translation_map[(d_idx, m_idx)] = translated
    
    print(f"[OK] Перевод завершён")
    
    # Анализируем каждый диалог
    print(f"\n[INFO] Анализ {len(dialogs)} диалогов с помощью 8 моделей...")
    print("=" * 70)
    
    results = []
    
    for d_idx, dialog in enumerate(dialogs):
        dialog_result = {
            'dialog_id': dialog['dialog_id'],
            'num_messages': dialog['num_messages'],
            'full_text': dialog['full_text'][:500] + '...' if len(dialog['full_text']) > 500 else dialog['full_text'],
            'actual_sentiment': dialog.get('actual_sentiment'),   # <-- НОВОЕ
        }
        
        # Анализируем каждую реплику
        message_results = []
        
        for m_idx, message in enumerate(dialog['messages']):
            message = str(message)
            translated = translation_map.get((d_idx, m_idx), message)
            
            msg_result = {
                'message_index': m_idx,
                'message_text': message[:200] + '...' if len(message) > 200 else message
            }
            
            # VADER (на переводе)
            if translated:
                scores = sia.polarity_scores(translated)
                msg_result['vader_sentiment'] = interpret_sentiment_scores(scores)
                msg_result['vader_compound'] = scores['compound']
                # VADER score для ensemble
                if scores['compound'] >= 0.33:
                    vader_score = 5
                elif scores['compound'] <= -0.33:
                    vader_score = 1
                else:
                    vader_score = 3
            else:
                msg_result['vader_sentiment'] = 'Neutral'
                msg_result['vader_compound'] = 0.0
                vader_score = 3
            
            msg_result['vader_score'] = vader_score
            
            # Flair (на оригинале)
            try:
                flair_score, flair_sentiment, flair_conf = get_flair_score(message)
                msg_result['flair_sentiment'] = flair_sentiment
                msg_result['flair_score'] = flair_score
                msg_result['flair_confidence'] = flair_conf
            except Exception as e:
                msg_result['flair_sentiment'] = 'Neutral'
                msg_result['flair_score'] = 3
                msg_result['flair_confidence'] = 0.0
            
            # RuBERT (на оригинале)
            try:
                _, rubert_score, rubert_sentiment, rubert_conf = get_transformer_sentiment_ru(message)
                msg_result['rubert_sentiment'] = rubert_sentiment
                msg_result['rubert_score'] = rubert_score
                msg_result['rubert_confidence'] = rubert_conf
            except Exception as e:
                msg_result['rubert_sentiment'] = 'Neutral'
                msg_result['rubert_score'] = 3
                msg_result['rubert_confidence'] = 0.0
            
            # RoBERTa (на переводе)
            try:
                _, roberta_score, roberta_sentiment, roberta_conf = get_transformer_sentiment_en(
                    translated if translated else "", 'roberta'
                )
                msg_result['roberta_sentiment'] = roberta_sentiment
                msg_result['roberta_score'] = roberta_score
                msg_result['roberta_confidence'] = roberta_conf
            except Exception as e:
                msg_result['roberta_sentiment'] = 'Neutral'
                msg_result['roberta_score'] = 3
                msg_result['roberta_confidence'] = 0.0
            
            # DistilBERT (на оригинале)
            try:
                _, distilbert_score, distilbert_sentiment, distilbert_conf = get_transformer_sentiment_ru_light(message)
                msg_result['distilbert_sentiment'] = distilbert_sentiment
                msg_result['distilbert_score'] = distilbert_score
                msg_result['distilbert_confidence'] = distilbert_conf
            except Exception as e:
                msg_result['distilbert_sentiment'] = 'Neutral'
                msg_result['distilbert_score'] = 3
                msg_result['distilbert_confidence'] = 0.0
            
            # scikit-learn модели (на оригинале)
            for model_name in sklearn_models.keys():
                try:
                    score, sentiment, conf = get_sklearn_sentiment(message, model_name, sklearn_models)
                    msg_result[f'{model_name}_sentiment'] = sentiment
                    msg_result[f'{model_name}_score'] = score
                    msg_result[f'{model_name}_confidence'] = conf
                except Exception as e:
                    msg_result[f'{model_name}_sentiment'] = 'Neutral'
                    msg_result[f'{model_name}_score'] = 3
                    msg_result[f'{model_name}_confidence'] = 0.0
            
            # Ensemble score для реплики
            scores_for_ensemble = [
                msg_result.get('vader_score', 3),
                msg_result.get('flair_score', 3),
                msg_result.get('rubert_score', 3),
                msg_result.get('roberta_score', 3),
                msg_result.get('distilbert_score', 3),
            ]
            
            # Добавляем sklearn scores
            for model_name in sklearn_models.keys():
                scores_for_ensemble.append(msg_result.get(f'{model_name}_score', 3))
            
            ensemble_score = round(np.mean(scores_for_ensemble))
            msg_result['ensemble_score'] = ensemble_score
            
            if ensemble_score >= 4:
                msg_result['ensemble_sentiment'] = 'Positive'
            elif ensemble_score <= 2:
                msg_result['ensemble_sentiment'] = 'Negative'
            else:
                msg_result['ensemble_sentiment'] = 'Neutral'
            
            message_results.append(msg_result)
        
        # Агрегируем результаты по диалогу
        dialog_result['messages'] = message_results
        
        # Считаем статистику по диалогу
        ensemble_scores = [m['ensemble_score'] for m in message_results]
        if ensemble_scores:
            dialog_result['ensemble_score_avg'] = float(np.mean(ensemble_scores))
            dialog_result['ensemble_score_rounded'] = round(dialog_result['ensemble_score_avg'])
        else:
            dialog_result['ensemble_score_avg'] = 3.0
            dialog_result['ensemble_score_rounded'] = 3
        
        if dialog_result['ensemble_score_rounded'] >= 4:
            dialog_result['dialog_sentiment'] = 'Positive'
        elif dialog_result['ensemble_score_rounded'] <= 2:
            dialog_result['dialog_sentiment'] = 'Negative'
        else:
            dialog_result['dialog_sentiment'] = 'Neutral'
        
        # Считаем количество по тональностям
        sentiments = [m['ensemble_sentiment'] for m in message_results]
        dialog_result['positive_messages'] = sentiments.count('Positive')
        dialog_result['negative_messages'] = sentiments.count('Negative')
        dialog_result['neutral_messages'] = sentiments.count('Neutral')
        
        # Определяем тренд
        if len(ensemble_scores) >= 2:
            mid = len(ensemble_scores) // 2
            first_half = np.mean(ensemble_scores[:mid])
            second_half = np.mean(ensemble_scores[mid:])
            
            if second_half - first_half > 0.5:
                dialog_result['sentiment_trend'] = 'improving'
            elif first_half - second_half > 0.5:
                dialog_result['sentiment_trend'] = 'worsening'
            else:
                dialog_result['sentiment_trend'] = 'stable'
        else:
            dialog_result['sentiment_trend'] = 'stable'
        
        results.append(dialog_result)
        
        print(f"  [{d_idx+1}/{len(dialogs)}] '{dialog['dialog_id']}': "
              f"{dialog_result['dialog_sentiment']} "
              f"(score: {dialog_result['ensemble_score_rounded']}, "
              f"реплик: {dialog_result['num_messages']}, "
              f"тренд: {dialog_result['sentiment_trend']})")
    
    print("=" * 70)
    
    # Сохраняем результаты
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Детальные результаты по репликам
    messages_rows = []
    for dialog_result in results:
        for msg in dialog_result['messages']:
            row = {
                'dialog_id': dialog_result['dialog_id'],
                'message_index': msg['message_index'],
                'message_text': msg['message_text'],
                'actual_sentiment': dialog_result.get('actual_sentiment'),
                'vader_sentiment': msg.get('vader_sentiment'),
                'vader_score': msg.get('vader_score'),
                'flair_sentiment': msg.get('flair_sentiment'),
                'flair_score': msg.get('flair_score'),
                'rubert_sentiment': msg.get('rubert_sentiment'),
                'rubert_score': msg.get('rubert_score'),
                'roberta_sentiment': msg.get('roberta_sentiment'),
                'roberta_score': msg.get('roberta_score'),
                'distilbert_sentiment': msg.get('distilbert_sentiment'),
                'distilbert_score': msg.get('distilbert_score'),
                'ensemble_sentiment': msg.get('ensemble_sentiment'),
                'ensemble_score': msg.get('ensemble_score')
            }
            for model_name in sklearn_models.keys():
                row[f'{model_name}_sentiment'] = msg.get(f'{model_name}_sentiment')
                row[f'{model_name}_score'] = msg.get(f'{model_name}_score')
            messages_rows.append(row)
    
    messages_df = pd.DataFrame(messages_rows)
    messages_file = os.path.join(output_dir, f'dialog_messages_analysis_{timestamp}.csv')
    messages_df.to_csv(messages_file, index=False, encoding='utf-8-sig')
    print(f"\n[OK] Анализ реплик сохранён: {messages_file}")
    
    # 2. Сводка по диалогам
    dialog_rows = []
    for dialog_result in results:
        # «text» и «rating» нужны для совместимости с VisualAnalisTonResults.py
        # и с логикой расчёта точности в AnalisTon.py
        actual_sent = dialog_result.get('actual_sentiment')

        # Приводим эталон к числовой оценке, чтобы AnalisTon мог считать is_correct
        if actual_sent == 'Positive':
            rating_num = 5
        elif actual_sent == 'Negative':
            rating_num = 1
        elif actual_sent == 'Neutral':
            rating_num = 3
        else:
            rating_num = None

        row = {
            'dialog_id': dialog_result['dialog_id'],
            'num_messages': dialog_result['num_messages'],
            'dialog_sentiment': dialog_result['dialog_sentiment'],
            'ensemble_score_avg': dialog_result['ensemble_score_avg'],
            'ensemble_score_rounded': dialog_result['ensemble_score_rounded'],
            'positive_messages': dialog_result['positive_messages'],
            'negative_messages': dialog_result['negative_messages'],
            'neutral_messages': dialog_result['neutral_messages'],
            'sentiment_trend': dialog_result['sentiment_trend'],
            'full_text': dialog_result['full_text'],

            # --- Для отчёта и расчёта accuracy ---
            'text': dialog_result['full_text'],          
            'rating': rating_num,                        
            'actual_sentiment': actual_sent,             

            # --- Псевдо-столбцы моделей для совместимости ---
            # (чтобы VisualAnalisTonResults.py построил те же графики)
            'ensemble_score': dialog_result['ensemble_score_rounded'],
            'ensemble_sentiment': dialog_result['dialog_sentiment']
        }
        dialog_rows.append(row)
    
    dialogs_df = pd.DataFrame(dialog_rows)
    dialogs_file = os.path.join(output_dir, f'dialogs_summary_{timestamp}.csv')
    dialogs_df.to_csv(dialogs_file, index=False, encoding='utf-8-sig')
    print(f"[OK] Сводка по диалогам сохранена: {dialogs_file}")
    
    # 3. Простой CSV со всеми репликами (для совместимости с AnalisTon)
    simple_rows = []
    for dialog_result in results:
        for msg in dialog_result['messages']:
            simple_rows.append({
                'text': msg['message_text'],
                'dialog_id': dialog_result['dialog_id'],
                'message_index': msg['message_index']
            })
    
    simple_df = pd.DataFrame(simple_rows)
    simple_file = os.path.join(output_dir, f'dialogs_simple_{timestamp}.csv')
    simple_df.to_csv(simple_file, index=False, encoding='utf-8-sig')
    print(f"[OK] Простой CSV сохранён: {simple_file}")
    
    return {
        'dialogs': results,
        'messages_file': messages_file,
        'dialogs_file': dialogs_file,
        'simple_file': simple_file,
        'messages_df': messages_df,
        'dialogs_df': dialogs_df
    }

def aggregate_dialogs_by_replicas(per_row_csv_path, output_csv_path):
    """
    Агрегирует результаты анализа реплик по диалогам.

    Логика:
    1. Для каждой реплики уже есть ensemble_score (среднее 8 моделей, округлено).
    2. Для диалога ensemble_score = round(mean(ensemble_score всех реплик)).
    3. ensemble_sentiment диалога = classify(ensemble_score диалога).
    4. is_correct_ensemble = (ensemble_sentiment == actual_sentiment).
    5. accuracy_ensemble = доля диалогов, где is_correct_ensemble == True.

    Parameters:
    per_row_csv_path: str - CSV, который пишет AnalisTon.analyze_sentiment_from_csv
                            (там есть столбцы: text, dialog_id, message_index,
                             is_dialog, actual_sentiment, ensemble_score,
                             ensemble_sentiment, is_correct_vader, ...)
    output_csv_path: str - куда сохранить агрегированный CSV (1 строка = 1 диалог)

    Returns:
    pd.DataFrame: агрегированный DataFrame
    """
    df = pd.read_csv(per_row_csv_path, encoding='utf-8-sig')

    # --- Оставляем только реплики (исключаем строки с is_dialog == True) ---
    if 'is_dialog' in df.columns:
        replicas = df[df['is_dialog'] == False].copy()
    else:
        # Если столбца нет — используем message_index >= 0
        if 'message_index' in df.columns:
            replicas = df[df['message_index'] >= 0].copy()
        else:
            replicas = df.copy()

    if 'dialog_id' not in replicas.columns:
        raise ValueError("В CSV нет столбца 'dialog_id' — агрегация по диалогам невозможна")

    # --- Группируем по диалогу ---
    aggregated = []
    for dialog_id, group in replicas.groupby('dialog_id', sort=False):
        # Эталонная тональность диалога (должна быть одинаковой во всех репликах)
        actual_values = group['actual_sentiment'].dropna().unique() if 'actual_sentiment' in group.columns else []
        actual_sentiment = actual_values[0] if len(actual_values) > 0 else None

        # Оценки реплик
        replica_scores = group['ensemble_score'].dropna().tolist() if 'ensemble_score' in group.columns else []

        if len(replica_scores) == 0:
            dialog_score = 3
        else:
            dialog_score = round(sum(replica_scores) / len(replica_scores))

        if dialog_score >= 4:
            dialog_sentiment = 'Positive'
        elif dialog_score <= 2:
            dialog_sentiment = 'Negative'
        else:
            dialog_sentiment = 'Neutral'

        is_correct = None
        if actual_sentiment is not None:
            is_correct = (dialog_sentiment == actual_sentiment)

        aggregated.append({
            'dialog_id': dialog_id,
            'num_messages': len(group),
            'actual_sentiment': actual_sentiment,
            'ensemble_score': dialog_score,
            'ensemble_sentiment': dialog_sentiment,
            'is_correct_ensemble': is_correct,
            # Для визуализаций
            'text': ' '.join(group['text'].astype(str).tolist()),
        })

    agg_df = pd.DataFrame(aggregated)

    # --- Совместимость с VisualAnalisTonResults.py ---
    # Он ожидает: rating, ensemble_score, ensemble_sentiment, actual_sentiment,
    #             vader_score, flair_score, ... для построения графиков.
    # Добавляем числовой rating из actual_sentiment
    sentiment_to_rating = {'Positive': 5, 'Neutral': 3, 'Negative': 1}
    agg_df['rating'] = agg_df['actual_sentiment'].map(sentiment_to_rating)

    # Прокидываем средние score по каждой из 8 моделей (для графиков MAE и т.п.)
    per_model_score_cols = [
        'vader_score', 'flair_score', 'rubert_score', 'roberta_score',
        'distilbert_score', 'logistic_regression_score',
        'svm_score', 'random_forest_score',
    ]
    for col in per_model_score_cols:
        if col in replicas.columns:
            agg_df[col] = replicas.groupby('dialog_id')[col].mean().reindex(
                agg_df['dialog_id']
            ).values

    # Прокидываем средние sentiment-метки моделей (по большинству)
    per_model_sent_cols = [c for c in replicas.columns if c.endswith('_sentiment')
                           and c not in ('ensemble_sentiment', 'actual_sentiment')]
    for col in per_model_sent_cols:
        def _majority(series):
            vals = series.dropna().tolist()
            if not vals:
                return 'Neutral'
            return max(set(vals), key=vals.count)
        agg_df[col.replace('_sentiment', '_sentiment')] = replicas.groupby(
            'dialog_id'
        )[col].apply(_majority).reindex(agg_df['dialog_id']).values

    # Метки моделей для совместимости с PDFReportGenerator
    for model_name in ['vader', 'flair', 'rubert', 'roberta', 'distilbert',
                       'logistic_regression', 'svm', 'random_forest']:
        sent_col = f'{model_name}_sentiment'
        if sent_col in agg_df.columns:
            score_col = f'{model_name}_score'
            if score_col not in agg_df.columns:
                agg_df[score_col] = agg_df[sent_col].map(sentiment_to_rating)

    agg_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"[OK] Агрегация по диалогам сохранена: {output_csv_path}")
    print(f"     Диалогов: {len(agg_df)}")
    if agg_df['is_correct_ensemble'].notna().any():
        acc = agg_df['is_correct_ensemble'].dropna().mean()
        print(f"     accuracy_ensemble (по диалогам): {acc:.1%}")

    return agg_df

def create_dialog_visualizations(results, output_dir='answer'):
    """
    Создаёт визуализации для анализа диалогов.
    
    Parameters:
    results: dict - результаты из analyze_dialogs_with_sentiment
    output_dir: str - директория для сохранения
    """
    import matplotlib
    matplotlib.use('Agg')  # Для работы без GUI
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.style.use('seaborn-v0_8-darkgrid')
    os.makedirs(output_dir, exist_ok=True)
    
    dialogs = results['dialogs']
    
    if not dialogs:
        print("[WARNING] Нет данных для визуализации")
        return None
    
    # Подготовка данных
    dialog_sentiments = [d['dialog_sentiment'] for d in dialogs]
    dialog_scores = [d['ensemble_score_rounded'] for d in dialogs]
    num_messages = [d['num_messages'] for d in dialogs]
    trends = [d.get('sentiment_trend', 'stable') for d in dialogs]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ========================================================================
    # 1. Распределение тональностей диалогов
    # ========================================================================
    ax = axes[0, 0]
    sentiment_counts = pd.Series(dialog_sentiments).value_counts()
    colors = {'Positive': '#4ECDC4', 'Neutral': '#FFE194', 'Negative': '#FF6B6B'}
    
    order = ['Negative', 'Neutral', 'Positive']
    values = [sentiment_counts.get(s, 0) for s in order]
    
    bars = ax.bar(order, values, color=[colors[s] for s in order], 
                  edgecolor='black', linewidth=2, alpha=0.8)
    
    for bar, val in zip(bars, values):
        if val > 0:
            pct = val / len(dialogs) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(0.5, max(values) * 0.02),
                   f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Тональность диалога', fontsize=12)
    ax.set_ylabel('Количество диалогов', fontsize=12)
    ax.set_title('Распределение тональностей диалогов', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    if max(values) > 0:
        ax.set_ylim(0, max(values) * 1.2)
    
    # ========================================================================
    # 2. Распределение оценок
    # ========================================================================
    ax = axes[0, 1]
    score_counts = pd.Series(dialog_scores).value_counts().sort_index()
    
    if len(score_counts) > 0:
        bars = ax.bar(score_counts.index.astype(str), score_counts.values, 
                      color='#45B7D1', edgecolor='black', alpha=0.8, linewidth=2)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Ensemble Score', fontsize=12)
    ax.set_ylabel('Количество диалогов', fontsize=12)
    ax.set_title('Распределение оценок диалогов', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # 3. Зависимость тональности от длины диалога
    # ========================================================================
    ax = axes[1, 0]
    colors_scatter = [colors.get(s, 'gray') for s in dialog_sentiments]
    scatter = ax.scatter(num_messages, dialog_scores, c=colors_scatter, 
                         s=120, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Добавляем легенду
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[s], edgecolor='black', label=s) 
                       for s in ['Positive', 'Neutral', 'Negative']]
    ax.legend(handles=legend_elements, loc='best')
    
    ax.set_xlabel('Количество реплик в диалоге', fontsize=12)
    ax.set_ylabel('Ensemble Score', fontsize=12)
    ax.set_title('Зависимость тональности от длины диалога', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yticks([1, 2, 3, 4, 5])
    
    # ========================================================================
    # 4. Тепловая карта тональностей реплик
    # ========================================================================
    ax = axes[1, 1]
    
    # Собираем данные по репликам
    all_messages = []
    for d in dialogs:
        for m in d['messages']:
            all_messages.append({
                'dialog_id': d['dialog_id'],
                'message_index': m['message_index'],
                'ensemble_score': m['ensemble_score']
            })
    
    if all_messages:
        msg_df = pd.DataFrame(all_messages)
        
        try:
            pivot = msg_df.pivot_table(
                index='dialog_id', 
                columns='message_index', 
                values='ensemble_score',
                aggfunc='mean'
            )
            
            # Ограничиваем размер для читаемости
            # if len(pivot) > 20:
            #     pivot = pivot.head(20)
            if len(pivot.columns) > 15:
                pivot = pivot.iloc[:, :15]
            
            sns.heatmap(pivot, cmap='RdYlGn', center=3, ax=ax, 
                        cbar_kws={'label': 'Score'}, 
                        vmin=1, vmax=5,
                        linewidths=0.5, linecolor='white')
            ax.set_title('Тепловая карта тональностей реплик', fontsize=14, fontweight='bold')
            ax.set_xlabel('Индекс реплики', fontsize=12)
            ax.set_ylabel('Диалог', fontsize=12)
        except Exception as e:
            ax.text(0.5, 0.5, f'Не удалось построить\nтепловую карту:\n{e}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('Тепловая карта тональностей реплик', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'dialog_analysis_dashboard.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Визуализация диалогов сохранена: {output_file}")
    return output_file

def create_dialog_statistics(results, output_dir='answer'):
    """
    Создаёт текстовый файл со статистикой по диалогам.
    """
    dialogs = results['dialogs']
    
    if not dialogs:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = os.path.join(output_dir, f'dialog_statistics_{timestamp}.txt')
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("СТАТИСТИКА АНАЛИЗА ДИАЛОГОВ\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Всего диалогов: {len(dialogs)}\n\n")
        
        # Распределение по тональностям
        f.write("-" * 70 + "\n")
        f.write("РАСПРЕДЕЛЕНИЕ ПО ТОНАЛЬНОСТЯМ\n")
        f.write("-" * 70 + "\n")
        
        sentiments = [d['dialog_sentiment'] for d in dialogs]
        for sent in ['Positive', 'Neutral', 'Negative']:
            count = sentiments.count(sent)
            pct = count / len(dialogs) * 100 if dialogs else 0
            f.write(f"  {sent:10}: {count:4} ({pct:5.1f}%)\n")
        
        # Распределение по трендам
        f.write("\n" + "-" * 70 + "\n")
        f.write("РАСПРЕДЕЛЕНИЕ ПО ТРЕНДАМ\n")
        f.write("-" * 70 + "\n")
        
        trends = [d.get('sentiment_trend', 'stable') for d in dialogs]
        for trend in ['improving', 'stable', 'worsening']:
            count = trends.count(trend)
            pct = count / len(dialogs) * 100 if dialogs else 0
            f.write(f"  {trend:12}: {count:4} ({pct:5.1f}%)\n")
        
        # Длина диалогов
        f.write("\n" + "-" * 70 + "\n")
        f.write("СТАТИСТИКА ПО ДЛИНЕ ДИАЛОГОВ\n")
        f.write("-" * 70 + "\n")
        
        lengths = [d['num_messages'] for d in dialogs]
        f.write(f"  Средняя длина: {np.mean(lengths):.1f} реплик\n")
        f.write(f"  Медиана:       {np.median(lengths):.1f} реплик\n")
        f.write(f"  Мин/Макс:      {min(lengths)}/{max(lengths)} реплик\n")
        f.write(f"  Всего реплик:  {sum(lengths)}\n")
        
        # Средние scores
        f.write("\n" + "-" * 70 + "\n")
        f.write("СРЕДНИЕ ENSEMBLE SCORES\n")
        f.write("-" * 70 + "\n")
        
        scores = [d['ensemble_score_avg'] for d in dialogs]
        f.write(f"  Средний score:  {np.mean(scores):.3f}\n")
        f.write(f"  Медиана:        {np.median(scores):.3f}\n")
        f.write(f"  Std:            {np.std(scores):.3f}\n")
        
        # Детальная информация по диалогам
        f.write("\n" + "=" * 70 + "\n")
        f.write("ДЕТАЛЬНАЯ ИНФОРМАЦИЯ ПО ДИАЛОГАМ\n")
        f.write("=" * 70 + "\n\n")
        
        for i, d in enumerate(dialogs, 1):
            f.write(f"Диалог #{i}: {d['dialog_id']}\n")
            f.write(f"  Реплик:       {d['num_messages']}\n")
            f.write(f"  Тональность:  {d['dialog_sentiment']}\n")
            f.write(f"  Score:        {d['ensemble_score_rounded']} (avg: {d['ensemble_score_avg']:.2f})\n")
            f.write(f"  Тренд:        {d.get('sentiment_trend', 'stable')}\n")
            f.write(f"  Реплики:      +{d['positive_messages']} / ={d['neutral_messages']} / -{d['negative_messages']}\n")
            f.write("\n")
    
    print(f"[OK] Статистика диалогов сохранена: {stats_file}")
    return stats_file


def main():
    """Точка входа для CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Анализ тональности диалогов (поддерживает JSON, JSONL, NDJSON, CSV)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python DialogProcessor.py dialogs.jsonl --prepare-only
  python DialogProcessor.py dialogs.jsonl --visualize
  python DialogProcessor.py dialogs.json --output-dir results
  python DialogProcessor.py dialogs.ndjson --visualize --output-dir answer
        """
    )
    parser.add_argument('input_file', 
                       help='Путь к файлу с диалогами (.json, .jsonl, .ndjson, .csv)')
    parser.add_argument('--output-dir', default='answer', 
                       help='Директория для результатов (по умолчанию: answer)')
    parser.add_argument('--prepare-only', action='store_true', 
                       help='Только подготовить CSV без анализа')
    parser.add_argument('--visualize', action='store_true',
                       help='Создать визуализации')
    parser.add_argument('--no-individual', action='store_true',
                       help='Не включать отдельные реплики в CSV (только полные диалоги)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"[X] Файл не найден: {args.input_file}")
        sys.exit(1)
    
    try:
        if args.prepare_only:
            # Только конвертация в CSV
            output_file = prepare_dialogs_for_analysis(
                args.input_file, 
                args.output_dir,
                include_individual_messages=not args.no_individual
            )
            print(f"\n[OK] Файл подготовлен: {output_file}")
        else:
            # Полный анализ
            results = analyze_dialogs_with_sentiment(args.input_file, args.output_dir)
            
            if results:
                if args.visualize:
                    create_dialog_visualizations(results, args.output_dir)
                create_dialog_statistics(results, args.output_dir)
                
                print(f"\n" + "=" * 70)
                print(f"[OK] Анализ завершён!")
                print(f"=" * 70)
                print(f"  Диалогов проанализировано: {len(results['dialogs'])}")
                print(f"  Сводка:     {results['dialogs_file']}")
                print(f"  Реплики:    {results['messages_file']}")
                print(f"  Простой CSV: {results['simple_file']}")
                if args.visualize:
                    print(f"  Графики:    {args.output_dir}/dialog_analysis_dashboard.png")
            else:
                print("[X] Анализ не дал результатов")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n[INFO] Анализ прерван пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"[X] Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()