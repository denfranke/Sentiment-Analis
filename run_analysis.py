import sys
import json
import os
import asyncio

# Настройка кодировки для Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def is_dialog_format(file_path):
    """
    Проверяет, содержит ли файл диалоги.
    Поддерживает .json, .jsonl, .ndjson
    """
    if isinstance(file_path, dict):
        # Старая логика для обратной совместимости
        for key, value in file_path.items():
            if isinstance(value, list):
                if all(isinstance(item, str) for item in value):
                    return True
                if all(isinstance(item, dict) and ('text' in item or 'message' in item) 
                       for item in value):
                    return True
        return False
    
    file_lower = str(file_path).lower()
    
    if file_lower.endswith(('.jsonl', '.ndjson')):
        return True
    
    if file_lower.endswith('.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        if all(isinstance(item, str) for item in value):
                            return True
                        if all(isinstance(item, dict) and 
                               ('text' in item or 'message' in item) 
                               for item in value):
                            return True
            elif isinstance(data, list):
                if data and isinstance(data[0], list):
                    return True
                if data and isinstance(data[0], dict) and 'messages' in data[0]:
                    return True
        except Exception:
            pass
    
    return False


def process_dialog_file(input_file, output_dir, max_rows=None):
    """
    Обрабатывает файл с диалогами (.json, .jsonl, .ndjson).
    Конвертирует в CSV и возвращает путь + список диалогов.
    """
    from DialogProcessor import DialogLoader, DialogToCSVConverter
    from datetime import datetime
    
    ext = os.path.splitext(input_file)[1].lower()
    print(f"[INFO] Обнаружен формат диалогов ({ext})")
    
    # Универсальный загрузчик — определяет формат автоматически
    dialogs = DialogLoader.load(input_file)
    
    if max_rows:
        dialogs = dialogs[:max_rows]
        print(f"[INFO] Ограничено до {max_rows} диалогов")
    
    if not dialogs:
        raise ValueError("Не удалось загрузить диалоги из файла")
    
    # Конвертируем в CSV
    converter = DialogToCSVConverter(include_individual_messages=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    temp_csv = os.path.join(output_dir, f"{base_name}_dialogs_{timestamp}.csv")
    
    df = converter.convert(dialogs, temp_csv)
    
    print(f"[INFO] Создан временный CSV: {temp_csv}")
    print(f"[INFO] Строк для анализа: {len(df)} (включая {len(dialogs)} полных диалогов)")
    
    return temp_csv, dialogs


def run_analysis(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from AnalisTon import analyze_sentiment_from_csv
    
    input_file = config['input_file']
    output_dir = config['output_dir']
    text_column = config.get('text_column', 'text')
    rating_column = config.get('rating_column')
    use_rating = config.get('use_rating', False)
    max_rows = config.get('max_rows')
    is_dialog = config.get('is_dialog', False)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Проверяем формат
    ext = os.path.splitext(input_file)[1].lower()
    is_dialog_file = (ext in ['.json', '.jsonl', '.ndjson'] or is_dialog)
    
    if is_dialog_file:
        print(f"[INFO] Обработка диалогов из {input_file}")

        temp_csv, dialogs = process_dialog_file(input_file, output_dir, max_rows)

        # Сначала — построчный анализ (реплики + склейки), пишет AnalisTon
        per_row_csv = os.path.join(output_dir, 'per_row_analysis.csv')
        summary_csv = os.path.join(output_dir, 'sentiment_summary.csv')
        stats_file = os.path.join(output_dir, 'sentiment_statistics.txt')

        asyncio.run(analyze_sentiment_from_csv(
            temp_csv, per_row_csv, summary_csv, stats_file,
            'text', 'actual_sentiment',
            'models', None
        ))

        # Затем — агрегация по диалогам (1 строка = 1 диалог)
        from DialogProcessor import aggregate_dialogs_by_replicas

        canonical_csv = os.path.join(output_dir, 'отзывы(результат работы).csv')
        agg_df = aggregate_dialogs_by_replicas(per_row_csv, canonical_csv)

        # Сохраняем accuracy по диалогам в отдельный summary
        if 'is_correct_ensemble' in agg_df.columns:
            correct = agg_df['is_correct_ensemble'].dropna()
            if len(correct) > 0:
                dialog_accuracy = {
                    'total_dialogs_with_rating': len(correct),
                    'correct_predictions_ensemble': int(correct.sum()),
                    'accuracy_ensemble': correct.mean(),
                }

                # По каждой модели — тоже по диалогам (по средней score)
                sentiment_to_rating = {'Positive': 5, 'Neutral': 3, 'Negative': 1}
                for model_name in ['vader', 'flair', 'rubert', 'roberta', 'distilbert',
                                   'logistic_regression', 'svm', 'random_forest']:
                    score_col = f'{model_name}_score'
                    if score_col in agg_df.columns:
                        pred = agg_df[score_col].apply(
                            lambda s: 'Positive' if s >= 4 else ('Negative' if s <= 2 else 'Neutral')
                        )
                        actual = agg_df['actual_sentiment']
                        mask = actual.notna()
                        if mask.any():
                            acc = (pred[mask] == actual[mask]).mean()
                            dialog_accuracy[f'accuracy_{model_name}_by_dialog'] = acc

                dialog_summary_path = os.path.join(output_dir, 'dialog_accuracy_summary.csv')
                pd.DataFrame(
                    list(dialog_accuracy.items()),
                    columns=['metric', 'value']
                ).to_csv(dialog_summary_path, index=False, encoding='utf-8-sig')
                print(f"[OK] Точность по диалогам сохранена: {dialog_summary_path}")

        print(f"\n[OK] Анализ диалогов завершён!")
        print(f"  Канонический CSV: {canonical_csv}")
        return
    
    # Обычная обработка CSV
    output_csv = os.path.join(output_dir, 'отзывы(результат работы).csv')
    summary_csv = os.path.join(output_dir, 'sentiment_summary.csv')
    stats_file = os.path.join(output_dir, 'sentiment_statistics.txt')
    
    print(f"Запуск анализа...")
    print(f"  Входной файл: {input_file}")
    print(f"  Выходная директория: {output_dir}")
    print(f"  Столбец текста: {text_column}")
    if use_rating and rating_column:
        print(f"  Столбец оценки: {rating_column}")
    
    asyncio.run(analyze_sentiment_from_csv(
        input_file, output_csv, summary_csv, stats_file,
        text_column, rating_column if use_rating else None,
        'models', max_rows
    ))
    
    print(f"\n[OK] Анализ завершён!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_analysis(sys.argv[1])
    else:
        print("Использование: python Run_analysis.py <config_file.json>")