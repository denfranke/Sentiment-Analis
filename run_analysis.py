#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import os
import asyncio

# Настройка кодировки для Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def run_analysis(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from AnalisTon import analyze_sentiment_from_csv
    
    input_file = config['input_file']
    output_dir = config['output_dir']
    text_column = config['text_column']
    rating_column = config['rating_column']
    use_rating = config.get('use_rating', False)
    
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
        'models'
    ))
    
    print(f"\n[OK] Анализ завершен!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_analysis(sys.argv[1])
    else:
        print("Использование: python run_analysis.py <config_file.json>")