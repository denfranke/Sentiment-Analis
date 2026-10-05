import sys
import os
import json
import asyncio
import subprocess
import webbrowser
import platform
import glob
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QMessageBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QProgressBar,
    QGroupBox, QCheckBox, QComboBox, QLineEdit, QGridLayout,
    QHeaderView, QStatusBar, QRadioButton, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QProcess
from PyQt5.QtGui import QFont, QColor, QTextCursor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import sys
# import io

# # Настройка кодировки для Windows
# if sys.platform == 'win32':
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
#     sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings('ignore')

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')


class AnalysisProcess(QProcess):
    """Процесс для запуска основного анализа"""
    
    output_received = pyqtSignal(str)
    finished_analysis = pyqtSignal(str, bool)
    
    def __init__(self):
        super().__init__()
        self.output_dir = None
        self.readyReadStandardOutput.connect(self.handle_stdout)
        self.readyReadStandardError.connect(self.handle_stderr)
        self.finished.connect(self.on_finished)
    
    def start_analysis(self, input_file, text_col, rating_col, use_rating, output_dir, max_rows=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        config = {
            'input_file': input_file,
            'output_dir': output_dir,
            'text_column': text_col,
            'rating_column': rating_col if use_rating and rating_col else None,
            'use_rating': use_rating,
            'max_rows': max_rows
        }
        
        config_file = os.path.join(output_dir, 'analysis_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # Запуск скрипта
        script_path = os.path.join(os.path.dirname(__file__), 'Run_analysis.py')
        if not os.path.exists(script_path):
            self.create_run_script(script_path)
        
        program = sys.executable
        self.start(program, [script_path, config_file])
    
    def create_run_script(self, script_path):
        """Создаёт Run_analysis.py если его нет"""
        content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import os
import asyncio

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
        'models',max_rows
    ))
    
    print(f"\\n✅ Анализ завершён!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_analysis(sys.argv[1])
    else:
        print("Использование: python Run_analysis.py <config_file.json>")
'''
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def handle_stdout(self):
        data = self.readAllStandardOutput()
        try:
            text = bytes(data).decode('utf-8', errors='replace')
        except:
            text = bytes(data).decode('cp1251', errors='replace')
        self.output_received.emit(text)
    
    def handle_stderr(self):
        data = self.readAllStandardError()
        try:
            text = bytes(data).decode('utf-8', errors='replace')
        except:
            text = bytes(data).decode('cp1251', errors='replace')
        self.output_received.emit(f"[INFO] {text}")
    
    def on_finished(self, exit_code, exit_status):
        success = (exit_code == 0)
        self.finished_analysis.emit(self.output_dir, success)


class VisualAnalysisProcess(QProcess):
    """Процесс для запуска визуализации (VisualAnalisTonResults)"""
    
    output_received = pyqtSignal(str)
    finished_visual = pyqtSignal(str, bool)
    
    def __init__(self):
        super().__init__()
        self.output_dir = None
        self.readyReadStandardOutput.connect(self.handle_stdout)
        self.readyReadStandardError.connect(self.handle_stderr)
        self.finished.connect(self.on_finished)
    
    def start_visualization(self, output_dir):
        """Запускает VisualAnalisTonResults для создания графиков"""
        self.output_dir = output_dir
        
        # Путь к скрипту визуализации
        visual_script = os.path.join(os.path.dirname(__file__), 'VisualAnalisTonResults.py')
        
        if not os.path.exists(visual_script):
            self.finished_visual.emit(output_dir, False)
            return
        
        # Копируем результат в папку answer/ для VisualAnalisTonResults
        import shutil
        result_file = os.path.join(output_dir, 'отзывы(результат работы).csv')
        if os.path.exists(result_file):
            # Копируем с кириллическим именем
            dest_cyrillic = os.path.join('answer', 'отзывы(результат работы).csv')
            shutil.copy2(result_file, dest_cyrillic)
            print(f"[VISUAL] Copied results to: {dest_cyrillic}")
            
            # Дополнительно копируем с латинским именем для совместимости
            dest_latin = os.path.join('answer', 'otzyvy_result.csv')
            shutil.copy2(result_file, dest_latin)
            print(f"[VISUAL] Copied results to: {dest_latin}")
        
        # Устанавливаем рабочую директорию
        self.setWorkingDirectory(os.path.dirname(os.path.abspath(__file__)))
        
        # Запускаем скрипт визуализации
        program = sys.executable
        self.start(program, [visual_script])
    
    def handle_stdout(self):
        data = self.readAllStandardOutput()
        try:
            text = bytes(data).decode('utf-8', errors='replace')
        except:
            text = bytes(data).decode('cp1251', errors='replace')
        self.output_received.emit(text)
    
    def handle_stderr(self):
        data = self.readAllStandardError()
        try:
            text = bytes(data).decode('utf-8', errors='replace')
        except:
            text = bytes(data).decode('cp1251', errors='replace')
        self.output_received.emit(f"[VISUAL] {text}")
    
    def on_finished(self, exit_code, exit_status):
        success = (exit_code == 0)
        self.finished_visual.emit(self.output_dir, success)


class PDFReportGenerator(QThread):
    """Генерация PDF отчёта """
    
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
    
    def run(self):
        try:
            result_file = os.path.join(self.output_dir, 'отзывы(результат работы).csv')
            if not os.path.exists(result_file):
                self.error.emit("Файл результатов не найден")
                return
            
            self.progress.emit("Загрузка результатов...")
            df = pd.read_csv(result_file, encoding='utf-8-sig')
            
            self.progress.emit("Поиск графиков из VisualAnalisTonResults...")
            
            # Создаём HTML отчёт
            html_path = os.path.join(self.output_dir, 'sentiment_report.html')
            self.create_html_report(df, html_path)
            
            # Конвертируем в PDF
            pdf_path = os.path.join(self.output_dir, 'sentiment_report.pdf')
            self.convert_to_pdf(html_path, pdf_path)
            
            if os.path.exists(pdf_path):
                self.finished.emit(pdf_path)
            else:
                self.error.emit("PDF не создан")
                
        except Exception as e:
            self.error.emit(str(e))
    
    def find_visual_graphs(self):
        """Ищет графики, созданные VisualAnalisTonResults.py"""
        graphs = {}
        
        # Папка answer/ в корне проекта
        answer_dir = os.path.join(os.path.dirname(__file__), 'answer')
        
        # Ищем графики - проверяем несколько вариантов
        graph_patterns = {
            'class_dist': 'ensemble_class_distribution.png',
            'dist_analysis': 'distribution_analysis.png',
            'dashboard': 'sentiment_analysis_dashboard.png',
            'additional': 'additional_analysis.png',
        }
        
        for key, filename in graph_patterns.items():
            filepath = os.path.join(answer_dir, filename)
            if os.path.exists(filepath):
                graphs[key] = filepath
                print(f"[PDF] Found graph: {filepath}")
            else:
                print(f"[PDF] Graph not found: {filepath}")
        
        # Если графики не найдены, ищем любые PNG в answer/
        if not graphs:
            print("[PDF] No expected graphs found, searching for any PNG files...")
            all_pngs = glob.glob(os.path.join(answer_dir, '*.png'))
            for i, png in enumerate(all_pngs):
                graphs[f'extra_{i}'] = png
                print(f"[PDF] Found extra graph: {png}")
        
        return graphs
    
    def copy_graphs_to_output(self, graphs):
        """Копирует графики в папку с результатами"""
        copied = {}
        for key, src_path in graphs.items():
            if os.path.exists(src_path):
                import shutil
                dest_name = os.path.basename(src_path)
                dest_path = os.path.join(self.output_dir, dest_name)
                shutil.copy2(src_path, dest_path)
                copied[key] = dest_path
        return copied
    
    def create_html_report(self, df, html_path):
        """Создаёт HTML отчёт ТОЛЬКО с графиками из VisualAnalisTonResults"""
        
        output_dir = self.output_dir
        
        # Ищем графики из VisualAnalisTonResults
        self.progress.emit("Поиск графиков визуализации...")
        visual_graphs = self.find_visual_graphs()
        
        # Копируем их в папку с результатами
        local_graphs = self.copy_graphs_to_output(visual_graphs)
        
        # Сортируем графики в нужном порядке
        graph_order = ['class_dist', 'dist_analysis', 'dashboard', 'additional']
        graph_titles = {
            'class_dist': 'Распределение Ensemble по классам тональности',
            'dist_analysis': 'Анализ распределения оценок моделей',
            'dashboard': 'Дашборд анализа тональности',
            'additional': 'Дополнительный анализ (точность и ошибки)',
        }
        
        # Формируем список доступных графиков
        available_graphs = []
        for key in graph_order:
            if key in local_graphs:
                # Получаем абсолютный путь для вставки в PDF
                abs_path = os.path.abspath(local_graphs[key])
                # Кодируем путь для URL (заменяем обратные слеши)
                file_url = f"file:///{abs_path.replace(os.sep, '/')}"
                available_graphs.append((key, file_url, os.path.basename(local_graphs[key])))
        
        if not available_graphs:
            self.progress.emit("Графики не найдены. Создаю отчёт без них.")
        
        # Статистика
        total = len(df)
        pos_count = df['ensemble_sentiment'].value_counts().get('Positive', 0) if 'ensemble_sentiment' in df.columns else 0
        neg_count = df['ensemble_sentiment'].value_counts().get('Negative', 0) if 'ensemble_sentiment' in df.columns else 0
        neu_count = df['ensemble_sentiment'].value_counts().get('Neutral', 0) if 'ensemble_sentiment' in df.columns else 0
        
        # Для accuracy
        accuracy_value = 0
        if 'actual_sentiment' in df.columns and 'ensemble_sentiment' in df.columns:
            valid = df[df['actual_sentiment'].notna()]
            if len(valid) > 0:
                accuracy_value = (valid['ensemble_sentiment'] == valid['actual_sentiment']).mean() * 100
        
        # Генерируем секции графиков - используем data:image для встраивания
        sections_html = []
        toc_items = []
        
        for idx, (key, file_url, filename) in enumerate(available_graphs, 1):
            title = graph_titles.get(key, f'График {idx}')
            
            # Пытаемся загрузить изображение и преобразовать в base64
            img_data_url = ""
            try:
                import base64
                with open(os.path.join(self.output_dir, filename), 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    img_data_url = f"data:image/png;base64,{img_data}"
                    self.progress.emit(f"  График {filename} преобразован в base64")
            except Exception as e:
                self.progress.emit(f"  Не удалось преобразовать {filename}: {e}")
                img_data_url = file_url
            
            toc_items.append(f'<li><a href="#section{idx}">{title.replace("📈 ", "").replace("📊 ", "").replace("📋 ", "").replace("🎯 ", "")}</a></li>')
            
            sections_html.append(f'''
            <div id="section{idx}" class="plot">
                <h2>{title}</h2>
                <img src="{img_data_url}" alt="{title}" style="max-width: 100%;">
            </div>''')
        
        # Секция с таблицей примеров
        table_section_idx = len(available_graphs) + 1
        
        # Создаём полный HTML
        html_content = f'''<!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <title>Отчёт анализа тональности</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 10px; padding: 30px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                h1 {{ color: #4CAF50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
                h2 {{ color: #333; margin-top: 40px; border-left: 4px solid #4CAF50; padding-left: 15px; }}
                h3 {{ color: #555; margin-top: 30px; }}
                .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 30px 0; }}
                .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .stat-card.success {{ background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); }}
                .stat-number {{ font-size: 36px; font-weight: bold; }}
                .plot {{ margin: 40px 0; text-align: center; background: #fafafa; padding: 20px; border-radius: 10px; }}
                .plot img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .plot-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 30px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 12px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #4CAF50; color: white; position: sticky; top: 0; }}
                .positive {{ color: #4CAF50; font-weight: bold; }}
                .negative {{ color: #f44336; font-weight: bold; }}
                .neutral {{ color: #FF9800; font-weight: bold; }}
                .footer {{ text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }}
                .toc {{ background: #f0f0f0; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .toc a {{ text-decoration: none; color: #4CAF50; }}
                .toc li {{ margin: 5px 0; }}
                .timestamp {{ color: #888; font-style: italic; }}
                .no-graphs {{ background: #fff3e0; padding: 20px; border-radius: 8px; text-align: center; color: #e65100; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Отчёт анализа тональности</h1>
                <p class="timestamp">Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number">{total}</div>
                        <div>Всего текстов</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{pos_count}</div>
                        <div>Позитивных</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{neg_count}</div>
                        <div>Негативных</div>
                    </div>
                    <div class="stat-card success">
                        <div class="stat-number">{accuracy_value:.1f}%</div>
                        <div>Точность Ensemble</div>
                    </div>
                </div>
                
                <div class="toc">
                    <h3>Содержание</h3>
                    <p><strong>Создано графиков: {len(available_graphs)}</strong></p>
                    <ul>
                        {''.join(toc_items) if toc_items else '<li>Графики не найдены</li>'}
                        <li><a href="#section{table_section_idx}">Примеры результатов анализа</a></li>
                    </ul>
                </div>
                
                {''.join(sections_html) if sections_html else '<div class="no-graphs"><p>Графики не были созданы. Проверьте, что VisualAnalisTonResults.py отработал успешно.</p></div>'}
                
                <div id="section{table_section_idx}">
                    <h2>Примеры результатов анализа</h2>
                    <div style="overflow-x: auto;">
                        <table>
                            <thead>
                                <tr>
                                    <th>№</th>
                                    <th>Текст</th>
                                    <th>VADER</th>
                                    <th>Flair</th>
                                    <th>RuBERT</th>
                                    <th>RoBERTa</th>
                                    <th>DistilBERT</th>
                                    <th>LR</th>
                                    <th>SVM</th>
                                    <th>RF</th>
                                    <th>Ensemble</th>
                                    <th>Факт</th>
                                </tr>
                            </thead>
                            <tbody>
                                {self.generate_table_rows(df.head(10))}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <div class="footer">
                    <p>Система анализа тональности | Модели: VADER, Flair, RuBERT, RoBERTa, DistilBERT, Logistic Regression, SVM, Random Forest</p>
                    <p>Финальная оценка: ансамбль 8 моделей | Отчёт сгенерирован автоматически</p>
                </div>
            </div>
        </body>
        </html>'''
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.progress.emit(f"HTML отчёт создан с {len(available_graphs)} графиками")
    
    def generate_table_rows(self, df):
        """Генерирует строки таблицы"""
        rows = []
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            text = str(row.get('text', ''))
            if len(text) > 100:
                text = text[:100] + '...'
            
            vader = row.get('vader_sentiment', 'Neutral')
            flair = row.get('flair_sentiment', 'Neutral')
            rubert = row.get('rubert_sentiment', 'Neutral')
            roberta = row.get('roberta_sentiment', 'Neutral')
            distil = row.get('distilbert_sentiment', 'Neutral')
            lr = row.get('logistic_regression_sentiment', 'Neutral')
            svm = row.get('svm_sentiment', 'Neutral')
            rf = row.get('random_forest_sentiment', 'Neutral')
            ensemble = row.get('ensemble_sentiment', 'Neutral')
            
            # Фактическая оценка
            actual = row.get('actual_sentiment', '')
            actual_str = str(actual) if actual and str(actual) != 'nan' else '-'
            
            def get_class(sent):
                sent_str = str(sent)
                if 'Positive' in sent_str:
                    return 'positive'
                elif 'Negative' in sent_str:
                    return 'negative'
                else:
                    return 'neutral'
            
            rows.append(f'''<tr>
                <td>{idx}</td>
                <td>{text}</td>
                <td class="{get_class(vader)}">{vader}</td>
                <td class="{get_class(flair)}">{flair}</td>
                <td class="{get_class(rubert)}">{rubert}</td>
                <td class="{get_class(roberta)}">{roberta}</td>
                <td class="{get_class(distil)}">{distil}</td>
                <td class="{get_class(lr)}">{lr}</td>
                <td class="{get_class(svm)}">{svm}</td>
                <td class="{get_class(rf)}">{rf}</td>
                <td class="{get_class(ensemble)}" style="font-weight: bold; font-size: 110%;">{ensemble}</td>
                <td class="{get_class(actual_str)}">{actual_str}</td>
            </tr>''')
        return '\n'.join(rows)
    
    def convert_to_pdf(self, html_path, pdf_path):
        """Конвертирует HTML в PDF"""
        # Пробуем сначала pdfkit (лучше работает в Windows)
        try:
            import pdfkit
            self.progress.emit("Конвертация в PDF через pdfkit...")
            
            # Проверяем, установлен ли wkhtmltopdf
            wkhtmltopdf_path = None
            
            # Стандартные пути установки wkhtmltopdf
            possible_paths = [
                r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe',
                r'C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe',
                os.path.join(os.environ.get('PROGRAMFILES', ''), 'wkhtmltopdf', 'bin', 'wkhtmltopdf.exe'),
                os.path.join(os.environ.get('PROGRAMFILES(X86)', ''), 'wkhtmltopdf', 'bin', 'wkhtmltopdf.exe'),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    wkhtmltopdf_path = path
                    break
            
            # Если не нашли по стандартным путям, ищем через where
            if not wkhtmltopdf_path:
                import subprocess
                try:
                    result = subprocess.run(['where', 'wkhtmltopdf'], capture_output=True, text=True)
                    if result.returncode == 0 and result.stdout.strip():
                        wkhtmltopdf_path = result.stdout.strip().split('\n')[0]
                except:
                    pass
            
            if wkhtmltopdf_path:
                config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
                pdfkit.from_file(html_path, pdf_path, configuration=config)
                self.progress.emit("PDF создан через pdfkit")
                return
            else:
                # Пробуем без указания пути (если добавлен в PATH)
                pdfkit.from_file(html_path, pdf_path)
                self.progress.emit("PDF создан через pdfkit (PATH)")
                return
                
        except ImportError:
            self.progress.emit("pdfkit не установлен. Пробую WeasyPrint...")
        except Exception as e:
            self.progress.emit(f"Ошибка pdfkit: {e}. Пробую WeasyPrint...")
        
        # Запасной вариант: WeasyPrint
        try:
            from weasyprint import HTML
            self.progress.emit("Конвертация в PDF через WeasyPrint...")
            HTML(html_path).write_pdf(pdf_path)
            self.progress.emit("PDF создан через WeasyPrint")
            return
        except ImportError:
            self.progress.emit("WeasyPrint не установлен")
        except Exception as e:
            self.progress.emit(f"Ошибка WeasyPrint: {e}")
        
        # Если ничего не сработало
        self.progress.emit("⚠️ PDF не создан. Доступен только HTML отчёт.")
        self.progress.emit(f"📄 HTML отчёт находится: {html_path}")
        
        # Пытаемся открыть HTML в браузере
        try:
            import webbrowser
            webbrowser.open(html_path)
            self.progress.emit("🌐 HTML отчёт открыт в браузере")
        except:
            pass


class SentimentAnalysisGUI(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.analysis_process = None
        self.visual_process = None
        self.pdf_generator = None
        self.current_output_dir = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Анализ тональности текстов и диалогов")
        self.setGeometry(100, 100, 1300, 850)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Вкладки
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Вкладка 1: Загрузка файла
        self.file_tab = QWidget()
        self.tabs.addTab(self.file_tab, "Анализ файла")
        self.setup_file_tab()
        
        # Вкладка 2: Диалоги (НОВАЯ)
        self.dialog_tab = QWidget()
        self.tabs.addTab(self.dialog_tab, "Анализ диалогов")
        self.setup_dialog_tab()
        
        # Вкладка 3: Ручной ввод
        self.text_tab = QWidget()
        self.tabs.addTab(self.text_tab, "Ручной ввод")
        self.setup_text_tab()
        
        # Вкладка 4: Результаты
        self.results_tab = QWidget()
        self.tabs.addTab(self.results_tab, "Результаты")
        self.setup_results_tab()
        
        # Статусбар
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Готов к работе")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar.addPermanentWidget(self.progress_bar)
        
        self.apply_styles()
    
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QTabWidget::pane { background: white; border: 1px solid #ccc; border-radius: 5px; }
            QTabBar::tab { padding: 10px 20px; margin-right: 2px; background: #e0e0e0; border-radius: 5px; }
            QTabBar::tab:selected { background: #4CAF50; color: white; }
            QPushButton { background: #4CAF50; color: white; border: none; padding: 8px 20px; border-radius: 4px; font-size: 12px; }
            QPushButton:hover { background: #45a049; }
            QPushButton:disabled { background: #cccccc; }
            QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px; margin-top: 12px; padding-top: 10px; }
            QTextEdit, QLineEdit, QComboBox { border: 1px solid #ccc; border-radius: 4px; padding: 5px; }
            QTableWidget { alternate-background-color: #f9f9f9; }
            QHeaderView::section { background: #4CAF50; color: white; padding: 5px; border: none; }
        """)
    
    def setup_file_tab(self):
        layout = QVBoxLayout(self.file_tab)
        
        # Выбор файла
        file_group = QGroupBox("Выбор файла")
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Путь к CSV файлу...")
        browse_btn = QPushButton("Обзор")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_path)
        file_layout.addWidget(browse_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Настройка столбцов
        columns_group = QGroupBox("Настройка столбцов")
        columns_layout = QGridLayout()
        
        columns_layout.addWidget(QLabel("Столбец с текстом:"), 0, 0)
        self.text_column = QComboBox()
        self.text_column.setEditable(True)
        self.text_column.addItems(["text"])
        columns_layout.addWidget(self.text_column, 0, 1)
        
        columns_layout.addWidget(QLabel("Столбец с оценкой:"), 1, 0)
        self.rating_column = QComboBox()
        self.rating_column.setEditable(True)
        self.rating_column.addItems(["label"])
        columns_layout.addWidget(self.rating_column, 1, 1)
        
        self.use_rating = QCheckBox("Использовать оценки для расчёта точности")
        self.use_rating.setChecked(True)
        columns_layout.addWidget(self.use_rating, 2, 0, 1, 2)
        
        columns_group.setLayout(columns_layout)
        layout.addWidget(columns_group)

        rows_group = QGroupBox("Настройка анализа")
        rows_layout = QHBoxLayout()
        
        rows_layout.addWidget(QLabel("Анализировать первые:"))
        self.max_rows_input = QLineEdit()
        self.max_rows_input.setPlaceholderText("Все строки (оставьте пустым)")
        self.max_rows_input.setText("100")
        self.max_rows_input.setMaximumWidth(150)
        rows_layout.addWidget(self.max_rows_input)
        
        rows_layout.addWidget(QLabel("строк(и)"))
        rows_layout.addStretch()
        
        info_rows = QLabel("Укажите число для анализа только первых N строк. Пустое поле = все строки.")
        info_rows.setStyleSheet("color: #666; font-size: 11px;")
        rows_layout.addWidget(info_rows)
        
        rows_group.setLayout(rows_layout)
        layout.addWidget(rows_group)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("Запустить анализ")
        self.analyze_btn.clicked.connect(self.start_file_analysis)
        self.analyze_btn.setMinimumHeight(50)
        self.analyze_btn.setStyleSheet("font-size: 14px; font-weight: bold;")
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.analyze_btn)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Информация
        info_label = QLabel(
            "Информация:\n"
            "• Будут использованы 8 моделей: VADER, Flair, RuBERT, RoBERTa, DistilBERT, Logistic Regression, SVM, Random Forest\n"
            "• После анализа автоматически запустится VisualAnalisTonResults для создания графиков\n"
            "• В PDF отчёт будут включены ТОЛЬКО графики из VisualAnalisTonResults\n"
            "• Результаты сохранятся в папку answer/ с временной меткой"
        )
        info_label.setStyleSheet("background: #e8f5e9; padding: 10px; border-radius: 5px; margin-top: 20px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
    
    def setup_text_tab(self):
        layout = QVBoxLayout(self.text_tab)
        
        # Выбор режима анализа
        mode_group = QGroupBox("Режим анализа")
        mode_layout = QHBoxLayout()
        
        self.radio_single = QRadioButton("Одиночный текст (отзыв)")
        self.radio_single.setChecked(True)
        self.radio_dialog = QRadioButton("Диалог (несколько реплик)")
        
        self.radio_single.toggled.connect(self.on_mode_changed)
        self.radio_dialog.toggled.connect(self.on_mode_changed)
        
        mode_layout.addWidget(self.radio_single)
        mode_layout.addWidget(self.radio_dialog)
        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Область ввода
        input_group = QGroupBox("Введите текст или диалог")
        input_layout = QVBoxLayout()
        
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText(
            "Введите отзыв для анализа тональности...\n\n"
            "Пример:\n"
            "Отличный сервис, очень доволен! Всё быстро и качественно."
        )
        self.text_input.setMinimumHeight(200)
        input_layout.addWidget(self.text_input)
        
        # Подсказка для диалога
        self.dialog_hint = QLabel(
            "💡 В режиме диалога: каждая строка — отдельная реплика.\n"
            "Пустые строки игнорируются. Можно также разделять реплики через '----'."
        )
        self.dialog_hint.setStyleSheet(
            "background: #fff3e0; padding: 8px; border-radius: 5px; color: #e65100;"
        )
        self.dialog_hint.setWordWrap(True)
        self.dialog_hint.setVisible(False)  # Скрыто по умолчанию
        input_layout.addWidget(self.dialog_hint)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("Очистить")
        self.clear_btn.clicked.connect(self.clear_text_input)
        self.clear_btn.setStyleSheet(
            "background: #9E9E9E; font-size: 13px; padding: 10px 25px;"
        )
        
        self.quick_analyze_btn = QPushButton("Анализировать")
        self.quick_analyze_btn.clicked.connect(self.start_quick_analysis)
        self.quick_analyze_btn.setMinimumHeight(50)
        self.quick_analyze_btn.setStyleSheet(
            "font-size: 14px; font-weight: bold; background: #FF9800; padding: 10px 40px;"
        )
        
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.clear_btn)
        buttons_layout.addWidget(self.quick_analyze_btn)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Результат быстрого анализа
        result_group = QGroupBox("Результат анализа")
        result_layout = QVBoxLayout()
        
        self.quick_result_text = QTextEdit()
        self.quick_result_text.setReadOnly(True)
        self.quick_result_text.setMinimumHeight(150)
        self.quick_result_text.setPlaceholderText(
            "Здесь появятся результаты анализа..."
        )
        result_layout.addWidget(self.quick_result_text)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        # Информация
        info_label = QLabel(
            "Как это работает:\n"
            "• Одиночный текст: анализируется как один отзыв всеми 8 моделями\n"
            "• Диалог: каждая реплика анализируется отдельно + агрегация по диалогу\n"
            "• Используются те же модели: VADER, Flair, RuBERT, RoBERTa, DistilBERT, LR, SVM, RF"
        )
        info_label.setStyleSheet(
            "background: #e3f2fd; padding: 10px; border-radius: 5px; color: #1565c0;"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
    
    def setup_results_tab(self):
        layout = QVBoxLayout(self.results_tab)
        
        # Панель управления
        control_layout = QHBoxLayout()
        self.open_pdf_btn = QPushButton("Открыть PDF отчёт")
        self.open_pdf_btn.clicked.connect(self.open_pdf)
        self.open_pdf_btn.setEnabled(False)
        
        self.open_folder_btn = QPushButton("Открыть папку")
        self.open_folder_btn.clicked.connect(self.open_folder)
        self.open_folder_btn.setEnabled(False)
        
        control_layout.addWidget(self.open_pdf_btn)
        control_layout.addWidget(self.open_folder_btn)
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Таблица результатов
        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        layout.addWidget(self.results_table)
        
        # Лог выполнения
        log_group = QGroupBox("Лог выполнения")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
    
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите CSV файл", "", "CSV файлы (*.csv);;Все файлы (*.*)"
        )
        if file_path:
            self.file_path.setText(file_path)
            try:
                df = pd.read_csv(file_path, nrows=5)
                columns = df.columns.tolist()
                self.text_column.clear()
                self.rating_column.clear()
                for col in columns:
                    self.text_column.addItem(col)
                    self.rating_column.addItem(col)
                self.statusBar.showMessage(f"Загружен файл: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось прочитать файл: {e}")
    
    def start_file_analysis(self):
        file_path = self.file_path.text().strip()
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Ошибка", "Выберите существующий CSV файл")
            return
        
        text_col = self.text_column.currentText()
        rating_col = self.rating_column.currentText() if self.use_rating.isChecked() else None
        use_rating = self.use_rating.isChecked()
        
        max_rows_text = self.max_rows_input.text().strip()
        max_rows = None
        if max_rows_text:
            try:
                max_rows = int(max_rows_text)
                if max_rows <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "Ошибка", "Количество строк должно быть положительным числом")
                return

        # Создаём выходную папку
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("answer", f"{base_name}_{timestamp}")
        
        self.current_output_dir = output_dir
        
        # Блокируем интерфейс
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("Анализ выполняется...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        self.log_text.clear()
        self.log_text.append("=" * 60)
        self.log_text.append("ЗАПУСК ПОЛНОГО ЦИКЛА АНАЛИЗА")
        self.log_text.append("=" * 60)
        self.log_text.append(f"Файл: {file_path}")
        self.log_text.append(f"Выходная папка: {output_dir}")
        self.log_text.append(f"Столбец текста: {text_col}")
        if use_rating:
            self.log_text.append(f"Столбец оценки: {rating_col}")
        if max_rows:
            self.log_text.append(f"Анализ первых {max_rows} строк")
        else:
            self.log_text.append(f"Анализ всех строк")
        self.log_text.append("")
        self.log_text.append("ЭТАП 1: Запуск основного анализа (8 моделей)...")
        self.log_text.append("-" * 50)
        
        # Запускаем основной анализ
        self.analysis_process = AnalysisProcess()
        self.analysis_process.output_received.connect(self.on_analysis_output)
        self.analysis_process.finished_analysis.connect(self.on_analysis_finished)
        self.analysis_process.start_analysis(file_path, text_col, rating_col, use_rating, output_dir, max_rows)
    
    def on_analysis_output(self, text):
        self.log_text.append(text.strip())
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        QApplication.processEvents()
    
    def on_analysis_finished(self, output_dir, success):
        if not success:
            self.analyze_btn.setEnabled(True)
            self.analyze_btn.setText("Запустить анализ")
            self.progress_bar.setVisible(False)
            self.log_text.append("\nОшибка на этапе основного анализа!")
            QMessageBox.critical(self, "Ошибка", "Не удалось выполнить основной анализ")
            return
        
        self.log_text.append("")
        self.log_text.append("Основной анализ завершён успешно!")
        self.log_text.append("")
        self.log_text.append("ЭТАП 2: Запуск визуализации (VisualAnalisTonResults)...")
        self.log_text.append("-" * 50)
        
        # Запускаем визуализацию
        self.visual_process = VisualAnalysisProcess()
        self.visual_process.output_received.connect(self.on_visual_output)
        self.visual_process.finished_visual.connect(self.on_visual_finished)
        self.visual_process.start_visualization(output_dir)
    
    def on_visual_output(self, text):
        self.log_text.append(text.strip())
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        QApplication.processEvents()
    
    def on_visual_finished(self, output_dir, success):
        self.current_output_dir = output_dir
        
        if success:
            self.log_text.append("")
            self.log_text.append("Визуализация завершена успешно!")
        else:
            self.log_text.append("")
            self.log_text.append("Визуализация завершена с ошибками (графики могут быть не созданы)")
            self.log_text.append("   Проверьте, что VisualAnalisTonResults.py существует и работает корректно.")
        
        self.log_text.append("")
        self.log_text.append("ЭТАП 3: Генерация PDF отчёта с графиками...")
        self.log_text.append("-" * 50)
        
        # Запускаем генерацию PDF
        self.pdf_generator = PDFReportGenerator(output_dir)
        self.pdf_generator.progress.connect(lambda msg: self.log_text.append(f"   {msg}"))
        self.pdf_generator.finished.connect(self.on_pdf_generated)
        self.pdf_generator.error.connect(lambda e: self.log_text.append(f"   ❌ {e}"))
        self.pdf_generator.start()
    
    def on_pdf_generated(self, pdf_path):
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Запустить анализ")
        self.progress_bar.setVisible(False)
        
        self.log_text.append(f"PDF отчёт создан: {pdf_path}")
        self.log_text.append("")
        self.log_text.append("=" * 60)
        self.log_text.append("ПОЛНЫЙ ЦИКЛ АНАЛИЗА ЗАВЕРШЁН!")
        self.log_text.append("=" * 60)
        
        self.current_pdf_path = pdf_path
        
        # Загружаем результаты в таблицу
        result_file = os.path.join(self.current_output_dir, 'отзывы(результат работы).csv')
        if os.path.exists(result_file):
            df = pd.read_csv(result_file, encoding='utf-8-sig')
            self.display_results_table(df)
        
        # Активируем кнопки
        self.open_pdf_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(True)
        
        # Переключаемся на вкладку результатов
        self.tabs.setCurrentIndex(3)
        
        QMessageBox.information(
            self, 
            "Анализ завершён!", 
            f"Все этапы выполнены успешно!\n\n"
            f"Папка с результатами: {self.current_output_dir}\n"
            f"PDF отчёт: {pdf_path}\n\n"
            f"В PDF включены графики из VisualAnalisTonResults."
        )
    
    def display_results_table(self, df):
        """Отображение результатов в таблице"""
        columns = ['№', 'Текст', 'VADER', 'Flair', 'RuBERT', 'RoBERTa', 'DistilBERT', 'Ensemble']
        if 'actual_sentiment' in df.columns and df['actual_sentiment'].notna().any():
            columns.append('Факт')
        
        self.results_table.setColumnCount(len(columns))
        self.results_table.setHorizontalHeaderLabels(columns)
        
        # Берём первые 100 строк для отображения
        display_df = df.head(100)
        self.results_table.setRowCount(len(display_df))
        
        colors = {
            'Positive': QColor(200, 230, 200),
            'Negative': QColor(230, 200, 200),
            'Neutral': QColor(230, 230, 200)
        }
        
        for row_idx, (_, row_data) in enumerate(display_df.iterrows()):
            text = str(row_data.get('text', ''))
            if len(text) > 80:
                text = text[:80] + '...'
            
            # Номер строки
            self.results_table.setItem(row_idx, 0, QTableWidgetItem(str(row_idx + 1)))
            self.results_table.setItem(row_idx, 1, QTableWidgetItem(text))
            
            vader = row_data.get('vader_sentiment', 'Neutral')
            flair = row_data.get('flair_sentiment', 'Neutral')
            rubert = row_data.get('rubert_sentiment', 'Neutral')
            roberta = row_data.get('roberta_sentiment', 'Neutral')
            distil = row_data.get('distilbert_sentiment', 'Neutral')
            ensemble = row_data.get('ensemble_sentiment', 'Neutral')
            
            for col_idx, val in enumerate([vader, flair, rubert, roberta, distil, ensemble], 2):
                item = QTableWidgetItem(str(val))
                if val in colors:
                    item.setBackground(colors[val])
                self.results_table.setItem(row_idx, col_idx, item)
            
            if 'actual_sentiment' in columns:
                actual = row_data.get('actual_sentiment', '')
                actual_str = str(actual) if actual and str(actual) != 'nan' else '-'
                if len(columns) - 1 < self.results_table.columnCount():
                    item = QTableWidgetItem(actual_str)
                    if actual_str in colors:
                        item.setBackground(colors[actual_str])
                    self.results_table.setItem(row_idx, len(columns)-1, item)
        
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        for i in range(len(columns)):
            if i != 1:
                header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
    
    def start_quick_analysis(self):
        """Быстрый анализ введённого текста или диалога"""
        raw_text = self.text_input.toPlainText().strip()
        
        if not raw_text:
            QMessageBox.warning(self, "Ошибка", "Введите текст для анализа")
            return
        
        is_dialog_mode = self.radio_dialog.isChecked()
        
        # Блокируем кнопку
        self.quick_analyze_btn.setEnabled(False)
        self.quick_analyze_btn.setText("Анализ...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.quick_result_text.clear()
        self.quick_result_text.append("⏳ Запуск анализа...\n")
        QApplication.processEvents()
        
        try:
            # Импортируем модули анализа
            from AnalisTon import (
                sia, load_sklearn_models, get_sklearn_sentiment,
                get_flair_score, get_transformer_sentiment_ru,
                get_transformer_sentiment_en, get_transformer_sentiment_ru_light,
                interpret_sentiment_scores, TextTranslator
            )
            import numpy as np
            import asyncio
            
            # Загружаем sklearn модели
            self.quick_result_text.append("📦 Загрузка моделей...")
            QApplication.processEvents()
            
            sklearn_models = load_sklearn_models('models')
            if sklearn_models:
                self.quick_result_text.append(
                    f"   Загружено {len(sklearn_models)} sklearn моделей\n"
                )
            else:
                self.quick_result_text.append(
                    "   ⚠️ sklearn модели не найдены, использую только трансформеры\n"
                )
            QApplication.processEvents()
            
            # Разбиваем на реплики
            if is_dialog_mode:
                messages = self._parse_dialog_input(raw_text)
                self.quick_result_text.append(
                    f"💬 Режим диалога: {len(messages)} реплик\n"
                )
            else:
                messages = [raw_text]
                self.quick_result_text.append("📝 Режим одиночного текста\n")
            
            if not messages:
                raise ValueError("Не удалось извлечь реплики из ввода")
            
            QApplication.processEvents()
            
            # Инициализируем переводчик для RU->EN
            translator = TextTranslator(max_concurrent_translations=10)
            
            # Переводим все реплики
            self.quick_result_text.append("🌐 Перевод реплик (RU→EN)...")
            QApplication.processEvents()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                translated_texts = loop.run_until_complete(
                    translator.translate_batch(messages, batch_size=50)
                )
            finally:
                loop.close()
            
            self.quick_result_text.append("   Готово\n")
            QApplication.processEvents()
            
            # Анализируем каждую реплику
            self.quick_result_text.append("🔍 Анализ моделями...\n")
            QApplication.processEvents()
            
            message_results = []
            for idx, (msg, trans) in enumerate(zip(messages, translated_texts)):
                self.quick_result_text.append(
                    f"   [{idx+1}/{len(messages)}] Анализ реплики..."
                )
                QApplication.processEvents()
                
                result = self._analyze_single_text(
                    msg, trans, sklearn_models
                )
                result['message_index'] = idx
                result['message_text'] = msg
                message_results.append(result)
            
            self.quick_result_text.append("   Готово\n")
            QApplication.processEvents()
            
            # Формируем отчёт
            self._display_quick_results(
                message_results, is_dialog_mode, sklearn_models
            )
            
            self.statusBar.showMessage("Анализ завершён")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.quick_result_text.append(f"\n❌ Ошибка: {e}\n\n{error_details}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось выполнить анализ:\n{e}")
        finally:
            self.quick_analyze_btn.setEnabled(True)
            self.quick_analyze_btn.setText("Анализировать")
            self.progress_bar.setVisible(False)
    
    def open_pdf(self):
        if hasattr(self, 'current_pdf_path') and os.path.exists(self.current_pdf_path):
            webbrowser.open(self.current_pdf_path)
        elif self.current_output_dir:
            # Пробуем открыть PDF
            pdf_path = os.path.join(self.current_output_dir, 'sentiment_report.pdf')
            if os.path.exists(pdf_path):
                webbrowser.open(pdf_path)
            else:
                # Если PDF нет, открываем HTML
                html_path = os.path.join(self.current_output_dir, 'sentiment_report.html')
                if os.path.exists(html_path):
                    webbrowser.open(html_path)
                    self.log_text.append("PDF не найден, открыт HTML отчёт")
                else:
                    QMessageBox.warning(self, "Ошибка", "Отчёт не найден")
    
    def open_folder(self):
        if self.current_output_dir and os.path.exists(self.current_output_dir):
            if platform.system() == "Windows":
                os.startfile(self.current_output_dir)
            else:
                subprocess.call(["open", self.current_output_dir])
        else:
            QMessageBox.warning(self, "Ошибка", "Папка не найдена")

    def setup_dialog_tab(self):
        """Настройка вкладки для анализа диалогов"""
        layout = QVBoxLayout(self.dialog_tab)
        
        # Выбор файла
        file_group = QGroupBox("Выбор файла с диалогами")
        file_layout = QHBoxLayout()
        self.dialog_file_path = QLineEdit()
        self.dialog_file_path.setPlaceholderText("Путь к JSON или CSV файлу с диалогами...")
        dialog_browse_btn = QPushButton("Обзор")
        dialog_browse_btn.clicked.connect(self.browse_dialog_file)
        file_layout.addWidget(self.dialog_file_path)
        file_layout.addWidget(dialog_browse_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Информация о формате
        format_info = QLabel(
            "Поддерживаемые форматы:\n"
            "• JSONL/NDJSON (по одной строке = один диалог):\n"
            "   {\"sample\": [\"реплика1\", \"реплика2\", ...]}\n"
            "• JSON: {\"sample\": [\"реплика1\", \"реплика2\", ...]}\n"
            "• JSON: [{\"dialog_id\": \"...\", \"messages\": [...]}, ...]\n"
            "• CSV: с колонками dialog_id и message"
        )
        format_info.setStyleSheet("background: #e3f2fd; padding: 10px; border-radius: 5px; color: #1565c0;")
        format_info.setWordWrap(True)
        layout.addWidget(format_info)
        
        # Предпросмотр диалогов
        preview_group = QGroupBox("Предпросмотр диалогов")
        preview_layout = QVBoxLayout()
        self.dialog_preview = QTextEdit()
        self.dialog_preview.setReadOnly(True)
        self.dialog_preview.setMaximumHeight(200)
        self.dialog_preview.setPlaceholderText("Здесь появятся загруженные диалоги...")
        preview_layout.addWidget(self.dialog_preview)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Настройки анализа
        settings_group = QGroupBox("Настройки анализа диалогов")
        settings_layout = QGridLayout()
        
        settings_layout.addWidget(QLabel("Максимум диалогов:"), 0, 0)
        self.dialog_max_rows = QLineEdit()
        self.dialog_max_rows.setPlaceholderText("Все (оставьте пустым)")
        self.dialog_max_rows.setText("50")
        settings_layout.addWidget(self.dialog_max_rows, 0, 1)
        
        self.analyze_individual = QCheckBox("Анализировать каждую реплику отдельно")
        self.analyze_individual.setChecked(True)
        settings_layout.addWidget(self.analyze_individual, 1, 0, 1, 2)
        
        self.create_visualizations = QCheckBox("Создать визуализации")
        self.create_visualizations.setChecked(True)
        settings_layout.addWidget(self.create_visualizations, 2, 0, 1, 2)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("Предпросмотр")
        self.preview_btn.clicked.connect(self.preview_dialogs)
        
        self.analyze_dialogs_btn = QPushButton("Анализировать диалоги")
        self.analyze_dialogs_btn.clicked.connect(self.start_dialog_analysis)
        self.analyze_dialogs_btn.setMinimumHeight(50)
        self.analyze_dialogs_btn.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        buttons_layout.addWidget(self.preview_btn)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.analyze_dialogs_btn)
        layout.addLayout(buttons_layout)
        
        # Информация
        info_label = QLabel(
            "Анализ диалогов:\n"
            "• Каждая реплика анализируется отдельно всеми 8 моделями\n"
            "• Результаты агрегируются для всего диалога\n"
            "• Определяется общая тональность и тренд диалога\n"
            "• Создаются визуализации по диалогам"
        )
        info_label.setStyleSheet("background: #e8f5e9; padding: 10px; border-radius: 5px; margin-top: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()

    def browse_dialog_file(self):
        """Выбор файла с диалогами"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с диалогами", "", 
            "Все поддерживаемые (*.json *.jsonl *.ndjson *.csv);;"
            "JSON файлы (*.json);;"
            "JSONL файлы (*.jsonl *.ndjson);;"
            "CSV файлы (*.csv);;"
            "Все файлы (*.*)"
        )
        if file_path:
            self.dialog_file_path.setText(file_path)
            self.preview_dialogs()

    def preview_dialogs(self):
        """Предпросмотр загруженных диалогов"""
        file_path = self.dialog_file_path.text().strip()
        
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Ошибка", "Выберите существующий файл")
            return
        
        try:
            from DialogProcessor import DialogLoader
            
            # Универсальный загрузчик — определяет формат автоматически
            dialogs = DialogLoader.load(file_path)
            
            if not dialogs:
                self.dialog_preview.setText("Не удалось загрузить диалоги")
                return
            
            preview_text = f"Загружено диалогов: {len(dialogs)}\n"
            preview_text += f"Формат файла: {os.path.splitext(file_path)[1].upper()}\n"
            preview_text += "=" * 50 + "\n\n"
            
            for i, dialog in enumerate(dialogs[:5]):
                preview_text += f"Диалог {i+1}: {dialog['dialog_id']}\n"
                preview_text += f"  Реплик: {dialog['num_messages']}\n"
                
                for j, msg in enumerate(dialog['messages'][:3]):
                    msg_preview = str(msg)[:80] + "..." if len(str(msg)) > 80 else str(msg)
                    preview_text += f"    [{j+1}] {msg_preview}\n"
                
                if dialog['num_messages'] > 3:
                    preview_text += f"    ... и ещё {dialog['num_messages'] - 3} реплик\n"
                
                preview_text += "\n"
            
            if len(dialogs) > 5:
                preview_text += f"... и ещё {len(dialogs) - 5} диалогов\n"
            
            self.dialog_preview.setText(preview_text)
            self.statusBar.showMessage(f"Загружено {len(dialogs)} диалогов")
            
        except Exception as e:
            self.dialog_preview.setText(f"Ошибка загрузки: {e}")
            QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить диалоги: {e}")

    def start_dialog_analysis(self):
        """Запуск анализа диалогов"""
        file_path = self.dialog_file_path.text().strip()
        
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Ошибка", "Выберите существующий файл с диалогами")
            return
        
        # Создаём выходную папку
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("answer", f"dialogs_{base_name}_{timestamp}")
        
        # ВАЖНО: создаём папку сразу
        os.makedirs(output_dir, exist_ok=True)
        
        self.current_output_dir = output_dir
        
        # Блокируем интерфейс
        self.analyze_dialogs_btn.setEnabled(False)
        self.analyze_dialogs_btn.setText("Анализ диалогов...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        self.log_text.clear()
        self.log_text.append("=" * 60)
        self.log_text.append("АНАЛИЗ ДИАЛОГОВ")
        self.log_text.append("=" * 60)
        self.log_text.append(f"Файл: {file_path}")
        self.log_text.append(f"Выходная папка: {output_dir}")
        self.log_text.append("")
        
        max_rows = None
        max_rows_text = self.dialog_max_rows.text().strip()
        if max_rows_text:
            try:
                max_rows = int(max_rows_text)
            except ValueError:
                pass
        
        config = {
            'input_file': file_path,
            'output_dir': output_dir,
            'is_dialog': True,
            'max_rows': max_rows
        }
        
        config_file = os.path.join(output_dir, 'dialog_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # Запускаем процесс
        self.analysis_process = AnalysisProcess()
        # ВАЖНО: устанавливаем output_dir ДО запуска!
        self.analysis_process.output_dir = output_dir
        self.analysis_process.output_received.connect(self.on_analysis_output)
        self.analysis_process.finished_analysis.connect(self.on_dialog_analysis_finished)
        
        script_path = os.path.join(os.path.dirname(__file__), 'Run_analysis.py')
        program = sys.executable
        self.analysis_process.start(program, [script_path, config_file])

    def on_dialog_analysis_finished(self, output_dir, success):
        """Обработка завершения анализа диалогов"""
        self.analyze_dialogs_btn.setEnabled(True)
        self.analyze_dialogs_btn.setText("Анализировать диалоги")
        self.progress_bar.setVisible(False)

        if success:
            self.log_text.append("\n[OK] Анализ диалогов завершён!")

            # Канонический файл уже лежит в output_dir как
            # 'отзывы(результат работы).csv' (его пишет Run_analysis.py).
            # Копируем его в answer/, чтобы VisualAnalisTonResults.py его нашёл.
            import shutil
            canonical_src = os.path.join(output_dir, 'отзывы(результат работы).csv')
            canonical_dst = os.path.join('answer', 'отзывы(результат работы).csv')

            if os.path.exists(canonical_src):
                os.makedirs('answer', exist_ok=True)
                shutil.copy2(canonical_src, canonical_dst)
                self.log_text.append(f"[OK] Скопировано для визуализации: {canonical_dst}")
            else:
                self.log_text.append(f"[!] Файл результатов не найден: {canonical_src}")

            # Запускаем визуализацию (те же графики, что и для отзывов)
            self.log_text.append("\n[ЭТАП 2] Запуск VisualAnalisTonResults...")
            self.visual_process = VisualAnalysisProcess()
            self.visual_process.output_received.connect(self.on_visual_output)
            self.visual_process.finished_visual.connect(self.on_dialog_visual_finished)

            # Передаём папку с результатами
            self.visual_process.output_dir = output_dir
            visual_script = os.path.join(os.path.dirname(__file__), 'VisualAnalisTonResults.py')

            program = sys.executable
            self.visual_process.setWorkingDirectory(os.path.dirname(os.path.abspath(__file__)))
            self.visual_process.start(program, [visual_script])
        else:
            self.log_text.append("\n[X] Ошибка анализа диалогов")
            QMessageBox.critical(self, "Ошибка", "Не удалось выполнить анализ диалогов")

    def on_dialog_visual_finished(self, output_dir, success):
        """После визуализации диалогов — генерируем PDF-отчёт"""
        if success:
            self.log_text.append("[OK] Графики построены")
        else:
            self.log_text.append("[!] Визуализация завершилась с ошибками")

        # Генерируем PDF так же, как для отзывов
        self.log_text.append("\n[ЭТАП 3] Генерация PDF отчёта...")
        self.pdf_generator = PDFReportGenerator(output_dir)
        self.pdf_generator.progress.connect(lambda msg: self.log_text.append(f"   {msg}"))
        self.pdf_generator.finished.connect(self.on_pdf_generated)
        self.pdf_generator.error.connect(lambda e: self.log_text.append(f"   ❌ {e}"))
        self.pdf_generator.start()

    def display_dialog_results(self, df):
        """Отображение результатов анализа диалогов в таблице"""
        columns = ['№', 'Dialog ID', 'Реплик', 'Тональность', 'Score', '+', '-', '=']
        
        self.results_table.setColumnCount(len(columns))
        self.results_table.setHorizontalHeaderLabels(columns)
        
        display_df = df.head(100)
        self.results_table.setRowCount(len(display_df))
        
        colors = {
            'Positive': QColor(200, 230, 200),
            'Negative': QColor(230, 200, 200),
            'Neutral': QColor(230, 230, 200)
        }
        
        for row_idx, (_, row_data) in enumerate(display_df.iterrows()):
            self.results_table.setItem(row_idx, 0, QTableWidgetItem(str(row_idx + 1)))
            self.results_table.setItem(row_idx, 1, QTableWidgetItem(str(row_data.get('dialog_id', ''))))
            self.results_table.setItem(row_idx, 2, QTableWidgetItem(str(row_data.get('num_messages', ''))))
            
            sentiment = row_data.get('dialog_sentiment', 'Neutral')
            item = QTableWidgetItem(str(sentiment))
            if sentiment in colors:
                item.setBackground(colors[sentiment])
            self.results_table.setItem(row_idx, 3, item)
            
            self.results_table.setItem(row_idx, 4, QTableWidgetItem(str(row_data.get('ensemble_score_rounded', ''))))
            self.results_table.setItem(row_idx, 5, QTableWidgetItem(str(row_data.get('positive_messages', 0))))
            self.results_table.setItem(row_idx, 6, QTableWidgetItem(str(row_data.get('negative_messages', 0))))
            self.results_table.setItem(row_idx, 7, QTableWidgetItem(str(row_data.get('neutral_messages', 0))))
        
        header = self.results_table.horizontalHeader()
        for i in range(len(columns)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

    def on_mode_changed(self):
        """Обработка переключения режима анализа"""
        is_dialog = self.radio_dialog.isChecked()
        self.dialog_hint.setVisible(is_dialog)
        
        if is_dialog:
            self.text_input.setPlaceholderText(
                "Введите диалог — каждая строка отдельная реплика:\n\n"
                "Здравствуйте, чем могу помочь?\n"
                "Хотел узнать о доставке\n"
                "Доставка бесплатная от 1000 рублей\n"
                "Отлично, спасибо!"
            )
        else:
            self.text_input.setPlaceholderText(
                "Введите отзыв для анализа тональности...\n\n"
                "Пример:\n"
                "Отличный сервис, очень доволен! Всё быстро и качественно."
            )

    def clear_text_input(self):
        """Очистка поля ввода и результатов"""
        self.text_input.clear()
        self.quick_result_text.clear()
        self.statusBar.showMessage("Поле ввода очищено")

    def _parse_dialog_input(self, raw_text):
        """
        Разбирает введённый текст на реплики.
        
        Поддерживает:
        - Разделение по строкам (каждая строка = реплика)
        - Разделение по '----'
        - Комбинацию обоих
        """
        # Сначала пробуем разделить по '----'
        if '----' in raw_text:
            parts = [p.strip() for p in raw_text.split('----')]
        else:
            # Иначе — по строкам
            parts = [line.strip() for line in raw_text.split('\n')]
        
        # Фильтруем пустые
        messages = [p for p in parts if p]
        
        return messages

    def _analyze_single_text(self, text, translated_text, sklearn_models):
        """
        Анализирует один текст всеми 8 моделями.
        
        Returns:
        dict: результаты по каждой модели + ensemble
        """
        from AnalisTon import (
            sia, get_sklearn_sentiment,
            get_flair_score, get_transformer_sentiment_ru,
            get_transformer_sentiment_en, get_transformer_sentiment_ru_light,
            interpret_sentiment_scores
        )
        import numpy as np
        
        text = str(text)
        translated = translated_text if translated_text else text
        
        result = {}
        
        # 1. VADER (на переводе)
        try:
            scores = sia.polarity_scores(translated)
            result['vader_sentiment'] = interpret_sentiment_scores(scores)
            result['vader_compound'] = scores['compound']
            if scores['compound'] >= 0.33:
                result['vader_score'] = 5
            elif scores['compound'] <= -0.33:
                result['vader_score'] = 1
            else:
                result['vader_score'] = 3
        except Exception:
            result['vader_sentiment'] = 'Neutral'
            result['vader_compound'] = 0.0
            result['vader_score'] = 3
        
        # 2. Flair (на оригинале)
        try:
            flair_score, flair_sent, flair_conf = get_flair_score(text)
            result['flair_sentiment'] = flair_sent
            result['flair_score'] = flair_score
            result['flair_confidence'] = flair_conf
        except Exception:
            result['flair_sentiment'] = 'Neutral'
            result['flair_score'] = 3
            result['flair_confidence'] = 0.0
        
        # 3. RuBERT (на оригинале)
        try:
            _, r_score, r_sent, r_conf = get_transformer_sentiment_ru(text)
            result['rubert_sentiment'] = r_sent
            result['rubert_score'] = r_score
            result['rubert_confidence'] = r_conf
        except Exception:
            result['rubert_sentiment'] = 'Neutral'
            result['rubert_score'] = 3
            result['rubert_confidence'] = 0.0
        
        # 4. RoBERTa (на переводе)
        try:
            _, r_score, r_sent, r_conf = get_transformer_sentiment_en(translated, 'roberta')
            result['roberta_sentiment'] = r_sent
            result['roberta_score'] = r_score
            result['roberta_confidence'] = r_conf
        except Exception:
            result['roberta_sentiment'] = 'Neutral'
            result['roberta_score'] = 3
            result['roberta_confidence'] = 0.0
        
        # 5. DistilBERT (на оригинале)
        try:
            _, d_score, d_sent, d_conf = get_transformer_sentiment_ru_light(text)
            result['distilbert_sentiment'] = d_sent
            result['distilbert_score'] = d_score
            result['distilbert_confidence'] = d_conf
        except Exception:
            result['distilbert_sentiment'] = 'Neutral'
            result['distilbert_score'] = 3
            result['distilbert_confidence'] = 0.0
        
        # 6-8. sklearn модели
        for model_name in sklearn_models.keys():
            try:
                s, sent, conf = get_sklearn_sentiment(text, model_name, sklearn_models)
                result[f'{model_name}_sentiment'] = sent
                result[f'{model_name}_score'] = s
                result[f'{model_name}_confidence'] = conf
            except Exception:
                result[f'{model_name}_sentiment'] = 'Neutral'
                result[f'{model_name}_score'] = 3
                result[f'{model_name}_confidence'] = 0.0
        
        # Ensemble
        all_scores = [
            result.get('vader_score', 3),
            result.get('flair_score', 3),
            result.get('rubert_score', 3),
            result.get('roberta_score', 3),
            result.get('distilbert_score', 3),
        ]
        for model_name in sklearn_models.keys():
            all_scores.append(result.get(f'{model_name}_score', 3))
        
        ensemble_avg = float(np.mean(all_scores))
        ensemble_rounded = round(ensemble_avg)
        result['ensemble_score_avg'] = ensemble_avg
        result['ensemble_score'] = ensemble_rounded
        
        if ensemble_rounded >= 4:
            result['ensemble_sentiment'] = 'Positive'
        elif ensemble_rounded <= 2:
            result['ensemble_sentiment'] = 'Negative'
        else:
            result['ensemble_sentiment'] = 'Neutral'
        
        return result

    def _display_quick_results(self, message_results, is_dialog_mode, sklearn_models):
        """Отображает результаты быстрого анализа"""
        import numpy as np
        
        self.quick_result_text.clear()
        
        # Цветовые маркеры
        def emoji(sent):
            if sent == 'Positive':
                return '🟢'
            elif sent == 'Negative':
                return '🔴'
            else:
                return '🟡'
        
        # Заголовок
        self.quick_result_text.append("=" * 60)
        if is_dialog_mode:
            self.quick_result_text.append(f"АНАЛИЗ ДИАЛОГА ({len(message_results)} реплик)")
        else:
            self.quick_result_text.append("АНАЛИЗ ОТЗЫВА")
        self.quick_result_text.append("=" * 60)
        self.quick_result_text.append("")
        
        # Результаты по каждой реплике
        for msg_result in message_results:
            idx = msg_result['message_index']
            text = msg_result['message_text']
            
            # Обрезаем длинный текст
            if len(text) > 120:
                text = text[:120] + '...'
            
            if is_dialog_mode:
                self.quick_result_text.append(f"📌 Реплика [{idx+1}]: {text}")
            else:
                self.quick_result_text.append(f"📝 Текст: {text}")
            
            self.quick_result_text.append("")
            self.quick_result_text.append("  Результаты моделей:")
            
            # Список моделей
            models_info = [
                ('VADER', 'vader_sentiment', 'vader_score'),
                ('Flair', 'flair_sentiment', 'flair_score'),
                ('RuBERT', 'rubert_sentiment', 'rubert_score'),
                ('RoBERTa', 'roberta_sentiment', 'roberta_score'),
                ('DistilBERT', 'distilbert_sentiment', 'distilbert_score'),
                ('Logistic Regression', 'logistic_regression_sentiment', 'logistic_regression_score'),
                ('SVM', 'svm_sentiment', 'svm_score'),
                ('Random Forest', 'random_forest_sentiment', 'random_forest_score'),
            ]
            
            for name, sent_key, score_key in models_info:
                if sent_key in msg_result:
                    sent = msg_result[sent_key]
                    score = msg_result.get(score_key, 3)
                    self.quick_result_text.append(
                        f"    {emoji(sent)} {name:20} : {sent:8} (score: {score})"
                    )
            
            # Ensemble
            ens_sent = msg_result.get('ensemble_sentiment', 'Neutral')
            ens_score = msg_result.get('ensemble_score', 3)
            ens_avg = msg_result.get('ensemble_score_avg', 3.0)
            
            self.quick_result_text.append("")
            self.quick_result_text.append(
                f"  🎯 ENSEMBLE: {emoji(ens_sent)} {ens_sent} "
                f"(score: {ens_score}, avg: {ens_avg:.2f})"
            )
            self.quick_result_text.append("")
            self.quick_result_text.append("-" * 60)
            self.quick_result_text.append("")
        
        # Итоговая агрегация для диалога
        if is_dialog_mode and len(message_results) > 1:
            scores = [m['ensemble_score'] for m in message_results]
            sentiments = [m['ensemble_sentiment'] for m in message_results]
            
            avg_score = float(np.mean(scores))
            rounded_score = round(avg_score)
            
            if rounded_score >= 4:
                overall_sent = 'Positive'
            elif rounded_score <= 2:
                overall_sent = 'Negative'
            else:
                overall_sent = 'Neutral'
            
            # Тренд
            if len(scores) >= 2:
                mid = len(scores) // 2
                first_half = np.mean(scores[:mid])
                second_half = np.mean(scores[mid:])
                
                if second_half - first_half > 0.5:
                    trend = 'улучшается ↗'
                elif first_half - second_half > 0.5:
                    trend = 'ухудшается ↘'
                else:
                    trend = 'стабильный →'
            else:
                trend = 'стабильный →'
            
            self.quick_result_text.append("=" * 60)
            self.quick_result_text.append("ИТОГОВАЯ ОЦЕНКА ДИАЛОГА")
            self.quick_result_text.append("=" * 60)
            self.quick_result_text.append("")
            self.quick_result_text.append(
                f"  Общая тональность:  {emoji(overall_sent)} {overall_sent}"
            )
            self.quick_result_text.append(
                f"  Средний score:      {avg_score:.2f} (округлён: {rounded_score})"
            )
            self.quick_result_text.append(f"  Тренд:              {trend}")
            self.quick_result_text.append("")
            self.quick_result_text.append(
                f"  Реплики:  "
                f"🟢 {sentiments.count('Positive')} позитивных, "
                f"🔴 {sentiments.count('Negative')} негативных, "
                f"🟡 {sentiments.count('Neutral')} нейтральных"
            )
            self.quick_result_text.append("")
        
        # Подсказка про полный отчёт
        self.quick_result_text.append("=" * 60)
        self.quick_result_text.append("💡 Для полного отчёта с графиками:")
        self.quick_result_text.append("   1. Сохраните текст в CSV файл")
        self.quick_result_text.append("   2. Используйте вкладку 'Анализ файла'")
        self.quick_result_text.append("   3. Или 'Анализ диалогов' для JSONL/JSON")
        self.quick_result_text.append("=" * 60)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = SentimentAnalysisGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()