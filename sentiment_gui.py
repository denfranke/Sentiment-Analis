#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Графический интерфейс для анализа тональности текстов и диалогов
Поддерживает:
- Загрузку CSV файлов с настройкой столбцов
- Ручной ввод текста/диалогов
- Запуск полного анализа с генерацией PDF отчёта
"""

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
import io

# Настройка кодировки для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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
    """Генерация PDF отчёта ТОЛЬКО с графиками из VisualAnalisTonResults"""
    
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
            'class_dist': '📈 Распределение Ensemble по классам тональности',
            'dist_analysis': '📊 Анализ распределения оценок моделей',
            'dashboard': '📋 Дашборд анализа тональности',
            'additional': '📊 Дополнительный анализ (точность и ошибки)',
        }
        
        # Формируем список доступных графиков
        available_graphs = []
        for key in graph_order:
            if key in local_graphs:
                available_graphs.append((key, local_graphs[key]))
        
        if not available_graphs:
            self.progress.emit("⚠️ Графики не найдены. Создаю отчёт без них.")
        
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
        
        # Генерируем секции графиков
        sections_html = []
        toc_items = []
        
        for idx, (key, path) in enumerate(available_graphs, 1):
            filename = os.path.basename(path)
            title = graph_titles.get(key, f'График {idx}')
            
            toc_items.append(f'<li><a href="#section{idx}">{title.replace("📈 ", "").replace("📊 ", "").replace("📋 ", "").replace("🎯 ", "")}</a></li>')
            
            sections_html.append(f'''
            <div id="section{idx}" class="plot">
                <h2>{title}</h2>
                <img src="{filename}" alt="{title}" onerror="this.parentElement.innerHTML='<p style=\\'color:red;\\'>График не загрузился: {filename}</p>'">
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
        <h1>📊 Отчёт анализа тональности</h1>
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
            <h3>📑 Содержание</h3>
            <p><strong>Создано графиков: {len(available_graphs)}</strong></p>
            <ul>
                {''.join(toc_items) if toc_items else '<li>Графики не найдены</li>'}
                <li><a href="#section{table_section_idx}">📋 Примеры результатов анализа</a></li>
            </ul>
        </div>
        
        {''.join(sections_html) if sections_html else '<div class="no-graphs"><p>⚠️ Графики не были созданы. Проверьте, что VisualAnalisTonResults.py отработал успешно.</p></div>'}
        
        <div id="section{table_section_idx}">
            <h2>📋 Примеры результатов анализа</h2>
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
        self.tabs.addTab(self.file_tab, "📁 Анализ файла")
        self.setup_file_tab()
        
        # Вкладка 2: Ручной ввод
        self.text_tab = QWidget()
        self.tabs.addTab(self.text_tab, "✏️ Ручной ввод")
        self.setup_text_tab()
        
        # Вкладка 3: Результаты
        self.results_tab = QWidget()
        self.tabs.addTab(self.results_tab, "📊 Результаты")
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
        
        info_rows = QLabel("💡 Укажите число для анализа только первых N строк. Пустое поле = все строки.")
        info_rows.setStyleSheet("color: #666; font-size: 11px;")
        rows_layout.addWidget(info_rows)
        
        rows_group.setLayout(rows_layout)
        layout.addWidget(rows_group)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("🚀 Запустить анализ")
        self.analyze_btn.clicked.connect(self.start_file_analysis)
        self.analyze_btn.setMinimumHeight(50)
        self.analyze_btn.setStyleSheet("font-size: 14px; font-weight: bold;")
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.analyze_btn)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Информация
        info_label = QLabel(
            "📌 Информация:\n"
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
        
        # Область ввода
        input_group = QGroupBox("Введите текст или диалог")
        input_layout = QVBoxLayout()
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText(
            "Введите текст для анализа...\n\n"
            "Пример отзыва:\n"
            "Отличный сервис, очень доволен! Всё быстро и качественно.\n\n"
            "Пример диалога (разделяйте реплики через ----):\n"
            "Продавец: Здравствуйте, чем могу помочь?\n"
            "----\n"
            "Покупатель: Здравствуйте, хотел узнать о доставке.\n"
            "----\n"
            "Продавец: Доставка бесплатная от 1000 рублей."
        )
        self.text_input.setMinimumHeight(200)
        input_layout.addWidget(self.text_input)
        
        self.dialog_mode = QCheckBox("Режим диалога (разделять реплики по '----')")
        input_layout.addWidget(self.dialog_mode)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        self.clear_btn = QPushButton("🗑 Очистить")
        self.clear_btn.clicked.connect(lambda: self.text_input.clear())
        self.quick_analyze_btn = QPushButton("🔍 Быстрый анализ")
        self.quick_analyze_btn.clicked.connect(self.start_quick_analysis)
        self.quick_analyze_btn.setMinimumHeight(50)
        self.quick_analyze_btn.setStyleSheet("font-size: 14px; font-weight: bold; background: #FF9800;")
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.clear_btn)
        buttons_layout.addWidget(self.quick_analyze_btn)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Информация
        info_label = QLabel(
            "📌 Для диалогов анализ выполняется для каждой реплики отдельно,\n"
            "после чего показывается общая динамика и итоговая тональность."
        )
        info_label.setStyleSheet("background: #fff3e0; padding: 10px; border-radius: 5px;")
        layout.addWidget(info_label)
        
        layout.addStretch()
    
    def setup_results_tab(self):
        layout = QVBoxLayout(self.results_tab)
        
        # Панель управления
        control_layout = QHBoxLayout()
        self.open_pdf_btn = QPushButton("📄 Открыть PDF отчёт")
        self.open_pdf_btn.clicked.connect(self.open_pdf)
        self.open_pdf_btn.setEnabled(False)
        
        self.open_folder_btn = QPushButton("📁 Открыть папку")
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
        self.analyze_btn.setText("⏳ Анализ выполняется...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        self.log_text.clear()
        self.log_text.append("=" * 60)
        self.log_text.append("🚀 ЗАПУСК ПОЛНОГО ЦИКЛА АНАЛИЗА")
        self.log_text.append("=" * 60)
        self.log_text.append(f"📂 Файл: {file_path}")
        self.log_text.append(f"📁 Выходная папка: {output_dir}")
        self.log_text.append(f"📝 Столбец текста: {text_col}")
        if use_rating:
            self.log_text.append(f"⭐ Столбец оценки: {rating_col}")
        if max_rows:
            self.log_text.append(f"🔢 Анализ первых {max_rows} строк")
        else:
            self.log_text.append(f"🔢 Анализ всех строк")
        self.log_text.append("")
        self.log_text.append("📊 ЭТАП 1: Запуск основного анализа (8 моделей)...")
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
            self.analyze_btn.setText("🚀 Запустить анализ")
            self.progress_bar.setVisible(False)
            self.log_text.append("\n❌ Ошибка на этапе основного анализа!")
            QMessageBox.critical(self, "Ошибка", "Не удалось выполнить основной анализ")
            return
        
        self.log_text.append("")
        self.log_text.append("✅ Основной анализ завершён успешно!")
        self.log_text.append("")
        self.log_text.append("📊 ЭТАП 2: Запуск визуализации (VisualAnalisTonResults)...")
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
            self.log_text.append("✅ Визуализация завершена успешно!")
        else:
            self.log_text.append("")
            self.log_text.append("⚠️ Визуализация завершена с ошибками (графики могут быть не созданы)")
            self.log_text.append("   Проверьте, что VisualAnalisTonResults.py существует и работает корректно.")
        
        self.log_text.append("")
        self.log_text.append("📄 ЭТАП 3: Генерация PDF отчёта с графиками...")
        self.log_text.append("-" * 50)
        
        # Запускаем генерацию PDF
        self.pdf_generator = PDFReportGenerator(output_dir)
        self.pdf_generator.progress.connect(lambda msg: self.log_text.append(f"   {msg}"))
        self.pdf_generator.finished.connect(self.on_pdf_generated)
        self.pdf_generator.error.connect(lambda e: self.log_text.append(f"   ❌ {e}"))
        self.pdf_generator.start()
    
    def on_pdf_generated(self, pdf_path):
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("🚀 Запустить анализ")
        self.progress_bar.setVisible(False)
        
        self.log_text.append(f"✅ PDF отчёт создан: {pdf_path}")
        self.log_text.append("")
        self.log_text.append("=" * 60)
        self.log_text.append("🎉 ПОЛНЫЙ ЦИКЛ АНАЛИЗА ЗАВЕРШЁН!")
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
        self.tabs.setCurrentIndex(2)
        
        QMessageBox.information(
            self, 
            "Анализ завершён!", 
            f"Все этапы выполнены успешно!\n\n"
            f"📁 Папка с результатами: {self.current_output_dir}\n"
            f"📄 PDF отчёт: {pdf_path}\n\n"
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
        """Быстрый анализ введённого текста"""
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Ошибка", "Введите текст для анализа")
            return
        
        self.tabs.setCurrentIndex(2)
        self.log_text.clear()
        self.log_text.append("🔄 Быстрый анализ не поддерживает полный цикл.")
        self.log_text.append("📝 Для полного анализа с PDF отчётом используйте вкладку 'Анализ файла'")
        self.log_text.append("\n💡 Совет: Сохраните текст в CSV файл и загрузите его для получения полного отчёта с графиками.")
        
        QMessageBox.information(
            self, 
            "Быстрый анализ",
            "Для получения полного отчёта с графиками и PDF:\n\n"
            "1. Сохраните текст в CSV файл\n"
            "2. Перейдите на вкладку 'Анализ файла'\n"
            "3. Выберите файл и настройте столбцы\n"
            "4. Нажмите 'Запустить анализ'\n\n"
            "Процесс включает 3 этапа:\n"
            "• Основной анализ (8 моделей)\n"
            "• Визуализация (VisualAnalisTonResults)\n"
            "• Генерация PDF отчёта"
        )
    
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
                    self.log_text.append("📄 PDF не найден, открыт HTML отчёт")
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


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = SentimentAnalysisGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()