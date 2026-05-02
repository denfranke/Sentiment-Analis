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

import warnings
warnings.filterwarnings('ignore')

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')


class AnalysisProcess(QProcess):
    """Процесс для запуска анализа"""
    
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
        
        # Создаём конфиг для run_analysis.py
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
        script_path = os.path.join(os.path.dirname(__file__), 'run_analysis.py')
        if not os.path.exists(script_path):
            # Если нет run_analysis.py, создаём его
            self.create_run_script(script_path)
        
        program = sys.executable
        self.start(program, [script_path, config_file])
    
    def create_run_script(self, script_path):
        """Создаёт run_analysis.py если его нет"""
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
        'models'
    ))
    
    print(f"\\n✅ Анализ завершён!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_analysis(sys.argv[1])
    else:
        print("Использование: python run_analysis.py <config_file.json>")
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


class PDFReportGenerator(QThread):
    """Генерация PDF отчёта из результатов"""
    
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
            
            self.progress.emit("Создание графиков...")
            
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
    
    def create_html_report(self, df, html_path):
        """Создаёт HTML отчёт с графиками"""
        
        # Создаём графики
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        if 'ensemble_sentiment' in df.columns:
            sentiments = df['ensemble_sentiment'].value_counts()
            colors = {'Positive': '#4ECDC4', 'Neutral': '#FFE194', 'Negative': '#FF6B6B'}
            bars = ax1.bar(sentiments.index, sentiments.values, 
                          color=[colors.get(s, '#888') for s in sentiments.index])
            ax1.set_title('Распределение тональности (Ensemble)')
            ax1.set_ylabel('Количество')
            for bar in bars:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(int(bar.get_height())), ha='center', va='bottom')
            plt.savefig(os.path.join(self.output_dir, 'plot1.png'), dpi=150, bbox_inches='tight')
        plt.close(fig1)
        
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        models = ['vader', 'flair', 'rubert', 'roberta', 'distilbert', 'ensemble']
        model_names = ['VADER', 'Flair', 'RuBERT', 'RoBERTa', 'DistilBERT', 'Ensemble']
        accuracies = []
        for model in models:
            col = f'{model}_sentiment'
            if col in df.columns and 'actual_sentiment' in df.columns:
                valid = df[df['actual_sentiment'].notna()]
                if len(valid) > 0:
                    acc = (valid[col] == valid['actual_sentiment']).mean() * 100
                    accuracies.append(acc)
                else:
                    accuracies.append(0)
            else:
                accuracies.append(0)
        
        bars = ax2.bar(model_names, accuracies, color='#4ECDC4', edgecolor='black')
        ax2.set_ylabel('Точность (%)')
        ax2.set_title('Сравнение точности моделей')
        ax2.set_ylim(0, 100)
        for bar, acc in zip(bars, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        plt.savefig(os.path.join(self.output_dir, 'plot2.png'), dpi=150, bbox_inches='tight')
        plt.close(fig2)
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        if 'ensemble_score' in df.columns:
            scores = df['ensemble_score'].value_counts().sort_index()
            bars = ax3.bar(scores.index.astype(str), scores.values, color='#3498DB', edgecolor='black')
            ax3.set_xlabel('Оценка')
            ax3.set_ylabel('Количество')
            ax3.set_title('Распределение оценок Ensemble')
            for bar in bars:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(int(bar.get_height())), ha='center', va='bottom')
            plt.savefig(os.path.join(self.output_dir, 'plot3.png'), dpi=150, bbox_inches='tight')
        plt.close(fig3)
        
        # HTML шаблон
        html_content = f'''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Отчёт анализа тональности</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; padding: 30px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #4CAF50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #333; margin-top: 30px; }}
        .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .stat-number {{ font-size: 36px; font-weight: bold; }}
        .plot {{ margin: 30px 0; text-align: center; }}
        .plot img {{ max-width: 100%; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        .positive {{ color: #4CAF50; font-weight: bold; }}
        .negative {{ color: #f44336; font-weight: bold; }}
        .neutral {{ color: #FF9800; font-weight: bold; }}
        .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Отчёт анализа тональности</h1>
        <p>Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{len(df)}</div>
                <div>Всего текстов</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{df['ensemble_sentiment'].value_counts().get('Positive', 0)}</div>
                <div>Позитивных</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{df['ensemble_sentiment'].value_counts().get('Negative', 0)}</div>
                <div>Негативных</div>
            </div>
        </div>
        
        <div class="plot">
            <h2>📈 Распределение тональности</h2>
            <img src="plot1.png" alt="Распределение тональности">
        </div>
        
        <div class="plot">
            <h2>🤖 Сравнение моделей</h2>
            <img src="plot2.png" alt="Сравнение моделей">
        </div>
        
        <div class="plot">
            <h2>⭐ Распределение оценок</h2>
            <img src="plot3.png" alt="Распределение оценок">
        </div>
        
        <h2>📋 Примеры результатов</h2>
        <table>
            <thead>
                <tr><th>Текст</th><th>VADER</th><th>Flair</th><th>RuBERT</th><th>RoBERTa</th><th>DistilBERT</th><th>Ensemble</th></tr>
            </thead>
            <tbody>
                {self.generate_table_rows(df.head(20))}
            </tbody>
        </table>
        
        <div class="footer">
            <p>Система анализа тональности | 8 моделей: VADER, Flair, RuBERT, RoBERTa, DistilBERT, Logistic Regression, SVM, Random Forest</p>
        </div>
    </div>
</body>
</html>'''
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def generate_table_rows(self, df):
        """Генерирует строки таблицы"""
        rows = []
        for _, row in df.iterrows():
            text = row.get('text', '')[:100] + '...' if len(str(row.get('text', ''))) > 100 else row.get('text', '')
            vader = row.get('vader_sentiment', 'Neutral')
            flair = row.get('flair_sentiment', 'Neutral')
            rubert = row.get('rubert_sentiment', 'Neutral')
            roberta = row.get('roberta_sentiment', 'Neutral')
            distil = row.get('distilbert_sentiment', 'Neutral')
            ensemble = row.get('ensemble_sentiment', 'Neutral')
            
            def get_class(sent):
                return 'positive' if sent == 'Positive' else 'negative' if sent == 'Negative' else 'neutral'
            
            rows.append(f'''<tr>
                <td>{text}</td>
                <td class="{get_class(vader)}">{vader}</td>
                <td class="{get_class(flair)}">{flair}</td>
                <td class="{get_class(rubert)}">{rubert}</td>
                <td class="{get_class(roberta)}">{roberta}</td>
                <td class="{get_class(distil)}">{distil}</td>
                <td class="{get_class(ensemble)}">{ensemble}</td>
            </tr>''')
        return '\n'.join(rows)
    
    def convert_to_pdf(self, html_path, pdf_path):
        """Конвертирует HTML в PDF"""
        try:
            from weasyprint import HTML
            HTML(html_path).write_pdf(pdf_path)
        except ImportError:
            # Если weasyprint не установлен, пробуем через wkhtmltopdf
            try:
                import pdfkit
                pdfkit.from_file(html_path, pdf_path)
            except:
                # Если ничего не работает, просто копируем HTML
                pass


class SentimentAnalysisGUI(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.analysis_process = None
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
        self.text_column.addItems(["text", "review", "comment", "content"])
        columns_layout.addWidget(self.text_column, 0, 1)
        
        columns_layout.addWidget(QLabel("Столбец с оценкой:"), 1, 0)
        self.rating_column = QComboBox()
        self.rating_column.setEditable(True)
        self.rating_column.addItems(["rating", "label", "score", "stars"])
        columns_layout.addWidget(self.rating_column, 1, 1)
        
        self.use_rating = QCheckBox("Использовать оценки для расчёта точности")
        self.use_rating.setChecked(True)
        columns_layout.addWidget(self.use_rating, 2, 0, 1, 2)
        
        columns_group.setLayout(columns_layout)
        layout.addWidget(columns_group)
        
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
            "• Результаты сохранятся в папку answer/с именем файла с временной меткой\n"
            "• Будет сгенерирован HTML и PDF отчёт с графиками"
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
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
    
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите CSV файл", "", "CSV файлы (*.csv);;Все файлы (*.*)"
        )
        if file_path:
            self.file_path.setText(file_path)
            # Читаем заголовки
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
        self.log_text.append(f"🚀 Запуск анализа файла: {file_path}")
        self.log_text.append(f"📁 Результаты будут сохранены в: {output_dir}")
        self.log_text.append(f"📝 Столбец с текстом: {text_col}")
        if use_rating:
            self.log_text.append(f"⭐ Столбец с оценкой: {rating_col}")
        self.log_text.append(f"🧠 Используются 8 моделей анализа")
        self.log_text.append("-" * 50)
        
        # Запускаем процесс
        self.analysis_process = AnalysisProcess()
        self.analysis_process.output_received.connect(self.on_analysis_output)
        self.analysis_process.finished_analysis.connect(self.on_analysis_finished)
        self.analysis_process.start_analysis(file_path, text_col, rating_col, use_rating, output_dir)
    
    def on_analysis_output(self, text):
        self.log_text.append(text)
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        QApplication.processEvents()
    
    def on_analysis_finished(self, output_dir, success):
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("🚀 Запустить анализ")
        self.progress_bar.setVisible(False)
        
        if success:
            self.log_text.append("\n✅ Анализ успешно завершён!")
            self.current_output_dir = output_dir
            
            # Генерация PDF
            self.log_text.append("\n📄 Генерация PDF отчёта...")
            self.pdf_generator = PDFReportGenerator(output_dir)
            self.pdf_generator.progress.connect(lambda msg: self.log_text.append(f"   {msg}"))
            self.pdf_generator.finished.connect(self.on_pdf_generated)
            self.pdf_generator.error.connect(lambda e: self.log_text.append(f"❌ {e}"))
            self.pdf_generator.start()
            
            # Активируем кнопки
            self.open_pdf_btn.setEnabled(True)
            self.open_folder_btn.setEnabled(True)
            
            # Переключаемся на вкладку результатов
            self.tabs.setCurrentIndex(2)
        else:
            self.log_text.append("\n❌ Ошибка при выполнении анализа")
            QMessageBox.critical(self, "Ошибка", "Не удалось выполнить анализ")
    
    def on_pdf_generated(self, pdf_path):
        self.log_text.append(f"✅ PDF отчёт создан: {pdf_path}")
        self.current_pdf_path = pdf_path
        
        # Загружаем результаты в таблицу
        result_file = os.path.join(self.current_output_dir, 'отзывы(результат работы).csv')
        if os.path.exists(result_file):
            df = pd.read_csv(result_file, encoding='utf-8-sig')
            self.display_results_table(df)
        
        QMessageBox.information(self, "Готово", f"Анализ завершён!\n\nPDF отчёт: {pdf_path}")
    
    def display_results_table(self, df):
        """Отображение результатов в таблице"""
        columns = ['Текст', 'VADER', 'Flair', 'RuBERT', 'RoBERTa', 'DistilBERT', 'Ensemble']
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
        
        for row, (_, row_data) in enumerate(display_df.iterrows()):
            text = str(row_data.get('text', ''))[:80] + '...' if len(str(row_data.get('text', ''))) > 80 else str(row_data.get('text', ''))
            self.results_table.setItem(row, 0, QTableWidgetItem(text))
            
            vader = row_data.get('vader_sentiment', 'Neutral')
            flair = row_data.get('flair_sentiment', 'Neutral')
            rubert = row_data.get('rubert_sentiment', 'Neutral')
            roberta = row_data.get('roberta_sentiment', 'Neutral')
            distil = row_data.get('distilbert_sentiment', 'Neutral')
            ensemble = row_data.get('ensemble_sentiment', 'Neutral')
            
            for col, val in enumerate([vader, flair, rubert, roberta, distil, ensemble], 1):
                item = QTableWidgetItem(val)
                if val in colors:
                    item.setBackground(colors[val])
                self.results_table.setItem(row, col, item)
            
            if 'actual_sentiment' in columns:
                actual = row_data.get('actual_sentiment', '')
                if actual:
                    item = QTableWidgetItem(str(actual))
                    if actual in colors:
                        item.setBackground(colors[actual])
                    self.results_table.setItem(row, len(columns)-1, item)
        
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(columns)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
    
    def start_quick_analysis(self):
        """Быстрый анализ введённого текста"""
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Ошибка", "Введите текст для анализа")
            return
        
        # Переключаемся на вкладку результатов и показываем сообщение
        self.tabs.setCurrentIndex(2)
        self.log_text.clear()
        self.log_text.append("🔄 Для быстрого анализа используется отдельный поток...")
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
            "Тогда будут использованы все 8 моделей и создан PDF отчёт."
        )
    
    def open_pdf(self):
        if hasattr(self, 'current_pdf_path') and os.path.exists(self.current_pdf_path):
            webbrowser.open(self.current_pdf_path)
        elif self.current_output_dir:
            pdf_path = os.path.join(self.current_output_dir, 'sentiment_report.pdf')
            if os.path.exists(pdf_path):
                webbrowser.open(pdf_path)
            else:
                html_path = os.path.join(self.current_output_dir, 'sentiment_report.html')
                if os.path.exists(html_path):
                    webbrowser.open(html_path)
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