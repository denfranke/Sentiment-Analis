import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля для графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'

def load_and_prepare_data(file_path):
    print("="*60)
    print("ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    print("="*60)
    
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # Фильтруем строки с валидными рейтингами
    df_clean = df.dropna(subset=['actual_sentiment', 'rating']).copy()
    
    # Преобразуем рейтинг в числовой формат
    df_clean['rating_numeric'] = pd.to_numeric(df_clean['rating'], errors='coerce')
    
    # Создаем числовые представления для предсказаний моделей
    sentiment_to_score = {
        'Positive': 5,
        'Neutral': 3,
        'Negative': 1
    }
    
    # Добавляем числовые колонки для каждой модели (если их нет)
    model_columns = {
        'vader_sentiment': 'vader_score',
        'flair_sentiment': 'flair_score',
        'rubert_sentiment': 'rubert_score',
        'roberta_sentiment': 'roberta_score',
        'distilbert_sentiment': 'distilbert_score'
    }
    
    for sent_col, score_col in model_columns.items():
        if sent_col in df_clean.columns and score_col not in df_clean.columns:
            df_clean[score_col] = df_clean[sent_col].map(sentiment_to_score)
    
    # Для scikit-learn моделей
    sklearn_sentiment_cols = [col for col in df_clean.columns if col.endswith('_sentiment') 
                              and col not in model_columns.keys()
                              and col != 'ensemble_sentiment'
                              and col != 'actual_sentiment']
    
    for sent_col in sklearn_sentiment_cols:
        score_col = sent_col.replace('_sentiment', '_score')
        if score_col not in df_clean.columns:
            df_clean[score_col] = df_clean[sent_col].map(sentiment_to_score)
    
    # ВАЖНО: Используем ensemble_score_raw из CSV, если есть
    if 'ensemble_score_raw' in df_clean.columns:
        # Используем RAW значение для точных расчётов
        df_clean['ensemble_score_for_analysis'] = df_clean['ensemble_score_raw']
        print("✓ Используется ensemble_score_raw из CSV для анализа")
    elif 'ensemble_score' in df_clean.columns:
        # Если нет raw, используем существующий ensemble_score
        df_clean['ensemble_score_for_analysis'] = df_clean['ensemble_score']
        print("✓ Используется ensemble_score из CSV для анализа")
    else:
        # Создаём ensemble_score из доступных моделей
        score_columns = ['vader_score', 'flair_score', 'rubert_score', 
                         'roberta_score', 'distilbert_score']
        sklearn_score_cols = [col for col in df_clean.columns if col.endswith('_score') 
                              and col not in score_columns]
        all_score_cols = score_columns + sklearn_score_cols
        available_scores = [col for col in all_score_cols if col in df_clean.columns]
        
        if available_scores:
            df_clean['ensemble_score_for_analysis'] = df_clean[available_scores].mean(axis=1)
            print(f"✓ Создан ensemble_score_for_analysis на основе {len(available_scores)} моделей")
    
    # Выводим информацию о загруженных данных
    print(f"Загружено {len(df_clean)} записей с валидными рейтингами")
    print(f"Доступные score колонки: {[col for col in df_clean.columns if col.endswith('_score')]}")
    
    return df_clean

def plot_ensemble_class_distribution(df):
    """
    Строит график распределения ensemble оценок по трём классам тональности
    """
    print("\n" + "="*60)
    print("ПОСТРОЕНИЕ ГРАФИКА РАСПРЕДЕЛЕНИЯ ENSEMBLE ПО КЛАССАМ")
    print("="*60)
    
    # Создаём копию данных
    plot_df = df.copy()
    
    # Используем ensemble_score (округлённый до 1,3,5) для классификации
    if 'ensemble_score' in plot_df.columns:
        ensemble_scores = plot_df['ensemble_score']
    elif 'ensemble_score_for_analysis' in plot_df.columns:
        # Округляем raw значения до 1,3,5
        ensemble_scores = plot_df['ensemble_score_for_analysis'].round().clip(1, 5)
    else:
        print("❌ Нет данных для ensemble")
        return None, None
    
    # Определяем класс тональности на основе округлённой оценки
    def classify_sentiment(score):
        if score >= 4:
            return "Positive"
        elif score <= 2:
            return "Negative"
        else:
            return "Neutral"
    
    plot_df['ensemble_class'] = ensemble_scores.apply(classify_sentiment)
    
    # Подсчитываем количество в каждом классе
    class_counts = plot_df['ensemble_class'].value_counts()
    class_percentages = class_counts / len(plot_df) * 100
    
    # Для проверки: выводим распределение округлённых оценок
    print("\nРаспределение округлённых оценок ensemble_score:")
    rounded_counts = ensemble_scores.value_counts().sort_index()
    for score, count in rounded_counts.items():
        print(f"  Оценка {score}: {count} ({count/len(plot_df)*100:.1f}%)")
    
    # Создаём график
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Цвета для классов
    colors = {'Negative': '#FF6B6B', 'Neutral': '#FFE194', 'Positive': '#4ECDC4'}
    
    # Строим столбцы в правильном порядке
    class_order = ['Negative', 'Neutral', 'Positive']
    bars_list = []
    for cls in class_order:
        if cls in class_counts:
            bar = ax.bar(cls, class_counts[cls], color=colors[cls], 
                        edgecolor='black', linewidth=2, alpha=0.8)
            bars_list.append(bar)
    
    # Добавляем значения на столбцы
    for bar in bars_list:
        height = bar[0].get_height()
        ax.text(bar[0].get_x() + bar[0].get_width()/2., height + 5,
                f'{int(height)}\n({height/len(plot_df)*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Настройка графика
    ax.set_xlabel('Класс тональности', fontsize=14, fontweight='bold')
    ax.set_ylabel('Количество отзывов', fontsize=14, fontweight='bold')
    ax.set_title('Распределение оценок Ensemble по классам тональности\n(на основе округлённого ensemble_score)',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Добавляем подпись с информацией
    mean_score = ensemble_scores.mean()
    median_score = ensemble_scores.median()
    ax.text(0.5, -0.12, f'Среднее (округл.): {mean_score:.3f} | Медиана (округл.): {median_score:.3f}',
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('answer/ensemble_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    print("\n✓ График распределения Ensemble по классам сохранён как 'answer/ensemble_class_distribution.png'")
    
    # Вывод статистики
    print("\nСтатистика распределения Ensemble:")
    print("-" * 40)
    for cls in class_order:
        if cls in class_counts:
            print(f"{cls:10}: {class_counts[cls]:4} ({class_percentages[cls]:.1f}%)")
    
    return class_counts, class_percentages

def plot_normality_comparison(df):
    print("\n" + "="*60)
    print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ ОЦЕНОК")
    print("="*60)
    
    # Определяем порядок колонок
    model_order = [
        'vader_score',
        'flair_score', 
        'rubert_score',
        'roberta_score',
        'distilbert_score',
        'logistic_regression_score',
        'random_forest_score',
        'svm_score',
        'ensemble_score_for_analysis',
        'rating_numeric'
    ]
    
    columns_to_test = [col for col in model_order if col in df.columns]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE194', 
              '#A8E6CF', '#D4A5A5', '#9B59B6', '#3498DB', '#F1C40F']
    
    results = {}
    
    print("\n" + "-" * 90)
    print(f"{'Модель':<20} | {'Уникальных значений':<18} | {'Распределение':<35} | {'Тест KS p-value':<15}")
    print("-" * 90)
    
    for idx, col in enumerate(columns_to_test):
        if col not in df.columns:
            continue
            
        data = df[col].dropna()
        unique_values = data.nunique()
        
        from scipy import stats
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        if unique_values <= 10:
            value_counts = data.value_counts().sort_index()
            theoretical = np.random.uniform(data.min(), data.max(), 10000)
            ks_stat, ks_p = stats.ks_2samp(data, theoretical)
            distr_items = [f"{v}: {c}" for v, c in value_counts.items()]
            distr_str = ", ".join(distr_items[:5])
            if len(distr_items) > 5:
                distr_str += "..."
        else:
            ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            distr_str = f"среднее={data.mean():.2f}, σ={data.std():.2f}"
        
        if col == 'rating_numeric':
            model_name = 'Пользователи'
        elif col == 'ensemble_score_for_analysis':
            model_name = 'СУММАТОР'
        else:
            model_name = col.replace('_score', '').replace('_', ' ').title()
        
        normality_note = f"p={ks_p:.4f}" + (" (≈ нормальное)" if ks_p > 0.05 else " (≠ нормальное)")
        
        print(f"{model_name:<20} | {unique_values:<18} | {distr_str:<35} | {normality_note:<15}")
        
        results[col] = {
            'ks_p': ks_p,
            'is_normal': ks_p > 0.05,
            'unique_values': unique_values,
            'mean': data.mean(),
            'std': data.std(),
            'skewness': skewness,
            'kurtosis': kurtosis,
            'shapiro_p': ks_p
        }
    
    # Визуализация
    n_cols = len(columns_to_test)
    n_rows = (n_cols + 4) // 5
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(columns_to_test):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        data = df[col].dropna()
        
        if col == 'rating_numeric':
            model_name = 'Пользователи'
        elif col == 'ensemble_score_for_analysis':
            model_name = 'СУММАТОР'
        else:
            model_name = col.replace('_score', '').replace('_', ' ').title()
        
        unique_values = data.nunique()
        
        if unique_values > 10:
            n, bins, patches = ax.hist(data, bins=20, color='#3498DB', 
                                       edgecolor='black', alpha=0.7)
            ax.set_xlabel('Оценка')
        else:
            value_counts = data.value_counts().sort_index()
            bars = ax.bar(value_counts.index.astype(str), value_counts.values, 
                        color=colors[idx % len(colors)], edgecolor='black', alpha=0.7)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=9)
            ax.set_xlabel('Оценка')
        
        if col == 'ensemble_score_for_analysis':
            # Округляем значения до ближайшего целого 1,2,3,4,5
            rounded_data = data.round().clip(1, 5)
            value_counts = rounded_data.value_counts().sort_index()
            
            # Убеждаемся, что есть все столбцы от 1 до 5
            for i in range(1, 6):
                if i not in value_counts.index:
                    value_counts[i] = 0
            value_counts = value_counts.sort_index()
            
            # ОЧИЩАЕМ ОСЬ ПЕРЕД ПОСТРОЕНИЕМ (убираем старую диаграмму)
            ax.clear()
            
            # Строим столбчатую диаграмму
            bars = ax.bar(value_counts.index.astype(str), value_counts.values, 
                        color='#3498DB', edgecolor='black', linewidth=1.5, alpha=0.8)
            
            # Числовые значения над столбцами
            max_height = max(value_counts.values) if len(value_counts) > 0 else 0
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max_height * 0.02,
                    f'{int(height)}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10, rotation=360)
            
            ax.set_xlabel('Оценка')
            ax.set_ylabel('Частота')
            ax.set_title('СУММАТОР', fontsize=11, fontweight='bold')
            ax.set_xticks(range(5))
            ax.set_xticklabels(['1', '2', '3', '4', '5'])
            
            # Вертикальные линии порогов
            ax.axvline(x=1.5, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.axvline(x=2.5, color='green', linestyle='--', linewidth=2, alpha=0.8)
            
            y_min, y_max = ax.get_ylim()
            
            # Закрашенные зоны
            ax.fill_betweenx([y_min, y_max], -0.5, 1.5, alpha=0.1, color='red')
            ax.fill_betweenx([y_min, y_max], 1.5, 2.5, alpha=0.1, color='gray')
            ax.fill_betweenx([y_min, y_max], 2.5, 4.5, alpha=0.1, color='green')
            
            ax.grid(True, alpha=0.3)
            
            # Пропускаем дальнейшую обработку для этой колонки
            continue
        
        ax.set_ylabel('Частота')
        ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(columns_to_test), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('answer/distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("\n✓ Анализ распределений сохранен как 'answer/distribution_analysis.png'")
    
    return results

def calculate_mathematical_expectation(df):
    """
    Расчет математического ожидания для каждой модели
    """
    print("\n" + "="*60)
    print("МАТЕМАТИЧЕСКОЕ ОЖИДАНИЕ ОЦЕНОК МОДЕЛЕЙ")
    print("="*60)
    
    # Колонки с оценками моделей
    score_columns = ['vader_score', 'flair_score', 'rubert_score', 
                       'roberta_score', 'distilbert_score','logistic_regression_score','svm_score','random_forest_score', 'ensemble_score']
    
    results = {}
    
    print("\nСредние значения оценок (математическое ожидание):")
    print("-" * 70)
    
    for col in score_columns:
        if col in df.columns:
            mean_value = df[col].mean()
            std_value = df[col].std()
            ci_lower, ci_upper = stats.t.interval(
                0.95, 
                len(df[col].dropna())-1, 
                loc=mean_value, 
                scale=stats.sem(df[col].dropna())
            )
            
            results[col] = {
                'mean': mean_value,
                'std': std_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'median': df[col].median(),
                'mode': df[col].mode()[0] if len(df[col].mode()) > 0 else np.nan
            }
            
            model_name = col.replace('_score', '').upper()
            if col == 'ensemble_score':
                model_name = 'СУММАТОР'
            print(f"{model_name:12} | Среднее: {mean_value:.3f} | Медиана: {results[col]['median']:.3f} | "
                  f"95% ДИ: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # Математическое ожидание пользовательских оценок
    user_mean = df['rating_numeric'].mean()
    user_std = df['rating_numeric'].std()
    print("-" * 70)
    print(f"Пользователи  | Среднее: {user_mean:.3f} | Медиана: {df['rating_numeric'].median():.3f}")
    
    results['user_rating'] = {
        'mean': user_mean,
        'std': user_std,
        'median': df['rating_numeric'].median()
    }
    
    return results

def analyze_deviations(df):
    """
    Анализ отклонений предсказаний моделей от пользовательских оценок
    """
    print("\n" + "="*60)
    print("АНАЛИЗ ОТКЛОНЕНИЙ ОТ ПОЛЬЗОВАТЕЛЬСКИХ ОЦЕНОК")
    print("="*60)
    
    score_columns = ['vader_score', 'flair_score', 'rubert_score', 
                       'roberta_score', 'distilbert_score','logistic_regression_score',
                       'svm_score','random_forest_score', 'ensemble_score']
    
    deviation_results = {}
    
    print("\nСредние абсолютные отклонения (MAE):")
    print("-" * 70)
    
    for col in score_columns:
        if col in df.columns:
            # Абсолютное отклонение
            abs_deviation = np.abs(df[col] - df['rating_numeric'])
            mae = abs_deviation.mean()
            mae_std = abs_deviation.std()
            
            # Относительное отклонение (в процентах)
            rel_deviation = (abs_deviation / df['rating_numeric'] * 100).mean()
            
            # Среднеквадратичная ошибка
            rmse = np.sqrt(((df[col] - df['rating_numeric']) ** 2).mean())
            
            deviation_results[col] = {
                'mae': mae,
                'mae_std': mae_std,
                'rmse': rmse,
                'rel_deviation': rel_deviation
            }
            
            model_name = col.replace('_score', '').upper()
            if col == 'ensemble_score':
                model_name = 'СУММАТОР'
            print(f"{model_name:12} | MAE: {mae:.3f} (±{mae_std:.3f}) | RMSE: {rmse:.3f} | "
                  f"Отн.откл.: {rel_deviation:.1f}%")
    
    return deviation_results

def analyze_by_rating_groups(df):
    """
    Анализ по группам рейтингов
    """
    print("\n" + "="*60)
    print("АНАЛИЗ ПО ГРУППАМ РЕЙТИНГОВ")
    print("="*60)
    
    # Создаем группы рейтингов
    df['rating_group'] = pd.cut(df['rating_numeric'], 
                                 bins=[0, 2, 3, 5], 
                                 labels=['Низкий (1-2)', 'Средний (3)', 'Высокий (4-5)'])
    
    score_columns = ['vader_score', 'flair_score', 'rubert_score', 
                       'roberta_score', 'distilbert_score','logistic_regression_score',
                       'svm_score','random_forest_score', 'ensemble_score']
    
    print("\nСредние оценки моделей по группам:")
    print("-" * 80)
    
    group_stats = []
    
    for group in df['rating_group'].unique():
        group_data = df[df['rating_group'] == group]
        print(f"\n{group}: {len(group_data)} отзывов")
        print(f"  Пользовательский рейтинг: {group_data['rating_numeric'].mean():.2f}")
        
        for col in score_columns:
            if col in df.columns:
                model_mean = group_data[col].mean()
                model_std = group_data[col].std()
                model_name = col.replace('_score', '').upper()
                if col == 'ensemble_score':
                    model_name = 'СУММАТОР'
                print(f"  {model_name:12} | {model_mean:.2f} ± {model_std:.2f}")
                
                group_stats.append({
                    'group': group,
                    'model': model_name,
                    'mean_score': model_mean,
                    'std_score': model_std,
                    'count': len(group_data)
                })
    
    return pd.DataFrame(group_stats)

def create_visualizations(df, results, deviation_results):
    print("\n" + "="*60)
    print("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("="*60)
    
    # 1. Сравнение средних оценок
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 9))
    axes = axes.flatten()
    
    score_columns = ['vader_score', 'flair_score', 'rubert_score', 
                       'roberta_score', 'distilbert_score','logistic_regression_score','svm_score','random_forest_score', 'ensemble_score']
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE194', 
              '#A8E6CF', '#D4A5A5', '#9B59B6', '#3498DB', '#F1C40F']
    
    # График 1: Сравнение средних значений
    ax = axes[0]
    models = []
    means = []
    errors = []
    
    for col in score_columns:
        if col in df.columns:
            model_name = col.replace('_score', '')
            if col == 'ensemble_score':
                model_name = 'СУММАТОР'
            models.append(model_name)
            means.append(results[col]['mean'])
            errors.append(results[col]['std'] / np.sqrt(len(df[col].dropna())))
    
    models.append('User')
    means.append(results['user_rating']['mean'])
    errors.append(results['user_rating']['std'] / np.sqrt(len(df['rating_numeric'])))
    
    bars = ax.bar(range(len(models)), means, yerr=errors, capsize=5, 
                  color=colors[:len(models)], alpha=0.8, edgecolor='black', linewidth=1)
    ax.axhline(y=results['user_rating']['mean'], color='red', linestyle='--', 
               linewidth=2, label=f"Средняя пользовательская: {results['user_rating']['mean']:.2f}")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45)
    ax.set_ylabel('Средняя оценка')
    ax.set_title('Сравнение средних оценок моделей\n(математическое ожидание)', fontsize=10, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
    
    
    
    # График 2 Сравнение MAE
    ax = axes[1]
    models_mae = []
    mae_values = []
    mae_errors = []
    
    for col in score_columns:
        if col in df.columns:
            model_name = col.replace('_score', '')
            if col == 'ensemble_score':
                model_name = 'СУММАТОР'
            models_mae.append(model_name)
            mae_values.append(deviation_results[col]['mae'])
            # mae_errors.append(deviation_results[col]['mae_std'])
            mae_errors.append(0)

    
    bars = ax.bar(range(len(models_mae)), mae_values, yerr=mae_errors, capsize=5,
                  color=colors[:len(models_mae)], alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(models_mae)))
    ax.set_xticklabels(models_mae, rotation=45)
    ax.set_ylabel('Среднее абсолютное отклонение (MAE)')
    ax.set_title('Ошибка моделей относительно пользовательских оценок', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for i, (bar, mae) in enumerate(zip(bars, mae_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{mae:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # График 3: Оценки по группам рейтингов
    ax = axes[2]
    df['rating_group'] = pd.cut(df['rating_numeric'], 
                                 bins=[0, 2, 3, 5], 
                                 labels=['Низкие (1-2)', 'Средние (3)', 'Высокие (4-5)'])
    
    x = np.arange(len(df['rating_group'].unique()))
    width = 0.1
    
    for i, (col, color) in enumerate(zip(score_columns, colors[:len(score_columns)])):
        if col in df.columns:
            means = df.groupby('rating_group')[col].mean()
            offset = width * (i - len(score_columns)/2 + 0.5)
            model_name = col.replace('_score', '')
            if col == 'ensemble_score':
                model_name = 'СУММАТОР'
            bars = ax.bar(x + offset, means, width, label=model_name, 
                         color=color, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Группа рейтинга')
    ax.set_ylabel('Средняя оценка модели')
    ax.set_title('Оценки моделей по группам пользовательских рейтингов', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['rating_group'].unique())
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=3, color='gray', linestyle=':', alpha=0.5)
    
    # График 4: Диаграмма рассеяния (сумматор vs пользователь)
    ax = axes[3]
    
    if 'ensemble_score' in df.columns:
        scatter = ax.scatter(df['rating_numeric'], df['ensemble_score'], 
                             c=df['rating_numeric'], cmap='viridis', 
                             alpha=0.6, edgecolors='black', linewidth=0.5, s=100)
        
        # Линия идеального соответствия
        ax.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Идеальное соответствие', alpha=0.7)
        
        # Линия регрессии
        z = np.polyfit(df['rating_numeric'], df['ensemble_score'], 1)
        p = np.poly1d(z)
        ax.plot([1, 5], p([1, 5]), 'b-', linewidth=2, 
                label=f'Тренд (R²={np.corrcoef(df["rating_numeric"], df["ensemble_score"])[0,1]**2:.2f})', alpha=0.7)
        
        ax.set_xlabel('Пользовательский рейтинг')
        ax.set_ylabel('Оценка СУММАТОРА')
        ax.set_title('Сравнение СУММАТОРА с пользовательскими оценками', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Пользовательский рейтинг')
    
    plt.tight_layout()
    plt.savefig('answer/sentiment_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Дашборд сохранен как 'answer/sentiment_analysis_dashboard.png'")
    
    return fig

def create_additional_visualizations(df, results):
    # График распределения ошибок
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE194', 
              '#A8E6CF', '#D4A5A5', '#9B59B6', '#3498DB', '#F1C40F']

    score_columns = ['vader_score', 'flair_score', 'rubert_score', 
                       'roberta_score', 'distilbert_score','logistic_regression_score',
                       'svm_score','random_forest_score', 'ensemble_score']
    
    # 1. Распределение ошибок
    ax = axes[0]
    for col in score_columns:
        if col in df.columns:
            errors = df[col] - df['rating_numeric']
            model_name = col.replace('_score', '').upper()
            if col == 'ensemble_score':
                model_name = 'СУММАТОР'
            ax.hist(errors, bins=20, alpha=0.5, label=model_name, density=True)
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Нулевая ошибка')
    ax.set_xlabel('Ошибка предсказания (модель - пользователь)')
    ax.set_ylabel('Плотность')
    ax.set_title('Распределение ошибок моделей', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Точность по классам
    ax = axes[1]
    
    # Создаем категории
    df['rating_cat'] = pd.cut(df['rating_numeric'], bins=[0, 2, 3, 5], 
                               labels=['Negative', 'Neutral', 'Positive'])
    
    accuracy_data = []
    models_list = []
    
    for col in score_columns:
        if col in df.columns:
            df['pred_cat'] = pd.cut(df[col], bins=[0, 2, 3, 5], 
                                     labels=['Negative', 'Neutral', 'Positive'])
            accuracy = (df['pred_cat'] == df['rating_cat']).mean()
            accuracy_data.append(accuracy)
            model_name = col.replace('_score', '')
            if col == 'ensemble_score':
                model_name = 'СУММАТОР'
            models_list.append(model_name)
    
    bars = ax.bar(models_list, accuracy_data, color=colors[:len(models_list)], 
                  alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_ylabel('Точность классификации')
    ax.set_title('Точность классификации тональности\n(совпадение категорий)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for i, (bar, acc) in enumerate(zip(bars, accuracy_data)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('answer/additional_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Дополнительный график сохранен как 'answer/additional_analysis.png'")

def main():
    # Путь к файлу с результатами
    file_path = "answer/отзывы(результат работы).csv"
    
    try:
        # Загрузка данных
        df = load_and_prepare_data(file_path)
        
        # НОВЫЙ ГРАФИК: распределение Ensemble по трём классам
        class_counts, class_percentages = plot_ensemble_class_distribution(df)
        
        # Визуализация нормальности (с правильным отображением Ensemble)
        normality_results = plot_normality_comparison(df)
        
        # Расчет математического ожидания
        results = calculate_mathematical_expectation(df)
        
        # Анализ отклонений
        deviation_results = analyze_deviations(df)
        
        # Анализ по группам
        group_stats = analyze_by_rating_groups(df)
        
        # Создание визуализаций
        create_visualizations(df, results, deviation_results)
        
        # Дополнительные визуализации
        create_additional_visualizations(df, results)
        
        # Сохранение результатов анализа
        summary_data = []
        for k in list(results.keys()):
            if k != 'user_rating' and k in deviation_results:
                model_name = k.replace('_score', '').upper()
                model_name = model_name.replace('_FOR_ANALYSIS', '')
                if k == 'ensemble_score_for_analysis':
                    model_name = 'СУММАТОР (raw)'
                summary_data.append({
                    'Модель': model_name,
                    'Средняя оценка': results[k]['mean'],
                    'Стд отклонение': results[k]['std'],
                    'MAE': deviation_results[k]['mae'],
                    'RMSE': deviation_results[k]['rmse'],
                    'Асимметрия': normality_results[k]['skewness'] if k in normality_results else np.nan,
                    'Эксцесс': normality_results[k]['kurtosis'] if k in normality_results else np.nan,
                    'Нормальность (p-value)': normality_results[k]['shapiro_p'] if k in normality_results else np.nan
                })
        
        summary = pd.DataFrame(summary_data)
        summary.to_csv('answer/model_performance_summary.csv', index=False, encoding='utf-8-sig')
        print("\n📊 Сводка производительности сохранена в 'answer/model_performance_summary.csv'")
        
        # Вывод итогов
        print("\n" + "="*60)
        print("ИТОГОВЫЙ ВЫВОД")
        print("="*60)
        
        # Статистика по классам Ensemble
        print("\n📊 Распределение Ensemble по классам тональности:")
        for cls in ['Negative', 'Neutral', 'Positive']:
            if cls in class_counts:
                print(f"   {cls}: {class_counts[cls]} ({class_percentages[cls]:.1f}%)")
        
        # Определяем лучшую модель (минимальная MAE)
        best_model = min(deviation_results, key=lambda x: deviation_results[x]['mae'])
        best_model_name = best_model.replace('_score', '').upper()
        best_model_name = best_model_name.replace('_FOR_ANALYSIS', ' (raw)')
        if best_model == 'ensemble_score_for_analysis':
            best_model_name = 'СУММАТОР (raw)'
        
        print(f"\n🏆 Лучшая модель: {best_model_name}")
        print(f"   MAE: {deviation_results[best_model]['mae']:.3f}")
        
        # Сравнение с пользовательскими оценками
        user_mean = results['user_rating']['mean']
        print(f"\n👥 Пользовательская средняя оценка: {user_mean:.3f}")

    except FileNotFoundError:
        print(f"❌ Файл {file_path} не найден. Проверьте путь к файлу.")
    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()