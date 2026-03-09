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
    
    # Добавляем числовые колонки для каждой модели
    model_columns = {
        'vader_sentiment': 'vader_score',
        'flair_sentiment': 'flair_score',
        'rubert_sentiment': 'rubert_score',
        'roberta_sentiment': 'roberta_score',
        'distilbert_sentiment': 'distilbert_score'
    }
    
    for sent_col, score_col in model_columns.items():
        if sent_col in df_clean.columns:
            df_clean[score_col] = df_clean[sent_col].map(sentiment_to_score)
    
    # СОЗДАЕМ СУММАТОР (усредненное предсказание всех моделей)
    score_columns = ['vader_score', 'flair_score', 'rubert_score', 
                         'roberta_score', 'distilbert_score','logistic_regression_score','svm_score','random_forest_score']
    available_scores = [col for col in score_columns if col in df_clean.columns]
    
    if available_scores:
        df_clean['ensemble_score'] = df_clean[available_scores].mean(axis=1)
        df_clean['ensemble_score'] = df_clean['ensemble_score'].round()  # Округляем до целого
    
    print(f"Загружено {len(df_clean)} записей с валидными рейтингами")
    print(f"Колонки в данных: {list(df_clean.columns)}")
    print(f"Создан сумматор (ensemble_score) на основе {len(available_scores)} моделей")
    
    return df_clean

def plot_normality_comparison(df):
    print("\n" + "="*60)
    print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ ОЦЕНОК")
    print("="*60)
    
    columns_to_test = ['vader_score', 'flair_score', 'rubert_score', 
                       'roberta_score', 'distilbert_score', 'logistic_regression_score',
                       'svm_score', 'random_forest_score', 'ensemble_score', 'rating_numeric']
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE194', 
              '#A8E6CF', '#D4A5A5', '#9B59B6', '#3498DB', '#F1C40F']

    results = {}
    
    print("\n" + "-" * 90)
    print(f"{'Модель':<15} | {'Уникальных значений':<18} | {'Распределение':<30} | {'Тест KS p-value':<15}")
    print("-" * 90)
    
    for col in columns_to_test:
        if col in df.columns:
            data = df[col].dropna()
            
            # Анализ частот для дискретных данных
            value_counts = data.value_counts().sort_index()
            unique_values = len(value_counts)
            
            from scipy import stats
            
            # Расчет асимметрии (skewness) и эксцесса (kurtosis)
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            
            # Эмпирическое распределение
            if unique_values <= 5:  # Дискретные данные
                # Создаем теоретическое равномерное распределение для сравнения
                theoretical = np.random.uniform(1, 5, 10000)
                ks_stat, ks_p = stats.ks_2samp(data, theoretical)
                
                # Формируем строку с распределением
                distr_str = ", ".join([f"{v}: {c}" for v, c in value_counts.items()])
                if len(distr_str) > 28:
                    distr_str = distr_str[:25] + "..."
            else:
                # Для непрерывных данных используем тест на нормальность
                ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                distr_str = f"среднее={data.mean():.2f}"
            
            model_name = col.replace('_score', '').replace('rating_numeric', 'Пользователи').replace('ensemble', 'СУММАТОР')
            if col == 'ensemble_score':
                model_name = 'СУММАТОР'
            
            # Интерпретация p-value
            if ks_p > 0.05:
                normality_note = f"p={ks_p:.4f} (≈ нормальное)"
            else:
                normality_note = f"p={ks_p:.4f} (≠ нормальное)"
            
            print(f"{model_name:<15} | {unique_values:<18} | {distr_str:<30} | {normality_note:<15}")
            
            results[col] = {
                'ks_p': ks_p,
                'is_normal': ks_p > 0.05,
                'unique_values': unique_values,
                'distribution': value_counts.to_dict(),
                'mean': data.mean(),
                'std': data.std(),
                'skewness': skewness,      # Добавляем асимметрию
                'kurtosis': kurtosis,       # Добавляем эксцесс
                'shapiro_p': ks_p           # Для совместимости используем ks_p как shapiro_p
            }
    
    # Добавляем визуализацию распределений
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, col in enumerate(columns_to_test):
        if col in df.columns and idx < len(axes):
            ax = axes[idx]
            data = df[col].dropna()
            
            model_name = col.replace('_score', '').replace('rating_numeric', 'Пользователи').replace('ensemble', 'СУММАТОР')
            if col == 'ensemble_score':
                model_name = 'СУММАТОР'
            
            # Столбчатая диаграмма для дискретных данных
            value_counts = data.value_counts().sort_index()
            bars = ax.bar(value_counts.index.astype(str), value_counts.values, 
                   color=colors[idx], edgecolor='black', alpha=0.7)
            
            # Добавляем значения на столбцы
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                
            # Добавляем вертикальную линию для средней пользовательской оценки
            # user_mean = df['rating_numeric'].mean()
            # ax.axvline(x=list(value_counts.index).index(round(user_mean)) if round(user_mean) in value_counts.index else len(value_counts)/2, 
            #           color='blue', linestyle='--', linewidth=2, 
            #           label=f'Средняя пользователя: {user_mean:.2f}')            

            # Добавляем кривую нормального распределения для справки
            # if len(value_counts) > 3:
            #     x = np.linspace(0, 4, 100)
            #     y = stats.norm.pdf(x, data.mean(), data.std())
            #     ax2 = ax.twinx()
            #     ax2.plot(x, y * len(data) * 0.5, 'r-', linewidth=2, alpha=0.5)
            #     ax2.set_ylabel('Плотность', color='r')
            #     ax2.tick_params(axis='y', labelcolor='r')
            
            ax.set_xlabel('Оценка')
            ax.set_ylabel('Частота')
            ax.set_title(f'{model_name}')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('answer/distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
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
    file_path = "answer/отзывы(результат работы).csv"  # замените на ваш файл
    
    try:
        # Загрузка данных
        df = load_and_prepare_data(file_path)
        
        score_columns = ['vader_score', 'flair_score', 'rubert_score', 
                         'roberta_score', 'distilbert_score','logistic_regression_score','svm_score','random_forest_score', 'ensemble_score']
        available_models = [col for col in score_columns if col in df.columns]

        # Визуализация нормальности
        normality_results = plot_normality_comparison(df)
        
        # Расчет математического ожидания
        results = calculate_mathematical_expectation(df)
        
        # Анализ отклонений
        deviation_results = analyze_deviations(df)
        
        # Анализ по группам
        group_stats = analyze_by_rating_groups(df)
             
        # Визуализация нормальности
        #plot_normality_comparison(df, normality_results)
        
        # Создание визуализаций
        create_visualizations(df, results, deviation_results)
        
        # Дополнительные визуализации
        create_additional_visualizations(df, results)
        
        # Сохранение результатов анализа
        summary_data = []
        for k in list(results.keys()):
            if k != 'user_rating' and k in deviation_results:
                model_name = k.replace('_score', '').upper()
                if k == 'ensemble_score':
                    model_name = 'СУММАТОР'
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
        
        print("\n" + "="*60)
        print("ИТОГОВЫЙ ВЫВОД")
        print("="*60)
        
        # Определяем лучшую модель (минимальная MAE)
        best_model = min(deviation_results, key=lambda x: deviation_results[x]['mae'])
        best_model_name = best_model.replace('_score', '').upper()
        if best_model == 'ensemble_score':
            best_model_name = 'СУММАТОР'
        
        print(f"\n🏆 Лучшая модель: {best_model_name}")
        print(f"   MAE: {deviation_results[best_model]['mae']:.3f}")
        print(f"   Средняя оценка: {results[best_model]['mean']:.3f}")
        
        # Сравнение с пользовательскими оценками
        user_mean = results['user_rating']['mean']
        print(f"\n👥 Пользовательская средняя оценка: {user_mean:.3f}")
        
        # Отклонение от пользовательских оценок
        print("\n📈 Отклонения от пользовательских оценок:")
        for model in summary_data:
            deviation = abs(model['Средняя оценка'] - user_mean)
            if deviation < 0.3:
                print(f"✅ {model['Модель']}: близка к пользовательским оценкам (отклонение {deviation:.3f})")
            elif deviation < 0.5:
                print(f"⚠️ {model['Модель']}: умеренное отклонение ({deviation:.3f})")
            else:
                print(f"❌ {model['Модель']}: сильное отклонение ({deviation:.3f})")
        
        # Оценка эффективности сумматора
        if 'ensemble_score' in deviation_results:
            ensemble_mae = deviation_results['ensemble_score']['mae']
            individual_maes = [deviation_results[col]['mae'] for col in score_columns 
                              if col in deviation_results and col != 'ensemble_score']
            avg_individual_mae = np.mean(individual_maes)
            
            print(f"\n🤖 Анализ эффективности СУММАТОРА:")
            print(f"   MAE сумматора: {ensemble_mae:.3f}")
            print(f"   Средняя MAE отдельных моделей: {avg_individual_mae:.3f}")
            
            if ensemble_mae < avg_individual_mae:
                improvement = (avg_individual_mae - ensemble_mae) / avg_individual_mae * 100
                print(f"   ✅ Сумматор работает лучше отдельных моделей на {improvement:.1f}%")
            else:
                print(f"   ⚠️ Сумматор работает хуже среднего по отдельным моделям")

    except FileNotFoundError:
        print(f"❌ Файл {file_path} не найден. Проверьте путь к файлу.")
    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()