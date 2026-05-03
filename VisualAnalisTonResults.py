import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import glob
import sys
import io

# Настройка кодировки для вывода
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

# Настройка стиля для графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'

def find_result_file():
    """Автоматический поиск файла с результатами"""
    answer_dir = "answer"
    
    if not os.path.exists(answer_dir):
        print("[X] Folder 'answer' does not exist")
        return None
    
    # Ищем ВСЕ CSV файлы с результатами - и латиницу, и кириллицу
    patterns = [
        os.path.join(answer_dir, "**", "отзывы(результат работы).csv"),
        os.path.join(answer_dir, "**", "otzyvy(result).csv"),
        os.path.join(answer_dir, "отзывы(результат работы).csv"),
        os.path.join(answer_dir, "otzyvy(result).csv"),
    ]
    
    all_files = []
    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        all_files.extend(found)
    
    if all_files:
        # Убираем дубликаты и сортируем по времени
        all_files = list(set(all_files))
        all_files.sort(key=os.path.getctime, reverse=True)
        latest_file = all_files[0]
        print(f"[OK] Found results file: {latest_file}")
        return latest_file
    
    # Если ничего не нашли по шаблонам, ищем все CSV в answer/
    all_csvs = glob.glob(os.path.join(answer_dir, "**", "*.csv"), recursive=True)
    if all_csvs:
        all_csvs.sort(key=os.path.getctime, reverse=True)
        latest_file = all_csvs[0]
        print(f"[OK] Found CSV file: {latest_file}")
        return latest_file
    
    print("[X] No result CSV files found in 'answer' folder")
    return None

def load_and_prepare_data(file_path=None):
    print("="*60)
    print("LOADING AND PREPARING DATA")
    print("="*60)
    
    if file_path is None:
        file_path = find_result_file()
    
    if file_path is None:
        return None
    
    # Пробуем разные кодировки
    for encoding in ['utf-8-sig', 'utf-8', 'cp1251']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except:
            continue
    else:
        print("[X] Could not read file with any encoding")
        return None
    
    print(f"\nAvailable columns in file:")
    for col in df.columns:
        print(f"   - {col}")
    
    print(f"\nFirst 5 rows (selected columns):")
    display_cols = ['rating', 'actual_sentiment', 'ensemble_sentiment', 'vader_sentiment']
    existing_cols = [c for c in display_cols if c in df.columns]
    if existing_cols:
        print(df[existing_cols].head())
    
    # Проверяем наличие столбца rating
    if 'rating' not in df.columns:
        print("[X] No 'rating' column")
        return None
    
    # Преобразуем rating в числовой формат
    df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')
    
    print(f"\nRating statistics:")
    print(f"   Total rows: {len(df)}")
    print(f"   Valid numeric ratings: {df['rating_numeric'].notna().sum()}")
    
    # Создаем actual_sentiment из rating если его нет или он пустой
    if 'actual_sentiment' not in df.columns:
        df['actual_sentiment'] = None
    
    df['actual_sentiment'] = df['actual_sentiment'].astype(object)
    
    def rating_to_sentiment(rating):
        try:
            r = float(rating)
            if r >= 4:
                return 'Positive'
            elif r <= 2:
                return 'Negative'
            else:
                return 'Neutral'
        except:
            return None
    
    missing_mask = df['actual_sentiment'].isna()
    if missing_mask.any():
        df.loc[missing_mask, 'actual_sentiment'] = df.loc[missing_mask, 'rating'].apply(rating_to_sentiment)
        print(f"[OK] Filled {missing_mask.sum()} missing values in actual_sentiment")
    
    print(f"\nactual_sentiment after filling:")
    print(f"   Valid: {df['actual_sentiment'].notna().sum()}")
    print(f"   Unique values: {df['actual_sentiment'].dropna().unique()[:10]}")
    
    # Фильтруем строки с валидными рейтингами
    mask = df['rating_numeric'].notna() & df['actual_sentiment'].notna()
    df_clean = df[mask].copy()
    
    print(f"\nAfter filtering:")
    print(f"   Filtered rows: {len(df_clean)} out of {len(df)}")
    
    if len(df_clean) == 0:
        print("[X] No records with valid ratings")
        return None
    
    # Заполняем пропуски в rating_numeric на основе actual_sentiment
    if 'rating_numeric' in df_clean.columns:
        missing_mask = df_clean['rating_numeric'].isna()
        if missing_mask.any():
            sentiment_to_rating = {'Positive': 5, 'Neutral': 3, 'Negative': 1}
            df_clean.loc[missing_mask, 'rating_numeric'] = df_clean.loc[missing_mask, 'actual_sentiment'].map(sentiment_to_rating)
            print(f"[OK] Filled {missing_mask.sum()} missing values in rating_numeric")
    
    # Создаем числовые представления для предсказаний моделей
    sentiment_to_score = {
        'Positive': 5,
        'Neutral': 3,
        'Negative': 1
    }
    
    sentiment_cols = [col for col in df_clean.columns if col.endswith('_sentiment') and col != 'actual_sentiment']
    
    for sent_col in sentiment_cols:
        score_col = sent_col.replace('_sentiment', '_score')
        if score_col not in df_clean.columns:
            df_clean[score_col] = df_clean[sent_col].map(sentiment_to_score)
            print(f"[OK] Created {score_col} from {sent_col}")
    
    # Обработка ensemble_score
    if 'ensemble_score_raw' in df_clean.columns:
        df_clean['ensemble_score_for_analysis'] = df_clean['ensemble_score_raw']
        print("[OK] Using ensemble_score_raw")
    elif 'ensemble_score' in df_clean.columns:
        df_clean['ensemble_score_for_analysis'] = df_clean['ensemble_score']
        print("[OK] Using ensemble_score")
    elif 'ensemble_sentiment' in df_clean.columns:
        df_clean['ensemble_score'] = df_clean['ensemble_sentiment'].map(sentiment_to_score)
        df_clean['ensemble_score_for_analysis'] = df_clean['ensemble_score']
        print("[OK] Created ensemble_score from ensemble_sentiment")
    
    if 'ensemble_score' not in df_clean.columns and 'ensemble_score_for_analysis' in df_clean.columns:
        df_clean['ensemble_score'] = df_clean['ensemble_score_for_analysis'].round().clip(1, 5)
    
    print(f"\n[OK] Loaded {len(df_clean)} records with valid ratings")
    score_cols = [col for col in df_clean.columns if col.endswith('_score')]
    print(f"Available score columns: {score_cols}")
    
    return df_clean

def plot_ensemble_class_distribution(df):
    """
    График распределения ensemble оценок по трём классам тональности
    """
    
    plot_df = df.copy()
    
    if 'ensemble_score' in plot_df.columns:
        ensemble_scores = plot_df['ensemble_score']
    elif 'ensemble_score_for_analysis' in plot_df.columns:
        # Только если ensemble_score отсутствует
        ensemble_scores = plot_df['ensemble_score_for_analysis'].round().clip(1, 5)
    else:
        print("[X] No ensemble data")
        return None, None
    
    def classify_sentiment(score):
        if score >= 4:
            return "Positive"
        elif score <= 2:
            return "Negative"
        else:
            return "Neutral"
    
    plot_df['ensemble_class'] = ensemble_scores.apply(classify_sentiment)
    
    class_counts = plot_df['ensemble_class'].value_counts()
    class_percentages = class_counts / len(plot_df) * 100
    
    print("\nRounded ensemble_score distribution:")
    rounded_counts = ensemble_scores.value_counts().sort_index()
    for score, count in rounded_counts.items():
        print(f"  Score {score}: {count} ({count/len(plot_df)*100:.1f}%)")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {'Negative': '#FF6B6B', 'Neutral': '#FFE194', 'Positive': '#4ECDC4'}
    class_order = ['Negative', 'Neutral', 'Positive']
    
    class_values = []
    class_labels = []
    
    for cls in class_order:
        count = class_counts.get(cls, 0)
        class_labels.append(cls)
        class_values.append(count)
    
    bars = ax.bar(class_labels, class_values, 
                  color=[colors[cls] for cls in class_labels], 
                  edgecolor='black', linewidth=2, alpha=0.8)
    
    for bar, count in zip(bars, class_values):
        if count > 0:
            percentage = count / len(plot_df) * 100
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(1, height * 0.02),
                    f'{count}\n({percentage:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_xlabel('Sentiment Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Reviews', fontsize=14, fontweight='bold')
    ax.set_title('Ensemble Score Distribution by Sentiment Class\n(based on rounded ensemble_score)',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    mean_score = ensemble_scores.mean()
    median_score = ensemble_scores.median()
    ax.text(0.5, -0.12, f'Mean: {mean_score:.3f} | Median: {median_score:.3f}',
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_ylim(0, max(class_values) * 1.15 if class_values else 10)
    
    plt.tight_layout()
    
    os.makedirs('answer', exist_ok=True)
    
    plt.savefig('answer/ensemble_class_distribution.png', dpi=300, bbox_inches='tight')
    # plt.show()
    # plt.close(fig)
    
    print("\n[OK] Ensemble class distribution plot saved as 'answer/ensemble_class_distribution.png'")
    
    print("\nEnsemble distribution statistics:")
    print("-" * 40)
    for cls, count in zip(class_labels, class_values):
        if count > 0:
            print(f"{cls:10}: {count:4} ({count/len(plot_df)*100:.1f}%)")
    
    return class_counts, class_percentages

def plot_normality_comparison(df):
    print("\n" + "="*60)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    model_order = [
        'vader_score',
        'flair_score', 
        'rubert_score',
        'roberta_score',
        'distilbert_score',
        'logistic_regression_score',
        'random_forest_score',
        'svm_score',
        'ensemble_score',       # Изменено: используем округлённый
        'rating_numeric'
    ]
    
    columns_to_test = [col for col in model_order if col in df.columns]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE194', 
              '#A8E6CF', '#D4A5A5', '#9B59B6', '#3498DB', '#F1C40F']
    
    results = {}
    
    print("\n" + "-" * 90)
    print(f"{'Model':<20} | {'Unique Values':<18} | {'Distribution':<35} | {'KS Test p-value':<15}")
    print("-" * 90)
    
    for idx, col in enumerate(columns_to_test):
        if col not in df.columns:
            continue
            
        data = df[col].dropna()
        unique_values = data.nunique()
        
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
            distr_str = f"mean={data.mean():.2f}, std={data.std():.2f}"
        
        if col == 'rating_numeric':
            model_name = 'Users'
        elif col == 'ensemble_score':
            model_name = 'ENSEMBLE'
        else:
            model_name = col.replace('_score', '').replace('_', ' ').title()
        
        normality_note = f"p={ks_p:.4f}" + (" (normal)" if ks_p > 0.05 else " (not normal)")
        
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
    
    # Визуализация - строим по одному графику за раз, избегая tight_layout
    for idx, col in enumerate(columns_to_test):
        if col not in df.columns:
            continue
            
        data = df[col].dropna()
        
        if col == 'rating_numeric':
            model_name = 'Users'
        elif col == 'ensemble_score':
            model_name = 'ENSEMBLE'
        else:
            model_name = col.replace('_score', '').replace('_', ' ').title()
        
        unique_values = data.nunique()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        if unique_values <= 10:
            value_counts = data.value_counts().sort_index()
            bars = ax.bar(value_counts.index.astype(str), value_counts.values, 
                        color=colors[idx % len(colors)], edgecolor='black', alpha=0.7)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=9)
        else:
            ax.hist(data, bins=20, color='#3498DB', edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Явно устанавливаем ticks для избежания ошибки
        if unique_values <= 10:
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels([str(v) for v in value_counts.index])
        
        plt.tight_layout()
        plt.close(fig)
    
    # Создаём сводный график с обработкой ошибок
    try:
        n_cols = min(len(columns_to_test), 5)
        n_rows = (len(columns_to_test) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, col in enumerate(columns_to_test):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            data = df[col].dropna()
            
            if col == 'rating_numeric':
                model_name = 'Users'
            elif col == 'ensemble_score':
                model_name = 'ENSEMBLE'
            else:
                model_name = col.replace('_score', '').replace('_', ' ').title()
            
            unique_values = data.nunique()
            
            if unique_values <= 10:
                value_counts = data.value_counts().sort_index()
                ax.bar(value_counts.index.astype(str), value_counts.values, 
                       color=colors[idx % len(colors)], edgecolor='black', alpha=0.7)
                # Явно устанавливаем ticks
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels([str(v) for v in value_counts.index], fontsize=8)
            else:
                ax.hist(data, bins=20, color='#3498DB', edgecolor='black', alpha=0.7)
            
            ax.set_xlabel('Score', fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            ax.set_title(f'{model_name}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        for idx in range(len(columns_to_test), len(axes)):
            axes[idx].set_visible(False)
        
        plt.savefig('answer/distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"[WARNING] Could not create summary plot: {e}")
        # Сохраняем хотя бы что-то
        try:
            plt.savefig('answer/distribution_analysis.png', dpi=150)
            plt.close()
        except:
            pass
    
    print("\n[OK] Distribution analysis saved as 'answer/distribution_analysis.png'")
    
    return results

def calculate_mathematical_expectation(df):
    """
    Расчет математического ожидания для каждой модели
    """
    print("\n" + "="*60)
    print("MATHEMATICAL EXPECTATION OF MODEL SCORES")
    print("="*60)
    
    score_columns = ['vader_score', 'flair_score', 'rubert_score', 
                       'roberta_score', 'distilbert_score','logistic_regression_score','svm_score','random_forest_score', 'ensemble_score']
    
    results = {}
    
    print("\nMean score values (mathematical expectation):")
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
                model_name = 'ENSEMBLE'
            print(f"{model_name:12} | Mean: {mean_value:.3f} | Median: {results[col]['median']:.3f} | "
                  f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    user_mean = df['rating_numeric'].mean()
    user_std = df['rating_numeric'].std()
    print("-" * 70)
    print(f"Users        | Mean: {user_mean:.3f} | Median: {df['rating_numeric'].median():.3f}")
    
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
    print("DEVIATION ANALYSIS FROM USER RATINGS")
    print("="*60)
    
    score_columns = ['vader_score', 'flair_score', 'rubert_score', 
                       'roberta_score', 'distilbert_score','logistic_regression_score',
                       'svm_score','random_forest_score', 'ensemble_score']
    
    deviation_results = {}
    
    print("\nMean Absolute Errors (MAE):")
    print("-" * 70)
    
    for col in score_columns:
        if col in df.columns:
            abs_deviation = np.abs(df[col] - df['rating_numeric'])
            mae = abs_deviation.mean()
            mae_std = abs_deviation.std()
            
            rel_deviation = (abs_deviation / df['rating_numeric'] * 100).mean()
            rmse = np.sqrt(((df[col] - df['rating_numeric']) ** 2).mean())
            
            deviation_results[col] = {
                'mae': mae,
                'mae_std': mae_std,
                'rmse': rmse,
                'rel_deviation': rel_deviation
            }
            
            model_name = col.replace('_score', '').upper()
            if col == 'ensemble_score':
                model_name = 'ENSEMBLE'
            print(f"{model_name:12} | MAE: {mae:.3f} (+/-{mae_std:.3f}) | RMSE: {rmse:.3f} | "
                  f"Rel.dev.: {rel_deviation:.1f}%")
    
    return deviation_results

def analyze_by_rating_groups(df):
    """
    Анализ по группам рейтингов
    """
    print("\n" + "="*60)
    print("ANALYSIS BY RATING GROUPS")
    print("="*60)
    
    df['rating_group'] = pd.cut(df['rating_numeric'], 
                                 bins=[0, 2, 3, 5], 
                                 labels=['Low (1-2)', 'Medium (3)', 'High (4-5)'])
    
    score_columns = ['vader_score', 'flair_score', 'rubert_score', 
                       'roberta_score', 'distilbert_score','logistic_regression_score',
                       'svm_score','random_forest_score', 'ensemble_score']
    
    print("\nMean model scores by group:")
    print("-" * 80)
    
    group_stats = []
    
    for group in df['rating_group'].unique():
        group_data = df[df['rating_group'] == group]
        print(f"\n{group}: {len(group_data)} reviews")
        print(f"  User rating: {group_data['rating_numeric'].mean():.2f}")
        
        for col in score_columns:
            if col in df.columns:
                model_mean = group_data[col].mean()
                model_std = group_data[col].std()
                model_name = col.replace('_score', '').upper()
                if col == 'ensemble_score':
                    model_name = 'ENSEMBLE'
                print(f"  {model_name:12} | {model_mean:.2f} +/- {model_std:.2f}")
                
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
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 9))
    axes = axes.flatten()
    
    score_columns = ['vader_score', 'flair_score', 'rubert_score', 
                       'roberta_score', 'distilbert_score','logistic_regression_score','svm_score','random_forest_score', 'ensemble_score']
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE194', 
              '#A8E6CF', '#D4A5A5', '#9B59B6', '#3498DB', '#F1C40F']
    
    # Plot 1: Mean score comparison
    ax = axes[0]
    models = []
    means = []
    errors = []
    
    for col in score_columns:
        if col in df.columns:
            model_name = col.replace('_score', '')
            if col == 'ensemble_score':
                model_name = 'ENSEMBLE'
            models.append(model_name)
            means.append(results[col]['mean'])
            errors.append(results[col]['std'] / np.sqrt(len(df[col].dropna())))
    
    models.append('User')
    means.append(results['user_rating']['mean'])
    errors.append(results['user_rating']['std'] / np.sqrt(len(df['rating_numeric'])))
    
    bars = ax.bar(range(len(models)), means, yerr=errors, capsize=5, 
                  color=colors[:len(models)], alpha=0.8, edgecolor='black', linewidth=1)
    ax.axhline(y=results['user_rating']['mean'], color='red', linestyle='--', 
               linewidth=2, label=f"User mean: {results['user_rating']['mean']:.2f}")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45)
    ax.set_ylabel('Mean Score')
    ax.set_title('Model Mean Score Comparison\n(Mathematical Expectation)', fontsize=10, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: MAE comparison
    ax = axes[1]
    models_mae = []
    mae_values = []
    mae_errors = []
    
    for col in score_columns:
        if col in df.columns:
            model_name = col.replace('_score', '')
            if col == 'ensemble_score':
                model_name = 'ENSEMBLE'
            models_mae.append(model_name)
            mae_values.append(deviation_results[col]['mae'])
            mae_errors.append(0)
    
    bars = ax.bar(range(len(models_mae)), mae_values, yerr=mae_errors, capsize=5,
                  color=colors[:len(models_mae)], alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(models_mae)))
    ax.set_xticklabels(models_mae, rotation=45)
    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_title('Model Error Relative to User Ratings', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for i, (bar, mae) in enumerate(zip(bars, mae_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{mae:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Scores by rating groups
    ax = axes[2]
    df['rating_group'] = pd.cut(df['rating_numeric'], 
                                 bins=[0, 2, 3, 5], 
                                 labels=['Low (1-2)', 'Medium (3)', 'High (4-5)'])
    
    x = np.arange(len(df['rating_group'].unique()))
    width = 0.1
    
    for i, (col, color) in enumerate(zip(score_columns, colors[:len(score_columns)])):
        if col in df.columns:
            means = df.groupby('rating_group')[col].mean()
            offset = width * (i - len(score_columns)/2 + 0.5)
            model_name = col.replace('_score', '')
            if col == 'ensemble_score':
                model_name = 'ENSEMBLE'
            bars = ax.bar(x + offset, means, width, label=model_name, 
                         color=color, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Rating Group')
    ax.set_ylabel('Mean Model Score')
    ax.set_title('Model Scores by User Rating Groups', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['rating_group'].unique())
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=3, color='gray', linestyle=':', alpha=0.5)
    
    # Plot 4: Scatter plot (ensemble vs user)
    ax = axes[3]
    
    if 'ensemble_score' in df.columns:
        scatter = ax.scatter(df['rating_numeric'], df['ensemble_score'], 
                             c=df['rating_numeric'], cmap='viridis', 
                             alpha=0.6, edgecolors='black', linewidth=0.5, s=100)
        
        ax.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect Match', alpha=0.7)
        
        z = np.polyfit(df['rating_numeric'], df['ensemble_score'], 1)
        p = np.poly1d(z)
        ax.plot([1, 5], p([1, 5]), 'b-', linewidth=2, 
                label=f'Trend (R^2={np.corrcoef(df["rating_numeric"], df["ensemble_score"])[0,1]**2:.2f})', alpha=0.7)
        
        ax.set_xlabel('User Rating')
        ax.set_ylabel('ENSEMBLE Score')
        ax.set_title('ENSEMBLE vs User Ratings', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='User Rating')
    
    plt.tight_layout()
    plt.savefig('answer/sentiment_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("\n[OK] Dashboard saved as 'answer/sentiment_analysis_dashboard.png'")
    
    return fig

def create_additional_visualizations(df, results):
    # Error distribution and accuracy plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE194', 
              '#A8E6CF', '#D4A5A5', '#9B59B6', '#3498DB', '#F1C40F']

    score_columns = ['vader_score', 'flair_score', 'rubert_score', 
                       'roberta_score', 'distilbert_score','logistic_regression_score',
                       'svm_score','random_forest_score', 'ensemble_score']
    
    # 1. Error distribution
    ax = axes[0]
    for col in score_columns:
        if col in df.columns:
            errors = df[col] - df['rating_numeric']
            model_name = col.replace('_score', '').upper()
            if col == 'ensemble_score':
                model_name = 'ENSEMBLE'
            ax.hist(errors, bins=20, alpha=0.5, label=model_name, density=True)
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Prediction Error (model - user)')
    ax.set_ylabel('Density')
    ax.set_title('Model Error Distributions', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Accuracy by class
    ax = axes[1]
    
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
                model_name = 'ENSEMBLE'
            models_list.append(model_name)
    
    bars = ax.bar(models_list, accuracy_data, color=colors[:len(models_list)], 
                  alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Sentiment Classification Accuracy\n(Category Match)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, acc) in enumerate(zip(bars, accuracy_data)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('answer/additional_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("[OK] Additional analysis saved as 'answer/additional_analysis.png'")

def main():
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS RESULTS VISUALIZATION")
    print("="*60)
    
    # Поиск файла
    file_path = find_result_file()
    
    if file_path is None:
        print("\n[X] Results file not found!")
        print("   Please run the analysis first.")
        return
    
    # Загрузка данных
    df = load_and_prepare_data(file_path)
    
    if df is None or len(df) == 0:
        print("\n[X] No data for analysis")
        return
    
    print(f"\n[OK] Loaded {len(df)} records for analysis")
    
    try:
        # Построение всех графиков
        plot_ensemble_class_distribution(df)
        plot_normality_comparison(df)
        results = calculate_mathematical_expectation(df)
        deviation_results = analyze_deviations(df)
        analyze_by_rating_groups(df)
        create_visualizations(df, results, deviation_results)
        create_additional_visualizations(df, results)
        
        # Сохранение сводки
        if results and deviation_results:
            summary_data = []
            for key in results:
                if key != 'user_rating' and key in deviation_results:
                    model_name = key.replace('_score', '').upper()
                    model_name = model_name.replace('_FOR_ANALYSIS', '')
                    if key == 'ensemble_score':
                        model_name = 'ENSEMBLE'
                    summary_data.append({
                        'Model': model_name,
                        'Mean Score': f"{results[key]['mean']:.3f}",
                        'MAE': f"{deviation_results[key]['mae']:.3f}",
                        'RMSE': f"{deviation_results[key]['rmse']:.3f}"
                    })
            
            if summary_data:
                summary = pd.DataFrame(summary_data)
                summary.to_csv('answer/model_performance_summary.csv', index=False, encoding='utf-8-sig')
                print("\n[OK] Performance summary saved as 'answer/model_performance_summary.csv'")
        
        print("\n" + "="*60)
        print("[OK] Analysis complete! All plots saved in 'answer' folder")
        print("="*60)
        
    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()