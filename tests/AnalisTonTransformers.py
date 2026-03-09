import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. Загрузка данных из CSV
def load_data(csv_path, text_col='text', label_col=None):
    """
    Загружает данные из CSV.
    :param csv_path: путь к CSV-файлу
    :param text_col: название столбца с текстом
    :param label_col: название столбца с истинными метками (если есть)
    :return: DataFrame с текстами и метками (если есть)
    """
    df = pd.read_csv(csv_path, nrows=1000)
    
    if text_col not in df.columns:
        raise ValueError(f"Столбец '{text_col}' не найден в CSV")
    
    # Оставляем только нужные столбцы
    cols = [text_col]
    if label_col and label_col in df.columns:
        cols.append(label_col)
    
    return df[cols].dropna(subset=[text_col])

# 2. Модель для тональности (русский язык)
class RussianSentimentAnalyzer:
    def __init__(self, model_name="cointegrated/rubert-tone-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def predict(self, texts):
        predictions = []
        
        for text in texts:
            # Токенизация
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs.to(self.device)
            
            # Инференс
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).item()
            
            # Декодирование класса
            if predicted_class == 0:
                label = "negative"
            elif predicted_class == 1:
                label = "neutral"
            else:
                label = "positive"
            
            predictions.append(label)
        
        return predictions

# 3. Оценка качества (если есть истинные метки)
def evaluate_predictions(y_true, y_pred):
    """
    Выводит метрики качества.
    :param y_true: истинные метки
    :param y_pred: предсказанные метки
    """
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

def main(csv_path, text_col='text', label_col=None, output_path=None):
    print("Загружаем данные...")
    df = load_data(csv_path, text_col, label_col)
    texts = df[text_col].tolist()
    
    print("Инициализируем модель...")
    analyzer = RussianSentimentAnalyzer()
    
    print("Анализируем тональность...")
    predictions = analyzer.predict(texts)
    df['predicted_sentiment'] = predictions
    
    # Оценка качества (если есть метки)
    if label_col and label_col in df.columns:
        y_true = df[label_col].tolist()
        y_pred = df['predicted_sentiment'].tolist()
        evaluate_predictions(y_true, y_pred)
    else:
        print("\nИстинные метки не предоставлены — оценка качества невозможна.")
    
    if output_path:
        print(f"Сохраняем результаты в {output_path}...")
        df.to_csv(output_path, index=False)
    
    return df

if __name__ == "__main__":
    CSV_PATH = "отзывы.csv"           # путь к вашему CSV
    TEXT_COL = "text"                 # столбец с текстами
    LABEL_COL = "rating"          # столбец с истинными метками (опционально)
    OUTPUT_PATH = "отзывы(результат работы transformers).csv"     # куда сохранить результаты
    
    results_df = main(
        csv_path=CSV_PATH,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        output_path=OUTPUT_PATH
    )
    
    print(results_df.all)
