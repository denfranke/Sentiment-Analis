import csv
import sys

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    words = text.split()
    text = ' '.join(words)
    
    allowed_chars = set('abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789 .,!?;:()\'"-')
    text = ''.join(char for char in text if char in allowed_chars)
    
    text = text.strip()
    
    return text

def filter_csv(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile, delimiter=',', quotechar='"', skipinitialspace=True)
        writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        rows_processed = 0
        rows_kept = 0
        invalid_rows = 0
        
        writer.writerow(['label', 'text'])

        for row in reader:
            rows_processed += 1
            
            # Проверяем, что строка не пустая и имеет минимум 2 поля
            if not row or len(row) < 2:
                invalid_rows += 1
                continue

            first_col = row[0].strip().upper()
            
            text = row[1] if len(row) > 1 else ""
            cleaned_text = clean_text(text)
            
            if not cleaned_text:
                invalid_rows += 1
                continue

            if first_col.startswith(('SKIP', 'SPEECH')):
                continue

            if first_col.startswith(('NEGATIVE', 'POSITIVE', 'NEUTRAL')):
                writer.writerow([first_col, cleaned_text])
                rows_kept += 1
            elif first_col == 'LABEL':  
                continue
            else:
                invalid_rows += 1

    print(f"Обработано строк: {rows_processed}, сохранено: {rows_kept}, пропущено: {invalid_rows}")
    return rows_kept

# if __name__ == "__main__":
#     input_filename = "data/rusentiment.csv"
#     output_filename = "data/rusentiment_clean.csv"

#     if len(sys.argv) > 2:
#         input_filename = sys.argv[1]
#         output_filename = sys.argv[2]
#     elif len(sys.argv) > 1:
#         input_filename = sys.argv[1]
#         print(f"Выходной файл не указан, будет использован {output_filename}")

#     filter_csv(input_filename, output_filename)




import csv
import sys

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    words = text.split()
    filtered_words = []
    for word in words:
        if 'http' in word or 'www.' in word or '.com' in word or '.ru' in word:
            continue
        if word.startswith('@'):
            continue
        if word.startswith('#'):
            continue
        filtered_words.append(word)
    
    text = ' '.join(filtered_words)
    
    # Удаляем повторяющиеся знаки препинания
    for punct in ['!', '?', '.']:
        while punct * 3 in text:
            text = text.replace(punct * 3, punct)
        while punct * 2 in text:
            text = text.replace(punct * 2, punct)
    
    allowed_chars = set('abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789 .,!?;:()\'"-')
    text = ''.join(char for char in text if char in allowed_chars)
    
    words = text.split()
    text = ' '.join(words)
    
    return text

def merge_csv_files(file1, file2, output_file):
    all_rows = []
    stats = {'file1': 0, 'file2': 0, 'invalid': 0}
    
    print(f"Чтение файла: {file1}")
    try:
        with open(file1, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"', skipinitialspace=True)
            header = next(reader)  # Пропускаем заголовок
            
            for row in reader:
                if len(row) >= 2:
                    label = row[0].strip().upper()
                    text = row[1].strip()
                    
                    cleaned_text = clean_text(text)
                    
                    word_count = len(cleaned_text.split())
                    if cleaned_text and word_count >= 3:
                        all_rows.append([label, cleaned_text])
                        stats['file1'] += 1
                    else:
                        stats['invalid'] += 1
    except Exception as e:
        print(f"  Ошибка при чтении {file1}: {e}")
    
    print(f"Чтение файла: {file2}")
    try:
        with open(file2, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            header = next(reader)  # Пропускаем заголовок
            
            for row in reader:
                if len(row) >= 2:
                    text = row[0].strip()
                    label_num = row[1].strip()
                    
                    if label_num == '0':
                        label = 'NEUTRAL'
                    elif label_num == '1':
                        label = 'POSITIVE'
                    elif label_num == '2':
                        label = 'NEGATIVE'
                    else:
                        label = label_num.upper()
                    
                    cleaned_text = clean_text(text)
                    
                    word_count = len(cleaned_text.split())
                    if cleaned_text and word_count >= 3:
                        all_rows.append([label, cleaned_text])
                        stats['file2'] += 1
                    else:
                        stats['invalid'] += 1
    except Exception as e:
        print(f"  Ошибка при чтении {file2}: {e}")
    
    print("Удаление дубликатов...")
    unique_rows = {}
    for label, text in all_rows:
        if text not in unique_rows:
            unique_rows[text] = label
    
    print(f"\nЗапись объединенного файла: {output_file}")
    print(f"Статистика:")
    print(f"  Из файла 1: {stats['file1']} строк")
    print(f"  Из файла 2: {stats['file2']} строк")
    print(f"  Пропущено (короткие/пустые): {stats['invalid']}")
    print(f"  После удаления дубликатов: {len(unique_rows)} уникальных строк")
    
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            writer.writerow(['label', 'text'])
            
            for text, label in unique_rows.items():
                writer.writerow([label, text])
                    
    except Exception as e:
        print(f"Ошибка при записи: {e}")

# if __name__ == "__main__":
#     file1 = "data/rusentiment_clean.csv"
#     file2 = "data/sentiment_dataset.csv"
#     output = "data/sentiment_dataset_merged.csv"
    
#     merge_csv_files(file1, file2, output)




# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.utils import resample

# def basic_text_clean(text):
#     if not isinstance(text, str):
#         return ""
    
#     text = text.lower()
#     words = text.split()
#     text = ' '.join(words)
#     return text

# df = pd.read_csv('data/sentiment_dataset_merged.csv', nrows=100000, delimiter=',')

# df['text'] = df['text'].apply(basic_text_clean)
# df = df[df['text'].str.len() > 0]

# print("Уникальные метки в датасете:", df['label'].unique())
# print("Распределение классов:")
# print(df['label'].value_counts())

# # Балансировка классов
# def balance_dataset(df, target_col='label', random_state=42):
#     classes = df[target_col].unique()
#     min_size = df[target_col].value_counts().min()
    
#     # Ограничиваем минимальный размер, чтобы не потерять слишком много данных
#     min_size = max(min_size, 1000)  # минимум 1000 примеров на класс
    
#     balanced_dfs = []
#     for class_name in classes:
#         class_df = df[df[target_col] == class_name]
#         if len(class_df) > min_size:
#             class_df = resample(class_df, 
#                               replace=False, 
#                               n_samples=min_size,
#                               random_state=random_state)
#         balanced_dfs.append(class_df)
    
#     return pd.concat(balanced_dfs, ignore_index=True)

# # Проверяем, нужно ли балансировать
# class_counts = df['label'].value_counts()
# if len(class_counts) > 1 and class_counts.max() / class_counts.min() > 1.5:
#     print("\nДисбаланс классов, выполняем балансировку...")
#     df = balance_dataset(df)
#     print("После балансировки:")
#     print(df['label'].value_counts())

# df_train_val, df_test = train_test_split(
#     df, 
#     test_size=0.15, 
#     random_state=42, 
#     stratify=df['label']
# )

# df_train, df_dev = train_test_split(
#     df_train_val, 
#     test_size=0.176,  # 15% от общего = ~17.6% от train_val
#     random_state=42, 
#     stratify=df_train_val['label']
# )

# print(f"\nРазмер обучающей выборки: {len(df_train)}")
# print(f"Размер валидационной выборки: {len(df_dev)}")
# print(f"Размер тестовой выборки: {len(df_test)}")

# print("\nРаспределение в обучающей выборке:")
# print(df_train['label'].value_counts(normalize=True))

# def save_flair_format(dataframe, filepath, text_col='text', label_col='label'):
#     with open(filepath, 'w', encoding='utf-8') as f:
#         for _, row in dataframe.iterrows():
#             text = str(row[text_col]).replace('\n', ' ').replace('\r', ' ').strip()
#             label = row[label_col]
            
#             word_count = len(text.split())
#             if text and word_count >= 2:
#                 f.write(f"{label}, {text}\n")

# # Создаем директорию, если её нет
# import os
# os.makedirs('data/rusentiment-flair-model/files', exist_ok=True)

# save_flair_format(df_train, 'data/rusentiment-flair-model/files/train.txt')
# save_flair_format(df_dev, 'data/rusentiment-flair-model/files/dev.txt')
# save_flair_format(df_test, 'data/rusentiment-flair-model/files/test.txt')

# # Исправляем пути для чтения файлов
# for filename in ['train.txt', 'dev.txt', 'test.txt']:
#     file_path = f'data/rusentiment-flair-model/files/{filename}'
#     with open(file_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         print(f"{filename}: {len(lines)} строк")





from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
import torch

# 1. Укажите папку с данными
data_folder = 'data/rusentiment-flair-model/files'

# 2. Настройка маппинга колонок
column_name_map = {0: "label", 1: "text"}

# 3. Загружаем корпус
corpus: Corpus = CSVClassificationCorpus(
    data_folder,
    column_name_map,
    label_type='sentiment',
    skip_header=False,
    delimiter=',',
    encoding='utf-8'
)

# 4. Создаем словарь меток
label_dict = corpus.make_label_dictionary(label_type='sentiment')

print(f"Метки в датасете: {label_dict.get_items()}")
print(f"Количество примеров в обучающей выборке: {len(corpus.train)}")
print(f"Количество примеров в валидационной выборке: {len(corpus.dev)}")
print(f"Количество примеров в тестовой выборке: {len(corpus.test)}")

# 5. Создаем эмбеддинги
print("Загрузка эмбеддингов...")
flair_embeddings_forward = FlairEmbeddings('multi-forward')
flair_embeddings_backward = FlairEmbeddings('multi-backward')

document_embeddings = DocumentRNNEmbeddings(
    embeddings=[flair_embeddings_forward, flair_embeddings_backward],
    hidden_size=512,
    reproject_words=True,
    reproject_words_dimension=256,
    bidirectional=True
)

# 6. Создаем классификатор
classifier = TextClassifier(
    document_embeddings, 
    label_dictionary=label_dict,
    label_type='sentiment',
    multi_label=False
)

# 7. Обучаем модель 
trainer = ModelTrainer(classifier, corpus)

# # Минимальный рабочий код для обучения Flair
# trainer.train(
#     base_path='data/rusentiment-flair-model',
#     learning_rate=0.01,
#     mini_batch_size=32,
#     max_epochs=10,
#     train_with_dev=True,
#     optimizer=torch.optim.SGD,  # Только класс оптимизатора
#     save_final_model=True
# )

# С Adam оптимизатором (часто работает лучше)
trainer.train(
    base_path='data/rusentiment-flair-model',
    learning_rate=5e-5,
    mini_batch_size=32,
    max_epochs=20,
    train_with_dev=True,
    
    # Adam оптимизатор
    optimizer=torch.optim.AdamW,
    weight_decay=1e-5,
    betas=(0.9, 0.999),
    
    # Параметры обучения
    anneal_factor=0.5,
    patience=3,
    min_learning_rate=1e-6,
    save_final_model=True,
    embeddings_storage_mode='gpu',
    shuffle=True
)

print("Обучение завершено!")


