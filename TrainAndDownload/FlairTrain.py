# import csv
# import sys

# def clean_text(text):
#     if not isinstance(text, str):
#         return ""
    
#     text = text.lower()
    
#     words = text.split()
#     text = ' '.join(words)
    
#     allowed_chars = set('abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789 .,!?;:()\'"-')
#     text = ''.join(char for char in text if char in allowed_chars)
    
#     text = text.strip()
    
#     return text

# def filter_csv(input_file, output_file):
    # with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
    #      open(output_file, 'w', newline='', encoding='utf-8') as outfile:

    #     reader = csv.reader(infile, delimiter=',', quotechar='"', skipinitialspace=True)
    #     writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    #     rows_processed = 0
    #     rows_kept = 0
    #     invalid_rows = 0
        
    #     writer.writerow(['label', 'text'])

    #     for row in reader:
    #         rows_processed += 1
            
    #         # Проверяем, что строка не пустая и имеет минимум 2 поля
    #         if not row or len(row) < 2:
    #             invalid_rows += 1
    #             continue

    #         first_col = row[0].strip().upper()
            
    #         text = row[1] if len(row) > 1 else ""
    #         cleaned_text = clean_text(text)
            
    #         if not cleaned_text:
    #             invalid_rows += 1
    #             continue

    #         if first_col.startswith(('SKIP', 'SPEECH')):
    #             continue

    #         if first_col.startswith(('NEGATIVE', 'POSITIVE', 'NEUTRAL')):
    #             writer.writerow([first_col, cleaned_text])
    #             rows_kept += 1
    #         elif first_col == 'LABEL':  
    #             continue
    #         else:
    #             invalid_rows += 1

    # print(f"Обработано строк: {rows_processed}, сохранено: {rows_kept}, пропущено: {invalid_rows}")
    # return rows_kept

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




# import csv
# import sys

# MAX_FIELD_SIZE = 10 * 1024 * 1024  # 10 МБ - достаточно для большинства текстовых данных

# try:
#     csv.field_size_limit(MAX_FIELD_SIZE)
# except OverflowError:
#     # Если 10 МБ слишком много, пробуем меньшее значение
#     MAX_FIELD_SIZE = 5 * 1024 * 1024  # 5 МБ
#     try:
#         csv.field_size_limit(MAX_FIELD_SIZE)
#     except OverflowError:
#         MAX_FIELD_SIZE = 1 * 1024 * 1024  # 1 МБ
#         csv.field_size_limit(MAX_FIELD_SIZE)

# print(f"Установлен лимит размера поля CSV: {MAX_FIELD_SIZE // 1024} КБ")

# def clean_text(text):
#     if not isinstance(text, str):
#         return ""
    
#     text = text.lower()
    
#     words = text.split()
#     filtered_words = []
#     for word in words:
#         if 'http' in word or 'www.' in word or '.com' in word or '.ru' in word:
#             continue
#         if word.startswith('@'):
#             continue
#         if word.startswith('#'):
#             continue
#         filtered_words.append(word)
    
#     text = ' '.join(filtered_words)
    
#     # Удаляем повторяющиеся знаки препинания
#     for punct in ['!', '?', '.']:
#         while punct * 3 in text:
#             text = text.replace(punct * 3, punct)
#         while punct * 2 in text:
#             text = text.replace(punct * 2, punct)
    
#     allowed_chars = set('abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789 .,!?;:()\'"-')
#     text = ''.join(char for char in text if char in allowed_chars)
    
#     words = text.split()
#     text = ' '.join(words)
    
#     return text

# def merge_csv_files(file1, file2, output_file):
#     all_rows = []
#     stats = {'file1': 0, 'file2': 0, 'invalid': 0}
    
#     print(f"Чтение файла: {file1}")
#     try:
#         with open(file1, 'r', encoding='utf-8', newline='') as f:
#             reader = csv.reader(f, delimiter=',', quotechar='"', skipinitialspace=True)
#             header = next(reader)  # Пропускаем заголовок
            
#             for i, row in enumerate(reader, 1):
#                 if len(row) >= 2:
#                     label = row[0].strip().upper()
#                     text = row[1].strip()
                    
#                     cleaned_text = clean_text(text)
                    
#                     word_count = len(cleaned_text.split())
#                     if cleaned_text and word_count >= 3:
#                         all_rows.append([label, cleaned_text])
#                         stats['file1'] += 1
#                     else:
#                         stats['invalid'] += 1
#                 else:
#                     print(f"  Предупреждение: строка {i} имеет {len(row)} полей, ожидалось >=2")
#                     stats['invalid'] += 1
                    
#             print(f"  Прочитано {stats['file1']} валидных строк из {i}")
                    
#     except Exception as e:
#         print(f"  Ошибка при чтении {file1}: {e}")
#         import traceback
#         traceback.print_exc()
    
#     print(f"Чтение файла: {file2}")
#     try:
#         with open(file2, 'r', encoding='utf-8', newline='') as f:
#             reader = csv.reader(f, delimiter=',', quotechar='"')
#             header = next(reader)  # Пропускаем заголовок
            
#             for i, row in enumerate(reader, 1):
#                 if len(row) >= 3:
#                     text = row[1].strip()
#                     label_num = row[2].strip()
                    
#                     if label_num == '0':
#                         label = 'NEUTRAL'
#                     elif label_num == '1':
#                         label = 'POSITIVE'
#                     elif label_num == '2':
#                         label = 'NEGATIVE'
#                     else:
#                         label = label_num.upper()
                    
#                     cleaned_text = clean_text(text)
                    
#                     word_count = len(cleaned_text.split())
#                     if cleaned_text and word_count >= 3:
#                         all_rows.append([label, cleaned_text])
#                         stats['file2'] += 1
#                     else:
#                         stats['invalid'] += 1
#                 else:
#                     print(f"  Предупреждение: строка {i} имеет {len(row)} полей, ожидалось >=3")
#                     stats['invalid'] += 1
                    
#             print(f"  Прочитано {stats['file2']} валидных строк из {i}")
                    
#     except Exception as e:
#         print(f"  Ошибка при чтении {file2}: {e}")
#         import traceback
#         traceback.print_exc()
    
#     print("Удаление дубликатов...")
#     unique_rows = {}
#     for label, text in all_rows:
#         if text not in unique_rows:
#             unique_rows[text] = label
    
#     print(f"\nЗапись объединенного файла: {output_file}")
#     print(f"Статистика:")
#     print(f"  Из файла 1: {stats['file1']} строк")
#     print(f"  Из файла 2: {stats['file2']} строк")
#     print(f"  Пропущено (короткие/пустые/ошибки): {stats['invalid']}")
#     print(f"  После удаления дубликатов: {len(unique_rows)} уникальных строк")
    
#     try:
#         with open(output_file, 'w', encoding='utf-8', newline='') as f:
#             writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
#             writer.writerow(['label', 'text'])
            
#             for text, label in unique_rows.items():
#                 writer.writerow([label, text])
        
#         print(f"  Успешно записано {len(unique_rows)} строк")
                    
#     except Exception as e:
#         print(f"Ошибка при записи: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     file1 = "data/rusentiment_clean.csv"
#     file2 = "data/sentiment_dataset.csv" # !!!поменять значения столбцов обработки!!!
#     output = "data/sentiment_dataset_merged_2.csv"
    
#     merge_csv_files(file1, file2, output)
# if __name__ == "__main__":
#     file1 = "data/sentiment_dataset_merged.csv"
#     file2 = "data/datasets.csv"  # !!!поменять значения столбцов обработки!!!
#     output = "data/sentiment_dataset_merged_3.csv"
    
#     merge_csv_files(file1, file2, output)











# import pandas as pd
# import numpy as np

# # Перемешиваем строки и сохраняем в новый файл
# print("Чтение файла...")
# df = pd.read_csv('data/sentiment_dataset_merged_2.csv')

# print(f"Исходное количество строк: {len(df)}")

# # Перемешиваем строки
# df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# # Сохраняем в новый файл (оригинал остается нетронутым)
# df_shuffled.to_csv('data/sentiment_dataset_merged_2_shuffled.csv', index=False)

# print("Готово! Перемешали файл")










# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.utils import resample
# import os

# def basic_text_clean(text):
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     words = text.split()
#     text = ' '.join(words)
#     return text

# def save_flair_format(dataframe, filepath, text_col='text', label_col='label'):
#     """Сохранение с фильтрацией пустых и коротких текстов"""
#     with open(filepath, 'w', encoding='utf-8') as f:
#         valid_count = 0
#         skipped_count = 0
#         for _, row in dataframe.iterrows():
#             text = str(row[text_col]).replace('\n', ' ').replace('\r', ' ').strip()
#             label = row[label_col]
            
#             # Фильтрация: минимум 3 слова и не пустой
#             word_count = len(text.split())
#             if text and word_count >= 3:  # Минимум 3 слова
#                 f.write(f"{label},{text}\n")
#                 valid_count += 1
#             else:
#                 skipped_count += 1
#                 if skipped_count <= 5:  # Показываем первые 5 пропущенных
#                     print(f"Пропущен текст (слов: {word_count}): '{text[:50]}...'")
        
#         print(f"Сохранено {valid_count} из {len(dataframe)} примеров (пропущено {skipped_count})")

# # Загрузка данных
# print("Загрузка данных...")
# df = pd.read_csv('data/sentiment_dataset_merged_2_shuffled.csv', nrows=100000, delimiter=',')

# # Очистка текстов
# print("Очистка текстов...")
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

# # Разделение на train/val/test
# print("\nРазделение данных...")
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

# # Создаем директорию
# os.makedirs('data/rusentiment-flair-model/files', exist_ok=True)

# # Сохраняем файлы с фильтрацией
# print("\nСохранение файлов...")
# save_flair_format(df_train, 'data/rusentiment-flair-model/files/train.csv')
# save_flair_format(df_dev, 'data/rusentiment-flair-model/files/dev.csv')
# save_flair_format(df_test, 'data/rusentiment-flair-model/files/test.csv')

# # Проверяем сохраненные файлы
# print("\nПроверка сохраненных файлов:")
# for filename in ['train.csv', 'dev.csv', 'test.csv']:
#     file_path = f'data/rusentiment-flair-model/files/{filename}'
#     with open(file_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         print(f"{filename}: {len(lines)} строк")
        
#         # Проверяем первые 3 строки
#         if lines:
#             print(f"  Пример: {lines[0].strip()[:100]}")

# print("\n✅ Данные успешно подготовлены!")













from flair.data import Corpus, Sentence
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
import torch
from torch.utils.data import Subset

# 1. Укажите папку с данными
data_folder = 'data/rusentiment-flair-model/files'

# 2. Настройка маппинга колонок
column_name_map = {0: "label", 1: "text"}

# 3. Загружаем корпус
print("Загрузка корпуса...")
corpus: Corpus = CSVClassificationCorpus(
    data_folder,
    column_name_map,
    label_type='sentiment',
    skip_header=False,
    delimiter=',',
    encoding='utf-8'
)

# 4. АГРЕССИВНАЯ ФИЛЬТРАЦИЯ: удаляем все пустые предложения
print("\nАгрессивная фильтрация пустых предложений...")

def filter_empty_sentences_from_corpus(sentences_list):
    """Фильтрует пустые предложения и возвращает список индексов валидных"""
    valid_indices = []
    empty_count = 0
    
    for i, sentence in enumerate(sentences_list):
        # Проверяем наличие токенов
        if len(sentence.tokens) > 0:
            # Дополнительная проверка: есть ли хоть один текстовый токен
            has_text = any(token.text.strip() for token in sentence.tokens)
            if has_text:
                valid_indices.append(i)
            else:
                empty_count += 1
        else:
            empty_count += 1
    
    return valid_indices, empty_count

# Создаем подмножества без пустых предложений
train_indices, train_empty = filter_empty_sentences_from_corpus(corpus.train)
dev_indices, dev_empty = filter_empty_sentences_from_corpus(corpus.dev)
test_indices, test_empty = filter_empty_sentences_from_corpus(corpus.test)

print(f"Обучающая: {len(corpus.train)} -> {len(train_indices)} (удалено {train_empty})")
print(f"Валидационная: {len(corpus.dev)} -> {len(dev_indices)} (удалено {dev_empty})")
print(f"Тестовая: {len(corpus.test)} -> {len(test_indices)} (удалено {test_empty})")

# Создаем отфильтрованные подмножества
from torch.utils.data import Subset
train_filtered = Subset(corpus.train, train_indices)
dev_filtered = Subset(corpus.dev, dev_indices)
test_filtered = Subset(corpus.test, test_indices)

# 5. Создаем новый корпус с отфильтрованными данными
from flair.data import Corpus as FlairCorpus

filtered_corpus = FlairCorpus(
    train=train_filtered,
    dev=dev_filtered,
    test=test_filtered,
    name="filtered_corpus"
)

# 6. Создаем словарь меток из отфильтрованного корпуса
label_dict = filtered_corpus.make_label_dictionary(label_type='sentiment')

print(f"\nМетки в датасете: {label_dict.get_items()}")
print(f"Размер словаря: {len(label_dict)}")

# 7. Создаем эмбеддинги
print("\nЗагрузка эмбеддингов...")
flair_embeddings_forward = FlairEmbeddings('multi-forward')
flair_embeddings_backward = FlairEmbeddings('multi-backward')

document_embeddings = DocumentRNNEmbeddings(
    embeddings=[flair_embeddings_forward, flair_embeddings_backward],
    hidden_size=128,
    reproject_words=True,
    reproject_words_dimension=64,
    bidirectional=True
)

# 8. Создаем классификатор
classifier = TextClassifier(
    document_embeddings, 
    label_dictionary=label_dict,
    label_type='sentiment',
    multi_label=False
)

# 9. Обучение с отфильтрованным корпусом
print("\nНачало обучения...")
trainer = ModelTrainer(classifier, filtered_corpus)

'''хотя бы запускается loss=12'''
# trainer.train(
#     base_path='data/rusentiment-flair-model',
#     learning_rate=0.1,
#     mini_batch_size=8,
#     max_epochs=5,
#     optimizer=torch.optim.SGD,
#     momentum=0.9,
#     weight_decay=1e-5,
#     embeddings_storage_mode='cpu',
#     shuffle=True,
#     patience=5,
#     min_learning_rate=1e-8,
#     save_final_model=True,
# )

'''запуск lost= 1.09 55.5%'''
trainer.train(
    base_path='data/rusentiment-flair-model',
    learning_rate=1e-4,  # Маленькая LR вместо 0.1
    mini_batch_size=8,
    max_epochs=5,
    
    # AdamW оптимизатор (лучше для сходимости)
    optimizer=torch.optim.AdamW,
    weight_decay=1e-5,
    betas=(0.9, 0.999),
    
    embeddings_storage_mode='cpu',
    shuffle=True,
    save_final_model=True,
)

print("\n✅ Обучение завершено!")


