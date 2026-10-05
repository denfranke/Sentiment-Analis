"""
Перевод первых 100 строк датасета с русского на английский через Argos Translate.
"""

import os
from pathlib import Path

# === 1. Путь к моделям ДО импорта argostranslate ===
ARGOS_DIR = r"C:\argos-packages"
os.environ["ARGOS_PACKAGES_DIR"] = ARGOS_DIR

import pandas as pd
import argostranslate.package
import argostranslate.translate


# === 2. Установка моделей (если ещё не установлены) ===
def ensure_models_installed():
    installed = argostranslate.package.get_installed_packages()
    pairs = {(p.from_code, p.to_code) for p in installed}
    required = {("ru", "en"), ("en", "ru")}

    if required.issubset(pairs):
        print("Модели уже установлены.")
        return

    for model_file in Path(ARGOS_DIR).glob("*.argosmodel"):
        print(f"Устанавливаю: {model_file.name}")
        argostranslate.package.install_from_path(str(model_file))

    print("Готово. Установленные пакеты:",
          argostranslate.package.get_installed_packages())


# === 3. Перевод ===
def translate_ru_en(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        return argostranslate.translate.translate(text, "ru", "en")
    except Exception as e:
        print(f"[Ошибка перевода] {e}")
        return text  # вернуть оригинал при ошибке


# === 4. Основной сценарий ===
def main():
    ensure_models_installed()

    input_path = Path(r"data\sentiment_dataset_merged_2_shuffled.csv")
    if not input_path.exists():
        print(f"Файл не найден: {input_path.resolve()}")
        return

    # Читаем CSV (первый столбец — label, второй — text)
    df = pd.read_csv(input_path)
    print(f"Всего строк в файле: {len(df)}")
    print(f"Колонки: {list(df.columns)}")

    df100 = df.head(100).copy()

    print("Начинаю перевод первых 100 строк...")
    df100["text_en"] = df100["text"].apply(translate_ru_en)

    output_path = input_path.with_name("sentiment_dataset_first100_translated.csv")
    df100.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Готово. Результат: {output_path.resolve()}")

    # Показать первые 5 строк результата
    print("\nПример результата:")
    for i, row in df100.head(5).iterrows():
        print(f"\n[{row['label']}]")
        print(f"  RU: {row['text'][:120]}...")
        print(f"  EN: {str(row['text_en'])[:120]}...")


if __name__ == "__main__":
    main()