# import argostranslate.package

# # Замените путь на реальный путь к вашему .argosmodel файлу
# path_to_model = r"D:\Documents\Visual Studio Code\MainFolder\Diplom\TrainAndDownload\book\translate-en_ru-1_7.argosmodel"

# argostranslate.package.install_from_path(path_to_model)



# import argostranslate.translate
# import argostranslate.package

# # Убедимся, что v1.7 нет, а v1.9 есть
# for p in argostranslate.package.get_installed_packages():
#     print(f"{p.from_code}->{p.to_code}, v{p.package_version}, {p.package_path}")

# # Прямой тест
# print(argostranslate.translate.translate("This book is quite good.", "en", "ru"))



import pandas as pd
from pathlib import Path

# Настройки: путь к папке с датасетами и соответствие файл -> домен
DATA_DIR = Path(r"TrainAndDownload/book/add3column/data")

FILES = {
    "phone_dataset_15k.csv": "phones",
    "movie_dataset_1k.csv": "movies",
    "sentiment_dataset_merged_2_shuffled.csv":   "apps",
    "book_dataset_15k_ru.csv":  "books",
}

def add_domain_column(file_path: Path, domain: str, has_header: bool = False) -> pd.DataFrame:
    """Читает CSV, добавляет колонку domain, возвращает DataFrame."""
    df = pd.read_csv(
        file_path,
        header=0 if has_header else None,
        names=None if has_header else ["label", "text"],
        encoding="utf-8",
    )

    # На случай, если колонки называются иначе — принудительно переименуем первые две
    if not has_header:
        df.columns = ["label", "text"]

    # Нормализуем метки и текст
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df["domain"] = domain

    # Убираем пустые строки
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"] != ""]

    return df[["text", "label", "domain"]]


def main():
    all_dfs = []
    for filename, domain in FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"[!] Файл не найден: {path}")
            continue

        df = add_domain_column(path, domain)
        out_path = DATA_DIR / f"{path.stem}_with_domain.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")

        print(f"[+] {filename}: {len(df)} строк -> {out_path.name}")
        print(f"    метки: {df['label'].value_counts().to_dict()}")
        all_dfs.append(df)

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        merged_path = DATA_DIR / "merged_all_domains.csv"
        merged.to_csv(merged_path, index=False, encoding="utf-8")

        print(f"\n[=] Объединённый файл: {merged_path} ({len(merged)} строк)")
        print("\nРаспределение по доменам:")
        print(merged["domain"].value_counts())
        print("\nРаспределение по домену и метке:")
        print(merged.groupby(["domain", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()