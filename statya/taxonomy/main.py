#!/usr/bin/env python3
"""
Программа построения таксономии ошибок ансамблевой модели
анализа тональности русскоязычных отзывов.

Использование:
    python main.py --input data.csv --stage all
    python main.py --input data.csv --stage annotate
    python main.py --input data.csv --stage analyze
"""

import argparse
import sys

from io_utils import read_cases, write_cases
from stages import (
    stage1_filter,
    stage2_cluster,
    stage3_annotate,
    stage4_hierarchy,
    stage5_analysis,
)


def main():
    parser = argparse.ArgumentParser(
        description="Таксономия ошибок ансамблевой модели анализа тональности"
    )
    parser.add_argument(
        "--input",
        default=r"D:\Documents\Visual Studio Code\MainFolder\Diplom\statya\taxonomy\data.csv",
        help="Путь к CSV-файлу с отзывами и предсказаниями "
            "(по умолчанию: D:\\Documents\\...\\taxonomy\\data.csv)",
    )
    parser.add_argument("--stage", default="all",
                        choices=["all", "filter", "cluster",
                                 "annotate", "hierarchy", "analyze"],
                        help="Какой этап выполнить")
    parser.add_argument("--output-dir", default="output",
                        help="Каталог для результатов")
    args = parser.parse_args()

    print(f"Чтение данных из {args.input}...")
    cases = read_cases(args.input)
    total = len(cases)
    print(f"Загружено наблюдений: {total}")

    if total == 0:
        print("Файл пуст или не содержит данных.", file=sys.stderr)
        sys.exit(1)

    # --- Этап 1 ---
    errors, disputes = stage1_filter.run(cases)
    stage1_filter.print_summary(errors, disputes, total)
    write_cases(errors, f"{args.output_dir}/stage1_errors.csv")
    write_cases(disputes, f"{args.output_dir}/stage1_disputes.csv")

    if args.stage == "filter":
        return

    # --- Этап 2 ---
    errors = stage2_cluster.run(errors)
    stage2_cluster.print_summary(errors)
    write_cases(errors, f"{args.output_dir}/stage2_clusters.csv")

    if args.stage == "cluster":
        return

    # --- Этап 3 ---
    errors = stage3_annotate.run(errors)
    write_cases(errors, f"{args.output_dir}/stage3_annotated.csv")

    if args.stage == "annotate":
        return

    # --- Этап 4 ---
    stage4_hierarchy.print_summary(errors)

    if args.stage == "hierarchy":
        return

    # --- Этап 5 ---
    stage5_analysis.run(errors)

    print(f"\nГотово. Результаты в каталоге: {args.output_dir}/")


if __name__ == "__main__":
    main()