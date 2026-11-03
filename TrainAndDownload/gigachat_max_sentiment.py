#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Разметка тональности через GigaChat 2 Max.
Входные данные и параметры прописаны в блоке КОНФИГУРАЦИЯ ниже.
"""

import os
import sys
import asyncio
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai_gigachat import GigaChatModel, GigaChatProvider

load_dotenv()

# ═════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ — здесь всё прописано жёстко
# ═════════════════════════════════════════════════════════════

# Пути (относительно папки Diplom, где лежит скрипт)
# Папка, где лежит сам скрипт (TrainAndDownload)
SCRIPT_DIR = Path(__file__).resolve().parent

# Папка проекта (Diplom) — на уровень выше
PROJECT_ROOT = SCRIPT_DIR.parent

# Пути от корня проекта
INPUT_FILE  = PROJECT_ROOT / "data" / "15k_ozon.csv"
OUTPUT_FILE = PROJECT_ROOT / "answer" / "15k_ozon_updated.csv"

# Колонки в CSV
TEXT_COLUMN = "text"

# Ограничение количества строк для теста (None = весь файл)
MAX_ROWS = None          # например: 10 для быстрой проверки

# Модель GigaChat
GIGACHAT_MODEL = "GigaChat-2-Max"
VERIFY_SSL     = False   # ⚠ для продакшена замените на ca_bundle_file

# Параметры прогона
CHECKPOINT_EVERY = 100   # сохранять промежуточный результат каждые N строк
CONCURRENCY      = 1     
SLEEP_BETWEEN    = 1.0   

# Максимальная длина текста (обрезаем, чтобы не улететь в лимит токенов)
MAX_TEXT_LEN = 4000

# Системный промпт
SYSTEM_PROMPT = (
    "Ты — эксперт по анализу тональности русскоязычных текстов. "
    "Определи тональность отзыва или комментария. "
    "Ответь СТРОГО одним словом из списка: POSITIVE, NEGATIVE, NEUTRAL. "
    "Без пояснений, без кавычек, без точек."
)

ALLOWED = {"POSITIVE", "NEGATIVE", "NEUTRAL"}

# ═════════════════════════════════════════════════════════════
# Вспомогательные функции
# ═════════════════════════════════════════════════════════════

def normalize_label(raw: str) -> str:
    if not raw:
        return "NEUTRAL"
    s = raw.strip().upper().replace(".", "").replace(",", "").replace('"', "").replace("'", "")
    for label in ALLOWED:
        if label in s:
            return label
    return "NEUTRAL"


async def ask_gigachat(agent: Agent, text: str, retries: int = 5) -> str:
    if not isinstance(text, str) or not text.strip():
        return "NEUTRAL"

    text = text.strip()
    if len(text) > MAX_TEXT_LEN:
        text = text[:MAX_TEXT_LEN]

    for attempt in range(retries):
        try:
            result = await agent.run(text)
            return normalize_label(result.output)
        except Exception as e:
            wait = 2 ** attempt
            tqdm.write(f"[retry {attempt+1}/{retries}] {e} (ждём {wait}s)")
            await asyncio.sleep(wait)
    return "NEUTRAL"


# ═════════════════════════════════════════════════════════════
# Основной сценарий
# ═════════════════════════════════════════════════════════════

async def run():
    if not INPUT_FILE.exists():
        sys.exit(f"Файл не найден: {INPUT_FILE.resolve()}")

    if not os.getenv("GIGACHAT_CREDENTIALS"):
        sys.exit("Не задан GIGACHAT_CREDENTIALS в .env")

    df = pd.read_csv(
        INPUT_FILE,
        engine="python",
        on_bad_lines="skip",          # пропускать битые строки
        encoding="utf-8-sig",
    )

    if TEXT_COLUMN not in df.columns:
        sys.exit(f"Нет колонки '{TEXT_COLUMN}' в {INPUT_FILE}")

    # Сохраняем исходный label для последующего сравнения
    df["label_original"] = df["label"]
    df["label"] = None

    if MAX_ROWS:
        df = df.head(MAX_ROWS).copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Провайдер и агент GigaChat 2 Max
    provider = GigaChatProvider(
        authorization_key=os.getenv("GIGACHAT_CREDENTIALS"),
        verify_ssl=VERIFY_SSL,
    )
    model = GigaChatModel(GIGACHAT_MODEL, provider=provider)
    agent = Agent(model, system_prompt=SYSTEM_PROMPT)

    # Resume — если файл уже частично размечен
    start_idx = 0
    if OUTPUT_FILE.exists():
        try:
            prev = pd.read_csv(OUTPUT_FILE)
            if "label" in prev.columns and len(prev) >= len(df):
                done = prev["label"].notna().sum()
                if done > 0:
                    df["label"] = prev["label"].values[:len(df)]
                    start_idx = done
                    print(f"[resume] уже размечено {done} строк, продолжаем с {done}")
        except Exception:
            pass

    sem = asyncio.Semaphore(CONCURRENCY)

    async def process_one(i: int, text: str) -> None:
        async with sem:
            label = await ask_gigachat(agent, text)
            df.at[i, "label"] = label
            await asyncio.sleep(SLEEP_BETWEEN)

    try:
        tasks = []
        for i, row in df.iterrows():
            if i < start_idx:
                continue
            if pd.notna(df.at[i, "label"]):
                continue
            tasks.append(process_one(i, row[TEXT_COLUMN]))

            if len(tasks) >= CHECKPOINT_EVERY:
                await asyncio.gather(*tasks)
                tasks = []
                df[["label", "text"]].to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
                tqdm.write(f"[checkpoint] {df['label'].notna().sum()}/{len(df)}")

        if tasks:
            await asyncio.gather(*tasks)

    finally:
        await provider.aclose()

    # Финальное сохранение — структура label,text как в исходнике
    df[["label", "text"]].to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # Сводка
    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)
    print("\nРаспределение предсказаний GigaChat 2 Max:")
    print(df["label"].value_counts(dropna=False).to_string())

    valid = df[df["label"].notna()].copy()
    if len(valid) > 0:
        acc = (valid["label"].str.upper() == valid["label_original"].str.upper()).mean()
        print(f"\nСовпадение с исходной разметкой: {acc:.2%} ({len(valid)} строк)")

    print(f"\nСохранено: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    asyncio.run(run())