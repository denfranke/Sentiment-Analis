"""Этап 2. Автоматическая кластеризация ошибок."""

from typing import List, Dict

from models import Case, MODEL_COLUMNS


def _confidence_bin(v: float) -> str:
    if v < 0.6:
        return "low"
    if v < 0.85:
        return "medium"
    return "high"


def _agreement(c: Case) -> str:
    counts: Dict[str, int] = {}
    for m in MODEL_COLUMNS:
        s = c.model_sentiments.get(m)
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return "unknown"
    top = max(counts.values())
    if top == len(MODEL_COLUMNS):
        return "unanimous"     # единогласие
    if top > len(MODEL_COLUMNS) / 2:
        return "majority"      # большинство
    return "split"             # раскол


def run(errors: List[Case]) -> List[Case]:
    for c in errors:
        conf = c.ensemble_confidence or 0.0
        c.agreement = _agreement(c)
        c.cluster = "|".join([
            c.error_direction or "unknown",
            _confidence_bin(conf),
            c.agreement,
        ])
    return errors


def print_summary(errors: List[Case]):
    print(f"\n=== Этап 2. Автоматическая кластеризация ===")
    clusters: Dict[str, int] = {}
    for c in errors:
        clusters[c.cluster] = clusters.get(c.cluster, 0) + 1

    print(f"Всего кластеров: {len(clusters)}")
    print("Топ-10 кластеров:")
    for cl, n in sorted(clusters.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cl:55s} {n:4d}")