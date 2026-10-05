"""Этап 5. Количественный анализ ошибок."""

from collections import defaultdict, Counter
from typing import List, Dict

from models import (
    Case, ANNOTATION_CATEGORIES, MODEL_COLUMNS,
    CATEGORY_TO_FIXABLE,
)
from io_utils import save_json


def run(errors: List[Case]) -> Dict:
    annotated = [c for c in errors if c.annotation_id is not None]
    total = len(errors)
    total_annot = len(annotated)

    by_category = defaultdict(list)
    for c in annotated:
        by_category[c.annotation_id].append(c)

    report = {
        "total_errors": total,
        "annotated": total_annot,
        "categories": {},
    }

    print(f"\n=== Этап 5. Количественный анализ ===")

    for cat_id, name in ANNOTATION_CATEGORIES.items():
        cases = by_category.get(cat_id, [])
        if not cases:
            continue
        share = len(cases) / total_annot * 100
        avg_conf = sum(c.ensemble_confidence or 0 for c in cases) / len(cases)

        culprits = Counter(c.culprit_model for c in cases if c.culprit_model)
        top_culprit = culprits.most_common(1)[0] if culprits else ("—", 0)

        fixable_share = CATEGORY_TO_FIXABLE.get(cat_id, 0.0)
        # уточняем по факту: если модель-виновник — словарная/переводная,
        # оценка устранимости выше
        if top_culprit[0] in ("vader",):
            fixable_share = min(1.0, fixable_share + 0.15)
        if top_culprit[0] in ("roberta", "distilbert") and cat_id == 7:
            fixable_share = 0.9

        report["categories"][cat_id] = {
            "name": name,
            "count": len(cases),
            "share_pct": round(share, 2),
            "avg_ensemble_confidence": round(avg_conf, 3),
            "top_culprit": top_culprit[0],
            "top_culprit_count": top_culprit[1],
            "culprits": dict(culprits),
            "fixable_share": round(fixable_share, 2),
            "fixable_count": round(len(cases) * fixable_share, 1),
        }

        print(f"\n{cat_id}. {name}")
        print(f"   Кейсов:              {len(cases)} ({share:.1f}%)")
        print(f"   Ср. уверенность:     {avg_conf:.2f}")
        print(f"   Модель-виновник:     {top_culprit[0]} ({top_culprit[1]})")
        print(f"   Устранимо:           {fixable_share*100:.0f}% "
              f"(~{len(cases)*fixable_share:.0f} кейсов)")

    # Итоговая сводка
    total_fixable = sum(v["fixable_count"] for v in report["categories"].values())
    report["total_fixable"] = round(total_fixable, 1)
    report["total_fixable_pct"] = round(total_fixable / total_annot * 100, 1) \
        if total_annot else 0

    print(f"\n--- ИТОГ ---")
    print(f"Устранимых ошибок: ~{total_fixable:.0f} "
          f"({report['total_fixable_pct']}% от корпуса ошибок)")

    save_json(report, "output/stage5_report.json")
    return report