"""Этап 4. Построение иерархической таксономии."""

from collections import defaultdict
from typing import List, Dict

from models import Case, ANNOTATION_CATEGORIES


def build(errors: List[Case]) -> Dict:
    """
    Строит дерево:
      divergence -> category -> cause -> список row_id
    """
    annotated = [c for c in errors if c.annotation_id is not None]
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for c in annotated:
        tree[c.divergence_level][c.annotation_id][c.cause].append(c.row_id)
    return tree


def print_summary(errors: List[Case]):
    annotated = [c for c in errors if c.annotation_id is not None]
    print(f"\n=== Этап 4. Иерархия таксономии ===")
    print(f"Размечено кейсов: {len(annotated)} из {len(errors)}")

    tree = build(errors)
    for div in sorted(tree.keys()):
        print(f"\n[{div}]")
        for cat_id in sorted(tree[div].keys()):
            cat_name = ANNOTATION_CATEGORIES.get(cat_id, "?")
            total_cat = sum(len(v) for v in tree[div][cat_id].values())
            print(f"  {cat_id}. {cat_name}  ({total_cat})")
            for cause, ids in tree[div][cat_id].items():
                print(f"      └─ {cause}: {len(ids)}")