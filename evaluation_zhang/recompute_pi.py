"""
recompute_pi.py — Recalcule uniquement le score PI et met à jour pi_scores.json.

Usage:
    python recompute_pi.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data" / "zhang_eval"


def main():
    # Charger les données nécessaires
    with open(DATA_DIR / "doc_index.json", encoding="utf-8") as f:
        doc_index = json.load(f)

    with open(DATA_DIR / "attack_queries.json", encoding="utf-8") as f:
        attacks = json.load(f)

    with open(DATA_DIR / "responses.json", encoding="utf-8") as f:
        responses = json.load(f)

    print(f"{len(attacks)} queries, {len(responses)} responses\n")

    from metric_pi import PIMetric

    pi_metric = PIMetric()
    pi_metric.build_claims_db(doc_index)

    print("Computing PI scores...")
    pi_scores = pi_metric.compute_pi_batch(responses, attacks, verbose=True)
    pi_score = PIMetric.aggregate_pi(pi_scores)
    print(f"\nPI = {pi_score:.4f}")

    # Sauvegarder le cache
    pi_cache_path = DATA_DIR / "pi_scores.json"
    with open(pi_cache_path, "w", encoding="utf-8") as f:
        json.dump(pi_scores, f, ensure_ascii=False, indent=2)
    print(f"Cache saved → {pi_cache_path}")


if __name__ == "__main__":
    main()
