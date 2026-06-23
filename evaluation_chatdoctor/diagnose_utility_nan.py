"""
diagnose_utility_nan.py — Inspecte les lignes CR/AR à NaN issues de RAGAS
(data/chatdoctor_eval/utility_chunks/chunk_*.json).

Mode par défaut : résumé compact (indices NaN, taille de chunk associée,
nombre de contextes vides) pour confirmer la cause sans noyer la sortie.
--detail affiche le détail (query/réponse/contexte/référence) par métrique.

Usage:
  python evaluation_chatdoctor/diagnose_utility_nan.py
  python evaluation_chatdoctor/diagnose_utility_nan.py --detail CR,AR
"""

import argparse
import json
import math
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "chatdoctor_eval"
CHUNK_SIZE = 25


def load(name: str):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail", default="", help="metrics à détailler, ex: CR,AR")
    args = parser.parse_args()
    detail_metrics = {m.strip().upper() for m in args.detail.split(",") if m.strip()}

    attacks = load("attack_queries.json")
    responses = load("responses.json")
    contexts_per_query = load("contexts.json")
    references = load("reference_responses.json")

    rows = []
    for chunk_path in sorted((DATA_DIR / "utility_chunks").glob("chunk_*.json")):
        rows.extend(json.load(open(chunk_path, encoding="utf-8")))

    for metric in ("CR", "AR"):
        nan_idx = [i for i, r in enumerate(rows) if math.isnan(r[metric])]
        chunks_hit = sorted({i // CHUNK_SIZE for i in nan_idx})
        n_empty_ctx = sum(1 for i in nan_idx if len(contexts_per_query[i]) == 0)

        print(f"\n=== {metric}: {len(nan_idx)}/{len(rows)} lignes NaN ===")
        print("indices:    ", nan_idx)
        print("chunks touchés (0-indexed, taille 25):", chunks_hit)
        print(f"dont contextes vides: {n_empty_ctx}/{len(nan_idx)}")

        if metric in detail_metrics:
            for i in nan_idx:
                print(f"\n--- index {i} ---")
                print("query:     ", attacks[i]["query"][:200])
                print("response:  ", responses[i][:200])
                print("n_contexts:", len(contexts_per_query[i]))
                print("reference: ", references[i][:200])


if __name__ == "__main__":
    main()
