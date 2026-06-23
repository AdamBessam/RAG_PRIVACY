"""
diagnose_utility_nan.py — Inspecte les lignes CR/AR à NaN issues de RAGAS
(data/chatdoctor_eval/utility_chunks/chunk_*.json) et affiche, pour chacune,
la query, la réponse CPB, le nombre de contextes récupérés et la référence
GPT-4o, afin d'identifier la cause réelle (contexte vide, réponse évasive,
référence vide, etc.) plutôt que de l'écraser dans la moyenne.

Usage: python evaluation_chatdoctor/diagnose_utility_nan.py
"""

import json
import math
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "chatdoctor_eval"


def load(name: str):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def main():
    attacks = load("attack_queries.json")
    responses = load("responses.json")
    contexts_per_query = load("contexts.json")
    references = load("reference_responses.json")

    rows = []
    for chunk_path in sorted((DATA_DIR / "utility_chunks").glob("chunk_*.json")):
        rows.extend(json.load(open(chunk_path, encoding="utf-8")))

    for metric in ("CR", "AR"):
        nan_idx = [i for i, r in enumerate(rows) if math.isnan(r[metric])]
        print(f"\n=== {metric}: {len(nan_idx)}/{len(rows)} lignes NaN ===")
        for i in nan_idx:
            print(f"\n--- index {i} ---")
            print("query:     ", attacks[i]["query"][:200])
            print("response:  ", responses[i][:200])
            print("n_contexts:", len(contexts_per_query[i]))
            print("reference: ", references[i][:200])


if __name__ == "__main__":
    main()
