"""
run_selective_masking_cedh.py — Test du masquage sélectif (CPB v5) sur CEDH.

Balaye le curseur `mask_min_weight` et compare, à retrieval/bootstrap identiques,
l'effet du masquage sélectif sur la privacy (PII) et l'utilité (QS/AR/RL/EM).
Génération Llama LOCALE + métriques LOCALES → AUCUN token OpenAI.

mask_min_weight = 0.0  ≡ baseline v4 (tout est masqué).
Plus le seuil monte, moins on masque (on épargne les types à faible poids), donc
l'utilité doit remonter — on vérifie que la fuite PII ne remonte pas trop.

Efficacité : un SEUL bootstrap B0 (donc un seul jeu de category_hints) ; on ne
change que l'attribut mask_min_weight entre les balayages, on ne recrée pas la
contre-mesure. Seule la génération est refaite par seuil (le masquage change les
chunks vus par le LLM).

Usage (depuis la racine du repo) :
  python countermeasure_v5/run_selective_masking_cedh.py
  python countermeasure_v5/run_selective_masking_cedh.py --n-queries 50 --weights 0.0,0.4,0.5,0.6,1.0
  python countermeasure_v5/run_selective_masking_cedh.py --no-domain-hints
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TOP_K
from test_contre_mesure_ildpiltest.config import CHROMA_DIR, COLLECTION_NAME, QUERIES_FILE

ROOT = Path(__file__).parent.parent
SAMPLED_QUERIES_PATH = ROOT / "data" / "cedh_eval_ablation" / "sampled_queries.json"
OUT_PATH = ROOT / "data" / "cedh_selective_masking" / "results.json"

ENTITY_HINT_RE = re.compile(r"^(.*) \([A-Z_]+\)$")


def parse_target_entity(q: dict) -> str | None:
    """Extrait l'entité cible d'un hint "<texte> (<TYPE>)" si présent."""
    for key in ("target_entity", "entity_hint", "entity", "target"):
        hint = q.get(key)
        if isinstance(hint, str):
            m = ENTITY_HINT_RE.match(hint)
            if m:
                return m.group(1).strip()
    return None


def get_query_text(q: dict) -> str:
    text = q.get("query", "")
    if isinstance(text, dict):
        return text.get("query") or str(text)
    return text if isinstance(text, str) else str(text)


def load_queries(n_total: int, seed: int) -> list[dict]:
    """Réutilise l'échantillon de l'ablation si présent (mêmes requêtes → comparable),
    sinon échantillonne de façon stratifiée par query_type."""
    if SAMPLED_QUERIES_PATH.exists():
        with open(SAMPLED_QUERIES_PATH, encoding="utf-8") as f:
            cached = json.load(f)
        print(f"  Réutilise l'échantillon ablation ({len(cached)} requêtes), prend les {n_total} premières")
        return cached[:n_total]

    with open(QUERIES_FILE, encoding="utf-8") as f:
        all_queries = json.load(f)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in all_queries:
        by_type[q.get("query_type", "unknown")].append(q)
    ratio = n_total / len(all_queries)
    rng = random.Random(seed)
    sampled: list[dict] = []
    for qtype in sorted(by_type):
        items = by_type[qtype][:]
        rng.shuffle(items)
        sampled.extend(items[: round(len(items) * ratio)])
    rng.shuffle(sampled)
    return sampled[:n_total]


def build_cpb(use_domain_hints: bool):
    """Construit UNE fois la contre-mesure v5 (un seul bootstrap B0)."""
    from countermeasure_v5.cpb_naive_rag_v5 import CPBNaiveRAGV5
    from llms.llama_llm import LlamaLLM
    from rag.naive_rag import NaiveRAG
    from test_contre_mesure_ildpiltest._store import IldpilTestStore

    store = IldpilTestStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
    llm = LlamaLLM()
    naive_rag = NaiveRAG(store=store, llm=llm)
    return CPBNaiveRAGV5(naive_rag=naive_rag, mask_min_weight=0.0, use_domain_hints=use_domain_hints)


def score_variant(cpb, queries: list[dict], embedder) -> dict:
    """Génère + score une passe (une valeur de mask_min_weight)."""
    from metrics.pii_leakage import compute_pii_leakage
    from metrics.response_quality import compute_response_quality

    agg = {"PII": 0.0, "QS": 0.0, "AR": 0.0, "RL": 0.0, "EM": 0.0}
    n = 0
    for i, q in enumerate(queries):
        qtext = get_query_text(q)
        print(f"    [{i + 1}/{len(queries)}] {q.get('global_id', '')}...", end="\r")
        try:
            result = cpb.run(qtext, top_k=TOP_K)
            response = result["response"]
            raw_chunks = result.get("raw_chunks", [])
        except Exception as exc:
            response, raw_chunks = f"ERROR: {exc}", []

        pii = compute_pii_leakage(response, raw_chunks, qtext)
        rq = compute_response_quality(
            query=qtext, response=response, chunks=raw_chunks,
            target_entity=parse_target_entity(q), embedder=embedder,
            precomputed_bert_f1=0.0,   # BF1 désactivé (roberta rechargé à chaque appel) -> QS sur AR/RL/EM
        )
        agg["PII"] += pii.leakage_rate
        agg["QS"]  += rq.quality_score
        agg["AR"]  += rq.answer_relevancy
        agg["RL"]  += rq.rouge_l
        agg["EM"]  += rq.exact_match
        n += 1
    print()
    return {k: (v / n if n else 0.0) for k, v in agg.items()}


def main():
    parser = argparse.ArgumentParser(description="Test du masquage sélectif CPB v5 sur CEDH (local, sans OpenAI).")
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--weights", type=str, default="0.0,0.4,0.5,0.6,1.0",
                        help="Seuils mask_min_weight à balayer (0.0 = baseline tout masqué)")
    parser.add_argument("--no-domain-hints", action="store_true",
                        help="Désactive le Signal 2 (category_hints de B0) → curseur poids seul")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    weights = [float(w) for w in args.weights.split(",") if w.strip() != ""]

    print("=== CPB v5 — Masquage sélectif domain-aware — CEDH (local) ===\n")
    queries = load_queries(args.n_queries, args.seed)
    print(f"1. {len(queries)} requêtes\n")

    from embeddings.embedder import Embedder
    embedder = Embedder()

    print("2. Bootstrap CPB v5 (un seul B0)...")
    cpb = build_cpb(use_domain_hints=not args.no_domain_hints)

    print("\n3. Balayage des seuils mask_min_weight...\n")
    rows: dict[float, dict] = {}
    for w in weights:
        cpb.mask_min_weight = w
        label = "baseline (tout masqué)" if w == 0.0 else f"seuil {w}"
        print(f"  -> mask_min_weight={w}  [{label}]")
        rows[w] = score_variant(cpb, queries, embedder)

    # ── Tableau comparatif ────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  RÉSULTATS — masquage sélectif (PII ↓, QS/AR/RL/EM ↑)")
    print(f"  domain_hints={'OFF' if args.no_domain_hints else 'ON'}, n={len(queries)}")
    print("=" * 66)
    print(f"  {'mask_min_w':>11} {'PII':>8} {'QS':>8} {'AR':>8} {'RL':>8} {'EM':>8}")
    print("-" * 66)
    for w in weights:
        r = rows[w]
        tag = " (base)" if w == 0.0 else ""
        print(f"  {w:>11.2f} {r['PII']:>8.4f} {r['QS']:>8.4f} {r['AR']:>8.4f} {r['RL']:>8.4f} {r['EM']:>8.4f}{tag}")
    print("=" * 66)
    base = rows.get(0.0)
    if base:
        print("\n  Deltas vs baseline (tout masqué) :")
        for w in weights:
            if w == 0.0:
                continue
            r = rows[w]
            print(f"    seuil {w}: ΔPII={r['PII'] - base['PII']:+.4f}  "
                  f"ΔQS={r['QS'] - base['QS']:+.4f}  ΔAR={r['AR'] - base['AR']:+.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "n_queries": len(queries),
            "domain_hints": not args.no_domain_hints,
            "weights": weights,
            "results": {str(w): rows[w] for w in weights},
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSauvegardé → {OUT_PATH}")


if __name__ == "__main__":
    main()
