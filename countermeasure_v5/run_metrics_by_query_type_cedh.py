"""
run_metrics_by_query_type_cedh.py — Métriques CPB v5 (contextuel) VENTILÉES PAR
TYPE DE QUESTION sur CEDH (ildpil/text-anonymization-benchmark, split test).

But : ne PAS agréger toutes les requêtes ensemble, mais comparer les métriques
(PII / QS / AR / RL / EM) SÉPARÉMENT pour chaque `query_type` du corpus :
    normal · direct · injection · dgea · mia
→ on voit quel type de question réagit comment à la contre-mesure.

Masquage = CPB v5 combo (masquage par COMBINAISONS ré-identifiantes générées par
B0/Llama pour le domaine détecté ; on masque pour casser toute combinaison
présente, on garde le reste ; identifiants forts toujours masqués).

100 % LOCAL : génération Llama locale + métriques locales → AUCUN token OpenAI.
Un SEUL bootstrap B0 (un seul jeu de combinaisons) pour tout le run.

Usage (depuis la racine du repo) :
  python countermeasure_v5/run_metrics_by_query_type_cedh.py
  python countermeasure_v5/run_metrics_by_query_type_cedh.py --per-type 30
  python countermeasure_v5/run_metrics_by_query_type_cedh.py --no-llm-combos   # fallback v5
  python countermeasure_v5/run_metrics_by_query_type_cedh.py --no-domain-hints
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
OUT_PATH = ROOT / "data" / "cedh_metrics_by_query_type" / "results.json"

ENTITY_HINT_RE = re.compile(r"^(.*) \([A-Z_]+\)$")
METRIC_KEYS = ("PII", "QS", "AR", "RL", "EM")


def parse_target_entity(q: dict) -> str | None:
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


def load_queries_by_type(per_type: int, seed: int) -> dict[str, list[dict]]:
    """Échantillonne `per_type` requêtes PAR query_type depuis le corpus 1000."""
    with open(QUERIES_FILE, encoding="utf-8") as f:
        all_queries = json.load(f)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in all_queries:
        by_type[q.get("query_type", "unknown")].append(q)
    rng = random.Random(seed)
    out: dict[str, list[dict]] = {}
    for qtype in sorted(by_type):
        items = by_type[qtype][:]
        rng.shuffle(items)
        out[qtype] = items[:per_type]
    return out


def build_cpb(mask_min_weight: float, use_domain_hints: bool, use_llm_combos: bool):
    """Construit UNE fois la contre-mesure (un seul bootstrap B0)."""
    from countermeasure_v5.cpb_naive_rag_v5_combo import CPBNaiveRAGV5Combo
    from llms.llama_llm import LlamaLLM
    from rag.naive_rag import NaiveRAG
    from test_contre_mesure_ildpiltest._store import IldpilTestStore

    store = IldpilTestStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
    llm = LlamaLLM()
    naive_rag = NaiveRAG(store=store, llm=llm)
    return CPBNaiveRAGV5Combo(
        naive_rag=naive_rag,
        mask_min_weight=mask_min_weight,
        use_domain_hints=use_domain_hints,
        use_llm_combos=use_llm_combos,
    )


def score_group(cpb, queries: list[dict], embedder) -> dict:
    """Génère + score un groupe de requêtes (un query_type)."""
    from metrics.pii_leakage import compute_pii_leakage
    from metrics.response_quality import compute_response_quality

    agg = {k: 0.0 for k in METRIC_KEYS}
    n = 0
    for i, q in enumerate(queries):
        qtext = get_query_text(q)
        print(f"      [{i + 1}/{len(queries)}] {q.get('global_id', q.get('query_id', ''))}...", end="\r")
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
            precomputed_bert_f1=0.0,   # BF1 désactivé → QS sur AR/RL/EM
        )
        agg["PII"] += pii.leakage_rate
        agg["QS"]  += rq.quality_score
        agg["AR"]  += rq.answer_relevancy
        agg["RL"]  += rq.rouge_l
        agg["EM"]  += rq.exact_match
        n += 1
    print()
    return {"n": n, **{k: (agg[k] / n if n else 0.0) for k in METRIC_KEYS}}


def main():
    parser = argparse.ArgumentParser(
        description="Métriques CPB v5 contextuel ventilées par type de question (CEDH, local, sans OpenAI).")
    parser.add_argument("--per-type", type=int, default=20,
                        help="Nb de requêtes échantillonnées par query_type (défaut 20).")
    parser.add_argument("--mask-min-weight", type=float, default=0.5,
                        help="Seuil v5 (utilisé par le fallback si les combos sont absents).")
    parser.add_argument("--no-domain-hints", action="store_true",
                        help="Désactive le Signal 2 (category_hints de B0).")
    parser.add_argument("--no-llm-combos", action="store_true",
                        help="Désactive la génération LLM des combinaisons → fallback masquage v5.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_llm_combos = not args.no_llm_combos

    print("=== CPB v5 combo — métriques PAR TYPE DE QUESTION — CEDH (local) ===\n")
    groups = load_queries_by_type(args.per_type, args.seed)
    total = sum(len(v) for v in groups.values())
    print(f"1. {total} requêtes : " + ", ".join(f"{t}={len(q)}" for t, q in groups.items()) + "\n")

    from embeddings.embedder import Embedder
    embedder = Embedder()

    print("2. Bootstrap CPB v5 combo (un seul B0)...")
    cpb = build_cpb(
        mask_min_weight=args.mask_min_weight,
        use_domain_hints=not args.no_domain_hints,
        use_llm_combos=use_llm_combos,
    )

    print("\n3. Scoring par type de question...\n")
    rows: dict[str, dict] = {}
    for qtype, queries in groups.items():
        if not queries:
            continue
        print(f"  -> query_type = {qtype}  (n={len(queries)})")
        rows[qtype] = score_group(cpb, queries, embedder)

    # ── Tableau comparatif par type ──────────────────────────────────────────
    print("\n" + "=" * 74)
    print("  MÉTRIQUES PAR TYPE DE QUESTION  (PII ↓ = mieux, QS/AR/RL/EM ↑ = mieux)")
    print(f"  llm_combos={'ON' if use_llm_combos else 'OFF (fallback v5)'}, "
          f"domain_hints={'OFF' if args.no_domain_hints else 'ON'}")
    print("=" * 74)
    print(f"  {'query_type':>11} {'n':>4} {'PII':>8} {'QS':>8} {'AR':>8} {'RL':>8} {'EM':>8}")
    print("-" * 74)
    for qtype, r in rows.items():
        print(f"  {qtype:>11} {r['n']:>4} {r['PII']:>8.4f} {r['QS']:>8.4f} "
              f"{r['AR']:>8.4f} {r['RL']:>8.4f} {r['EM']:>8.4f}")
    # Moyenne globale (pondérée par n)
    tot_n = sum(r["n"] for r in rows.values())
    if tot_n:
        glob = {k: sum(r[k] * r["n"] for r in rows.values()) / tot_n for k in METRIC_KEYS}
        print("-" * 74)
        print(f"  {'GLOBAL':>11} {tot_n:>4} {glob['PII']:>8.4f} {glob['QS']:>8.4f} "
              f"{glob['AR']:>8.4f} {glob['RL']:>8.4f} {glob['EM']:>8.4f}")
    print("=" * 74)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "per_type": args.per_type,
            "mask_min_weight": args.mask_min_weight,
            "llm_combos": use_llm_combos,
            "domain_hints": not args.no_domain_hints,
            "risky_combinations": [sorted(c) for c in getattr(cpb, "risky_combos", [])],
            "by_query_type": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSauvegardé → {OUT_PATH}")


if __name__ == "__main__":
    main()
