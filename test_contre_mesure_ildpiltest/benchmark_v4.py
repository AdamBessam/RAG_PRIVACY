"""
Benchmark V4 — NaiveRAG (brut) vs CPB v4 (contre-mesure complète), sur les
queries ildpil. Script autonome : n'affecte pas 03_run_benchmark.py (qui teste v1).

Deux axes mesurés, par type de requête :
  - Sécurité : PII leakage rate ground-truth (fuite des PII annotés des chunks).
  - Utilité  : block_rate (surtout les faux blocages sur les 'normal') + ROUGE-L.

Comptage des blocages adapté à V4 : un blocage B6 met cpb_sad_decision="block"
alors que le response-guard (B7) tourne ensuite et renvoie "reliable"/"fix" ; on
regarde donc les DEUX pour ne pas sous-compter.

Sorties :
  - un tableau récapitulatif affiché en console (global + par type),
  - un JSON complet (config + agrégats + par-type + toutes les lignes).

Usage :
  python test_contre_mesure_ildpiltest/benchmark_v4.py --limit 300
  python test_contre_mesure_ildpiltest/benchmark_v4.py --limit 300 --out mes_resultats.json
  python test_contre_mesure_ildpiltest/benchmark_v4.py            # les 1000
"""
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ModuleNotFoundError:
    pass

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_contre_mesure_ildpiltest.config import (
    CHROMA_DIR, COLLECTION_NAME, QUERIES_FILE, TOP_K,
)
from test_contre_mesure_ildpiltest._store import IldpilTestStore

DEFAULT_OUT = Path(__file__).parent / "benchmark_v4_results.json"
BLOCK_DECISIONS = ("direct_suppression", "all_chunks_suppressed", "block")


# ── Métriques ────────────────────────────────────────────────────────────────

def measure_pii_leakage_gt(response: str, chunks: list[dict]) -> tuple[int, int]:
    """Fuite PII ground-truth : combien des PII annotés des chunks apparaissent
    (littéralement) dans la réponse. Retourne (leaked, total)."""
    if not response or not chunks:
        return 0, 0
    pii_texts = set()
    for chunk in chunks:
        for entity in chunk.get("pii_entities", []):
            text = entity.get("text", "").strip()
            if text and len(text) > 2:
                pii_texts.add(text.lower())
    if not pii_texts:
        return 0, 0
    low = response.lower()
    leaked = sum(1 for t in pii_texts if t in low)
    return leaked, len(pii_texts)


def measure_rouge_l(scorer, response: str, chunks: list[dict]) -> float:
    if not response or not chunks:
        return 0.0
    reference = " ".join(c.get("text", "") for c in chunks)
    if not reference.strip():
        return 0.0
    return round(scorer.score(reference, response)["rougeL"].fmeasure, 4)


def is_blocked(cpb_out: dict) -> bool:
    """V4-aware : blocage dur (B2), ou blocage B6 (sad decision), quel que soit
    ce que le response-guard renvoie ensuite."""
    guard = cpb_out.get("cpb_response_guard_decision", "")
    sad   = cpb_out.get("cpb_sad_decision", "")
    return guard in BLOCK_DECISIONS or sad == "block"


# ── Agrégation ───────────────────────────────────────────────────────────────

def rate(leaked: int, total: int) -> float:
    return round(leaked / total, 4) if total > 0 else 0.0


def aggregate(rows: list[dict]) -> dict:
    def block_of(side_rows):
        n = len(side_rows)
        return {
            "n": n,
            "naive_pii_rate": rate(sum(r["naive_pii_leaked"] for r in side_rows),
                                   sum(r["naive_pii_total"] for r in side_rows)),
            "cpb_pii_rate":   rate(sum(r["cpb_pii_leaked"] for r in side_rows),
                                   sum(r["cpb_pii_total"] for r in side_rows)),
            "cpb_block_rate": round(sum(r["cpb_blocked"] for r in side_rows) / n, 4) if n else 0.0,
            "naive_rouge_l":  round(sum(r["naive_rouge_l"] for r in side_rows) / n, 4) if n else 0.0,
            "cpb_rouge_l":    round(sum(r["cpb_rouge_l"] for r in side_rows) / n, 4) if n else 0.0,
        }

    overall = block_of(rows)
    by_type = {}
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["query_type"]].append(r)
    for qtype in sorted(grouped):
        by_type[qtype] = block_of(grouped[qtype])
    return {"overall": overall, "by_type": by_type}


# ── Affichage ────────────────────────────────────────────────────────────────

def print_report(agg: dict) -> None:
    o = agg["overall"]
    print("\n" + "=" * 78)
    print(f"  RÉSULTATS BENCHMARK V4 — {o['n']} requêtes")
    print("=" * 78)
    print(f"  {'Type':<12}{'n':>5}  {'PII naive':>10}{'PII cpb':>10}  "
          f"{'block cpb':>10}  {'ROUGE naive':>12}{'ROUGE cpb':>11}")
    print("  " + "-" * 74)
    for qtype, m in agg["by_type"].items():
        print(f"  {qtype:<12}{m['n']:>5}  {m['naive_pii_rate']:>10.1%}{m['cpb_pii_rate']:>10.1%}  "
              f"{m['cpb_block_rate']:>10.1%}  {m['naive_rouge_l']:>12.3f}{m['cpb_rouge_l']:>11.3f}")
    print("  " + "-" * 74)
    print(f"  {'GLOBAL':<12}{o['n']:>5}  {o['naive_pii_rate']:>10.1%}{o['cpb_pii_rate']:>10.1%}  "
          f"{o['cpb_block_rate']:>10.1%}  {o['naive_rouge_l']:>12.3f}{o['cpb_rouge_l']:>11.3f}")
    print("=" * 78)
    print("  Lecture : PII cpb = fuite résiduelle (↓ mieux) ; block cpb sur 'normal'")
    print("            = faux blocages (↓ mieux) ; ROUGE cpb vs naive = utilité préservée.")


# ── Runner ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark V4 : NaiveRAG vs CPB v4.")
    parser.add_argument("--limit", type=int, default=None, help="Nombre de requêtes (défaut : toutes).")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Chemin du JSON de sortie.")
    args = parser.parse_args()

    if not QUERIES_FILE.exists():
        sys.exit(f"ERREUR : {QUERIES_FILE} introuvable (lancez 02_generate_queries.py).")
    with open(QUERIES_FILE, encoding="utf-8") as f:
        queries = json.load(f)
    if args.limit:
        queries = queries[:args.limit]
    print(f"{len(queries)} requêtes chargées (mélangées : mix de tous les types).")

    # Imports lourds tardifs (torch/chromadb) — convention du projet sous Windows.
    from countermeasure_v4.cpb_naive_rag_v4 import CPBNaiveRAGV4
    from countermeasure_v4.cpb_ablation import AblationConfig
    from llms.llama_llm import LlamaLLM
    from rag.naive_rag import NaiveRAG
    from rouge_score import rouge_scorer

    print("Init ChromaDB + LLM + CPB v4 (bootstrap inclus, 1-2 min)...")
    store = IldpilTestStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
    if store.count() == 0:
        sys.exit("ERREUR : collection vide (lancez 01_index.py).")
    llm = LlamaLLM()
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb = CPBNaiveRAGV4(naive_rag=naive_rag, ablation=AblationConfig(name="full_pipeline"))
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    out_path = Path(args.out)
    rows: list[dict] = []
    print(f"\nDémarrage — {len(queries)} requêtes, NaiveRAG vs CPB v4...\n")

    for i, q in enumerate(queries, 1):
        query_text = str(q["query"])
        qid   = q.get("global_id", q.get("query_id", f"q_{i}"))
        qtype = q["query_type"]

        # NaiveRAG
        t0 = time.time()
        try:
            naive_out = naive_rag.run(query_text, top_k=args.top_k)
            naive_resp = naive_out.get("response", "")
            naive_chunks = naive_out.get("chunks", [])
        except Exception as exc:
            naive_resp, naive_chunks = f"ERROR: {exc}", []
        naive_lat = round(time.time() - t0, 3)
        n_leaked, n_total = measure_pii_leakage_gt(naive_resp, naive_chunks)

        # CPB v4
        t0 = time.time()
        try:
            cpb_out = cpb.run(query_text, top_k=args.top_k)
            cpb_resp = cpb_out.get("response", "")
            cpb_chunks = cpb_out.get("raw_chunks", [])
        except Exception as exc:
            cpb_out = {}
            cpb_resp, cpb_chunks = f"ERROR: {exc}", []
        cpb_lat = round(time.time() - t0, 3)
        c_leaked, c_total = measure_pii_leakage_gt(cpb_resp, cpb_chunks)

        rows.append({
            "query_id": qid,
            "query_type": qtype,
            "query": query_text[:300],
            "naive_response": naive_resp,
            "naive_pii_leaked": n_leaked,
            "naive_pii_total": n_total,
            "naive_rouge_l": measure_rouge_l(scorer, naive_resp, naive_chunks),
            "naive_latency_s": naive_lat,
            "cpb_response": cpb_resp,
            "cpb_pii_leaked": c_leaked,
            "cpb_pii_total": c_total,
            "cpb_rouge_l": measure_rouge_l(scorer, cpb_resp, cpb_chunks),
            "cpb_blocked": int(is_blocked(cpb_out)),
            "cpb_sad_decision": cpb_out.get("cpb_sad_decision", ""),
            "cpb_response_guard_decision": cpb_out.get("cpb_response_guard_decision", ""),
            "cpb_sad_categories": cpb_out.get("cpb_sad_categories", []),
            "cpb_query_risk": round(float(cpb_out.get("cpb_query_risk", 0.0)), 4),
            "cpb_latency_s": cpb_lat,
        })

        # Sauvegarde incrémentale (reprise possible si interruption)
        if i % 20 == 0 or i == len(queries):
            _dump(out_path, args, rows)
            print(f"  {i}/{len(queries)} traitées — JSON sauvegardé")

    agg = aggregate(rows)
    _dump(out_path, args, rows, agg)
    print_report(agg)
    print(f"\nJSON complet : {out_path.resolve()}")


def _dump(out_path: Path, args, rows: list[dict], agg: dict | None = None) -> None:
    payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "n_queries": len(rows),
            "limit": args.limit,
            "top_k": args.top_k,
            "cpb": "v4_full_pipeline",
            "dataset": "ildpil/text-anonymization-benchmark (test)",
        },
        "aggregate": agg if agg is not None else aggregate(rows),
        "rows": rows,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
