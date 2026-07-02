"""
run_evaluation_openai_v2_nodedup.py — Variante "stack OpenAI" du harness Zhang
et al. avec la CONTRE-MESURE v4 (CPBNaiveRAGV4) et un retrieval SANS
déduplication par document.

Identique à run_evaluation_openai_v2.py (mêmes 300 requêtes, mêmes métriques,
mêmes juges GPT-4o, mêmes réponses gold, même LLM/embedding, même contre-mesure)
SAUF une seule variable :
  - store = OpenAIZhangChromaStore(doc_index, dedup=False)
    → le retrieval renvoie le top_k chunks bruts par similarité, PLUSIEURS du
      même document autorisés. Le LLM reçoit donc plus de contexte du doc source.

But : améliorer la qualité de réponse (CR/SS/AR remontent car le contexte couvre
tout le doc source) et mesurer l'impact sur la privacy (LO/AE/PI). À comparer au
run v4 dédupliqué (data/zhang_eval_openai_v2/), une seule variable changée.

Résultats propres à ce run dans data/zhang_eval_openai_v2_nodedup/ (n'écrase
aucun run existant).

Usage:
  python run_evaluation_openai_v2_nodedup.py [--skip-generation]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation_zhang"))
sys.path.insert(0, str(Path(__file__).parent))

from config import MLFLOW_TRACKING_URI, OPENAI_EMBEDDING_MODEL

# Réutilise le ChromaStore OpenAI (même embedding, même collection persistée).
from run_evaluation_openai import OpenAIZhangChromaStore

ZHANG_TABLE_2 = {
    "LO_F1": None, "AE": None, "PI": None, "CR": None, "SS": None, "AR": None,
}

SHARED_DATA_DIR = Path(__file__).parent.parent / "data" / "zhang_eval"                     # partagés
DATA_DIR        = Path(__file__).parent.parent / "data" / "zhang_eval_openai_v2_nodedup"   # propres à ce run sans dédup

RESPONSES_PATH = DATA_DIR / "responses.json"
CONTEXTS_PATH  = DATA_DIR / "contexts.json"
RESULTS_PATH   = DATA_DIR / "results.json"
CSV_PATH       = DATA_DIR / "results_per_query.csv"
EXPERIMENT_NAME = "zhang_evaluation"


# ── CPB v4 inference, retrieval SANS dédup (gpt-4o-mini) ──────────────────────

def run_cpb_v4_nodedup(doc_index: dict, attacks: list[dict]) -> tuple[list[str], list[list[str]]]:
    from countermeasure_v4.cpb_naive_rag_v4 import CPBNaiveRAGV4
    from llms.gpt4o_mini_llm import GPT4oMiniLLM
    from rag.naive_rag import NaiveRAG

    store = OpenAIZhangChromaStore(doc_index, dedup=False)   # ← plusieurs chunks/doc autorisés
    llm = GPT4oMiniLLM()
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb = CPBNaiveRAGV4(naive_rag=naive_rag)

    responses: list[str] = []
    contexts_per_query: list[list[str]] = []

    for i, attack in enumerate(attacks):
        print(f"  CPB v4 nodedup [{i + 1}/{len(attacks)}] {attack['doc_id']}...", end="\r")
        result = cpb.run(attack["query"])
        responses.append(result["response"])
        chunk_texts = [c.get("text", "") for c in result.get("raw_chunks", [])]
        contexts_per_query.append(chunk_texts)

    print()
    return responses, contexts_per_query


def load_or_run_cpb(doc_index: dict, attacks: list[dict], skip_generation: bool) -> tuple[list[str], list[list[str]]]:
    if skip_generation and RESPONSES_PATH.exists() and CONTEXTS_PATH.exists():
        print("Loading cached responses...")
        with open(RESPONSES_PATH, encoding="utf-8") as f:
            responses = json.load(f)
        with open(CONTEXTS_PATH, encoding="utf-8") as f:
            contexts_per_query = json.load(f)
        print(f"{len(responses)} responses loaded from cache.")
        return responses, contexts_per_query

    print("Running CPB v4 (no-dedup retrieval, gpt-4o-mini + text-embedding-3-small)...")
    responses, contexts_per_query = run_cpb_v4_nodedup(doc_index, attacks)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESPONSES_PATH, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    with open(CONTEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(contexts_per_query, f, ensure_ascii=False, indent=2)

    print(f"Responses saved → {RESPONSES_PATH}")
    return responses, contexts_per_query


def print_results_table(cpb_metrics: dict) -> None:
    order = ["LO_F1", "AE", "PI", "CR", "SS", "AR"]
    directions = {"LO_F1": "↓", "AE": "↑", "PI": "↓", "CR": "↑", "SS": "↑", "AR": "↑"}

    print("\n" + "=" * 70)
    print("  RESULTS — CPB v4 NO-DEDUP (gpt-4o-mini)  vs  Zhang et al. Table 2")
    print("=" * 70)
    print(f"  {'Metric':<10} {'Dir':>4} {'CPB v4 ND':>10} {'Zhang':>12} {'Delta':>10}")
    print("-" * 70)
    for m in order:
        our = cpb_metrics.get(m)
        zhang = ZHANG_TABLE_2.get(m)
        dir_str = directions.get(m, "")
        our_str = f"{our:.4f}" if our is not None else "   N/A"
        zhang_str = f"{zhang:.4f}" if zhang is not None else "  TODO"
        delta_str = f"{our - zhang:+.4f}" if (our is not None and zhang is not None) else "   N/A"
        print(f"  {m:<10} {dir_str:>4} {our_str:>10} {zhang_str:>12} {delta_str:>10}")
    print("=" * 70)


def main(skip_generation: bool = False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Zhang et al. Evaluation Harness — CPB v4 NO-DEDUP (OpenAI stack) ===\n")

    print("1. Loading shared data (doc_index, attack_queries)...")
    with open(SHARED_DATA_DIR / "doc_index.json", encoding="utf-8") as f:
        doc_index = json.load(f)
    attacks = json.loads((SHARED_DATA_DIR / "attack_queries.json").read_text(encoding="utf-8"))
    print(f"   {len(doc_index)} documents, {len(attacks)} attack queries\n")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="cpb_v4_nodedup_gpt4o_mini_openai_embed"):
        mlflow.log_param("system", "cpb_v4_openai_stack_nodedup")
        mlflow.log_param("retrieval_dedup", False)
        mlflow.log_param("llm_generation", "gpt-4o-mini")
        mlflow.log_param("embedding_model", OPENAI_EMBEDDING_MODEL)
        mlflow.log_param("llm_evaluation", "gpt-4o")
        mlflow.log_param("dataset", "umarbutler/open-australian-legal-corpus")
        mlflow.log_param("n_queries", len(attacks))

        print("2. Generating CPB v4 (no-dedup) responses...")
        responses, contexts_per_query = load_or_run_cpb(doc_index, attacks, skip_generation)
        mlflow.log_param("n_responses", len(responses))
        print()

        print("3. Privacy metrics")

        print("   [LO] ROUGE-L...")
        from metric_lo import aggregate_lo, compute_lo
        lo_results = [
            compute_lo(resp, doc_index.get(attack["doc_id"], {}).get("text", ""))
            for resp, attack in zip(responses, attacks)
        ]
        lo_agg = aggregate_lo(lo_results)
        print(f"       P={lo_agg['precision']:.4f}  R={lo_agg['recall']:.4f}  F1={lo_agg['f1']:.4f}")

        print("   [AE] GPT-4o judge...")
        from metric_ae import aggregate_ae, compute_ae_batch
        ae_cache_path = DATA_DIR / "ae_results.json"
        if ae_cache_path.exists():
            print("       Loading AE from cache...")
            with open(ae_cache_path, encoding="utf-8") as f:
                ae_results = json.load(f)
        else:
            ae_results = compute_ae_batch(responses, attacks, verbose=True)
            with open(ae_cache_path, "w", encoding="utf-8") as f:
                json.dump(ae_results, f, ensure_ascii=False, indent=2)
        ae_score = aggregate_ae(ae_results)
        print(f"       AE={ae_score:.4f}")

        print("   [PI] Personal Identification...")
        from metric_pi import PIMetric
        pi_cache_path = DATA_DIR / "pi_scores.json"
        pi_metric = PIMetric()
        if pi_cache_path.exists():
            print("       Loading PI from cache...")
            with open(pi_cache_path, encoding="utf-8") as f:
                pi_scores = json.load(f)
        else:
            pi_metric.build_claims_db(doc_index)
            pi_scores = pi_metric.compute_pi_batch(responses, attacks, verbose=True)
            with open(pi_cache_path, "w", encoding="utf-8") as f:
                json.dump(pi_scores, f, ensure_ascii=False, indent=2)
        pi_score = PIMetric.aggregate_pi(pi_scores)
        print(f"       PI={pi_score:.4f}")

        print("\n4. Utility metrics (RAGAS + GPT-4o)...")
        from metric_utility import compute_utility, generate_reference_responses
        utility_cache_path = DATA_DIR / "utility_scores.json"
        references = generate_reference_responses(attacks, doc_index)
        if utility_cache_path.exists():
            print("   Loading utility from cache...")
            with open(utility_cache_path, encoding="utf-8") as f:
                utility = json.load(f)
        else:
            utility = compute_utility(attacks, responses, contexts_per_query, references)
            with open(utility_cache_path, "w", encoding="utf-8") as f:
                json.dump(utility, f, ensure_ascii=False, indent=2)
        print(f"   CR={utility['CR']:.4f}  SS={utility['SS']:.4f}  AR={utility['AR']:.4f}")

        print("\n5. CSV export...")
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "index", "doc_id", "query", "response",
                "LO_precision", "LO_recall", "LO_f1",
                "AE", "PI",
            ])
            writer.writeheader()
            for i, (attack, resp) in enumerate(zip(attacks, responses)):
                lo = lo_results[i] if i < len(lo_results) else {}
                writer.writerow({
                    "index":        i,
                    "doc_id":       attack.get("doc_id", ""),
                    "query":        attack.get("query", ""),
                    "response":     resp,
                    "LO_precision": round(lo.get("precision", 0.0), 4),
                    "LO_recall":    round(lo.get("recall",    0.0), 4),
                    "LO_f1":        round(lo.get("f1",        0.0), 4),
                    "AE":           ae_results[i]["score"] if i < len(ae_results) else "",
                    "PI":           round(pi_scores[i],   4) if i < len(pi_scores)   else "",
                })
        print(f"   CSV saved → {CSV_PATH}")

        print("\n6. MLflow logging...")
        mlflow.log_metric("LO_precision", lo_agg["precision"])
        mlflow.log_metric("LO_recall", lo_agg["recall"])
        mlflow.log_metric("LO_f1", lo_agg["f1"])
        mlflow.log_metric("AE", ae_score)
        mlflow.log_metric("PI", pi_score)
        mlflow.log_metric("CR", utility["CR"])
        mlflow.log_metric("SS", utility["SS"])
        mlflow.log_metric("AR", utility["AR"])

        cpb_metrics = {
            "LO_F1": lo_agg["f1"], "AE": ae_score, "PI": pi_score,
            "CR": utility["CR"], "SS": utility["SS"], "AR": utility["AR"],
        }
        all_results = {
            "system":      "cpb_v4_openai_stack_nodedup",
            "retrieval_dedup": False,
            "llm":         "gpt-4o-mini",
            "embedding":   OPENAI_EMBEDDING_MODEL,
            "n_instances": len(attacks),
            "metrics":     cpb_metrics,
            "per_instance": {"LO": lo_results, "AE": ae_results, "PI": pi_scores},
        }
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(str(RESULTS_PATH))
        mlflow.log_artifact(str(CSV_PATH))

        print_results_table(cpb_metrics)

    print(f"\nDone. Full results → {RESULTS_PATH}")
    print(f"      Per-query CSV  → {CSV_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zhang et al. evaluation harness for CPB v4 with NO-DEDUP retrieval "
                     "(several chunks of the same doc allowed) — OpenAI stack"
    )
    parser.add_argument("--skip-generation", action="store_true", help="Reuse cached responses")
    args = parser.parse_args()
    main(skip_generation=args.skip_generation)
