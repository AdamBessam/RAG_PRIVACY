"""
run_evaluation.py — Orchestrateur d'évaluation Zhang et al. (IPM 2026).

Pipeline linéaire:
  1. Charger le doc_index et les 300 requêtes d'attaque
  2. CPB v3 (llama3.1:8b) → 300 réponses
  3. Métriques privacy  : LO (ROUGE-L), AE (juge GPT-4o), PI (ChromaDB + GPT-4o)
  4. Métriques utilité  : CR, SS, AR (RAGAS + GPT-4o, réponses de référence GPT-4o)
  5. MLflow logging
  6. Tableau comparatif CPB v3 vs Zhang et al. Table 2

Usage:
  python run_evaluation.py [--skip-generation]
    --skip-generation  : reuse responses already saved in data/zhang_eval/responses.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LLAMA_MODEL, MLFLOW_TRACKING_URI

# ── Published results (Zhang et al. Table 2) — fill in manually ───────────────
ZHANG_TABLE_2 = {
    "LO_F1": None,   # TODO
    "AE":    None,   # TODO
    "PI":    None,   # TODO
    "CR":    None,   # TODO
    "SS":    None,   # TODO
    "AR":    None,   # TODO
}

DATA_DIR = Path(__file__).parent.parent / "data" / "zhang_eval"
CHROMA_ZHANG_DIR = Path(__file__).parent.parent / "data" / "chroma_zhang"
RESPONSES_PATH = DATA_DIR / "responses.json"
RESULTS_PATH   = DATA_DIR / "results.json"
CSV_PATH       = DATA_DIR / "results_per_query.csv"
EXPERIMENT_NAME = "zhang_evaluation"


# ── ChromaStore wrapper ────────────────────────────────────────────────────────

class ZhangChromaStore:
    """
    Wraps the zhang_eval_corpus ChromaDB collection behind the ChromaStore
    interface expected by NaiveRAG and CPBNaiveRAGV3.
    """

    def __init__(self):
        import chromadb
        from chromadb.config import Settings

        from embeddings.embedder import Embedder

        self._embedder = Embedder()
        client = chromadb.PersistentClient(
            path=str(CHROMA_ZHANG_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection("zhang_eval_corpus")
        print(f"ZhangChromaStore ready: {self.collection.count()} chunks")

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        query_emb = self._embedder.embed_single(query_text).tolist()
        n_results = min(top_k * 3, self.collection.count())

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        seen_doc_ids: set[str] = set()

        for j in range(len(results["ids"][0])):
            meta = results["metadatas"][0][j]
            doc_id = meta.get("source_doc_id", results["ids"][0][j])

            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            chunks.append({
                "chunk_id":         results["ids"][0][j],
                "text":             results["documents"][0][j],
                "similarity_score": 1.0 - results["distances"][0][j],
                "doc_id":           doc_id,
                "n_pii":            0,
                "pii_entities":     [],
            })

            if len(chunks) >= top_k:
                break

        return chunks

    def get(self, limit: int = 50, include: list | None = None) -> dict:
        return self.collection.get(limit=limit, include=include or ["documents"])

    def count(self) -> int:
        return self.collection.count()


# ── CPB v3 inference ───────────────────────────────────────────────────────────

def run_cpb_v3(attacks: list[dict]) -> tuple[list[str], list[list[str]]]:
    """
    Instantiates CPB v3 with llama3.1:8b and runs it on all attack queries.
    Returns (responses, contexts_per_query).
    """
    from countermeasure_v3.cpb_naive_rag_v3 import CPBNaiveRAGV3
    from llms.llama_llm import LlamaLLM
    from rag.naive_rag import NaiveRAG

    store = ZhangChromaStore()
    llm = LlamaLLM()
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb = CPBNaiveRAGV3(naive_rag=naive_rag)

    responses: list[str] = []
    contexts_per_query: list[list[str]] = []

    for i, attack in enumerate(attacks):
        print(f"  CPB v3 [{i + 1}/{len(attacks)}] {attack['doc_id']}...", end="\r")
        result = cpb.run(attack["query"])
        responses.append(result["response"])
        chunk_texts = [c.get("text", "") for c in result.get("chunks", [])]
        contexts_per_query.append(chunk_texts)

    print()
    return responses, contexts_per_query


def load_or_run_cpb(attacks: list[dict], skip_generation: bool) -> tuple[list[str], list[list[str]]]:
    contexts_path = DATA_DIR / "contexts.json"

    if skip_generation and RESPONSES_PATH.exists() and contexts_path.exists():
        print("Loading cached responses...")
        with open(RESPONSES_PATH, encoding="utf-8") as f:
            responses = json.load(f)
        with open(contexts_path, encoding="utf-8") as f:
            contexts_per_query = json.load(f)
        print(f"{len(responses)} responses loaded from cache.")
        return responses, contexts_per_query

    print("Running CPB v3 (llama3.1:8b)...")
    responses, contexts_per_query = run_cpb_v3(attacks)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESPONSES_PATH, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    with open(contexts_path, "w", encoding="utf-8") as f:
        json.dump(contexts_per_query, f, ensure_ascii=False, indent=2)

    print(f"Responses saved → {RESPONSES_PATH}")
    return responses, contexts_per_query


# ── Results table ──────────────────────────────────────────────────────────────

def print_results_table(cpb_metrics: dict) -> None:
    order = ["LO_F1", "AE", "PI", "CR", "SS", "AR"]
    directions = {"LO_F1": "↓", "AE": "↑", "PI": "↓", "CR": "↑", "SS": "↑", "AR": "↑"}

    print("\n" + "=" * 58)
    print("  RESULTS — CPB v3  vs  Zhang et al. Table 2")
    print("=" * 58)
    print(f"  {'Metric':<10} {'Dir':>4} {'CPB v3':>10} {'Zhang':>12} {'Delta':>10}")
    print("-" * 58)
    for m in order:
        our = cpb_metrics.get(m)
        zhang = ZHANG_TABLE_2.get(m)
        dir_str = directions.get(m, "")
        our_str = f"{our:.4f}" if our is not None else "   N/A"
        zhang_str = f"{zhang:.4f}" if zhang is not None else "  TODO"
        delta_str = f"{our - zhang:+.4f}" if (our is not None and zhang is not None) else "   N/A"
        print(f"  {m:<10} {dir_str:>4} {our_str:>10} {zhang_str:>12} {delta_str:>10}")
    print("=" * 58)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(skip_generation: bool = False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Zhang et al. Evaluation Harness — CPB v3 ===\n")

    # 1. Load data
    print("1. Loading data...")
    with open(DATA_DIR / "doc_index.json", encoding="utf-8") as f:
        doc_index = json.load(f)
    attacks = json.loads((DATA_DIR / "attack_queries.json").read_text(encoding="utf-8"))
    print(f"   {len(doc_index)} documents, {len(attacks)} attack queries\n")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"cpb_v3_{LLAMA_MODEL.replace(':', '_')}"):
        mlflow.log_param("system", "cpb_v3")
        mlflow.log_param("llm_generation", LLAMA_MODEL)
        mlflow.log_param("llm_evaluation", "gpt-4o")
        mlflow.log_param("dataset", "umarbutler/open-australian-legal-corpus")
        mlflow.log_param("n_queries", len(attacks))

        # 2. Generate responses
        print("2. Generating CPB v3 responses...")
        responses, contexts_per_query = load_or_run_cpb(attacks, skip_generation)
        mlflow.log_param("n_responses", len(responses))
        print()

        # 3. Privacy metrics
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
        ae_results = compute_ae_batch(responses, attacks, verbose=True)
        ae_score = aggregate_ae(ae_results)
        print(f"       AE={ae_score:.4f}")

        print("   [PI] Personal Identification...")
        from metric_pi import PIMetric
        pi_metric = PIMetric()
        pi_metric.build_claims_db(doc_index)
        pi_scores = pi_metric.compute_pi_batch(responses, attacks, verbose=True)
        pi_score = PIMetric.aggregate_pi(pi_scores)
        print(f"       PI={pi_score:.4f}")

        # 4. Utility metrics
        print("\n4. Utility metrics (RAGAS + GPT-4o)...")
        from metric_utility import compute_utility, generate_reference_responses
        references = generate_reference_responses(attacks, doc_index)
        utility = compute_utility(attacks, responses, contexts_per_query, references)
        print(f"   CR={utility['CR']:.4f}  SS={utility['SS']:.4f}  AR={utility['AR']:.4f}")

        # 5. CSV export — one row per query
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

        # 6. MLflow logging
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
            "LO_F1": lo_agg["f1"],
            "AE":    ae_score,
            "PI":    pi_score,
            "CR":    utility["CR"],
            "SS":    utility["SS"],
            "AR":    utility["AR"],
        }
        all_results = {
            "system":      "cpb_v3",
            "llm":         LLAMA_MODEL,
            "n_instances": len(attacks),
            "metrics":     cpb_metrics,
            "per_instance": {
                "LO": lo_results,
                "AE": ae_results,
                "PI": pi_scores,
            },
        }
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(str(RESULTS_PATH))
        mlflow.log_artifact(str(CSV_PATH))

        # 7. Results table
        print_results_table(cpb_metrics)

    print(f"\nDone. Full results → {RESULTS_PATH}")
    print(f"      Per-query CSV  → {CSV_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zhang et al. evaluation harness for CPB v3")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Reuse cached responses from a previous run",
    )
    args = parser.parse_args()
    main(skip_generation=args.skip_generation)
