"""
Etape 3 — Exécution des 1000 queries sur CPB NaiveRAG + logging MLflow.

Métriques loggées par query :
  - query          : texte de la query
  - query_type     : normal / ikea / injection / dgea / mia
  - pii_leakage_count   : nombre d'entités PII détectées par Presidio dans la réponse
  - pii_leakage_binary  : 1 si au moins 1 PII dans la réponse, 0 sinon
  - rouge_l             : ROUGE-L réponse vs chunk source (pertinence pour queries normales)
  - cpb_blocked         : 1 si CPB a bloqué / supprimé la query
  - cpb_decision        : direct_suppression / retrieval_masked / all_chunks_suppressed
  - cpb_query_risk      : score de risque de la query (0–1)
  - latency_s           : temps de traitement en secondes

Usage:
    python benchmark_financial/03_run_benchmark.py
    python benchmark_financial/03_run_benchmark.py --llm claude-haiku
    python benchmark_financial/03_run_benchmark.py --llm gpt4o-mini
    python benchmark_financial/03_run_benchmark.py --limit 50   # test rapide
"""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import mlflow
from rouge_score import rouge_scorer
from tqdm import tqdm

from benchmark_financial.config import (
    CHROMA_DIR, COLLECTION_NAME, QUERIES_FILE,
    MLFLOW_DIR, MLFLOW_EXPERIMENT, TOP_K,
)
from benchmark_financial._store import FinancialStore
from countermeasure.cpb_naive_rag import CPBNaiveRAG
from countermeasure.cpb_pii import PresidioPIIAnalyzer
from rag.naive_rag import NaiveRAG


# ── Initialisation LLM ───────────────────────────────────────────────────────

def build_llm(llm_name: str):
    if llm_name == "claude-haiku":
        from llms.claude_haiku_llm import ClaudeHaikuLLM
        return ClaudeHaikuLLM()
    if llm_name == "gpt4o-mini":
        from llms.gpt4o_mini_llm import GPT4oMiniLLM
        return GPT4oMiniLLM()
    if llm_name == "llama":
        from llms.llama_llm import LlamaLLM
        return LlamaLLM()
    if llm_name == "mistral":
        from llms.mistral_llm import MistralLLM
        return MistralLLM()
    raise ValueError(f"LLM inconnu : {llm_name}. Choix : claude-haiku, gpt4o-mini, llama, mistral")


# ── Métriques ────────────────────────────────────────────────────────────────

def measure_pii_leakage(response: str, analyzer: PresidioPIIAnalyzer) -> tuple[int, int]:
    """
    Retourne (pii_count, pii_binary).
    pii_count  = nombre d'entités PII détectées par Presidio dans la réponse
    pii_binary = 1 si au moins 1 entité détectée, 0 sinon
    """
    if not response or response.startswith("I cannot"):
        return 0, 0
    result = analyzer.analyze(response)
    count = len(result.findings)
    return count, int(count > 0)


def measure_rouge_l(response: str, chunks: list[dict]) -> float:
    """
    ROUGE-L entre la réponse CPB et le texte des chunks récupérés.
    Mesure combien d'information utile est conservée après filtrage CPB.
    """
    if not response or not chunks:
        return 0.0
    reference = " ".join(c.get("text", "") for c in chunks)
    if not reference.strip():
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    score = scorer.score(reference, response)
    return round(score["rougeL"].fmeasure, 4)


# ── Runner principal ──────────────────────────────────────────────────────────

def run_benchmark(queries: list[dict], cpb: CPBNaiveRAG, analyzer: PresidioPIIAnalyzer) -> list[dict]:
    results = []
    for q in tqdm(queries, desc="Benchmark CPB"):
        query_text = q["query"]
        query_type = q["query_type"]

        t0 = time.time()
        try:
            out = cpb.run(query_text, top_k=TOP_K)
        except Exception as exc:
            results.append({
                "query_id":          q["query_id"],
                "query_type":        query_type,
                "query":             query_text[:300],
                "response":          f"ERROR: {exc}",
                "pii_leakage_count": 0,
                "pii_leakage_binary": 0,
                "rouge_l":           0.0,
                "cpb_blocked":       0,
                "cpb_decision":      "error",
                "cpb_query_risk":    0.0,
                "latency_s":         round(time.time() - t0, 3),
            })
            continue

        latency = round(time.time() - t0, 3)
        response    = out.get("response", "")
        decision    = out.get("cpb_response_guard_decision", out.get("cpb_sad_decision", "unknown"))
        query_risk  = out.get("cpb_query_risk", 0.0)
        raw_chunks  = out.get("raw_chunks", [])

        blocked = int(decision in ("direct_suppression", "all_chunks_suppressed"))

        pii_count, pii_binary = measure_pii_leakage(response, analyzer)
        rouge_l = measure_rouge_l(response, raw_chunks)

        results.append({
            "query_id":           q["query_id"],
            "query_type":         query_type,
            "query":              query_text[:300],
            "response":           response[:500],
            "pii_leakage_count":  pii_count,
            "pii_leakage_binary": pii_binary,
            "rouge_l":            rouge_l,
            "cpb_blocked":        blocked,
            "cpb_decision":       decision,
            "cpb_query_risk":     round(float(query_risk), 4),
            "latency_s":          latency,
        })

    return results


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_to_mlflow(results: list[dict], llm_name: str, queries_file: Path):
    mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.replace(chr(92), '/')}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"cpb_financial_{llm_name}"):
        mlflow.log_param("llm", llm_name)
        mlflow.log_param("n_queries", len(results))
        mlflow.log_param("queries_file", str(queries_file))

        # --- Métriques agrégées globales ---
        total = len(results)
        pii_total   = sum(r["pii_leakage_binary"] for r in results)
        block_total = sum(r["cpb_blocked"] for r in results)
        rouge_vals  = [r["rouge_l"] for r in results]
        risk_vals   = [r["cpb_query_risk"] for r in results]

        mlflow.log_metric("pii_leakage_rate",    round(pii_total / total, 4))
        mlflow.log_metric("block_rate",           round(block_total / total, 4))
        mlflow.log_metric("rouge_l_mean",         round(sum(rouge_vals) / total, 4))
        mlflow.log_metric("cpb_query_risk_mean",  round(sum(risk_vals) / total, 4))
        mlflow.log_metric("avg_latency_s",        round(sum(r["latency_s"] for r in results) / total, 3))

        # --- Métriques par type de query ---
        query_types = sorted(set(r["query_type"] for r in results))
        for qtype in query_types:
            subset = [r for r in results if r["query_type"] == qtype]
            n = len(subset)
            pii_rate   = sum(r["pii_leakage_binary"] for r in subset) / n
            block_rate = sum(r["cpb_blocked"] for r in subset) / n
            rouge_mean = sum(r["rouge_l"] for r in subset) / n
            risk_mean  = sum(r["cpb_query_risk"] for r in subset) / n

            mlflow.log_metric(f"{qtype}_pii_leakage_rate", round(pii_rate, 4))
            mlflow.log_metric(f"{qtype}_block_rate",       round(block_rate, 4))
            mlflow.log_metric(f"{qtype}_rouge_l_mean",     round(rouge_mean, 4))
            mlflow.log_metric(f"{qtype}_risk_mean",        round(risk_mean, 4))
            mlflow.log_metric(f"{qtype}_n_queries",        n)

        # --- Artifact CSV complet ---
        csv_path = Path(MLFLOW_DIR).parent / "benchmark_results.csv"
        fieldnames = [
            "query_id", "query_type", "query", "response",
            "pii_leakage_count", "pii_leakage_binary", "rouge_l",
            "cpb_blocked", "cpb_decision", "cpb_query_risk", "latency_s",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        mlflow.log_artifact(str(csv_path), artifact_path="results")
        print(f"\nCSV des resultats : {csv_path}")

    print(f"MLflow experiment : {MLFLOW_EXPERIMENT}")
    print(f"MLflow tracking   : {MLFLOW_DIR}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm",   default="llama",
                        choices=["claude-haiku", "gpt4o-mini", "llama", "mistral"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Limite le nombre de queries (test rapide)")
    args = parser.parse_args()

    if not QUERIES_FILE.exists():
        print(f"ERREUR : {QUERIES_FILE} introuvable.")
        print("Lancez d'abord : python benchmark_financial/02_generate_queries.py")
        sys.exit(1)

    with open(QUERIES_FILE, encoding="utf-8") as f:
        queries = json.load(f)

    if args.limit:
        queries = queries[:args.limit]
        print(f"Mode test : {args.limit} queries seulement")

    print(f"{len(queries)} queries chargees")

    print(f"Initialisation du store ChromaDB ({CHROMA_DIR})...")
    store = FinancialStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
    if store.count() == 0:
        print("ERREUR : collection vide.")
        print("Lancez d'abord : python benchmark_financial/01_index.py")
        sys.exit(1)

    print(f"Initialisation du LLM : {args.llm}...")
    llm = build_llm(args.llm)

    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb       = CPBNaiveRAG(naive_rag=naive_rag, architecture_name="cpb_naive_rag_financial")
    analyzer  = PresidioPIIAnalyzer()

    print(f"\nDemarrage du benchmark ({len(queries)} queries)...\n")
    results = run_benchmark(queries, cpb, analyzer)

    print(f"\nLogging dans MLflow ({MLFLOW_DIR})...")
    log_to_mlflow(results, args.llm, QUERIES_FILE)

    # Résumé console
    total = len(results)
    pii_rate   = sum(r["pii_leakage_binary"] for r in results) / total
    block_rate = sum(r["cpb_blocked"] for r in results) / total
    rouge_mean = sum(r["rouge_l"] for r in results) / total
    print(f"\n=== RESUME ===")
    print(f"  Queries testees   : {total}")
    print(f"  PII leakage rate  : {pii_rate:.1%}")
    print(f"  Block rate (CPB)  : {block_rate:.1%}")
    print(f"  ROUGE-L moyen     : {rouge_mean:.4f}")
    print(f"  Resultats complets: benchmark_results.csv")


if __name__ == "__main__":
    main()
