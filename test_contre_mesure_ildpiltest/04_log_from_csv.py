"""
Étape 4 — Re-logging MLflow depuis le CSV de résultats.

À utiliser quand le benchmark s'est terminé mais que MLflow n'a pas été alimenté.

Usage:
    python test_contre_mesure_ildpiltest/04_log_from_csv.py
    python test_contre_mesure_ildpiltest/04_log_from_csv.py --llm llama
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import mlflow

from test_contre_mesure_ildpiltest.config import (
    RESULTS_CSV, MLFLOW_DIR, MLFLOW_EXPERIMENT,
)


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        for col in ("naive_pii_binary", "cpb_pii_binary", "cpb_blocked",
                    "naive_pii_count", "cpb_pii_count"):
            r[col] = int(r[col]) if r[col] not in ("", None) else 0
        for col in ("naive_rouge_l", "cpb_rouge_l", "naive_latency_s",
                    "cpb_latency_s", "cpb_query_risk"):
            r[col] = float(r[col]) if r[col] not in ("", None) else 0.0
    return rows


def log_to_mlflow(results: list[dict], llm_name: str):
    mlflow_uri = f"file:///{MLFLOW_DIR.replace(chr(92), '/')}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = f"naiveRAG_vs_CPB_{llm_name}"
    with mlflow.start_run(run_name=run_name):
        total = len(results)
        mlflow.log_param("llm",       llm_name)
        mlflow.log_param("n_queries", total)
        mlflow.log_param("dataset",   "ildpil/text-anonymization-benchmark")
        mlflow.log_param("split",     "test")

        naive_pii = sum(r["naive_pii_binary"] for r in results) / total
        naive_rl  = sum(r["naive_rouge_l"]    for r in results) / total
        naive_lat = sum(r["naive_latency_s"]  for r in results) / total

        mlflow.log_metric("naive_pii_leakage_rate", round(naive_pii, 4))
        mlflow.log_metric("naive_rouge_l_mean",     round(naive_rl,  4))
        mlflow.log_metric("naive_latency_mean_s",   round(naive_lat, 3))

        cpb_pii   = sum(r["cpb_pii_binary"] for r in results) / total
        cpb_rl    = sum(r["cpb_rouge_l"]    for r in results) / total
        cpb_block = sum(r["cpb_blocked"]    for r in results) / total
        cpb_risk  = sum(r["cpb_query_risk"] for r in results) / total
        cpb_lat   = sum(r["cpb_latency_s"]  for r in results) / total

        mlflow.log_metric("cpb_pii_leakage_rate",  round(cpb_pii,   4))
        mlflow.log_metric("cpb_rouge_l_mean",      round(cpb_rl,    4))
        mlflow.log_metric("cpb_block_rate",        round(cpb_block, 4))
        mlflow.log_metric("cpb_query_risk_mean",   round(cpb_risk,  4))
        mlflow.log_metric("cpb_latency_mean_s",    round(cpb_lat,   3))

        pii_reduction = (naive_pii - cpb_pii) / naive_pii if naive_pii > 0 else 0.0
        mlflow.log_metric("pii_reduction_rate", round(pii_reduction, 4))

        query_types = sorted(set(r["query_type"] for r in results))
        for qtype in query_types:
            subset = [r for r in results if r["query_type"] == qtype]
            n = len(subset)
            if n == 0:
                continue
            mlflow.log_metric(f"{qtype}_naive_pii_rate",  round(sum(r["naive_pii_binary"] for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_cpb_pii_rate",    round(sum(r["cpb_pii_binary"]   for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_cpb_block_rate",  round(sum(r["cpb_blocked"]       for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_cpb_risk_mean",   round(sum(r["cpb_query_risk"]    for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_naive_rouge_l",   round(sum(r["naive_rouge_l"]     for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_cpb_rouge_l",     round(sum(r["cpb_rouge_l"]       for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_n_queries",       n)

        mlflow.log_artifact(str(RESULTS_CSV), artifact_path="results")

        print(f"\n{'='*50}")
        print(f"  RÉSULTATS — {total} queries")
        print(f"{'='*50}")
        print(f"  {'Métrique':<28} {'NaiveRAG':>10}  {'CPB':>10}")
        print(f"  {'-'*50}")
        print(f"  {'PII leakage rate':<28} {naive_pii:>10.1%}  {cpb_pii:>10.1%}")
        print(f"  {'ROUGE-L moyen':<28} {naive_rl:>10.4f}  {cpb_rl:>10.4f}")
        print(f"  {'Block rate':<28} {'—':>10}  {cpb_block:>10.1%}")
        print(f"  {'-'*50}")
        print(f"  Réduction PII grâce au CPB : {pii_reduction:.1%}")
        print(f"{'='*50}")
        print(f"\n  MLflow experiment : {MLFLOW_EXPERIMENT}")
        print(f"  MLflow tracking   : {MLFLOW_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="llama",
                        choices=["llama", "mistral", "gpt4o-mini", "claude-haiku"])
    args = parser.parse_args()

    if not RESULTS_CSV.exists():
        print(f"ERREUR : {RESULTS_CSV} introuvable.")
        sys.exit(1)

    print(f"Chargement du CSV : {RESULTS_CSV}")
    results = load_csv(RESULTS_CSV)
    print(f"{len(results)} lignes chargées")

    print(f"\nLogging dans MLflow ({MLFLOW_DIR})...")
    log_to_mlflow(results, args.llm)


if __name__ == "__main__":
    main()
