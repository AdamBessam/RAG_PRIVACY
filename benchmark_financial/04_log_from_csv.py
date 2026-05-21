"""
Etape 4 — Relogging MLflow depuis un CSV existant.

Usage:
    python benchmark_financial/04_log_from_csv.py
    python benchmark_financial/04_log_from_csv.py --csv benchmark_financial/benchmark_results.csv
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow

from benchmark_financial.config import MLFLOW_DIR, MLFLOW_EXPERIMENT


def load_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for r in rows:
        r["pii_leakage_count"]  = int(r["pii_leakage_count"])
        r["pii_leakage_binary"] = int(r["pii_leakage_binary"])
        r["rouge_l"]            = float(r["rouge_l"])
        r["cpb_blocked"]        = int(r["cpb_blocked"])
        r["cpb_query_risk"]     = float(r["cpb_query_risk"])
        r["latency_s"]          = float(r["latency_s"])
    return rows


def log_to_mlflow(results: list[dict], csv_path: Path):
    mlflow.set_tracking_uri(Path(MLFLOW_DIR).resolve().as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="cpb_financial_from_csv"):
        mlflow.log_param("source_csv",  str(csv_path))
        mlflow.log_param("n_queries",   len(results))

        total       = len(results)
        pii_total   = sum(r["pii_leakage_binary"] for r in results)
        block_total = sum(r["cpb_blocked"]        for r in results)
        rouge_vals  = [r["rouge_l"]        for r in results]
        risk_vals   = [r["cpb_query_risk"] for r in results]

        mlflow.log_metric("pii_leakage_rate",   round(pii_total   / total, 4))
        mlflow.log_metric("block_rate",          round(block_total / total, 4))
        mlflow.log_metric("rouge_l_mean",        round(sum(rouge_vals) / total, 4))
        mlflow.log_metric("cpb_query_risk_mean", round(sum(risk_vals)  / total, 4))
        mlflow.log_metric("avg_latency_s",       round(sum(r["latency_s"] for r in results) / total, 3))

        query_types = sorted(set(r["query_type"] for r in results))
        for qtype in query_types:
            subset = [r for r in results if r["query_type"] == qtype]
            n = len(subset)
            mlflow.log_metric(f"{qtype}_pii_leakage_rate", round(sum(r["pii_leakage_binary"] for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_block_rate",       round(sum(r["cpb_blocked"]        for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_rouge_l_mean",     round(sum(r["rouge_l"]            for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_risk_mean",        round(sum(r["cpb_query_risk"]     for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_n_queries",        n)

        try:
            mlflow.log_artifact(str(csv_path), artifact_path="results")
        except Exception as e:
            print(f"Warning: artifact non loggue ({e})")

    print(f"MLflow experiment : {MLFLOW_EXPERIMENT}")
    print(f"MLflow tracking   : {MLFLOW_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).parent.parent / "benchmark_financial" / "benchmark_results.csv"),
        help="Chemin vers le CSV de resultats",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERREUR : fichier introuvable — {csv_path}")
        sys.exit(1)

    print(f"Lecture de {csv_path}...")
    results = load_csv(csv_path)
    print(f"{len(results)} lignes chargees")

    total       = len(results)
    pii_rate    = sum(r["pii_leakage_binary"] for r in results) / total
    block_rate  = sum(r["cpb_blocked"]        for r in results) / total
    rouge_mean  = sum(r["rouge_l"]            for r in results) / total

    print(f"\n=== RESUME ===")
    print(f"  Queries          : {total}")
    print(f"  PII leakage rate : {pii_rate:.1%}")
    print(f"  Block rate (CPB) : {block_rate:.1%}")
    print(f"  ROUGE-L moyen    : {rouge_mean:.4f}")

    print(f"\nLogging dans MLflow ({MLFLOW_DIR})...")
    log_to_mlflow(results, csv_path)
    print("Done.")


if __name__ == "__main__":
    main()
