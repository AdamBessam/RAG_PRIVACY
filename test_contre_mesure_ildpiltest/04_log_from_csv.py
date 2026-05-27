"""
Étape 4 — Re-logging MLflow depuis le CSV de résultats.

À utiliser quand le benchmark s'est terminé mais que MLflow n'a pas été alimenté.

Usage:
    python test_contre_mesure_ildpiltest/04_log_from_csv.py
    python test_contre_mesure_ildpiltest/04_log_from_csv.py --llm llama
"""
import csv
import shutil
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
        for col in ("naive_pii_leaked", "naive_pii_total",
                    "cpb_pii_leaked",   "cpb_pii_total", "cpb_blocked"):
            r[col] = int(r[col]) if r.get(col) not in ("", None) else 0
        for col in ("naive_pii_rate", "cpb_pii_rate",
                    "naive_rouge_l",  "cpb_rouge_l",
                    "naive_latency_s", "cpb_latency_s", "cpb_query_risk"):
            r[col] = float(r[col]) if r.get(col) not in ("", None) else 0.0
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

        # --- NaiveRAG ground-truth ---
        naive_leaked = sum(r["naive_pii_leaked"] for r in results)
        naive_total  = sum(r["naive_pii_total"]  for r in results)
        naive_pii    = naive_leaked / naive_total if naive_total > 0 else 0.0
        naive_rl     = sum(r["naive_rouge_l"]    for r in results) / total
        naive_lat    = sum(r["naive_latency_s"]  for r in results) / total

        mlflow.log_metric("naive_pii_leaked_total", naive_leaked)
        mlflow.log_metric("naive_pii_total",        naive_total)
        mlflow.log_metric("naive_pii_leakage_rate", round(naive_pii, 4))
        mlflow.log_metric("naive_rouge_l_mean",     round(naive_rl,  4))
        mlflow.log_metric("naive_latency_mean_s",   round(naive_lat, 3))

        # --- CPB ground-truth ---
        cpb_leaked = sum(r["cpb_pii_leaked"] for r in results)
        cpb_total  = sum(r["cpb_pii_total"]  for r in results)
        cpb_pii    = cpb_leaked / cpb_total if cpb_total > 0 else 0.0
        cpb_rl     = sum(r["cpb_rouge_l"]    for r in results) / total
        cpb_block  = sum(r["cpb_blocked"]    for r in results) / total
        cpb_risk   = sum(r["cpb_query_risk"] for r in results) / total
        cpb_lat    = sum(r["cpb_latency_s"]  for r in results) / total

        mlflow.log_metric("cpb_pii_leaked_total",  cpb_leaked)
        mlflow.log_metric("cpb_pii_total",         cpb_total)
        mlflow.log_metric("cpb_pii_leakage_rate",  round(cpb_pii,   4))
        mlflow.log_metric("cpb_rouge_l_mean",      round(cpb_rl,    4))
        mlflow.log_metric("cpb_block_rate",        round(cpb_block, 4))
        mlflow.log_metric("cpb_query_risk_mean",   round(cpb_risk,  4))
        mlflow.log_metric("cpb_latency_mean_s",    round(cpb_lat,   3))

        pii_reduction = (naive_pii - cpb_pii) / naive_pii if naive_pii > 0 else 0.0
        mlflow.log_metric("pii_reduction_rate", round(pii_reduction, 4))

        # --- Par type de query ---
        query_types = sorted(set(r["query_type"] for r in results))
        for qtype in query_types:
            subset = [r for r in results if r["query_type"] == qtype]
            n = len(subset)
            if n == 0:
                continue
            s_naive_leaked = sum(r["naive_pii_leaked"] for r in subset)
            s_naive_total  = sum(r["naive_pii_total"]  for r in subset)
            s_cpb_leaked   = sum(r["cpb_pii_leaked"]   for r in subset)
            s_cpb_total    = sum(r["cpb_pii_total"]    for r in subset)
            mlflow.log_metric(f"{qtype}_naive_pii_rate",  round(s_naive_leaked / s_naive_total, 4) if s_naive_total > 0 else 0.0)
            mlflow.log_metric(f"{qtype}_cpb_pii_rate",    round(s_cpb_leaked   / s_cpb_total,   4) if s_cpb_total   > 0 else 0.0)
            mlflow.log_metric(f"{qtype}_cpb_block_rate",  round(sum(r["cpb_blocked"]    for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_cpb_risk_mean",   round(sum(r["cpb_query_risk"] for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_naive_rouge_l",   round(sum(r["naive_rouge_l"]  for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_cpb_rouge_l",     round(sum(r["cpb_rouge_l"]    for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_n_queries",       n)

        # --- Table complète (queries + réponses + métriques par ligne) ---
        # On écrit directement dans mlruns pour éviter les problèmes de chemin Linux/Windows
        try:
            import json as _json
            run        = mlflow.active_run()
            run_id     = run.info.run_id
            exp_id     = run.info.experiment_id
            art_dir    = Path(MLFLOW_DIR) / exp_id / run_id / "artifacts" / "results"
            art_dir.mkdir(parents=True, exist_ok=True)

            # Copie du CSV brut
            shutil.copy(str(RESULTS_CSV), str(art_dir / RESULTS_CSV.name))

            # Table JSON (colonnes + lignes) lisible dans MLflow UI
            cols = list(results[0].keys())
            table = {
                "columns": cols,
                "data":    [[r.get(c, "") for c in cols] for r in results],
            }
            with open(art_dir / "queries_responses.json", "w", encoding="utf-8") as fj:
                _json.dump(table, fj, ensure_ascii=False, indent=2)

            print(f"  OK - CSV + table JSON copies dans : {art_dir}")
        except Exception as e:
            print(f"  (artifact non copié : {e})")

        print(f"\n{'='*55}")
        print(f"  RÉSULTATS — {total} queries")
        print(f"{'='*55}")
        print(f"  {'Métrique':<30} {'NaiveRAG':>10}  {'CPB':>10}")
        print(f"  {'-'*55}")
        print(f"  {'PII leakage rate (GT)':<30} {naive_pii:>10.1%}  {cpb_pii:>10.1%}")
        print(f"  {'PII leaked / total':<30} {naive_leaked}/{naive_total}  {cpb_leaked}/{cpb_total}")
        print(f"  {'ROUGE-L moyen':<30} {naive_rl:>10.4f}  {cpb_rl:>10.4f}")
        print(f"  {'Block rate':<30} {'—':>10}  {cpb_block:>10.1%}")
        print(f"  {'-'*55}")
        print(f"  Réduction PII grâce au CPB : {pii_reduction:.1%}")
        print(f"{'='*55}")
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
