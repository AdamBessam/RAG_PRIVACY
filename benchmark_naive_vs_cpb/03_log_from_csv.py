"""
Étape 3 — Re-log MLflow depuis le CSV existant
================================================
Lit benchmark_results.csv et logue TOUT dans MLflow :
  - Métriques agrégées (PII leakage rate, block rate, signaux CPB...)
  - Métriques par type de query
  - Table interactive (questions + réponses visibles dans l'UI MLflow)
  - CSV en artifact

Usage:
    python benchmark_naive_vs_cpb/03_log_from_csv.py
    python benchmark_naive_vs_cpb/03_log_from_csv.py --llm llama
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import mlflow

from benchmark_naive_vs_cpb.config import (
    RESULTS_CSV, MLFLOW_DIR, MLFLOW_EXPERIMENT,
)


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  {len(rows)} lignes lues depuis {path.name}")
    return rows


def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_int(val, default=0) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def log_to_mlflow(results: list[dict], llm_name: str):
    mlflow_uri = f"file:///{MLFLOW_DIR.replace(chr(92), '/')}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = f"naive_instruction_vs_CPB_{llm_name}_from_csv"
    with mlflow.start_run(run_name=run_name):

        total = len(results)
        mlflow.log_param("llm",         llm_name)
        mlflow.log_param("n_queries",   total)
        mlflow.log_param("dataset",     "ildpil/text-anonymization-benchmark")
        mlflow.log_param("split",       "test")
        mlflow.log_param("source",      "re-logged from benchmark_results.csv")
        mlflow.log_param("condition_a", "NaiveRAG + instruction naïve")
        mlflow.log_param("condition_b", "CPBNaiveRAG (contre-mesure complète)")

        # ── Condition A ───────────────────────────────────────────────────────
        naive_leaked = sum(safe_int(r["naive_pii_leaked"]) for r in results)
        naive_total  = sum(safe_int(r["naive_pii_total"])  for r in results)
        naive_rate   = naive_leaked / naive_total if naive_total > 0 else 0.0
        naive_lat    = sum(safe_float(r["naive_latency_s"]) for r in results) / total

        mlflow.log_metric("naive_pii_leaked_total",   naive_leaked)
        mlflow.log_metric("naive_pii_total",          naive_total)
        mlflow.log_metric("naive_pii_leakage_rate",   round(naive_rate, 4))
        mlflow.log_metric("naive_latency_mean_s",     round(naive_lat,  3))

        # ── Condition B ───────────────────────────────────────────────────────
        cpb_leaked = sum(safe_int(r["cpb_pii_leaked"])  for r in results)
        cpb_total  = sum(safe_int(r["cpb_pii_total"])   for r in results)
        cpb_rate   = cpb_leaked / cpb_total if cpb_total > 0 else 0.0
        cpb_block  = sum(safe_int(r["cpb_blocked"])     for r in results) / total
        cpb_lat    = sum(safe_float(r["cpb_latency_s"]) for r in results) / total

        mlflow.log_metric("cpb_pii_leaked_total",     cpb_leaked)
        mlflow.log_metric("cpb_pii_total",            cpb_total)
        mlflow.log_metric("cpb_pii_leakage_rate",     round(cpb_rate,  4))
        mlflow.log_metric("cpb_block_rate",           round(cpb_block, 4))
        mlflow.log_metric("cpb_latency_mean_s",       round(cpb_lat,   3))

        # ── Signaux CPB moyens ────────────────────────────────────────────────
        def mean(key):
            vals = [safe_float(r.get(key, 0)) for r in results]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        mlflow.log_metric("cpb_risk_mean",            mean("cpb_query_risk"))
        mlflow.log_metric("cpb_s1_ner_mean",          mean("cpb_s1_ner"))
        mlflow.log_metric("cpb_s2_extractive_mean",   mean("cpb_s2_extractive"))
        mlflow.log_metric("cpb_s3_jailbreak_mean",    mean("cpb_s3_jailbreak"))
        mlflow.log_metric("cpb_s4_session_mean",      mean("cpb_s4_session"))
        mlflow.log_metric("cpb_s5_semantic_mean",     mean("cpb_s5_semantic"))
        mlflow.log_metric("cpb_sad_detected_rate",    round(
            sum(safe_int(r.get("cpb_sad_detected", 0)) for r in results) / total, 4))
        mlflow.log_metric("cpb_sad_confidence_mean",  mean("cpb_sad_confidence"))
        mlflow.log_metric("cpb_chunks_masked_mean",   mean("cpb_n_chunks_masked"))

        # ── Réduction PII ─────────────────────────────────────────────────────
        pii_reduction = (naive_rate - cpb_rate) / naive_rate if naive_rate > 0 else 0.0
        mlflow.log_metric("pii_reduction_vs_naive_instruction", round(pii_reduction, 4))

        # ── Métriques par type de query ───────────────────────────────────────
        query_types = sorted(set(r["query_type"] for r in results))
        for qtype in query_types:
            subset = [r for r in results if r["query_type"] == qtype]
            n = len(subset)
            if n == 0:
                continue
            s_naive_leaked = sum(safe_int(r["naive_pii_leaked"]) for r in subset)
            s_naive_total  = sum(safe_int(r["naive_pii_total"])  for r in subset)
            s_cpb_leaked   = sum(safe_int(r["cpb_pii_leaked"])   for r in subset)
            s_cpb_total    = sum(safe_int(r["cpb_pii_total"])    for r in subset)

            mlflow.log_metric(f"{qtype}_naive_pii_rate",
                round(s_naive_leaked / s_naive_total, 4) if s_naive_total > 0 else 0.0)
            mlflow.log_metric(f"{qtype}_cpb_pii_rate",
                round(s_cpb_leaked / s_cpb_total, 4) if s_cpb_total > 0 else 0.0)
            mlflow.log_metric(f"{qtype}_cpb_block_rate",
                round(sum(safe_int(r["cpb_blocked"])      for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_cpb_risk_mean",
                round(sum(safe_float(r["cpb_query_risk"]) for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_cpb_s3_jailbreak_mean",
                round(sum(safe_float(r["cpb_s3_jailbreak"]) for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_cpb_sad_detected_rate",
                round(sum(safe_int(r.get("cpb_sad_detected", 0)) for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_n_queries", n)

        # ── Table interactive (questions + réponses visibles dans l'UI) ───────
        # mlflow.log_table() → artefact JSON affiché comme tableau dans l'UI MLflow
        table_data = {
            "query_id":           [r.get("query_id",   "")         for r in results],
            "query_type":         [r.get("query_type", "")         for r in results],
            "query":              [r.get("query",      "")[:300]   for r in results],
            # ── Condition A
            "naive_response":     [r.get("naive_response", "")[:600] for r in results],
            "naive_pii_leaked":   [safe_int(r.get("naive_pii_leaked",   0)) for r in results],
            "naive_pii_total":    [safe_int(r.get("naive_pii_total",    0)) for r in results],
            "naive_pii_rate":     [safe_float(r.get("naive_pii_rate",   0)) for r in results],
            # ── Condition B
            "cpb_response":       [r.get("cpb_response", "")[:600] for r in results],
            "cpb_pii_leaked":     [safe_int(r.get("cpb_pii_leaked",    0)) for r in results],
            "cpb_pii_total":      [safe_int(r.get("cpb_pii_total",     0)) for r in results],
            "cpb_pii_rate":       [safe_float(r.get("cpb_pii_rate",    0)) for r in results],
            "cpb_blocked":        [safe_int(r.get("cpb_blocked",       0)) for r in results],
            "cpb_global_decision":[r.get("cpb_global_decision", "")        for r in results],
            # ── Signaux CPB
            "cpb_query_risk":     [safe_float(r.get("cpb_query_risk",  0)) for r in results],
            "cpb_s1_ner":         [safe_float(r.get("cpb_s1_ner",      0)) for r in results],
            "cpb_s2_extractive":  [safe_float(r.get("cpb_s2_extractive", 0)) for r in results],
            "cpb_s3_jailbreak":   [safe_float(r.get("cpb_s3_jailbreak", 0)) for r in results],
            "cpb_s4_session":     [safe_float(r.get("cpb_s4_session",   0)) for r in results],
            "cpb_s5_semantic":    [safe_float(r.get("cpb_s5_semantic",  0)) for r in results],
            "cpb_sad_detected":   [safe_int(r.get("cpb_sad_detected",   0)) for r in results],
            "cpb_sad_decision":   [r.get("cpb_sad_decision", "")            for r in results],
            "cpb_sad_confidence": [safe_float(r.get("cpb_sad_confidence", 0)) for r in results],
            "cpb_n_chunks_masked":[safe_int(r.get("cpb_n_chunks_masked", 0)) for r in results],
        }
        mlflow.log_table(table_data, artifact_file="results/query_responses_table.json")
        print("  Table interactive (questions + réponses) : ✓")

        # ── CSV brut en artifact ──────────────────────────────────────────────
        mlflow.log_artifact(str(RESULTS_CSV), artifact_path="results")
        print(f"  CSV artifact : {RESULTS_CSV.name} ✓")

    print(f"\n  Expérience MLflow : {MLFLOW_EXPERIMENT}")
    print(f"  Tracking URI      : {MLFLOW_DIR}")
    print(f"\nPour visualiser : mlflow ui --backend-store-uri benchmark_naive_vs_cpb/mlruns")
    print(f"Puis ouvre      : http://127.0.0.1:5000")


def print_summary(results: list[dict], llm_name: str):
    total = len(results)
    naive_leaked = sum(safe_int(r["naive_pii_leaked"]) for r in results)
    naive_total  = sum(safe_int(r["naive_pii_total"])  for r in results)
    naive_rate   = naive_leaked / naive_total if naive_total > 0 else 0.0
    cpb_leaked   = sum(safe_int(r["cpb_pii_leaked"])   for r in results)
    cpb_total    = sum(safe_int(r["cpb_pii_total"])    for r in results)
    cpb_rate     = cpb_leaked / cpb_total if cpb_total > 0 else 0.0
    cpb_block    = sum(safe_int(r["cpb_blocked"])      for r in results) / total
    reduction    = (naive_rate - cpb_rate) / naive_rate * 100 if naive_rate > 0 else 0.0

    print(f"\n{'='*65}")
    print(f"  RÉSULTATS — {total} queries — LLM : {llm_name}")
    print(f"{'='*65}")
    print(f"  {'Métrique':<35} {'Naïf':>10}  {'CPB':>10}")
    print(f"  {'-'*55}")
    print(f"  {'PII leakage rate':<35} {naive_rate:>10.1%}  {cpb_rate:>10.1%}")
    print(f"  {'PII leaked / total':<35} {naive_leaked}/{naive_total}  {cpb_leaked}/{cpb_total}")
    print(f"  {'Block rate (CPB seulement)':<35} {'—':>10}  {cpb_block:>10.1%}")
    print(f"  {'-'*55}")
    print(f"  Réduction PII (CPB vs instruction naïve) : {reduction:.1f}%")
    print(f"{'='*65}")

    print(f"\n  Détail par type de query :")
    print(f"  {'Type':<12} {'N':>4}  {'Naïf PII%':>10}  {'CPB PII%':>10}  {'CPB bloqué':>12}")
    print(f"  {'-'*55}")
    for qtype in sorted(set(r["query_type"] for r in results)):
        s  = [r for r in results if r["query_type"] == qtype]
        n  = len(s)
        nl = sum(safe_int(r["naive_pii_leaked"]) for r in s)
        nt = sum(safe_int(r["naive_pii_total"])  for r in s)
        cl = sum(safe_int(r["cpb_pii_leaked"])   for r in s)
        ct = sum(safe_int(r["cpb_pii_total"])    for r in s)
        cb = sum(safe_int(r["cpb_blocked"])      for r in s) / n
        nr = nl / nt if nt > 0 else 0.0
        cr = cl / ct if ct > 0 else 0.0
        print(f"  {qtype:<12} {n:>4}  {nr:>10.1%}  {cr:>10.1%}  {cb:>12.1%}")


def main():
    parser = argparse.ArgumentParser(description="Re-log MLflow depuis benchmark_results.csv")
    parser.add_argument("--llm", default="llama",
                        help="Nom du LLM utilisé (pour le nom du run MLflow)")
    args = parser.parse_args()

    if not RESULTS_CSV.exists():
        print(f"ERREUR : {RESULTS_CSV} introuvable.")
        print("Lancez d'abord : python benchmark_naive_vs_cpb/02_run_benchmark.py")
        sys.exit(1)

    print(f"Chargement du CSV : {RESULTS_CSV}")
    results = load_csv(RESULTS_CSV)

    if not results:
        print("ERREUR : CSV vide.")
        sys.exit(1)

    print(f"\nLogging dans MLflow ({MLFLOW_DIR})...")
    log_to_mlflow(results, args.llm)
    print_summary(results, args.llm)


if __name__ == "__main__":
    main()
