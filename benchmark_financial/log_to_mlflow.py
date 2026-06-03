import pandas as pd
import mlflow
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "benchmark_results.csv")
MLRUNS_PATH = os.path.join(os.path.dirname(__file__), "mlruns")
EXPERIMENT_NAME = "benchmark_financial_per_query"

mlflow.set_tracking_uri(f"file:///{MLRUNS_PATH}")
mlflow.set_experiment(EXPERIMENT_NAME)

df = pd.read_csv(CSV_PATH)

print(f"Logging {len(df)} requêtes dans MLflow...")

for i, row in df.iterrows():
    with mlflow.start_run(run_name=row["query_id"]):
        # Tags
        mlflow.set_tags({
            "query_type": row["query_type"],
            "cpb_decision": row["cpb_decision"],
            "query_id": row["query_id"],
        })

        # Métriques numériques
        mlflow.log_metrics({
            "pii_leakage_count": float(row["pii_leakage_count"]),
            "pii_total": float(row["pii_total"]),
            "pii_leakage_binary": float(row["pii_leakage_binary"]),
            "pii_leakage_rate": float(row["pii_leakage_count"]) / float(row["pii_total"]) if row["pii_total"] > 0 else 0.0,
            "rouge_l": float(row["rouge_l"]),
            "cpb_blocked": float(row["cpb_blocked"]),
            "cpb_query_risk": float(row["cpb_query_risk"]),
            "latency_s": float(row["latency_s"]),
        })

    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(df)} runs loggés...")

print(f"\nTerminé. Lance: mlflow ui --backend-store-uri {MLRUNS_PATH}")
print("Puis ouvre: http://localhost:5000")
print(f"Expérience: {EXPERIMENT_NAME}")
