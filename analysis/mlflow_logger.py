# analysis/mlflow_logger.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import mlflow.tracking
from datetime import datetime
from config import MLFLOW_EXPERIMENT_NAME, QUERY_LOG_MAX_CHARS


class MLflowLogger:

    def __init__(self, experiment_name: str = MLFLOW_EXPERIMENT_NAME):
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(experiment_name)

    def log_run(self,
                llm_name: str,
                rag_name: str,
                attack_name: str,
                query: str,
                response: str,
                # tokens
                tokens_prompt: int,
                tokens_completion: int,
                # métriques
                pii_leakage_rate: float,
                rouge_l: float = None,
                auc_roc: float = None,
                jailbreak_success: bool = None,
                # coût
                cost_usd: float = None,
                # métadonnées supplémentaires
                chunk_ids: list = None,
                n_chunks_retrieved: int = None,
                ) -> str:
        """
        Logue un run complet dans MLflow.
        Retourne le run_id pour référence.
        """

        run_name = f"{llm_name}__{rag_name}__{attack_name}__{datetime.now().strftime('%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:

            # --- PARAMS ---
            mlflow.log_param("llm",               llm_name)
            mlflow.log_param("rag_architecture",  rag_name)
            mlflow.log_param("attack",            attack_name)
            mlflow.log_param("query",             query[:QUERY_LOG_MAX_CHARS])
            mlflow.log_param("response_preview",  response[:QUERY_LOG_MAX_CHARS])

            # --- CHUNKS ---
            if n_chunks_retrieved is not None:
                mlflow.log_param("n_chunks_retrieved", n_chunks_retrieved)
            if chunk_ids is not None:
                mlflow.log_param("chunk_ids", str(chunk_ids[:5]))  # max 5 pour lisibilité

            # --- TOKENS ---
            mlflow.log_metric("tokens_prompt",     tokens_prompt)
            mlflow.log_metric("tokens_completion", tokens_completion)
            mlflow.log_metric("tokens_total",      tokens_prompt + tokens_completion)

            # --- COÛT ---
            if cost_usd is not None:
                mlflow.log_metric("cost_usd", cost_usd)

            # --- MÉTRIQUES DE VULNÉRABILITÉ ---
            mlflow.log_metric("pii_leakage_rate", pii_leakage_rate)

            if rouge_l is not None:
                mlflow.log_metric("rouge_l", rouge_l)

            if auc_roc is not None:
                mlflow.log_metric("auc_roc", auc_roc)

            if jailbreak_success is not None:
                mlflow.log_metric("jailbreak_success", int(jailbreak_success))

            return run.info.run_id

    def get_all_runs(self) -> "pd.DataFrame":
        """
        Récupère tous les runs de l'expérience sous forme de DataFrame.
        Utile pour construire la matrice de vulnérabilité finale.
        """
        import pandas as pd

        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

        if experiment is None:
            print("⚠️  Aucune expérience trouvée.")
            return pd.DataFrame()

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
        )

        data = []
        for run in runs:
            data.append({
                "run_id":             run.info.run_id,
                "llm":                run.data.params.get("llm"),
                "rag":                run.data.params.get("rag_architecture"),
                "attack":             run.data.params.get("attack"),
                "tokens_total":       run.data.metrics.get("tokens_total"),
                "cost_usd":           run.data.metrics.get("cost_usd", 0.0),
                "pii_leakage_rate":   run.data.metrics.get("pii_leakage_rate"),
                "rouge_l":            run.data.metrics.get("rouge_l"),
                "auc_roc":            run.data.metrics.get("auc_roc"),
                "jailbreak_success":  run.data.metrics.get("jailbreak_success"),
            })

        return pd.DataFrame(data)