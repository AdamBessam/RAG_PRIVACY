import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("mlruns")
client = MlflowClient()

experiment = client.get_experiment_by_name("rag_privacy_benchmark")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="params.attack = 'prompt_injection'",
)

for run in runs:
    client.delete_run(run.info.run_id)
    print(f"Supprimé : {run.info.run_id}")

print(f"✅ {len(runs)} runs supprimés.")