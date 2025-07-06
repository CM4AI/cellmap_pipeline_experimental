import mlflow
import os


def get_run_uri(run_id):
    client = mlflow.tracking.MlflowClient()

    # Get the experiment ID
    upstream_run = client.get_run(run_id)
    experiment_id = upstream_run.info.experiment_id

    # Build the URL (edit if you're using a custom MLflow UI base)
    mlflow_ui_base = mlflow.get_tracking_uri().rstrip("/")  # e.g. http://localhost:5000
    run_url = f"{mlflow_ui_base}/#/experiments/{experiment_id}/runs/{run_id}"

    return run_url

def log_artifact_directory(dir_path, ignore_path=None):
    dir_path = os.path.abspath(dir_path)
    for root, _, files in os.walk(dir_path):
        for file in files:
            if ignore_path is not None and ignore_path in root:
                continue
            mlflow.log_artifact(os.path.join(root, file), "rocrate")