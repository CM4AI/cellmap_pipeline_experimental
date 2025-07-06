from cellmaps_hierarchyeval.runner import CellmapshierarchyevalRunner
import mlflow
from fairops.mlops.autolog import LoggerFactory
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

def log_artifact_directory(dir_path):
    dir_path = os.path.abspath(dir_path)
    for root, _, files in os.walk(dir_path):
            for file in files:
                mlflow.log_artifact(os.path.join(root, file), "rocrate")

mlflow.set_experiment("hierarchy-eval")
ml_logger = LoggerFactory.get_logger("mlflow")

configs = [{
    "hierarchy_run_id": "3c1a27fd8bc24f7c9be6dcc845559bab",
    "max_fdr": 0.05,
    "min_jaccard_index": 0.1,
    "min_comp_size": 4,
    "corum": '633291aa-6e1d-11ef-a7fd-005056ae23aa',
    "go_cc": '6722d74d-6e20-11ef-a7fd-005056ae23aa',
    "hpa": '68c2f2c0-6e20-11ef-a7fd-005056ae23aa'
}]

with mlflow.start_run() as parent_run:
    for config in configs:
        with mlflow.start_run(nested=True) as child_run:
            hiergen_dir = f"data/hierarchy/generator/{config['hierarchy_run_id']}"
            config["hierarchy_run_uri"] = get_run_uri(config['hierarchy_run_id'])

            mlflow.log_params(config)

            hiereval_dir = f"data/hierarchy/eval/{child_run.info.run_id}"
            
            eval_hierarchy = CellmapshierarchyevalRunner(
                outdir=hiereval_dir,
                hierarchy_dir=hiergen_dir,
                max_fdr=config["max_fdr"],
                min_jaccard_index=config["min_jaccard_index"],
                min_comp_size=config["min_comp_size"],
                corum=config["corum"],
                go_cc=config["go_cc"],
                hpa=config["hpa"],
            )

            eval_hierarchy.run()

            hierarchyeval_rocrate_path = log_artifact_directory(hiereval_dir)
            
            ml_logger.export_logs_as_artifact()
            mlflow.end_run()
    
    ml_logger.export_logs_as_artifact()
    mlflow.end_run()
