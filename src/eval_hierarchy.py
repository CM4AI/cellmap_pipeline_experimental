import os

import mlflow
from mlops_helper import get_run_uri, log_artifact_directory

from cellmaps_hierarchyeval.runner import CellmapshierarchyevalRunner
from fairops.mlops.autolog import LoggerFactory


mlflow.set_experiment("hierarchyeval")
ml_logger = LoggerFactory.get_logger("mlflow")

configs = [{
    "hierarchy_run_id": "3ded3377cc1841008ce1aa01351ae9ed",
    "max_fdr": 0.05,
    "min_jaccard_index": 0.1,
    "min_comp_size": 4,
    "corum": '633291aa-6e1d-11ef-a7fd-005056ae23aa',
    "go_cc": '6722d74d-6e20-11ef-a7fd-005056ae23aa',
    "hpa": '68c2f2c0-6e20-11ef-a7fd-005056ae23aa'
}]

with mlflow.start_run() as parent_run:
    mlflow.set_tag("pipeline_step", "cellmaps_hierarchyeval_parent")
    mlflow.log_param("n_trials", len(configs))

    for config in configs:
        with mlflow.start_run(nested=True) as child_run:
            hiergen_dir = f"data/hierarchy/generator/{config['hierarchy_run_id']}"
            config["hierarchy_run_uri"] = get_run_uri(config['hierarchy_run_id'])

            mlflow.set_tag("pipeline_step", "cellmaps_hierarchyeval")
            mlflow.log_param("hierarchy_run_id", config["hierarchy_run_id"])
            mlflow.log_param("hierarchy_run_uri", config["hierarchy_run_uri"])

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
                log_fairops=True
            )

            eval_hierarchy.run()

            hierarchyeval_rocrate_path = log_artifact_directory(hiereval_dir)
            
            ml_logger.export_logs_as_artifact()
            mlflow.end_run()
    
    ml_logger.export_logs_as_artifact()
    mlflow.end_run()
