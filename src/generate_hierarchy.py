import os

import mlflow
from mlops_helper import get_run_uri

from cellmaps_generate_hierarchy.hcx import HCXFromCDAPSCXHierarchy
from cellmaps_generate_hierarchy.hierarchy import CDAPSHiDeFHierarchyGenerator
from cellmaps_generate_hierarchy.maturehierarchy import HiDeFHierarchyRefiner
from cellmaps_generate_hierarchy.ppi import CosineSimilarityPPIGenerator
from cellmaps_generate_hierarchy.runner import CellmapsGenerateHierarchy
from fairops.mlops.autolog import LoggerFactory


mlflow.set_experiment("hierarchy")
ml_logger = LoggerFactory.get_logger("mlflow")

configs = [{
    "coembed_run_id": "1a5835e81b8c4d72be2907e3e053a5d9",
    "algorithm": "leiden",
    "k": 10,
    "maxres": 80,
    "containment_threshold": 0.75,
    "jaccard_threshold": 0.9,
    "min_diff": 1,
    "min_system_size": 4,
    "ppi_cutoffs": [0.001, 0.002, 0.003],
    "parent_ppi_cutoff": 0.1,
    "bootstrap_edges": 0
}]

with mlflow.start_run() as parent_run:
    for config in configs:
        with mlflow.start_run(nested=True) as child_run:
            coembed_dir = f"data/embedding/coembed/{config['coembed_run_id']}"
            config["coembed_run_uri"] = get_run_uri(config['coembed_run_id'])
            
            mlflow.log_params(config)

            hiergen_dir = f"data/hierarchy/generator/{child_run.info.run_id}"

            ppigen = CosineSimilarityPPIGenerator(
                embeddingdirs=[coembed_dir],
                cutoffs=config["ppi_cutoffs"]
            )
            
            refiner = HiDeFHierarchyRefiner(
                ci_thre=config["containment_threshold"],
                ji_thre=config["jaccard_threshold"],
                min_term_size=config["min_system_size"],
                min_diff=config["min_diff"]
            )
            converter = HCXFromCDAPSCXHierarchy()
            hiergen = CDAPSHiDeFHierarchyGenerator(
                refiner=refiner,
                hcxconverter=converter,
                hierarchy_parent_cutoff=config["parent_ppi_cutoff"],
                bootstrap_edges=config["bootstrap_edges"]
            )

            generate_hierarchy = CellmapsGenerateHierarchy(
                outdir=hiergen_dir,
                inputdirs=coembed_dir,
                ppigen=ppigen,
                hiergen=hiergen,
                algorithm=config["algorithm"],
                maxres=config["maxres"],
                k=config["k"]
            )

            generate_hierarchy.run()

            hierarchy_rocrate_path = log_artifact_directory(hiergen_dir)

            ml_logger.export_logs_as_artifact()
            mlflow.end_run()
    
    ml_logger.export_logs_as_artifact()
    mlflow.end_run()
