import os
import json

import mlflow
import networkx as nx
from mlops_helper import log_artifact_directory

from cellmaps_ppi_embedding.runner import (CellMapsPPIEmbedder,
                                           EmbeddingGenerator,
                                           Node2VecEmbeddingGenerator)
from fairops.mlops.autolog import LoggerFactory


mlflow.set_experiment("ppi_embedding")
ml_logger = LoggerFactory.get_logger("mlflow")

configs_file_path = "./configs/embed_ppi_configs.json"

configs = []
with open (configs_file_path, 'r') as f:
    configs.append(json.load(f))

for config in configs:
    for k, v in config.items():
        if k.endswith("_run_id") and (not v or len(v.strip()) < 1):
            raise Exception(f"'{k}' needs to be provided")

with mlflow.start_run() as parent_run:
    mlflow.set_tag("pipeline_step", "cellmaps_ppi_embedding_parent")
    mlflow.log_param("n_trials", len(configs))
    
    for config in configs:
        with mlflow.start_run(nested=True) as child_run:
            mlflow.set_tag("pipeline_step", "cellmaps_ppi_embedding")
            mlflow.log_param("ppi_downloader_run_id", config['ppi_downloader_run_id'])

            input_dir = f"data/ppi/{config['ppi_downloader_run_id']}"
            out_dir = f"data/embedding/{child_run.info.run_id}"
            
            gen = Node2VecEmbeddingGenerator(
                nx_network=nx.read_edgelist(
                    CellMapsPPIEmbedder.get_apms_edgelist_file(input_dir), delimiter='\t'
                ),
                dimensions=config["dimensions"],
                p=config["p"],
                q=config["q"],
                walk_length=config["walk_length"],
                num_walks=config["num_walks"],
                workers=config["workers"],
                seed=config["seed"],
                window=config["window"],
                min_count=config["min_count"],
                sg=config["sg"],
                epochs=config["epochs"],
                log_fairops=True
            )

            ppi_embedder = CellMapsPPIEmbedder(
                inputdir=input_dir,
                outdir=out_dir,
                embedding_generator=gen,
                skip_logging=True
            )
            ppi_embedder.run()

            embed_rocrate_path = log_artifact_directory(out_dir)

            ml_logger.export_logs_as_artifact()
            mlflow.end_run()
    ml_logger.export_logs_as_artifact()
    mlflow.end_run()