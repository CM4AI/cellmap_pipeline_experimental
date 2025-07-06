import os

import mlflow
import networkx as nx

from cellmaps_ppi_embedding.runner import (CellMapsPPIEmbedder,
                                           EmbeddingGenerator,
                                           Node2VecEmbeddingGenerator)
from fairops.mlops.autolog import LoggerFactory


def log_artifact_directory(dir_path):
    dir_path = os.path.abspath(dir_path)
    parent_dir, dir_name = os.path.split(dir_path)
    for root, _, files in os.walk(dir_path):
            for file in files:
                mlflow.log_artifact(os.path.join(root, file), "rocrate")

mlflow.set_experiment("ppi_embedding")
ml_logger = LoggerFactory.get_logger("mlflow")

input_dir = "data/ppi"

configs = [{
    "dimensions": 1024,
    "walk_length": 80,
    "num_walks": 10,
    "workers": 8,
    "p": 2,
    "q": 1,
    "seed": None,
    "window": 10,
    "min_count": 0,
    "sg": 1,
    "epochs": 1
}]

with mlflow.start_run() as parent_run:
    for config in configs:
        with mlflow.start_run(nested=True) as child_run:
            out_dir = f"data/embedding/ppi/{child_run.info.run_id}"
            
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