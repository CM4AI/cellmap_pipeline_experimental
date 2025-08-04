import os

import json
import mlflow
from mlops_helper import log_artifact_directory

from cellmaps_image_embedding.runner import (CellmapsImageEmbedder,
                                             DensenetEmbeddingGenerator,
                                             EmbeddingGenerator)
from fairops.mlops.autolog import LoggerFactory


mlflow.set_experiment("image_embedding")
ml_logger = LoggerFactory.get_logger("mlflow")

configs_file_path = "./configs/embed_image_configs.json"

configs = []
with open (configs_file_path, 'r') as f:
    configs.append(json.load(f))

for config in configs:
    for k, v in config.items():
        if k.endswith("_run_id") and (not v or len(v.strip()) < 1):
            raise Exception (f"'{k}' needs to be provided")

with mlflow.start_run() as parent_run:
    mlflow.set_tag("pipeline_step", "cellmaps_image_embedding_parent")
    mlflow.log_param("n_trials", len(configs))
    
    for config in configs:
        with mlflow.start_run(nested=True) as child_run:
            out_dir = f"data/embedding/{child_run.info.run_id}"
            mlflow.log_params(config)
            mlflow.set_tag("pipeline_step", "cellmaps_image_embedding")

            input_dir = f"data/images/{config['image_downloader_run_id']}"

            gen = DensenetEmbeddingGenerator(
                inputdir=input_dir,
                dimensions=config["dimensions"],
                outdir=out_dir,
                model_path=config["embedding_model"],
                suffix=EmbeddingGenerator.SUFFIX,
                fold=config["fold"]
            )

            img_embedder = CellmapsImageEmbedder(
                inputdir=input_dir,
                outdir=out_dir,
                embedding_generator=gen,
                skip_logging=True
            )

            img_embedder.run()

            embed_rocrate_path = log_artifact_directory(out_dir, "resize")

            ml_logger.export_logs_as_artifact()
            mlflow.end_run()
    ml_logger.export_logs_as_artifact()
    mlflow.end_run()