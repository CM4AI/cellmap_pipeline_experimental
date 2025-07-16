import os

import mlflow
from mlops_helper import log_artifact_directory

from cellmaps_image_embedding.runner import (CellmapsImageEmbedder,
                                             DensenetEmbeddingGenerator,
                                             EmbeddingGenerator)
from fairops.mlops.autolog import LoggerFactory

mlflow.set_tracking_uri("http://127.0.0.1:5005")
mlflow.set_experiment("image_embedding")
ml_logger = LoggerFactory.get_logger("mlflow")

input_dir = "data/images"

configs = [{
    "dimensions": EmbeddingGenerator.DIMENSIONS,
    "fold": EmbeddingGenerator.DEFAULT_FOLD,
    "embedding_model": "https://github.com/CellProfiling/densenet/releases/download/v0.1.0/external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds_fold0_final.pth"
}]

with mlflow.start_run() as parent_run:
    for config in configs:
        with mlflow.start_run(nested=True) as child_run:
            out_dir = f"data/embedding/images/{child_run.info.run_id}"
            mlflow.log_params(config)

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