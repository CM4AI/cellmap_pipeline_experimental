import os
import json

import mlflow
from mlops_helper import get_run_uri, log_artifact_directory

from cellmaps_coembedding.runner import (CellmapsCoEmbedder,
                                         EmbeddingGenerator,
                                         MuseCoEmbeddingGenerator,
                                         ProteinGPSCoEmbeddingGenerator)
from fairops.mlops.autolog import LoggerFactory


mlflow.set_experiment("coembedding")
ml_logger = LoggerFactory.get_logger("mlflow")

base_path = "data/embedding"

configs_file_path = "./configs/coembedding_configs.json"

with open (configs_file_path, 'r') as f:
    configs = json.load(f)

for config in configs:
    algorithm = config.get("algorithm", "Unknown")
    for k,v in config.items():
        if k.endswith("_run_id") and (not v or len(v.strip()) < 1):
            raise Exception(f"'{k}' for {algorithm} algorithm needs to be provided")

with mlflow.start_run() as parent_run:
    mlflow.set_tag("pipeline_step", "cellmaps_coembedding_parent")
    mlflow.log_param("n_trials", len(configs))

    for config in configs:
        with mlflow.start_run(nested=True) as child_run:
            embedding_paths = []
            embedding_names = []

            for k,v in config.items():
                if k.endswith("_embed_run_id"):
                    embedding_names.append(k.split("_")[0])
                    embedding_paths.append(f"data/embedding/{config[k]}")
        
            coemb_outdir = f"data/embedding/coembed/{child_run.info.run_id}"
            mlflow.log_params({
                "img_embed_run_id": config['img_embed_run_id'],
                "img_embed_run_uri": get_run_uri(config['img_embed_run_id']),
                "ppi_embed_run_id": config['ppi_embed_run_id'],
                "ppi_embed_run_uri": get_run_uri(config['ppi_embed_run_id'])
            })
            mlflow.set_tag("pipeline_step", "cellmaps_coembedding")

            gen = EmbeddingGenerator()

            if config["algorithm"] == 'muse':
                gen = MuseCoEmbeddingGenerator(dimensions=config["latent_dimensions"],
                                               n_epochs=config["n_epochs"],
                                               n_epochs_init=config["n_epochs_init"],
                                               jackknife_percent=config["jackknife_percent"],
                                               dropout=config["dropout"],
                                               triplet_margin=config["triplet_margin"],
                                               k=config["k"],
                                               outdir=coemb_outdir,
                                               embeddings=embedding_paths,
                                               embedding_names=embedding_names,
                                               log_fairops=True)
            elif config["algorithm"] == 'proteingps':
                gen = ProteinGPSCoEmbeddingGenerator(dimensions=config["latent_dimensions"],
                                                     n_epochs=config["n_epochs"],
                                                     jackknife_percent=config["jackknife_percent"],
                                                     dropout=config["dropout"],
                                                     l2_norm=config["l2_norm"],
                                                     mean_losses=config["mean_losses"],
                                                     lambda_reconstruction=config["lambda_reconstruction"],
                                                     lambda_l2=config["lambda_l2"],
                                                     lambda_triplet=config["triplet_margin"],
                                                     batch_size=config["batch_size"],
                                                     triplet_margin=config["triplet_margin"],
                                                     learn_rate=config["learn_rate"],
                                                     hidden_size_1=config["hidden_size_1"],
                                                     hidden_size_2=config["hidden_size_2"],
                                                     save_update_epochs=True,
                                                     negative_from_batch=config["negative_from_batch"],
                                                     outdir=coemb_outdir,
                                                     embeddings=embedding_paths,
                                                     embedding_names=embedding_names,
                                                     log_fairops=True)
                
            inputdirs = gen.get_embedding_inputdirs()
            coembedder = CellmapsCoEmbedder(
                outdir=coemb_outdir,
                inputdirs=inputdirs,
                embedding_generator=gen,
                skip_logging=True
            )

            coembedder.run()
            coembed_rocrate_path = log_artifact_directory(coemb_outdir)

            ml_logger.export_logs_as_artifact()
            mlflow.end_run()
    ml_logger.export_logs_as_artifact()
    mlflow.end_run()