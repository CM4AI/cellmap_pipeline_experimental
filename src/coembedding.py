import os

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

muse_config = {
    "img_embed_run_id": "3aacc6d85ff44e26b556b4ac2ea4e3cd",
    "ppi_embed_run_id": "a55a44ba6f324866b64f1af70608b272",
    "algorithm": "muse",
    "latent_dimensions": 128,
    "n_epochs": 100,
    "jackknife_percent": 0.0,
    "dropout": 0.5,
    "triplet_margin": 0.1,
    "n_epochs_init": 100,
    "k": 10
}

pgps_config = {
    "img_embed_run_id": "3aacc6d85ff44e26b556b4ac2ea4e3cd",
    "ppi_embed_run_id": "a55a44ba6f324866b64f1af70608b272",
    "algorithm": "proteingps",
    "latent_dimensions": 128,
    "n_epochs": 100,
    "jackknife_percent": 0.0,
    "dropout": 0.5,
    "triplet_margin": 0.2,
    "l2_norm": False,
    "lambda_triplet": 1.0,
    "mean_losses": False,
    "batch_size": 16,
    "lambda_reconstruction": 1.0,
    "lambda_l2": 0.001,
    "learn_rate": 1e-4,
    "hidden_size_1": 512,
    "hidden_size_2": 256,
    "negative_from_batch": False
}

configs = [muse_config, pgps_config]

with mlflow.start_run() as parent_run:
    mlflow.set_tag("pipeline_step", "cellmaps_coembedding_parent")
    mlflow.log_param("n_trials", len(configs))

    for config in configs:
        with mlflow.start_run(nested=True) as child_run:
            img_emb_path = f"data/embedding/images/{config['img_embed_run_id']}"
            ppi_emb_path = f"data/embedding/ppi/{config['ppi_embed_run_id']}"
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
                                               embeddings=[ppi_emb_path, img_emb_path],
                                               embedding_names=["ppi", "img"],
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
                                                     embeddings=[ppi_emb_path, img_emb_path],
                                                     embedding_names=["ppi", "img"],
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