import mlflow
from fairops.mlops.autolog import LoggerFactory
import os
from cellmaps_coembedding.runner import EmbeddingGenerator, ProteinGPSCoEmbeddingGenerator
from cellmaps_coembedding.runner import MuseCoEmbeddingGenerator
from cellmaps_coembedding.runner import CellmapsCoEmbedder


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

mlflow.set_experiment("coembedding")
ml_logger = LoggerFactory.get_logger("mlflow")

base_path = "data/embedding"

muse_config = {
    "img_embed_run_id": "55a72e22d8094436bdb88ddf6cfef081",
    "ppi_embed_run_id": "425a708b21984258961b4927a0d9d0ff",
    "algorithm": "muse",
    "latent_dimensions": 128,
    "n_epochs": 100,
    "jackknife_percent": 0.0,
    "dropout": 0.5,
    "triplet_margin": 0.1,
    "n_epochs_init": 100, # Only for Muse
    "k": 10
}

# ProteinGPS not working with installed version of tf
# pgps_config = {
#     "algorithm": "proteingps",
#     "latent_dimensions": 128,
#     "n_epochs": 100,
#     "jackknife_percent": 0.0,
#     "dropout": 0.5,
#     "triplet_margin": 0.2,
#     "l2_norm": False, # Only for ProteinGPS
#     "lambda_triplet": 1.0,
#     "mean_losses": False,
#     "batch_size": 16,
#     "lambda_reconstruction": 1.0,
#     "lambda_l2": 0.001,
#     "learn_rate": 1e-4,
#     "hidden_size_1": 512,
#     "hidden_size_2": 256,
#     "negative_from_batch": False
# }

configs = [muse_config]

with mlflow.start_run() as parent_run:
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
                                                     l2_norm=["l2_norm"],
                                                     mean_losses=["mean_losses"],
                                                     lambda_reconstruction=["lambda_reconstruction"],
                                                     lambda_l2=["lambda_l2"],
                                                     lambda_triplet=["triplet_margin"],
                                                     batch_size=["batch_size"],
                                                     triplet_margin=["triplet_margin"],
                                                     learn_rate=["learn_rate"],
                                                     hidden_size_1=["hidden_size_1"],
                                                     hidden_size_2=["hidden_size_2"],
                                                     save_update_epochs=True,
                                                     negative_from_batch=["negative_from_batch"],
                                                     outdir=coemb_outdir,
                                                     embeddings=[ppi_emb_path, img_emb_path],
                                                     embedding_names=["ppi", "img"])
                
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