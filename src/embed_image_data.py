from cellmaps_image_embedding.runner import CellmapsImageEmbedder, EmbeddingGenerator
from cellmaps_image_embedding.runner import DensenetEmbeddingGenerator
import os
import zipfile
import mlflow
from fairops.mlops.autolog import LoggerFactory


def log_artifact_directory(dir_path):
    dir_path = os.path.abspath(dir_path)
    parent_dir, dir_name = os.path.split(dir_path)
    for root, _, files in os.walk(dir_path):
            for file in files:
                if file == f"{dir_name}.zip" or 'resize' in file:
                    continue

                mlflow.log_artifact(os.path.join(root, file), "rocrate")

def zip_directory(dir_path):
    dir_path = os.path.abspath(dir_path)
    parent_dir, dir_name = os.path.split(dir_path)
    zip_filename = os.path.join(dir_path, f"{dir_name}.zip")

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                # Skip the zip file if it's already in the directory
                if file == f"{dir_name}.zip" or 'resize' in file:
                    continue
                abs_file = os.path.join(root, file)
                rel_path = os.path.relpath(abs_file, dir_path)
                zipf.write(abs_file, arcname=rel_path)

    return zip_filename


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

            embed_rocrate_path = log_artifact_directory(out_dir)

            ml_logger.export_logs_as_artifact()
            mlflow.end_run()
    ml_logger.export_logs_as_artifact()
    mlflow.end_run()