import json
from cellmaps_image_embedding.runner import (CellmapsImageEmbedder,
                                             DensenetEmbeddingGenerator,
                                             EmbeddingGenerator)

configs_file_path = './configs/embed_image_configs.json'

configs = []
with open (configs_file_path, 'r') as f:
    configs.append(json.load(f))

for config in configs:
    if config.get("dimensions") == "USE_DEFAULT_DIMENSIONS":
        config["dimensions"] = EmbeddingGenerator.DIMENSIONS

    if config.get("fold") == "USE_DEFAULT_FOLD":
        config["fold"] = EmbeddingGenerator.DEFAULT_FOLD

    for k, v in config.items():
        if k.endswith("_run_id") and (not v or len(v.strip()) < 1):
            raise Exception (f"'{k}' needs to be provided")

print(configs)