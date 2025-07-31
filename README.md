# Experimental Cellmap Pipeline with FAIROps/MLFlow Logging

## Development Environment Setup
1. Install miniconda
2. Create and activate environment
```
conda create -n cellmaps_dev python=3.10
conda activate cellmaps_dev
```
3. Clone repositories
```
git clone git@github.com:CM4AI/cellmap_pipeline_experimental.git
git clone git@github.com:wadeschulz/cellmaps_utils_experimental.git
git clone git@github.com:wadeschulz/cellmaps_imagedownloader_experimental.git
git clone git@github.com:wadeschulz/cellmaps_ppidownloader_experimental.git
git clone git@github.com:wadeschulz/cellmaps_image_embedding_experimental.git
git clone git@github.com:wadeschulz/cellmaps_ppi_embedding_experimental.git
git clone git@github.com:wadeschulz/cellmaps_coembedding_experimental.git
git clone git@github.com:wadeschulz/cellmaps_generate_hierarchy_experimental.git
git clone git@github.com:wadeschulz/cellmaps_hierarchyeval_experimental.git
git clone git@github.com:acomphealth/fairops.git
```
4. Install libraries from cloned repos
```
pip install -e cellmaps_utils_experimental
pip install -e cellmaps_imagedownloader_experimental
pip install -e cellmaps_ppidownloader_experimental
pip install -e cellmaps_image_embedding_experimental
pip install -e cellmaps_ppi_embedding_experimental
pip install -e cellmaps_coembedding_experimental
pip install -e cellmaps_generate_hierarchy_experimental
pip install -e cellmaps_hierarchyeval_experimental
pip install -e fairops
pip install mlflow
```

## Starting MLFlow Server
To visualize MLFlow logging, open a second terminal, change to the cellmap_pipeline_experimental directory, activate the environment, and start MLFlow server
```
cd cellmap_pipeline_experimental
conda activate cellmaps_dev
mlflow server
```

## Running the Examples
1. Change to the cellmap_pipeline_experimental directory and activate the environment
```
cd cellmap_pipeline_experimental
conda activate cellmaps_dev
```
2. Download example data
```
python src/download_ppi_data.py
python src/download_image_data.py
```
3. Get PPI and image downloader runIDs from MLFlow and add to configs/embed_ppi_configs.json and configs/embed_image_configs.json
    1) configs/embed_ppi_configs.json
    ```
    {
        "ppi_downloader_run_id": "",
        "dimensions": 1024,
        "walk_length": 80,
        "num_walks": 10,
        "workers": 8,
        "p": 2,
        "q": 1,
        "seed": null,
        "window": 10,
        "min_count": 0,
        "sg": 1,
        "epochs": 1
    }
    ```
    2) configs/embed_image_configs.json
    ```
    {
    "image_downloader_run_id": "IMG_DOWNLOADER_RUN_ID",
    "dimensions": 1024,
    "fold": 1,
    "embedding_model": "https://github.com/CellProfiling/densenet/releases/download/v0.1.0/external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds_fold0_final.pth"
    }
    ```
4. Generate PPI and image embeddings
```
python src/embed_ppi_data.py
python src/embed_image_data.py
```
5. Get PPI and image embedding run IDs from MLFlow and add to configs/coembedding_configs.json
```
[
    {
    "img_embed_run_id": "",
    "ppi_embed_run_id": "",
    "algorithm": "muse",
    "latent_dimensions": 128,
    "n_epochs": 100,
    "jackknife_percent": 0.0,
    "dropout": 0.5,
    "triplet_margin": 0.1,
    "n_epochs_init": 100,
    "k": 10
    },

    {
    "img_embed_run_id": "",
    "ppi_embed_run_id": "",
    "algorithm": "proteingps",
    "latent_dimensions": 128,
    "n_epochs": 100,
    "jackknife_percent": 0.0,
    "dropout": 0.5,
    "triplet_margin": 0.2,
    "l2_norm": false,
    "lambda_triplet": 1.0,
    "mean_losses": false,
    "batch_size": 16,
    "lambda_reconstruction": 1.0,
    "lambda_l2": 0.001,
    "learn_rate": 1e-4,
    "hidden_size_1": 512,
    "hidden_size_2": 256,
    "negative_from_batch": false
    }
]
```
6. Run co-embedding
```
python src/coembedding.py
```
7. Get coembedding run IDs from MLFlow and add to configs/generate_hierarchy_configs.json
```
[
    {
        "coembed_run_id": "",
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
    },
    {
        "coembed_run_id": "",
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
    }
]
```
8. Generate cell map hierarchy
```
python src/generate_hierarchy.py
```
9. Get hierarchy run IDs from MLFlow and add to configs/eval_hierarchy_configs.json
```
[
    {
        "hierarchy_run_id": "",
        "max_fdr": 0.05,
        "min_jaccard_index": 0.1,
        "min_comp_size": 4,
        "corum": "633291aa-6e1d-11ef-a7fd-005056ae23aa",
        "go_cc": "6722d74d-6e20-11ef-a7fd-005056ae23aa",
        "hpa": "68c2f2c0-6e20-11ef-a7fd-005056ae23aa"
    },
    {
        "hierarchy_run_id": "",
        "max_fdr": 0.05,
        "min_jaccard_index": 0.1,
        "min_comp_size": 4,
        "corum": "633291aa-6e1d-11ef-a7fd-005056ae23aa",
        "go_cc": "6722d74d-6e20-11ef-a7fd-005056ae23aa",
        "hpa": "68c2f2c0-6e20-11ef-a7fd-005056ae23aa"
    }
]
```
10. Run hierarchy evaluation
```
python src/eval_hierarchy.py
```