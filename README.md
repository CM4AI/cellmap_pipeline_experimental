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
3. Generate image and PPI embeddings
```
python src/embed_ppi_data.py
python src/embed_image_data.py
```
4. Get PPI and image embedding run IDs from MLFlow and add to src/coembedding.py config
```
muse_config = {
    "img_embed_run_id": "IMG_EMBED_RUN_ID",
    "ppi_embed_run_id": "PPI_EMBED_RUN_ID",
    "algorithm": "muse",
    "latent_dimensions": 128,
    "n_epochs": 100,
    "jackknife_percent": 0.0,
    "dropout": 0.5,
    "triplet_margin": 0.1,
    "n_epochs_init": 100,
    "k": 10
}
```
5. Run co-embedding
```
python src/coembedding.py
```
6. Get coembedding run ID from MLFlow and add to src/generate_hierarchy.py config
```
configs = [{
    "coembed_run_id": "COEMBED_RUN_ID",
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
}]
```
7. Generate cell map hierarchy
```
python src/generate_hierarchy.py
```
8. Get hierarchy run ID from MLFlow and add to src/eval_hierarchy.py config
```
configs = [{
    "hierarchy_run_id": "HIERARCHY_RUN_ID",
    "max_fdr": 0.05,
    "min_jaccard_index": 0.1,
    "min_comp_size": 4,
    "corum": '633291aa-6e1d-11ef-a7fd-005056ae23aa',
    "go_cc": '6722d74d-6e20-11ef-a7fd-005056ae23aa',
    "hpa": '68c2f2c0-6e20-11ef-a7fd-005056ae23aa'
}]
```
9. Run hierarchy evaluation
```
python src/eval_hierarchy.py
```