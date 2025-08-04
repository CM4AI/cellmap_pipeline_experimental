#!/bin/bash
#SBATCH --job-name=image_download
#SBATCH --output=image_download.o%j
#SBATCH --error=image_download.e%j
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

conda activate cellmaps_dev

python src/embed_ppi_data.py