#!/bin/bash
#SBATCH --job-name=download_ppi
#SBATCH --output=download_ppi.o%j
#SBATCH --error=download_ppi.e%j
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate cellmaps_dev

python src/download_ppi_data.py 