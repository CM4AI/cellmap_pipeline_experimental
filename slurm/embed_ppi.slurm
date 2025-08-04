#!/bin/bash
#SBATCH --job-name=embed_ppi
#SBATCH --output=embed_ppi.o%j
#SBATCH --error=embed_ppi.e%j
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

soource ~/.bashrc
conda activate cellmaps_dev

python src/embed_ppi_data.py