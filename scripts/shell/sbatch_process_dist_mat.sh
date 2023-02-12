#!/bin/bash
#SBATCH -c 1
#SBATCH -p priority
#SBATCH -t 0-12:00
#SBATCH --mem=100G
#SBATCH -e sbatch_logs/process_dist_mat_%j.err

module load miniconda3/4.10.3

conda activate tcrbind

python scripts/run329_process.py