#!/usr/bin/env bash
#SBATCH --account=kumarv
#SBATCH --job-name=PFNs
#SBATCH --output=logs/eval_out_%A_%a.txt
#SBATCH --error=logs/eval_err_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --partition=msismall 
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=1

source ~/.bashrc
conda activate pfns

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# remove the stale miniconda path
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v miniconda3_4.8.3 | tr '\n' ':')

python3 sample_layouts.py --n_runs 1000
