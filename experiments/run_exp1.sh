#!/usr/bin/env bash
#SBATCH --account=kumarv
#SBATCH --job-name=PFNs
#SBATCH --output=logs/eval_out_%A_%a.txt
#SBATCH --error=logs/eval_err_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --partition=msigpu #kgml03
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1

source ~/.bashrc
conda activate pfns

python3 experiment_1.py --n_seeds 3
