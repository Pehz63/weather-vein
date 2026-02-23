#!/usr/bin/env bash
#SBATCH --account=kumarv
#SBATCH --job-name=EVAL
#SBATCH --output=logs/eval_out_%A_%a.txt
#SBATCH --error=logs/eval_err_%A_%a.txt
#SBATCH --time=06:30:00
#SBATCH --partition=msigpu #msigpu #kgml03
#SBATCH --gres=gpu:3 #a100:1 
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=1

source ~/.bashrc
conda activate ppo

python3 sample_efficiency.py --n_seeds 3
