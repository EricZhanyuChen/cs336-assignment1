#!/bin/bash
#SBATCH --job-name=owt_wt
#SBATCH --output=../logs/owt_wt_%j.out
#SBATCH --error=../logs/owt_wt_%j.err
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=32G
#SBATCH --gres=gpu:a100:1 --time=04:00:00 --partition=gpu
# TODO: Add weight tying to transformer.py, then run this
cd /home5/s6398820/projects/cs336/assignment1
module purge; module load CUDA/11.8.0; source $HOME/.local/bin/env
# TORCH_COMPILE_DISABLE=1 uv run python cs336_basics/train.py \
#     (same as baseline but with weight tying in model)
echo "NOT YET IMPLEMENTED"
